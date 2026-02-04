"""AgentExecutor for managing tool registration and agentic LLM interactions.

This module provides the AgentExecutor class which handles:
- Tool registration via Tool class instances
- Agentic loop with streaming support
- Tool execution with injected context
"""

import asyncio
from collections.abc import AsyncIterator
import inspect
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel

from ._logging import logger
from .exceptions import ApprovalRequired
from .providers.base import Provider
from .tools.tool import Tool
from .types import (
    AssistantMessagePart,
    MessagePart,
    StreamEndEvent,
    StreamEvent,
    TextPart,
    ToolResultEvent,
    ToolResultPart,
    ToolStreamEvent,
    ToolUseEndEvent,
    ToolUsePart,
    UserMessagePart,
)

OUTPUT_TOOL_NAME = "final_result"


class ToolCallResult(NamedTuple):
    """Result from executing a single tool call."""

    tool_name: str
    tool_call_id: str
    result: Any
    is_error: bool


class AgentExecutor[ContextT, OutputT: BaseModel]:
    """Manages tool registration, execution, and LLM interactions with streaming support.

    Type Parameters:
        ContextT: Type of context object passed to tools via Injected[ContextT]
        OutputT: Type of structured output (Pydantic model) when output_type is set

    Attributes:
        provider: The LLM provider type ('openai', 'azure-openai', 'gemini', 'anthropic')
        llm: The LLM provider instance
        output_type: Pydantic model for structured output (optional)
        output_mode: How to get structured output ('tool' or 'native')
        last_output: The last validated structured output (if output_type was set)
    """

    def __init__(
        self,
        provider: Literal["openai", "azure-openai", "gemini", "anthropic"],
        llm: Provider,
        tools: list[Tool] | None = None,
        output_type: type[OutputT] | None = None,
        output_mode: Literal["tool", "native"] = "tool",
    ):
        """Initialize the AgentExecutor.

        Args:
            provider: LLM provider type for schema formatting
            llm: LLM provider instance for chat completions
            tools: List of Tool instances to register
            output_type: Pydantic BaseModel for structured output
            output_mode: 'tool' (default) or 'native' (NotImplementedError)
        """
        self.provider = provider
        self.llm = llm
        self.output_type = output_type
        self.output_mode = output_mode
        self.last_output: OutputT | None = None

        self._tools: dict[str, Tool] = {}
        self._formatted_tools: list | None = None

        if output_type and output_mode == "native":
            raise NotImplementedError("Native output mode not yet supported. Use 'tool' mode.")

        # Create output tool schema if output_type is set
        if output_type and output_mode == "tool":
            description = (
                output_type.model_json_schema().get("description")
                or f"Return the final structured result as {output_type.__name__}"
            )
            output_tool = Tool.from_model(output_type, name=OUTPUT_TOOL_NAME, description=description)
            self._tools[output_tool.name] = output_tool

        if tools:
            for tool in tools:
                self._tools[tool.name] = tool
                logger.debug(f"Registered tool: {tool.name}")

    @property
    def tools(self) -> dict[str, Tool]:
        """Get all registered tools by name.

        Returns:
            Dictionary mapping tool names to Tool instances.
        """
        return self._tools

    def get_tools_schema(self) -> list:
        """Get tool schemas formatted for the LLM provider.

        Returns:
            List of tool schemas in provider-specific format
        """
        if self._formatted_tools is None:
            match self.provider:
                case "openai" | "azure-openai":
                    self._formatted_tools = [
                        tool.to_openai_format() for tool in self._tools.values()
                    ]
                case "gemini":
                    self._formatted_tools = [
                        tool.to_gemini_format() for tool in self._tools.values()
                    ]
                case "anthropic":
                    self._formatted_tools = [
                        tool.to_anthropic_format() for tool in self._tools.values()
                    ]
        return self._formatted_tools

    async def _invoke_tool(
        self,
        tool: Tool,
        inputs: dict[str, Any],
        context: ContextT | None,
    ) -> Any:
        """Invoke a non-streaming tool, handling context injection and sync/async dispatch.

        Args:
            tool: The Tool instance to invoke
            inputs: Tool input arguments from LLM
            context: Context to inject if tool takes_ctx

        Returns:
            The tool's return value
        """
        logger.debug(f"Executing tool: {tool.name}")
        kwargs = dict(inputs)
        if tool.takes_ctx and context is not None:
            if inspect.iscoroutinefunction(tool.__call__):
                return await tool(context, **kwargs)
            return tool(context, **kwargs)
        if inspect.iscoroutinefunction(tool.__call__):
            return await tool(**kwargs)
        return tool(**kwargs)

    def _emit_tool_result(
        self,
        tool_call: ToolUsePart,
        result: Any,
        is_error: bool,
        working_messages: list[MessagePart],
        *,
        is_terminal: bool = False,
    ) -> tuple[ToolResultEvent, ToolUseEndEvent]:
        """Build result events and append tool round-trip messages.

        Args:
            tool_call: The original tool call from the LLM
            result: The tool execution result (or error dict)
            is_error: Whether the result represents an error
            working_messages: Conversation message list to append to
            is_terminal: Whether this tool ends the agent loop

        Returns:
            Tuple of (ToolResultEvent, ToolUseEndEvent) for the caller to yield
        """
        working_messages.append(AssistantMessagePart(parts=[tool_call]))
        working_messages.append(
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id=tool_call.id,
                        name=tool_call.name,
                        parts=[TextPart(text=str(result))],
                        is_error=is_error,
                    )
                ]
            )
        )
        return (
            ToolResultEvent(
                tool_use_id=tool_call.id,
                name=tool_call.name,
                result=result,
                is_error=is_error,
                is_terminal=is_terminal,
            ),
            ToolUseEndEvent(
                type="tool_use_end",
                tool_use_id=tool_call.id,
                tool_name=tool_call.name,
            ),
        )

    async def run_stream(
        self,
        messages: list[MessagePart],
        system_prompt: str | None = None,
        context: ContextT | None = None,
        max_iterations: int = 10,
        reasoning_effort: str | None = None,
        approved_tool_calls: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run agent with streaming, yielding all events including tool stream events.

        Implements the agentic loop:
        1. Send messages to LLM with tools
        2. If LLM returns tool calls, execute them
        3. Add tool results to messages
        4. Repeat until LLM returns without tool calls or max iterations

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            context: Context to inject into tools (e.g., RunToolContext)
            max_iterations: Maximum tool calling iterations
            reasoning_effort: Optional reasoning effort override
            approved_tool_calls: Set of tool_call_ids that have been approved
                for tools with requires_approval=True. If a tool requires
                approval and its call_id is not in this set, ApprovalRequired
                is raised.

        Yields:
            StreamEvent: LLM streaming events
            ToolStreamEvent: Mid-execution tool events (for streaming tools)
            ToolUsePart: Tool call info
            ToolResultPart: Tool execution results

        Raises:
            ApprovalRequired: When a tool with requires_approval=True is called
                and its tool_call_id is not in approved_tool_calls
        """
        tools = self.get_tools_schema() if self._tools else None
        working_messages = list(messages)
        approved = approved_tool_calls or set()

        for _ in range(max_iterations):
            tool_calls: list[ToolUsePart] = []

            async for event in await self.llm.chat(
                working_messages,
                system_prompt=system_prompt,
                tools=tools,
                stream=True,
                reasoning_effort=reasoning_effort,
            ):
                yield event

                if isinstance(event, StreamEndEvent):
                    for part in event.parts:
                        if isinstance(part, ToolUsePart):
                            tool_calls.append(part)

            if not tool_calls:
                break

            # Handle output tool (final_result)
            for tc in tool_calls:
                if tc.name == OUTPUT_TOOL_NAME and self.output_type:
                    try:
                        self.last_output = self.output_type.model_validate(tc.inputs)
                        logger.debug(f"Validated output: {self.last_output}")
                        yield ToolResultEvent(
                            tool_use_id=tc.id,
                            name=tc.name,
                            result=tc.inputs,
                            is_error=False,
                        )
                        yield ToolUseEndEvent(
                            type="tool_use_end",
                            tool_use_id=tc.id,
                            tool_name=tc.name,
                        )
                        return
                    except Exception as e:
                        logger.error(f"Output validation error: {e}")
                        # TODO: Implement retry logic
                        raise

            # Categorize tool calls
            streaming_calls: list[ToolUsePart] = []
            parallel_calls: list[ToolUsePart] = []
            terminal_calls: list[ToolUsePart] = []
            for tc in tool_calls:
                if tc.name == OUTPUT_TOOL_NAME:
                    continue
                tool = self._tools.get(tc.name)
                if not tool:
                    logger.warning(f"Tool not found: {tc.name}")
                    continue
                if tool.terminal:
                    terminal_calls.append(tc)
                elif tool.stream_output:
                    streaming_calls.append(tc)
                else:
                    parallel_calls.append(tc)

            # Any sequential tool in the batch forces the entire batch to run
            # sequentially to preserve execution-order guarantees
            force_sequential = any(
                self._tools[tc.name].sequential for tc in parallel_calls
            )

            # Execute non-streaming tools (parallel or sequential)
            results: list[ToolCallResult | BaseException] = []
            if parallel_calls:
                if force_sequential or len(parallel_calls) == 1:
                    for tc in parallel_calls:
                        call_result = await self._execute_tool_call(
                            tc.name, tc.id, tc.inputs, context, approved
                        )
                        results.append(call_result)
                else:
                    results = await asyncio.gather(
                        *[
                            self._execute_tool_call(
                                tc.name, tc.id, tc.inputs, context, approved
                            )
                            for tc in parallel_calls
                        ],
                        return_exceptions=True,
                    )

            for tc, call_result in zip(parallel_calls, results, strict=True):
                if isinstance(call_result, ApprovalRequired):
                    raise call_result
                if isinstance(call_result, BaseException):
                    call_result = ToolCallResult(
                        tc.name, tc.id, {"error": str(call_result)}, True,
                    )

                result_event, end_event = self._emit_tool_result(
                    tc, call_result.result, call_result.is_error, working_messages,
                )
                yield result_event
                yield end_event

            # Execute streaming tools sequentially
            for tc in streaming_calls:
                result = None
                is_error = False
                try:
                    async for event_or_result in self._execute_tool_stream(
                        tc.name, tc.id, tc.inputs, context, approved
                    ):
                        if isinstance(event_or_result, ToolStreamEvent):
                            yield event_or_result
                        else:
                            result = event_or_result
                except Exception as e:
                    logger.error(f"Error executing streaming tool {tc.name}: {e}")
                    result = {"error": str(e)}
                    is_error = True

                result_event, end_event = self._emit_tool_result(
                    tc, result, is_error, working_messages,
                )
                yield result_event
                yield end_event

            # Terminal tool exits the loop
            if terminal_calls:
                tc = terminal_calls[0]
                result = None
                is_error = False
                try:
                    async for event_or_result in self._execute_tool_stream(
                        tc.name, tc.id, tc.inputs, context, approved
                    ):
                        if isinstance(event_or_result, ToolStreamEvent):
                            yield event_or_result
                        else:
                            result = event_or_result
                except Exception as e:
                    logger.error(f"Error executing terminal tool {tc.name}: {e}")
                    result = {"error": str(e)}
                    is_error = True
                result_event, _ = self._emit_tool_result(
                    tc, result, is_error, working_messages, is_terminal=True,
                )
                yield result_event
                return

    async def _execute_tool_call(
        self,
        tool_name: str,
        tool_call_id: str,
        inputs: dict[str, Any],
        context: ContextT | None,
        approved_tool_calls: set[str],
    ) -> ToolCallResult:
        """Execute a single non-streaming tool call.

        Args:
            tool_name: Name of the tool to execute
            tool_call_id: Unique ID for this tool call
            inputs: Tool input arguments from LLM
            context: Context to inject if tool takes_ctx
            approved_tool_calls: Set of approved tool call IDs

        Returns:
            ToolCallResult with tool_name, tool_call_id, result, and is_error

        Raises:
            ApprovalRequired: If tool requires approval and not approved
        """
        tool = self._tools[tool_name]

        if tool.requires_approval and tool_call_id not in approved_tool_calls:
            raise ApprovalRequired(tool_call_id, tool_name, inputs)

        try:
            result = await self._invoke_tool(tool, inputs, context)
            return ToolCallResult(tool_name, tool_call_id, result, False)
        except ApprovalRequired:
            raise
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolCallResult(tool_name, tool_call_id, {"error": str(e)}, True)

    async def _execute_tool_stream(
        self,
        tool_name: str,
        tool_call_id: str,
        inputs: dict[str, Any],
        context: ContextT | None,
        approved_tool_calls: set[str],
    ) -> AsyncIterator[ToolStreamEvent | Any]:
        """Execute a tool, yielding streaming events for streaming tools or the final result.

        For streaming tools (stream_output=True), yields ToolStreamEvent instances as
        the tool produces them. The last yielded value is the final result.

        For non-streaming tools, delegates to _invoke_tool and yields the result once.

        Args:
            tool_name: Name of the tool to execute
            tool_call_id: Unique ID for this tool call
            inputs: Tool input arguments from LLM
            context: Context to inject if tool takes_ctx
            approved_tool_calls: Set of approved tool call IDs

        Yields:
            ToolStreamEvent: Mid-execution streaming events (streaming tools only)
            Any: The final tool result as the last yielded value

        Raises:
            ApprovalRequired: If tool requires approval and not approved
        """
        tool = self._tools[tool_name]

        if tool.requires_approval and tool_call_id not in approved_tool_calls:
            raise ApprovalRequired(tool_call_id, tool_name, inputs)

        logger.debug(f"Executing tool: {tool_name}")

        kwargs = dict(inputs)

        if tool.stream_output:
            if tool.takes_ctx and context is not None:
                async for event in tool(context, **kwargs):  # type: ignore[misc]
                    yield event
            else:
                async for event in tool(**kwargs):  # type: ignore[misc]
                    yield event
        else:
            result = await self._invoke_tool(tool, inputs, context)
            yield result
