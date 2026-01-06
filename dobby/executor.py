"""AgentExecutor for managing tool registration and agentic LLM interactions.

This module provides the AgentExecutor class which handles:
- Tool registration via Tool class instances
- Agentic loop with streaming support
- Tool execution with injected context
"""

from collections.abc import AsyncIterator
import inspect
from typing import Any, Literal

from pydantic import BaseModel

from ._logging import logger
from .exceptions import ApprovalRequired
from .providers.openai import OpenAIProvider
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


class AgentExecutor[ContextT, OutputT: BaseModel]:
    """Manages tool registration, execution, and LLM interactions with streaming support.

    Type Parameters:
        ContextT: Type of context object passed to tools via Injected[ContextT]
        OutputT: Type of structured output (Pydantic model) when output_type is set

    Attributes:
        provider: The LLM provider type ('openai', 'azure-openai', 'anthropic')
        llm: The LLM provider instance
        output_type: Pydantic model for structured output (optional)
        output_mode: How to get structured output ('tool' or 'native')
        last_output: The last validated structured output (if output_type was set)
    """

    def __init__(
        self,
        provider: Literal["openai", "azure-openai", "anthropic"],
        llm: OpenAIProvider,
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
        self._output_tool_schema: dict | None = None
        
        # Validate output_mode
        if output_type and output_mode == "native":
            raise NotImplementedError("Native output mode not yet supported. Use 'tool' mode.")
        
        # Create output tool schema if output_type is set
        if output_type and output_mode == "tool":
            self._output_tool_schema = self._create_output_tool_schema(output_type)
        
        if tools:
            for tool in tools:
                self._tools[tool._tool_schema.name] = tool
                logger.debug(f"Registered tool: {tool._tool_schema.name}")
    
    def _create_output_tool_schema(self, model: type[BaseModel]) -> dict:
        """Create output tool schema from Pydantic model.
        
        Uses Pydantic's native JSON schema output directly to preserve all
        type information including enums, nested objects, arrays, dates, etc.
        
        Args:
            model: Pydantic BaseModel class for structured output
            
        Returns:
            OpenAI-compatible tool schema dict with full JSON schema
        """
        json_schema = model.model_json_schema()
        
        return {
            "type": "function",
            "name": OUTPUT_TOOL_NAME,
            "description": json_schema.get("description") or f"Return the final structured result as {model.__name__}",
            "parameters": json_schema,
        }

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
                        tool.to_openai_format()
                        for tool in self._tools.values()
                    ]
                    # Add output tool if output_type is set (already in OpenAI format)
                    if self._output_tool_schema:
                        self._formatted_tools.append(self._output_tool_schema)
                case "anthropic":
                    self._formatted_tools = [
                        tool.to_anthropic_format()
                        for tool in self._tools.values()
                    ]
                    # Add output tool for Anthropic format
                    if self._output_tool_schema:
                        # Convert OpenAI format to Anthropic format
                        self._formatted_tools.append({
                            "name": self._output_tool_schema["name"],
                            "description": self._output_tool_schema["description"],
                            "input_schema": self._output_tool_schema["parameters"],
                        })
        return self._formatted_tools

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

                # Collect tool calls from stream_end event
                if isinstance(event, StreamEndEvent):
                    for part in event.parts:
                        if isinstance(part, ToolUsePart):
                            tool_calls.append(part)

            # No tool calls = done
            if not tool_calls:
                break

            for tool_call in tool_calls:
                # Provider already yielded the ToolUseEvent
                # Execute tool and yield stream events
                tool_name = tool_call.name
                tool_id = tool_call.id
                tool_inputs = tool_call.inputs

                # Check if this is the output tool (final_result)
                if tool_name == OUTPUT_TOOL_NAME and self.output_type:
                    # Validate output with Pydantic
                    try:
                        self.last_output = self.output_type.model_validate(tool_inputs)
                        logger.debug(f"Validated output: {self.last_output}")
                        # Yield result event for the output tool
                        yield ToolResultEvent(
                            tool_use_id=tool_id,
                            name=tool_name,
                            result=tool_inputs,
                            is_error=False,
                        )
                        yield ToolUseEndEvent(
                            type="tool_use_end",
                            tool_use_id=tool_id,
                            tool_name=tool_name,
                        )
                        return  # End run - structured output received
                    except Exception as e:
                        logger.error(f"Output validation error: {e}")
                        # TODO: Implement retry logic
                        raise

                result = None
                is_error = False
                try:
                    async for event_or_result in self._execute_tool_stream(
                        tool_name, tool_id, tool_inputs, context, approved
                    ):
                        if isinstance(event_or_result, ToolStreamEvent):
                            yield event_or_result
                        else:
                            result = event_or_result
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    result = {"error": str(e)}
                    is_error = True

                yield ToolResultEvent(
                    tool_use_id=tool_id,
                    name=tool_name,
                    result=result,
                    is_error=is_error,
                )
                
                # TODO: see what is the requirement of this event ?, as if 
                # tool_result is generated that indicated too_use_end
                yield ToolUseEndEvent(
                    type="tool_use_end",
                    tool_use_id=tool_id,
                    tool_name=tool_name,
                )

                # Add to messages for next iteration
                working_messages.append(
                    AssistantMessagePart(parts=[tool_call])
                )
                working_messages.append(
                    UserMessagePart(parts=[
                        ToolResultPart(
                            tool_use_id=tool_id,
                            name=tool_name,
                            parts=[TextPart(text=str(result))],
                            is_error=is_error,
                        )
                    ])
                )

    async def _execute_tool_stream(
        self,
        tool_name: str,
        tool_call_id: str,
        inputs: dict[str, Any],
        context: ContextT | None,
        approved_tool_calls: set[str],
    ) -> AsyncIterator[ToolStreamEvent | Any]:
        """Execute a tool.

        Args:
            tool_name: Name of the tool to execute
            tool_call_id: Unique ID for this tool call
            inputs: Tool input arguments from LLM
            context: Context to inject if tool takes_ctx
            approved_tool_calls: Set of approved tool call IDs

        Yields:
            ToolStreamEvent for streaming tools, then final result

        Raises:
            ApprovalRequired: If tool requires approval and not approved
        """
        tool = self._tools[tool_name]

        if tool.requires_approval and tool_call_id not in approved_tool_calls:
            raise ApprovalRequired(tool_call_id, tool_name, inputs)

        logger.debug(f"[x] Executing tool: {tool_name}")

        kwargs = dict(inputs)

        if tool.stream_output:
            # Streaming tool - yields events then final result
            if tool.takes_ctx and context is not None:
                async for event in tool(context, **kwargs):  # type: ignore[misc]
                    yield event
            else:
                async for event in tool(**kwargs):  # type: ignore[misc]
                    yield event
        else:
            # Non-streaming tool
            if tool.takes_ctx and context is not None:
                if inspect.iscoroutinefunction(tool.__call__):
                    result = await tool(context, **kwargs)
                else:
                    result = tool(context, **kwargs)
            else:
                if inspect.iscoroutinefunction(tool.__call__):
                    result = await tool(**kwargs)
                else:
                    result = tool(**kwargs)
            yield result
