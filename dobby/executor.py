"""AgentExecutor for managing tool registration and agentic LLM interactions.

This module provides the AgentExecutor class which handles:
- Tool registration via @agent.tool() decorator
- Agentic loop with streaming support
- Tool execution with injected context
"""

from collections.abc import AsyncIterator, Callable
import inspect
from typing import Any, Literal

from ._logging import logger

from .providers.openai import OpenAIProvider
from .tools.base import ToolSchema
from .tools.schema_utils import process_tool_definition
from .types import (
    MessagePart,
    StreamEndEvent,
    StreamEvent,
    ToolResultPart,
    ToolStreamEvent,
    ToolUseEndEvent,
    ToolUsePart,
)


class AgentExecutor:
    """Manages tool registration, execution, and LLM interactions with streaming support.

    Attributes:
        provider: The LLM provider type ('openai', 'azure-openai', 'anthropic')
        llm: The LLM provider instance
    """

    def __init__(
        self,
        provider: Literal["openai", "azure-openai", "anthropic"],
        llm: OpenAIProvider,
    ):
        """Initialize the AgentExecutor.

        Args:
            provider: LLM provider type for schema formatting
            llm: LLM provider instance for chat completions
        """
        self.provider = provider
        self.llm = llm
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: dict[str, ToolSchema] = {}
        self._tool_metadata: dict[str, dict[str, Any]] = {}
        self._formatted_tools: list | None = None

    def tool(
        self,
        description: str,
        name: str | None = None,
        stream_output: bool = False,
        version: str = "1.0.0",
    ) -> Callable[[Callable], Callable]:
        """Decorator to register tools with the agent.

        Args:
            description: Tool description for LLM
            name: Optional tool name (defaults to function name)
            stream_output: If True, tool is async generator yielding ToolStreamEvent
            version: Tool version

        Returns:
            Decorator function that registers the tool

        Example:
            @agent.tool(description="Create document", stream_output=True)
            async def create_document(ctx: Injected[ToolContext], title: str):
                yield ToolStreamEvent(type="data-title", data=title)
                yield {"id": "123"}  # Final result
        """

        def decorator(func: Callable) -> Callable:
            schema, injected_params = process_tool_definition(
                func, name, description, version
            )

            tool_name = schema.name
            self._tools[tool_name] = func
            self._tool_schemas[tool_name] = schema
            self._tool_metadata[tool_name] = {
                "injected_params": injected_params,
                "stream_output": stream_output,
            }
            self._formatted_tools = None  # Clear cache

            logger.debug(f"Registering tool: {schema.name}")

            return func

        return decorator

    def get_tools_schema(self) -> list:
        """Get tool schemas formatted for the LLM provider.

        Returns:
            List of tool schemas in provider-specific format
        """
        if self._formatted_tools is None:
            match self.provider:
                case "openai" | "azure-openai":
                    self._formatted_tools = [
                        schema.to_openai_format()
                        for schema in self._tool_schemas.values()
                    ]
                case "anthropic":
                    self._formatted_tools = [
                        schema.to_anthropic_format()
                        for schema in self._tool_schemas.values()
                    ]
        return self._formatted_tools

    async def run_stream(
        self,
        messages: list[MessagePart],
        system_prompt: str | None = None,
        context: Any | None = None,
        max_iterations: int = 10,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamEvent | ToolStreamEvent | ToolResultPart | ToolUseEndEvent]:
        """Run agent with streaming, yielding all events including tool stream events.

        Implements the agentic loop:
        1. Send messages to LLM with tools
        2. If LLM returns tool calls, execute them
        3. Add tool results to messages
        4. Repeat until LLM returns without tool calls or max iterations

        Currently, the AgentExecutor does not support parallel execution.

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            context: Context to inject into tools (e.g., ToolContext)
            max_iterations: Maximum tool calling iterations
            reasoning_effort: Optional reasoning effort override

        Yields:
            StreamEvent: LLM streaming events
            ToolStreamEvent: Mid-execution tool events (for streaming tools)
            ToolUsePart: Tool call info
            ToolResultPart: Tool execution results
        """
        tools = self.get_tools_schema() if self._tools else None
        working_messages = list(messages)

        for _ in range(max_iterations):
            # Stream from LLM
            tool_calls: list[ToolUsePart] = []

            async for event in await self.llm.chat(
                working_messages,
                system_prompt=system_prompt,
                tools=tools,
                stream=True,
                reasoning_effort=reasoning_effort,
            ):
                yield event  # Forward LLM events (including ToolUseEvent from provider)

                # Collect tool calls from stream_end event
                if isinstance(event, StreamEndEvent):
                    for part in event.parts:
                        if isinstance(part, dict) and part.get("type") == "tool_use":
                            # Cast to ToolUsePart since we verified the type
                            tool_calls.append(part)  # type: ignore[arg-type]

            # No tool calls = done
            if not tool_calls:
                break

            for tool_call in tool_calls:
                # Provider already yielded the ToolUseEvent
                # Execute tool and yield stream events
                tool_name = tool_call["name"]
                tool_id = tool_call["id"]
                tool_inputs = tool_call["inputs"]

                result = None
                is_error = False
                try:
                    async for event_or_result in self._execute_tool_stream(
                        tool_name, tool_inputs, context
                    ):
                        if (
                            isinstance(event_or_result, dict)
                            and event_or_result.get("type", "").startswith("data-")
                        ):
                            # This is a ToolStreamEvent (TypedDict with type starting with 'data-')
                            yield ToolStreamEvent(type=event_or_result["type"], data=event_or_result.get("data"))
                        else:
                            result = event_or_result
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    result = {"error": str(e)}
                    is_error = True

                yield ToolResultPart(
                    type="tool_result_event",
                    tool_use_id=tool_id,
                    name=tool_name,
                    result=result,
                    is_error=is_error,
                )
                
                yield ToolUseEndEvent(
                    type="tool_use_end",
                    tool_use_id=tool_id,
                    tool_name=tool_name,
                )

                # Add to messages for next iteration
                working_messages.append(
                    {
                        "role": "assistant",
                        "parts": [tool_call],
                    }
                )
                working_messages.append(
                    {
                        "role": "tool_result",
                        "tool_use_id": tool_id,
                        "name": tool_name,
                        "parts": [{"type": "text", "text": str(result)}],
                        "is_error": is_error,
                    }
                )

    async def _execute_tool_stream(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        context: Any | None,
    ) -> AsyncIterator[ToolStreamEvent | Any]:
        """Execute a tool, yielding stream events if stream_output=True.

        Supports three tool types:
        1. Sync tools: def func() -> result
        2. Async tools: async def func() -> result
        3. Streaming tools: async def func() with stream_output=True -> yields events

        Args:
            tool_name: Name of the tool to execute
            inputs: Tool input arguments from LLM
            context: Context to inject into Injected parameters

        Yields:
            ToolStreamEvent objects during execution (streaming only), then final result
        """
        func = self._tools[tool_name]
        metadata = self._tool_metadata[tool_name]
        injected_params = metadata["injected_params"]
        stream_output = metadata["stream_output"]

        logger.debug(f"[x] Executing tool: {tool_name}")

        # Build kwargs with injected context
        kwargs = dict(inputs)
        if injected_params and context is not None:
            for param_name in injected_params:
                kwargs[param_name] = context

        if stream_output:
            # Streaming tool - async generator yielding events then final result
            result = None
            async for event in func(**kwargs):
                if (
                    isinstance(event, dict)
                    and event.get("type", "").startswith("data-")
                ):
                    yield event
                else:
                    result = event
            yield result
        elif inspect.iscoroutinefunction(func):
            # Async tool - await and return
            result = await func(**kwargs)
            yield result
        else:
            # Sync tool - call directly
            result = func(**kwargs)
            yield result
