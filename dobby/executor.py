"""AgentExecutor for managing tool registration and agentic LLM interactions.

This module provides the AgentExecutor class which handles:
- Tool registration via Tool class instances
- Agentic loop with streaming support
- Tool execution with injected context
"""

import asyncio
from collections.abc import AsyncIterator
import inspect
import traceback
from typing import Any, Literal, NamedTuple
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from ._context import (
    correlation_fields,
    run_id_var,
    tool_attempt_var,
    tool_call_id_var,
    tool_max_attempts_var,
    tool_name_var,
)
from ._logging import logger
from .exceptions import (
    ApprovalRequired,
    ErrorCode,
    ModelRetry,
    ModelRetryExhaustedError,
    ToolFailure,
    classify_tool_error,
    format_model_error,
)
from .providers.base import Provider, ProviderError
from .providers.vertexai.converters import to_vertexai_tool
from .tools.retry import (
    is_retryable_tool_exception,
    max_tool_invocations,
    retry_backoff_seconds,
)
from .tools.tool import Tool
from .types import (
    AssistantMessagePart,
    MessagePart,
    StreamEndEvent,
    StreamEvent,
    TextPart,
    ToolErrorDetails,
    ToolResultEvent,
    ToolResultPart,
    ToolStreamEvent,
    ToolUseEndEvent,
    ToolUsePart,
    UserMessagePart,
)

OUTPUT_TOOL_NAME = "final_result"
_DEFAULT_MAX_MODEL_CORRECTIONS = 3


def _resolve_max_model_corrections(max_model_corrections: int | None) -> int:
    """Resolve the run-level model-correction budget."""
    if max_model_corrections is not None:
        return max_model_corrections
    return _DEFAULT_MAX_MODEL_CORRECTIONS


def _tool_error_details(
    exception: Exception,
    *,
    error_code: ErrorCode | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
) -> ToolErrorDetails:
    """Create structured diagnostics for a tool execution exception."""
    decision = classify_tool_error(exception)
    return ToolErrorDetails(
        exception_type=type(exception).__qualname__,
        exception_module=type(exception).__module__,
        message=str(exception),
        traceback="".join(traceback.format_exception(exception)),
        error_code=(error_code or (decision.code if decision else None)),
        run_id=run_id_var.get(),
        tool_name=tool_name or tool_name_var.get(),
        tool_call_id=tool_call_id or tool_call_id_var.get(),
        attempt=tool_attempt_var.get(),
        max_attempts=tool_max_attempts_var.get(),
    )


def _log_tool_exception(exception: Exception) -> None:
    """Log a tool exception according to its semantic classification."""
    fields = correlation_fields()
    context = " ".join(f"{key}={value}" for key, value in fields.items())
    if isinstance(exception, ModelRetry):
        logger.warning(
            f"Tool requested model correction {context} error_code={exception.code.value}",
            extra={**fields, "layer": "tool", "error_code": exception.code.value},
        )
    elif isinstance(exception, ToolFailure):
        logger.error(
            f"Tool reported failure {context} error_code={exception.code.value}: {exception}",
            extra={**fields, "layer": "tool", "error_code": exception.code.value},
        )
    else:
        logger.exception(
            f"Unexpected tool execution error {context} "
            f"error_code={ErrorCode.TOOL_EXECUTION_ERROR.value}",
            extra={
                **fields,
                "layer": "tool",
                "error_code": ErrorCode.TOOL_EXECUTION_ERROR.value,
            },
        )


class ToolCallResult(NamedTuple):
    """Result from executing a single tool call."""

    tool_name: str
    tool_call_id: str
    result: Any
    is_error: bool
    error_details: ToolErrorDetails | None = None
    retry_model: bool = False


def _classified_tool_result(
    tool_name: str,
    tool_call_id: str,
    exception: Exception,
    *,
    error_details: ToolErrorDetails | None = None,
) -> ToolCallResult:
    """Build a model-facing result for a classified tool-call error."""
    decision = classify_tool_error(exception)
    if decision is None:
        raise RuntimeError("tool error must produce an error decision")
    return ToolCallResult(
        tool_name,
        tool_call_id,
        format_model_error(decision),
        True,
        error_details or _tool_error_details(exception),
        decision.retry_model,
    )


def _model_retry_result(
    tool_name: str,
    tool_call_id: str,
    exception: ModelRetry,
    *,
    error_details: ToolErrorDetails | None = None,
) -> ToolCallResult:
    """Build a model-facing result for a correctable tool-call error."""
    return _classified_tool_result(
        tool_name,
        tool_call_id,
        exception,
        error_details=error_details,
    )


def _final_result_validation_message(exception: ValidationError) -> str:
    """Build field-level final-result validation feedback."""
    issues = []
    for error in exception.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        issues.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "Invalid final result: " + "; ".join(issues)


def _control_flow_result(
    tool_call: ToolUsePart,
    exception: ApprovalRequired | asyncio.CancelledError,
) -> ToolCallResult:
    """Build an unsuccessful placeholder without classifying host control flow.

    Approval and cancellation remain host signals. The placeholder is marked
    unsuccessful so history does not look like the tool completed, but it is
    not routed through error classification.
    """
    if isinstance(exception, ApprovalRequired):
        result = {"approval_required": True}
    else:
        result = {"cancelled": True}
    return ToolCallResult(tool_call.name, tool_call.id, result, True)


def _unexecuted_result(tool_call: ToolUsePart, *, reason: str) -> ToolCallResult:
    """Build a placeholder for a tool call intentionally skipped by the executor."""
    return ToolCallResult(
        tool_call.name,
        tool_call.id,
        {"skipped": True, "reason": reason},
        True,
    )


class AgentExecutor[ContextT, OutputT: BaseModel]:
    """Manages tool registration, execution, and LLM interactions with streaming support.

    Type Parameters:
        ContextT: Type of context object passed to tools via Injected[ContextT]
        OutputT: Type of structured output (Pydantic model) when output_type is set

    Attributes:
        provider: The tool-schema wire format ('openai', 'azure-openai', 'gemini',
            'anthropic', 'vertexai'). This selects a schema shape, not a provider
            identity -- a provider's `.name` may differ (e.g. AnthropicProvider
            reports "azure-anthropic" on Azure but pairs with 'anthropic' here).
        llm: The LLM provider instance
        output_type: Pydantic model for structured output (optional)
        output_mode: How to get structured output ('tool' or 'native')
        last_output: The last validated structured output (if output_type was set)
    """

    def __init__(
        self,
        provider: Literal["openai", "azure-openai", "gemini", "anthropic", "vertexai"],
        llm: Provider,
        tools: list[Tool] | None = None,
        output_type: type[OutputT] | None = None,
        output_mode: Literal["tool", "native"] = "tool",
    ):
        """Initialize the AgentExecutor.

        Args:
            provider: Tool-schema wire format. Use 'openai' for OpenAIProvider
                (Responses API), 'vertexai' for VertexAIProvider (Chat
                Completions), 'anthropic' for AnthropicProvider in either mode
                (direct or Azure), and 'gemini' for GeminiProvider.
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
            output_tool = Tool.from_model(
                output_type, name=OUTPUT_TOOL_NAME, description=description
            )
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
                case "vertexai":
                    self._formatted_tools = [
                        to_vertexai_tool(tool) for tool in self._tools.values()
                    ]
                case _:
                    # `provider` is a Literal, so this is unreachable for valid
                    # callers. Present so every path assigns and the declared
                    # `list` return type holds even if the Literal grows.
                    self._formatted_tools = []
        return self._formatted_tools

    async def _call_tool(
        self,
        tool: Tool,
        inputs: dict[str, Any],
        context: ContextT | None,
    ) -> Any:
        """Dispatch a non-streaming tool once, including context injection."""
        kwargs = dict(inputs)
        if tool.takes_ctx and context is not None:
            if inspect.iscoroutinefunction(tool.__call__):
                return await tool(context, **kwargs)
            return tool(context, **kwargs)
        if inspect.iscoroutinefunction(tool.__call__):
            return await tool(**kwargs)
        return tool(**kwargs)

    async def _invoke_tool(
        self,
        tool: Tool,
        inputs: dict[str, Any],
        context: ContextT | None,
    ) -> Any:
        """Invoke a non-streaming tool, retrying listed transient failures.

        Lookup, input validation, and approval happen before this method.
        Retry state is local to this call.

        Args:
            tool: The Tool instance to invoke
            inputs: Validated tool input arguments from LLM
            context: Context to inject if tool takes_ctx

        Returns:
            The tool's return value
        """
        logger.debug(f"Executing tool: {tool.name}")
        policy = tool.retry_policy()
        max_attempts = max_tool_invocations(policy)
        tool_max_attempts_var.set(max_attempts)
        attempt = 0
        while True:
            attempt += 1
            tool_attempt_var.set(attempt)
            try:
                return await self._call_tool(tool, inputs, context)
            except Exception as exception:
                if attempt >= max_attempts or not is_retryable_tool_exception(exception, policy):
                    raise
                delay = retry_backoff_seconds(policy, failed_attempt=attempt)
                fields = correlation_fields()
                logger.warning(
                    f"Retrying tool layer=tool run_id={run_id_var.get()} "
                    f"tool_name={tool.name} tool_call_id={tool_call_id_var.get()} "
                    f"attempt={attempt}/{max_attempts} in {delay:.1f}s "
                    f"after {type(exception).__name__}: {exception}",
                    extra={**fields, "layer": "tool", "exception_type": type(exception).__name__},
                )
                await asyncio.sleep(delay)

    async def _stream_tool(
        self,
        tool: Tool,
        inputs: dict[str, Any],
        context: ContextT | None,
    ) -> AsyncIterator[Any]:
        """Stream a tool, retrying only if failure occurs before the first yield."""
        kwargs = dict(inputs)
        policy = tool.retry_policy()
        max_attempts = max_tool_invocations(policy)
        tool_max_attempts_var.set(max_attempts)
        attempt = 0
        while True:
            attempt += 1
            tool_attempt_var.set(attempt)
            yielded = False
            try:
                if tool.takes_ctx and context is not None:
                    stream = tool(context, **kwargs)  # type: ignore[misc]
                else:
                    stream = tool(**kwargs)  # type: ignore[misc]
                async for event in stream:
                    yielded = True
                    yield event
                return
            except Exception as exception:
                if (
                    yielded
                    or attempt >= max_attempts
                    or not is_retryable_tool_exception(exception, policy)
                ):
                    raise
                delay = retry_backoff_seconds(policy, failed_attempt=attempt)
                fields = correlation_fields()
                logger.warning(
                    f"Retrying tool layer=tool run_id={run_id_var.get()} "
                    f"tool_name={tool.name} tool_call_id={tool_call_id_var.get()} "
                    f"attempt={attempt}/{max_attempts} in {delay:.1f}s "
                    f"after {type(exception).__name__}: {exception}",
                    extra={**fields, "layer": "tool", "exception_type": type(exception).__name__},
                )
                await asyncio.sleep(delay)

    def _emit_tool_result(
        self,
        tool_call: ToolUsePart,
        result: Any,
        is_error: bool,
        working_messages: list[MessagePart],
        *,
        error_details: ToolErrorDetails | None = None,
        is_terminal: bool = False,
    ) -> tuple[ToolResultEvent, ToolUseEndEvent]:
        """Build result events and append tool round-trip messages.

        Args:
            tool_call: The original tool call from the LLM
            result: The tool execution result (or error dict)
            is_error: Whether the result represents an error
            working_messages: Conversation message list to append to
            error_details: Structured diagnostics for a tool execution exception
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
                error_details=error_details,
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
        reasoning_effort: str | int | None = None,
        max_tokens: int | None = None,
        approved_tool_calls: set[str] | None = None,
        max_model_corrections: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run the agent under a run-scoped correlation context.

        ``max_model_corrections`` is the run-level model-correction budget
        shared by tool-call and final-result corrections.
        """
        resolved_max_model_corrections = _resolve_max_model_corrections(max_model_corrections)
        self.last_output = None
        run_id = uuid4().hex
        token = run_id_var.set(run_id)
        logger.debug(
            f"Starting agent run run_id={run_id}",
            extra={"run_id": run_id, "layer": "executor"},
        )
        try:
            async for event in self._run_stream(
                messages=messages,
                system_prompt=system_prompt,
                context=context,
                max_iterations=max_iterations,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
                approved_tool_calls=approved_tool_calls,
                max_model_corrections=resolved_max_model_corrections,
            ):
                yield event
        except ProviderError as exception:
            logger.error(
                f"Provider call failed layer=provider run_id={run_id} "
                f"provider={exception.provider or self.llm.name} "
                f"exception_type={type(exception).__name__}: {exception}",
                extra={
                    "layer": "provider",
                    "run_id": run_id,
                    "provider": exception.provider or self.llm.name,
                    "exception_type": type(exception).__name__,
                },
            )
            raise
        finally:
            run_id_var.reset(token)

    async def _run_stream(
        self,
        messages: list[MessagePart],
        system_prompt: str | None = None,
        context: ContextT | None = None,
        max_iterations: int = 10,
        reasoning_effort: str | int | None = None,
        max_tokens: int | None = None,
        approved_tool_calls: set[str] | None = None,
        max_model_corrections: int = 3,
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
            max_tokens: Optional maximum number of model output tokens.
            approved_tool_calls: Set of tool_call_ids that have been approved
                for tools with requires_approval=True. If a tool requires
                approval and its call_id is not in this set, ApprovalRequired
                is raised.
            max_model_corrections: Maximum model-correction turns permitted per
                run, shared by tool-call and final-result corrections.

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
        model_corrections = 0
        last_correction_error: ToolErrorDetails | None = None

        for _ in range(max_iterations):
            tool_calls: list[ToolUsePart] = []
            batch_has_model_correction = False
            batch_last_correction_error: ToolErrorDetails | None = None

            async for event in await self.llm.chat(
                working_messages,
                system_prompt=system_prompt,
                tools=tools,
                stream=True,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
            ):
                yield event

                if isinstance(event, StreamEndEvent):
                    for part in event.parts:
                        if isinstance(part, ToolUsePart):
                            tool_calls.append(part)

            if not tool_calls:
                break

            # Handle output tool (final_result)
            final_result_invalid = False
            for tc in tool_calls:
                if tc.name == OUTPUT_TOOL_NAME and self.output_type:
                    try:
                        validated_output = self.output_type.model_validate(tc.inputs)
                    except ValidationError as exception:
                        retry = ModelRetry(
                            _final_result_validation_message(exception),
                            code=ErrorCode.FINAL_RESULT_INVALID,
                        )
                        call_result = _model_retry_result(
                            tc.name,
                            tc.id,
                            retry,
                            error_details=_tool_error_details(
                                retry,
                                tool_name=tc.name,
                                tool_call_id=tc.id,
                            ),
                        )
                        logger.warning(
                            f"Final result requires model correction layer=tool "
                            f"run_id={run_id_var.get()} tool_name={tc.name} "
                            f"tool_call_id={tc.id} "
                            f"error_code={ErrorCode.FINAL_RESULT_INVALID.value}",
                            extra={
                                "layer": "tool",
                                "run_id": run_id_var.get(),
                                "tool_name": tc.name,
                                "tool_call_id": tc.id,
                                "error_code": ErrorCode.FINAL_RESULT_INVALID.value,
                            },
                        )
                        result_event, end_event = self._emit_tool_result(
                            tc,
                            call_result.result,
                            call_result.is_error,
                            working_messages,
                            error_details=call_result.error_details,
                        )
                        yield result_event
                        yield end_event

                        model_corrections += 1
                        last_correction_error = call_result.error_details
                        if model_corrections > max_model_corrections:
                            raise ModelRetryExhaustedError(
                                model_corrections,
                                last_error=last_correction_error,
                            ) from exception
                        for sibling in tool_calls:
                            if sibling is tc:
                                continue
                            sibling_result = _unexecuted_result(
                                sibling,
                                reason=ErrorCode.FINAL_RESULT_INVALID.value,
                            )
                            sibling_event, sibling_end = self._emit_tool_result(
                                sibling,
                                sibling_result.result,
                                sibling_result.is_error,
                                working_messages,
                            )
                            yield sibling_event
                            yield sibling_end
                        final_result_invalid = True
                        break
                    else:
                        self.last_output = validated_output
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

            if final_result_invalid:
                continue

            # Categorize tool calls
            streaming_calls: list[ToolUsePart] = []
            parallel_calls: list[ToolUsePart] = []
            terminal_calls: list[ToolUsePart] = []
            for tc in tool_calls:
                if tc.name == OUTPUT_TOOL_NAME and self.output_type:
                    continue
                tool = self._tools.get(tc.name)
                if not tool:
                    logger.warning(
                        f"Tool not found layer=tool run_id={run_id_var.get()} "
                        f"tool_name={tc.name} tool_call_id={tc.id} "
                        f"error_code={ErrorCode.TOOL_NOT_FOUND.value}",
                        extra={
                            "layer": "tool",
                            "run_id": run_id_var.get(),
                            "tool_name": tc.name,
                            "tool_call_id": tc.id,
                            "error_code": ErrorCode.TOOL_NOT_FOUND.value,
                        },
                    )
                    parallel_calls.append(tc)
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
                self._tools[tc.name].sequential for tc in parallel_calls if tc.name in self._tools
            )

            # Execute non-streaming tools (parallel or sequential)
            results: list[ToolCallResult | BaseException] = []
            if parallel_calls:
                if force_sequential or len(parallel_calls) == 1:
                    for tc in parallel_calls:
                        try:
                            call_result = await self._execute_tool_call(
                                tc.name, tc.id, tc.inputs, context, approved
                            )
                        except (ApprovalRequired, asyncio.CancelledError) as exception:
                            results.append(exception)
                            break
                        results.append(call_result)
                        if call_result.retry_model:
                            break
                else:
                    results = await asyncio.gather(
                        *[
                            self._execute_tool_call(tc.name, tc.id, tc.inputs, context, approved)
                            for tc in parallel_calls
                        ],
                        return_exceptions=True,
                    )

            assembled_calls = parallel_calls[: len(results)]
            control_flow: ApprovalRequired | asyncio.CancelledError | None = None
            for tc, call_result in zip(assembled_calls, results, strict=True):
                if isinstance(call_result, (ApprovalRequired, asyncio.CancelledError)):
                    if control_flow is None:
                        control_flow = call_result
                    call_result = _control_flow_result(tc, call_result)
                elif isinstance(call_result, Exception):
                    call_result = _classified_tool_result(tc.name, tc.id, call_result)
                elif isinstance(call_result, BaseException):
                    raise call_result

                if call_result.retry_model:
                    batch_has_model_correction = True
                    batch_last_correction_error = call_result.error_details

                result_event, end_event = self._emit_tool_result(
                    tc,
                    call_result.result,
                    call_result.is_error,
                    working_messages,
                    error_details=call_result.error_details,
                )
                yield result_event
                yield end_event

            if control_flow is not None:
                remaining_regular_calls = parallel_calls[len(results) :]
                for remaining in remaining_regular_calls + streaming_calls + terminal_calls:
                    remaining_result = _control_flow_result(remaining, control_flow)
                    remaining_event, remaining_end = self._emit_tool_result(
                        remaining,
                        remaining_result.result,
                        remaining_result.is_error,
                        working_messages,
                    )
                    yield remaining_event
                    yield remaining_end
                raise control_flow

            terminal_completed = False
            if batch_has_model_correction:
                for remaining in parallel_calls[len(results) :] + streaming_calls + terminal_calls:
                    remaining_result = _unexecuted_result(
                        remaining,
                        reason="model_correction",
                    )
                    remaining_event, remaining_end = self._emit_tool_result(
                        remaining,
                        remaining_result.result,
                        remaining_result.is_error,
                        working_messages,
                    )
                    yield remaining_event
                    yield remaining_end
            else:
                # Execute streaming tools sequentially
                streaming_correction = False
                for index, tc in enumerate(streaming_calls):
                    result = None
                    is_error = False
                    error_details: ToolErrorDetails | None = None
                    try:
                        async for event_or_result in self._execute_tool_stream(
                            tc.name, tc.id, tc.inputs, context, approved
                        ):
                            if isinstance(event_or_result, ToolCallResult):
                                result = event_or_result.result
                                is_error = event_or_result.is_error
                                error_details = event_or_result.error_details
                                if event_or_result.retry_model:
                                    batch_has_model_correction = True
                                    batch_last_correction_error = event_or_result.error_details
                                    streaming_correction = True
                            elif isinstance(event_or_result, ToolStreamEvent):
                                yield event_or_result
                            else:
                                result = event_or_result
                    except (ApprovalRequired, asyncio.CancelledError) as exception:
                        call_result = _control_flow_result(tc, exception)
                        result_event, end_event = self._emit_tool_result(
                            tc,
                            call_result.result,
                            call_result.is_error,
                            working_messages,
                        )
                        yield result_event
                        yield end_event
                        unexecuted = streaming_calls[index + 1 :] + terminal_calls
                        for remaining in unexecuted:
                            remaining_result = _control_flow_result(remaining, exception)
                            remaining_event, remaining_end = self._emit_tool_result(
                                remaining,
                                remaining_result.result,
                                remaining_result.is_error,
                                working_messages,
                            )
                            yield remaining_event
                            yield remaining_end
                        raise

                    result_event, end_event = self._emit_tool_result(
                        tc,
                        result,
                        is_error,
                        working_messages,
                        error_details=error_details,
                    )
                    yield result_event
                    yield end_event

                    if streaming_correction:
                        for remaining in streaming_calls[index + 1 :] + terminal_calls:
                            remaining_result = _unexecuted_result(
                                remaining,
                                reason="model_correction",
                            )
                            remaining_event, remaining_end = self._emit_tool_result(
                                remaining,
                                remaining_result.result,
                                remaining_result.is_error,
                                working_messages,
                            )
                            yield remaining_event
                            yield remaining_end
                        break
                else:
                    # Terminal tool exits the loop
                    if terminal_calls:
                        tc = terminal_calls[0]
                        result = None
                        is_error = False
                        error_details = None
                        skip_terminal_exit = False
                        try:
                            async for event_or_result in self._execute_tool_stream(
                                tc.name, tc.id, tc.inputs, context, approved
                            ):
                                if isinstance(event_or_result, ToolCallResult):
                                    result = event_or_result.result
                                    is_error = event_or_result.is_error
                                    error_details = event_or_result.error_details
                                    if event_or_result.retry_model:
                                        batch_has_model_correction = True
                                        batch_last_correction_error = event_or_result.error_details
                                        skip_terminal_exit = True
                                elif isinstance(event_or_result, ToolStreamEvent):
                                    yield event_or_result
                                else:
                                    result = event_or_result
                        except (ApprovalRequired, asyncio.CancelledError) as exception:
                            call_result = _control_flow_result(tc, exception)
                            result_event, end_event = self._emit_tool_result(
                                tc,
                                call_result.result,
                                call_result.is_error,
                                working_messages,
                            )
                            yield result_event
                            yield end_event
                            for remaining in terminal_calls[1:]:
                                remaining_result = _control_flow_result(remaining, exception)
                                remaining_event, remaining_end = self._emit_tool_result(
                                    remaining,
                                    remaining_result.result,
                                    remaining_result.is_error,
                                    working_messages,
                                )
                                yield remaining_event
                                yield remaining_end
                            raise
                        result_event, end_event = self._emit_tool_result(
                            tc,
                            result,
                            is_error,
                            working_messages,
                            is_terminal=not skip_terminal_exit,
                            error_details=error_details,
                        )
                        yield result_event
                        if skip_terminal_exit:
                            yield end_event
                        else:
                            terminal_completed = True

                        skip_reason = "model_correction" if skip_terminal_exit else "terminal"
                        for remaining in terminal_calls[1:]:
                            remaining_result = _unexecuted_result(
                                remaining,
                                reason=skip_reason,
                            )
                            remaining_event, remaining_end = self._emit_tool_result(
                                remaining,
                                remaining_result.result,
                                remaining_result.is_error,
                                working_messages,
                            )
                            yield remaining_event
                            yield remaining_end

            if batch_has_model_correction:
                model_corrections += 1
                last_correction_error = batch_last_correction_error
                if model_corrections > max_model_corrections:
                    raise ModelRetryExhaustedError(
                        model_corrections,
                        last_error=last_correction_error,
                    )

            if terminal_completed:
                return

    async def _execute_tool_call(
        self,
        tool_name: str,
        tool_call_id: str,
        inputs: dict[str, Any],
        context: ContextT | None,
        approved_tool_calls: set[str],
    ) -> ToolCallResult:
        """Execute a tool call with call-scoped correlation context."""
        name_token = tool_name_var.set(tool_name)
        call_token = tool_call_id_var.set(tool_call_id)
        attempt_token = tool_attempt_var.set(None)
        max_attempts_token = tool_max_attempts_var.set(None)
        try:
            return await self._execute_tool_call_impl(
                tool_name,
                tool_call_id,
                inputs,
                context,
                approved_tool_calls,
            )
        finally:
            tool_max_attempts_var.reset(max_attempts_token)
            tool_attempt_var.reset(attempt_token)
            tool_call_id_var.reset(call_token)
            tool_name_var.reset(name_token)

    async def _execute_tool_call_impl(
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
        tool = self._tools.get(tool_name)
        if tool is None:
            return _model_retry_result(
                tool_name,
                tool_call_id,
                ModelRetry(
                    f"The requested tool '{tool_name}' is not available.",
                    code=ErrorCode.TOOL_NOT_FOUND,
                ),
            )

        try:
            validated_inputs = tool.validate_inputs(inputs)
        except ModelRetry as exception:
            _log_tool_exception(exception)
            return _model_retry_result(tool_name, tool_call_id, exception)

        if tool.requires_approval and tool_call_id not in approved_tool_calls:
            raise ApprovalRequired(tool_call_id, tool_name, validated_inputs)

        try:
            result = await self._invoke_tool(tool, validated_inputs, context)
            return ToolCallResult(tool_name, tool_call_id, result, False)
        except ApprovalRequired:
            raise
        except Exception as exception:
            _log_tool_exception(exception)
            return _classified_tool_result(tool_name, tool_call_id, exception)

    async def _execute_tool_stream(
        self,
        tool_name: str,
        tool_call_id: str,
        inputs: dict[str, Any],
        context: ContextT | None,
        approved_tool_calls: set[str],
    ) -> AsyncIterator[ToolStreamEvent | ToolCallResult | Any]:
        """Execute a streaming tool with call-scoped correlation context."""
        name_token = tool_name_var.set(tool_name)
        call_token = tool_call_id_var.set(tool_call_id)
        attempt_token = tool_attempt_var.set(None)
        max_attempts_token = tool_max_attempts_var.set(None)
        try:
            async for event in self._execute_tool_stream_impl(
                tool_name,
                tool_call_id,
                inputs,
                context,
                approved_tool_calls,
            ):
                yield event
        finally:
            tool_max_attempts_var.reset(max_attempts_token)
            tool_attempt_var.reset(attempt_token)
            tool_call_id_var.reset(call_token)
            tool_name_var.reset(name_token)

    async def _execute_tool_stream_impl(
        self,
        tool_name: str,
        tool_call_id: str,
        inputs: dict[str, Any],
        context: ContextT | None,
        approved_tool_calls: set[str],
    ) -> AsyncIterator[ToolStreamEvent | ToolCallResult | Any]:
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

        try:
            validated_inputs = tool.validate_inputs(inputs)
        except ModelRetry as exception:
            _log_tool_exception(exception)
            yield _model_retry_result(tool_name, tool_call_id, exception)
            return

        if tool.requires_approval and tool_call_id not in approved_tool_calls:
            raise ApprovalRequired(tool_call_id, tool_name, validated_inputs)

        try:
            if tool.stream_output:
                logger.debug(f"Executing tool: {tool_name}")
                async for event in self._stream_tool(tool, validated_inputs, context):
                    yield event
            else:
                result = await self._invoke_tool(tool, validated_inputs, context)
                yield result
        except ApprovalRequired:
            raise
        except Exception as exception:
            _log_tool_exception(exception)
            yield _classified_tool_result(tool_name, tool_call_id, exception)
