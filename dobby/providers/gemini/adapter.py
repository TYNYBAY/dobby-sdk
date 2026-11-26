"""Gemini provider for Google's Gemini models.

This module provides the GeminiProvider class for interacting with
Google's Gemini API via the google-genai SDK.
"""

import base64
from collections.abc import AsyncIterator, Iterable
import json
from typing import Any, Literal, NoReturn, overload

from google import genai
from google.genai import (
    errors as gemini_errors,
    types as genai_types,
)

from ...types import (
    AssistantMessagePart,
    MessagePart,
    ResponsePart,
    StopReason,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolResultPart,
    ToolUseErrorEvent,
    ToolUseEvent,
    ToolUsePart,
    Usage,
    UserMessagePart,
)
from .._retry import with_retries
from ..base import (
    APITimeoutError as DobbyAPITimeoutError,
    InternalServerError as DobbyInternalServerError,
    Provider,
    ProviderError as DobbyProviderError,
    RateLimitError as DobbyRateLimitError,
    ToolCallTruncatedError,
    _iter_translated,
)
from .converters import to_gemini_messages

__all__ = ["GeminiProvider"]

# Text of the executor's control-flow placeholder results (see
# `dobby.executor._control_flow_result`), which reach the history as
# `TextPart(text=str(result))`. A call answered only by one of these was paused,
# not completed, so its ID stays free for the resumed call.
_PLACEHOLDER_RESULT_TEXTS = frozenset(
    str(result) for result in ({"approval_required": True}, {"cancelled": True})
)


def _is_placeholder_result(part: ToolResultPart) -> bool:
    """Return whether a tool result is an approval or cancellation placeholder."""
    return (
        part.is_error
        and len(part.parts) == 1
        and isinstance(part.parts[0], TextPart)
        and part.parts[0].text in _PLACEHOLDER_RESULT_TEXTS
    )


def _taken_tool_call_ids(messages: list[MessagePart]) -> frozenset[str]:
    """Collect history tool-call IDs that a new Gemini call must not reuse.

    An ID is taken when an assistant `ToolUsePart` used it and a later user
    message holds a real (non-placeholder) `ToolResultPart` for it. A call with
    no result, or only approval/cancellation placeholders, does not take its ID,
    so a call regenerated after a pause keeps `call_{name}` and still matches
    the host's `approved_tool_calls`.

    Args:
        messages: The request's conversation history.

    Returns:
        IDs already bound to completed tool calls.
    """
    used: set[str] = set()
    taken: set[str] = set()
    for message in messages:
        for part in message.parts:
            if isinstance(message, AssistantMessagePart) and isinstance(part, ToolUsePart):
                used.add(part.id)
            elif (
                isinstance(message, UserMessagePart)
                and isinstance(part, ToolResultPart)
                and part.tool_use_id in used
                and not _is_placeholder_result(part)
            ):
                taken.add(part.tool_use_id)
    return frozenset(taken)


def _mint_tool_call_id(function_call: genai_types.FunctionCall, taken: set[str]) -> str:
    """Return a unique ID for a Gemini function call and mark it as taken.

    A Gemini-provided ID is used as is. A missing ID becomes `call_{name}`, or
    `call_{name}_2`, `call_{name}_3`, ... when that is already taken.

    Args:
        function_call: The function call from the Gemini response.
        taken: IDs unavailable for this response. Updated in place.

    Returns:
        The tool-call ID to expose.
    """
    if function_call.id:
        taken.add(function_call.id)
        return function_call.id
    base = f"call_{function_call.name}"
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _thinking_config(reasoning_effort: str | int | None) -> genai_types.ThinkingConfig | None:
    """Map dobby's `reasoning_effort` onto a Gemini `ThinkingConfig`.

    Args:
        reasoning_effort: `None` for no thinking config, an `int` thinking
            budget in tokens, or a `str` thinking level (e.g. "low", "high").

    Returns:
        The thinking config, or None when `reasoning_effort` is None.

    Raises:
        TypeError: If `reasoning_effort` is not None, int or str (bool is
            rejected even though it subclasses int).
    """
    if reasoning_effort is None:
        return None
    if isinstance(reasoning_effort, str):
        return genai_types.ThinkingConfig(thinking_level=reasoning_effort)  # type: ignore[arg-type]
    if isinstance(reasoning_effort, int) and not isinstance(reasoning_effort, bool):
        return genai_types.ThinkingConfig(thinking_budget=reasoning_effort)
    raise TypeError(
        "Gemini provider requires reasoning_effort as int (thinking_budget), "
        f"str (thinking_level) or None, got {type(reasoning_effort).__name__}"
    )


class GeminiProvider(Provider[genai.Client]):
    """Provider for Google Gemini models using the google-genai SDK.

    Targets the Gemini Developer API. Implements streaming and non-streaming
    chat completions with tool support. For Vertex-hosted models use
    `VertexAIProvider`.

    Attributes:
        api_key: API key for the Gemini Developer API

    Example:
        ```python
        # Gemini Developer API
        provider = GeminiProvider(model="gemini-2.5-flash", api_key="...")
        ```
    """

    api_key: str | None
    _model: str
    _client: genai.Client
    max_retries: int

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        max_retries: int = 3,
    ):
        """Initialize Gemini provider.

        Args:
            model: Model name (default: 'gemini-2.5-flash' for stable production use).
                Options: 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
                        'gemini-3-flash-preview', 'gemini-3-pro-preview'
            api_key: API key for Gemini Developer API.
                Uses GEMINI_API_KEY or GOOGLE_API_KEY env var if not provided.
            max_retries: Keyword-only. Maximum retry attempts for transient errors
                (default: 3).

        `max_retries` is keyword-only because `vertexai` used to occupy the third
        positional slot. Without the boundary, a caller still writing
        `GeminiProvider("gemini-2.5-flash", None, True)` would bind `True` to the
        retry count and be silently routed to the Developer API when they meant
        Vertex. Keyword-only turns that into a loud `TypeError`.
        """
        self.api_key = api_key
        self._model = model
        self.max_retries = max_retries

        self._client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        """Provider name."""
        return "gemini"

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    @property
    def client(self) -> genai.Client:
        """Authenticated client instance."""
        return self._client

    def _translate_error(self, e: Exception) -> NoReturn:
        """Map Gemini SDK exceptions to unified dobby errors.

        Always raises — never returns normally.

        Args:
            e: The original Gemini SDK exception.

        Raises:
            DobbyRateLimitError: For rate limit errors (429).
            DobbyAPITimeoutError: For timeout errors (408).
            DobbyAPIConnectionError: For connection failures.
            DobbyInternalServerError: For server errors (5xx).
            DobbyProviderError: For all other API errors.
        """
        match e:
            case gemini_errors.ClientError() if getattr(e, "status", None) == 429:
                raise DobbyRateLimitError(str(e), provider=self.name) from e
            case gemini_errors.ClientError() if getattr(e, "status", None) == 408:
                raise DobbyAPITimeoutError(str(e), provider=self.name) from e
            case gemini_errors.ServerError():
                status = getattr(e, "status", 500)
                raise DobbyInternalServerError(
                    str(e), provider=self.name, status_code=status
                ) from e
            case gemini_errors.ClientError():
                status = getattr(e, "status", None)
                raise DobbyProviderError(str(e), provider=self.name, status_code=status) from e
            case _:
                raise DobbyProviderError(str(e), provider=self.name) from e

    @overload
    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: Literal[False] = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[genai_types.Tool] | None = None,
        model: str | None = None,
        reasoning_effort: str | int | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamEndEvent: ...

    @overload
    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: Literal[True],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[genai_types.Tool] | None = None,
        model: str | None = None,
        reasoning_effort: str | int | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]: ...

    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[genai_types.Tool] | None = None,
        model: str | None = None,
        reasoning_effort: str | int | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> StreamEndEvent | AsyncIterator[StreamEvent]:
        """Generate response from messages using Gemini API.

        Converts provider-agnostic messages to Gemini format and handles both
        streaming and non-streaming responses.

        Args:
            messages: Conversation history with user/assistant/tool messages
            stream: Whether to stream response chunks
            system_prompt: Optional system message to guide behavior
            temperature: Controls randomness (0.0-2.0, default 0.0)
            tools: List of Gemini Tool objects (pre-converted by executor)
            model: Per-call model override. Falls back to the instance model.
            reasoning_effort: Thinking control. `None` sets no thinking config, an
                `int` sets `thinking_budget` (tokens), a `str` sets
                `thinking_level` (e.g. "low", "high").
            max_tokens: Maximum output tokens, sent as `max_output_tokens`.
            **kwargs: Additional Gemini-specific parameters forwarded verbatim to
                GenerateContentConfig (e.g. top_p, stop_sequences). Passing
                `thinking_config` here together with a non-None
                `reasoning_effort` is ambiguous and rejected.

        Returns:
            StreamEndEvent for non-streaming, AsyncIterator[StreamEvent] for streaming

        Raises:
            TypeError: If `reasoning_effort` is not None, int or str.
            ValueError: If both `reasoning_effort` and `thinking_config` are given.
        """
        thinking_config = _thinking_config(reasoning_effort)
        if thinking_config is not None:
            if "thinking_config" in kwargs:
                raise ValueError(
                    "Pass either reasoning_effort or thinking_config to GeminiProvider.chat, "
                    "not both"
                )
            kwargs["thinking_config"] = thinking_config

        messages = list(messages)
        taken_ids = _taken_tool_call_ids(messages)
        # Convert messages to Gemini format
        gemini_contents = to_gemini_messages(messages)
        target_model = model or self._model

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            # Disable automatic function calling - we handle it manually
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            # Forward provider-native params verbatim (e.g. top_p, stop_sequences).
            **kwargs,
        )

        if system_prompt:
            config.system_instruction = system_prompt

        if tools:
            config.tools = tools  # type: ignore[assignment]

        if max_tokens is not None:
            config.max_output_tokens = max_tokens

        if stream:
            return self._stream_chat_completion(gemini_contents, config, target_model, taken_ids)

        return await self._non_stream_chat_completion(
            gemini_contents, config, target_model, taken_ids
        )

    @with_retries
    async def _non_stream_chat_completion(
        self,
        contents: list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
        model: str,
        taken_ids: frozenset[str] = frozenset(),
    ) -> StreamEndEvent:
        """Non-streaming chat completion with retry support.

        Args:
            contents: Gemini-formatted content list.
            config: Generation configuration.
            model: Model id to generate with.
            taken_ids: History tool-call IDs new calls must not reuse.

        Returns:
            StreamEndEvent with complete response.
        """
        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            self._translate_error(e)

        return self._parse_response(response, model, taken_ids)

    def _parse_response(
        self,
        response: genai_types.GenerateContentResponse,
        model: str,
        taken_ids: frozenset[str] = frozenset(),
    ) -> StreamEndEvent:
        """Parse a non-streaming Gemini response into StreamEndEvent.

        Args:
            response: Gemini GenerateContentResponse object
            model: Model id used for the request
            taken_ids: History tool-call IDs new calls must not reuse.

        Returns:
            StreamEndEvent with parsed parts and usage
        """
        parts: list[ResponsePart] = []
        stop_reason: StopReason = "end_turn"
        truncated = False
        taken = set(taken_ids)

        # Extract parts from the response
        if response.candidates:
            candidate = response.candidates[0]

            # Check finish reason
            if candidate.finish_reason:
                match candidate.finish_reason:
                    case "STOP":
                        stop_reason = "end_turn"
                    case "MAX_TOKENS":
                        stop_reason = "max_tokens"
                        truncated = True
                    case "SAFETY":
                        stop_reason = "content_filter"
                    case _:
                        stop_reason = "end_turn"

            # Extract content parts
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        parts.append(TextPart(text=part.text))
                    elif part.function_call:
                        stop_reason = "tool_use"
                        func_args = part.function_call.args
                        # Per https://ai.google.dev/gemini-api/docs/thought-signatures#faqs:
                        # > You can set the following dummy signatures of either "context_engineering_is_the_way_to_go"
                        # > or "skip_thought_signature_validator"
                        # We use "skip_thought_signature_validator" as it works for both Gemini API and Vertex AI.
                        sig_bytes = part.thought_signature or b"skip_thought_signature_validator"
                        signature = base64.b64encode(sig_bytes).decode("utf-8")

                        parts.append(
                            ToolUsePart(
                                id=_mint_tool_call_id(part.function_call, taken),
                                name=part.function_call.name,  # type: ignore[assignment]
                                inputs=dict(func_args) if func_args else {},
                                metadata={"signature": signature},
                            )
                        )

        # Gemini parses tool args server-side (no client-side json.loads), so a
        # max_tokens cutoff yields a partial/empty `args` dict rather than a parse
        # error. Conservatively treat any tool call under MAX_TOKENS as truncated
        # and surface a typed error instead of executing the tool on partial input.
        if truncated:
            truncated_tool = next((p for p in parts if isinstance(p, ToolUsePart)), None)
            if truncated_tool is not None:
                raise ToolCallTruncatedError(
                    f"Tool call '{truncated_tool.name}' was truncated by max_tokens",
                    provider=self.name,
                    tool_name=truncated_tool.name,
                    tool_id=truncated_tool.id,
                    partial_inputs=truncated_tool.inputs,
                )

        # Extract usage
        usage: Usage | None = None
        if response.usage_metadata:
            usage = Usage(
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
                total_tokens=response.usage_metadata.total_token_count or 0,
            )

        return StreamEndEvent(
            model=model,
            parts=parts,
            stop_reason=stop_reason,
            usage=usage,
        )

    @with_retries
    async def _stream_chat_completion(
        self,
        contents: list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
        model: str,
        taken_ids: frozenset[str] = frozenset(),
    ) -> AsyncIterator[StreamEvent]:
        """Stream chat completion yielding discriminated events.

        Processes streaming events from Gemini API and yields
        typed StreamEvent objects for text and tool calls.
        Accumulates usage metadata from the stream.

        Args:
            contents: Gemini-formatted content list
            config: Generation configuration
            model: Model id to generate with
            taken_ids: History tool-call IDs new calls must not reuse.

        Yields:
            StreamEvent objects: StreamStartEvent, TextDeltaEvent,
            ToolUseEvent, StreamEndEvent
        """
        accumulated_text: str = ""
        function_calls: list[ToolUseEvent] = []
        stream_started = False
        final_usage: Usage | None = None
        stop_reason: StopReason = "end_turn"
        taken = set(taken_ids)

        try:
            stream_response = await self._client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            self._translate_error(e)

        async for chunk in _iter_translated(stream_response, self._translate_error):
            # Emit stream start on first chunk
            if not stream_started:
                yield StreamStartEvent(
                    id=f"gemini_{model}",
                    model=model,
                )
                stream_started = True

            # Trigger error if prompt was blocked
            if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                yield StreamErrorEvent(
                    error_code=str(chunk.prompt_feedback.block_reason),
                    error_message=f"Prompt blocked: {chunk.prompt_feedback.block_reason}",
                )
                return

            # Capture usage if present (typically in the last chunk)
            if chunk.usage_metadata:
                final_usage = Usage(
                    input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                    output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                    total_tokens=chunk.usage_metadata.total_token_count or 0,
                )

            # Process candidates
            if chunk.candidates:
                for candidate in chunk.candidates:
                    candidate_truncated = False
                    # Capture finish reason from the final candidate chunk
                    if candidate.finish_reason:
                        match candidate.finish_reason:
                            case "STOP":
                                stop_reason = "end_turn"
                            case "MAX_TOKENS":
                                stop_reason = "max_tokens"
                                candidate_truncated = True
                            case "SAFETY":
                                stop_reason = "content_filter"

                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                accumulated_text += part.text
                                yield TextDeltaEvent(delta=part.text)
                            elif part.function_call:
                                func_args = part.function_call.args
                                # A max_tokens cutoff leaves partial/empty args.
                                # Surface a typed event and keep the stream alive
                                # rather than emitting a tool call on partial input.
                                if candidate_truncated:
                                    yield ToolUseErrorEvent(
                                        id=_mint_tool_call_id(part.function_call, taken),
                                        name=part.function_call.name or "",
                                        raw_arguments=(
                                            json.dumps(dict(func_args)) if func_args else ""
                                        ),
                                        error="Tool call truncated by max_tokens",
                                    )
                                    continue
                                # Per https://ai.google.dev/gemini-api/docs/thought-signatures#faqs:
                                # > You can set the following dummy signatures of either "context_engineering_is_the_way_to_go"
                                # > or "skip_thought_signature_validator"
                                # Per https://cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures#using-rest-or-manual-handling:
                                # > You can set thought_signature to skip_thought_signature_validator
                                # We use "skip_thought_signature_validator" as it works for both Gemini API and Vertex AI.
                                sig_bytes = (
                                    part.thought_signature or b"skip_thought_signature_validator"
                                )
                                signature = base64.b64encode(sig_bytes).decode("utf-8")

                                tool_event = ToolUseEvent(
                                    id=_mint_tool_call_id(part.function_call, taken),
                                    name=part.function_call.name,  # type: ignore[assignment]
                                    inputs=dict(func_args) if func_args else {},
                                    metadata={"signature": signature},
                                )
                                function_calls.append(tool_event)
                                yield tool_event

        parts: list[ResponsePart] = []
        if accumulated_text:
            parts.append(TextPart(text=accumulated_text))

        for tool_event in function_calls:
            parts.append(
                ToolUsePart(
                    id=tool_event.id,
                    name=tool_event.name,
                    inputs=tool_event.inputs,
                    metadata=tool_event.metadata,
                )
            )

        # If tools were called, override the stop reason
        if function_calls:
            stop_reason = "tool_use"

        yield StreamEndEvent(
            model=model,
            parts=parts,
            stop_reason=stop_reason,
            usage=final_usage,
        )
