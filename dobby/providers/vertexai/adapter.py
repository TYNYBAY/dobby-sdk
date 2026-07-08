"""Vertex AI provider for Google Cloud's OpenAI-compatible Model-as-a-Service endpoint.

This module provides the VertexAIProvider class for interacting with Vertex AI's
Model Garden catalog (e.g. Llama, self-deployed containers) via its OpenAI-compatible
Chat Completions endpoint. Authentication uses Application Default Credentials (or an
explicitly supplied `google.auth.credentials.Credentials` object) and refreshes the
bearer token transparently on every request via the OpenAI SDK's native async-callable
`api_key` hook.
"""

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
import json
from typing import Any, Literal, NoReturn, overload

import google.auth
import google.auth.credentials
import google.auth.transport.requests
import openai
from openai import AsyncOpenAI

from ..._logging import logger
from ...types import (
    MessagePart,
    ResponsePart,
    StopReason,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolUseEvent,
    ToolUsePart,
    Usage,
)
from .._retry import with_retries
from ..base import (
    APIConnectionError as DobbyAPIConnectionError,
    APITimeoutError as DobbyAPITimeoutError,
    InternalServerError as DobbyInternalServerError,
    Provider,
    ProviderError as DobbyProviderError,
    RateLimitError as DobbyRateLimitError,
)
from .converters import to_vertexai_messages

__all__ = ["VertexAIProvider"]

# Vertex Chat Completions finish_reason values that map 1:1 (via this table)
# onto Dobby's StopReason. "tool_calls" is included for completeness, but
# `_non_stream_chat_completion` always prefers the presence of `tool_calls` on
# the message itself over this table (some OpenAI-compatible servers report a
# different finish_reason alongside a populated `tool_calls` list).
_FINISH_REASON_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "content_filter",
}


# Bounds for the streaming tool-call accumulator (see _stream_chat_completion).
# The target of this provider includes self-deployed/third-party Model Garden
# containers (per the origin document's scope) — a less-trusted boundary than
# native OpenAI — so an unbounded per-`index` accumulator keyed by
# backend-supplied data is a resource-exhaustion vector worth closing.
#
# 64 distinct tool-call indices comfortably covers any realistic parallel
# tool-call fan-out (real-world usage rarely exceeds single digits) while
# still bounding worst-case dict growth.
_MAX_TOOL_CALL_INDICES = 64
# ~1MB across all indices combined comfortably covers realistic tool-call
# argument payloads (JSON objects with many/large string fields) while
# bounding worst-case memory growth from a misbehaving or malicious stream.
_MAX_ACCUMULATED_TOOL_CALL_ARGUMENTS_LENGTH = 1_000_000


def _map_finish_reason(reason: str | None) -> StopReason:
    """Map a Vertex AI Chat Completions finish_reason onto Dobby's StopReason.

    Unknown or absent reasons collapse to ``"end_turn"`` so a future/unexpected
    value never produces an out-of-contract StopReason.
    """
    mapped = _FINISH_REASON_MAP.get(reason) if reason is not None else None
    if mapped is not None:
        return mapped
    if reason is not None:
        logger.debug(f"Unhandled Vertex AI finish_reason: {reason}")
    return "end_turn"


class VertexAIProvider(Provider[AsyncOpenAI]):
    """Provider for Vertex AI's OpenAI-compatible Model-as-a-Service (MaaS) endpoint.

    Targets Vertex AI's Model Garden catalog (Llama, self-deployed containers, and any
    other serving container that speaks OpenAI's Chat Completions wire format) through
    `POST https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/
    {location}/endpoints/openapi/chat/completions`.

    Auth is via Application Default Credentials (ADC) by default, or an explicitly
    supplied `google.auth.credentials.Credentials` object. The bearer token is kept
    fresh across calls by passing a bound async method as the OpenAI SDK's `api_key`
    parameter — the SDK invokes it before every request (including retries, streaming
    and non-streaming), so no manual header injection or eager refresh is needed.

    Attributes:
        project: GCP project ID.
        location: GCP location (default: us-central1).
        scopes: OAuth scopes forwarded to `google.auth.default()` (only used when no
            explicit `credentials` is supplied).
        max_retries: Maximum retry attempts for transient errors.

    Example:
        ```python
        provider = VertexAIProvider(
            model="meta/llama-3.1-405b-instruct-maas",
            project="my-project",
        )
        ```
    """

    project: str
    location: str
    scopes: Sequence[str] | None
    _model: str
    _credentials: google.auth.credentials.Credentials
    _client: AsyncOpenAI
    max_retries: int

    def __init__(
        self,
        model: str,
        project: str,
        location: str = "us-central1",
        credentials: google.auth.credentials.Credentials | None = None,
        scopes: Sequence[str] | None = None,
        max_retries: int = 3,
    ):
        """Initialize Vertex AI provider.

        Args:
            model: Publisher-qualified model id (e.g. "meta/llama-3.1-405b-instruct-maas"),
                forwarded verbatim — no hardcoded allow-list.
            project: GCP project ID.
            location: GCP location (default: "us-central1").
            credentials: Pre-built credentials object. When omitted, resolved via
                `google.auth.default(scopes=scopes)` at construction time.
            scopes: OAuth scopes forwarded to `google.auth.default()`. Only used when
                `credentials` is not supplied. Defaults to google-auth's own default
                resolution (typically the broad `cloud-platform` scope) — pass a
                narrower list for least-privilege where the ADC source supports it.
            max_retries: Maximum retry attempts for transient errors (default: 3).
        """
        self._model = model
        self.project = project
        self.location = location
        self.scopes = scopes
        self.max_retries = max_retries

        if credentials is not None:
            self._credentials = credentials
        else:
            self._credentials, _ = google.auth.default(scopes=scopes)

        self._refresh_lock = asyncio.Lock()

        self._client = AsyncOpenAI(
            base_url=(
                f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
                f"/locations/{location}/endpoints/openapi"
            ),
            api_key=self._bearer_token,
        )

    @property
    def name(self) -> str:
        """Provider name (distinct from GeminiProvider's "gemini-vertexai")."""
        return "vertexai"

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    @property
    def client(self) -> AsyncOpenAI:
        """Authenticated client instance.

        Fully self-sufficient: the callable `api_key` is invoked by the SDK's own
        request-preparation hook on every request, so direct use of
        `provider.client.chat.completions.create(...)` also gets a fresh token.
        """
        return self._client

    async def _bearer_token(self) -> str:
        """Return a valid bearer token, refreshing it first if necessary.

        Passed as the `api_key` callable to `AsyncOpenAI` — invoked automatically by
        the SDK before every request. Guarded by `self._refresh_lock` so concurrent
        requests racing an expired token only trigger one refresh.
        """
        async with self._refresh_lock:
            if not self._credentials.valid:
                await asyncio.to_thread(
                    self._credentials.refresh, google.auth.transport.requests.Request()
                )
            return self._credentials.token

    @overload
    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: Literal[False] = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        model: str | None = None,
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
        tools: list[Any] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]: ...

    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> StreamEndEvent | AsyncIterator[StreamEvent]:
        """Generate a response from conversation messages.

        Converts provider-agnostic messages to Chat Completions format and
        delegates to the non-streaming or streaming implementation.

        Args:
            messages: Conversation history with user/assistant/tool messages.
            stream: Whether to stream response chunks.
            system_prompt: Optional system message to guide behavior.
            temperature: Controls randomness (0.0-2.0, default 0.0).
            tools: Tool definitions already formatted for Vertex's Chat
                Completions endpoint (see `to_vertexai_tool()`), matching how
                sibling providers (`OpenAIProvider`, `GeminiProvider`) expect
                pre-formatted tools rather than converting them here.
            model: Per-call model override. Falls back to the instance model.
            **kwargs: Reserved for future Vertex-specific parameters.

        Returns:
            StreamEndEvent for non-streaming, AsyncIterator[StreamEvent] for streaming.
        """
        vertexai_messages = to_vertexai_messages(messages)
        if system_prompt is not None:
            vertexai_messages.insert(0, {"role": "system", "content": system_prompt})

        target_model = model or self._model

        if stream:
            return self._stream_chat_completion(
                vertexai_messages, target_model, temperature, tools
            )

        return await self._non_stream_chat_completion(
            vertexai_messages, target_model, temperature, tools
        )

    @staticmethod
    def _build_kwargs(
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for chat.completions.create(), excluding unset optionals.

        Args:
            model: Model id.
            messages: Chat-Completions-formatted messages.
            temperature: Sampling temperature.
            tools: Optional, already Chat-Completions-formatted tool schemas.

        Returns:
            Dictionary of kwargs to pass to chat.completions.create().
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools is not None:
            kwargs["tools"] = tools
        return kwargs

    def _translate_error(self, e: Exception) -> NoReturn:
        """Map OpenAI SDK exceptions to unified dobby errors.

        Same exception hierarchy as `OpenAIProvider._translate_error` since
        `AsyncOpenAI` raises `openai.*` errors regardless of which endpoint or
        sub-resource is hit.

        Always raises — never returns normally.

        Args:
            e: The original OpenAI SDK exception.

        Raises:
            DobbyRateLimitError: For rate limit errors (429).
            DobbyAPIConnectionError: For connection failures.
            DobbyAPITimeoutError: For request timeouts.
            DobbyInternalServerError: For server errors (5xx).
            DobbyProviderError: For all other API errors.
        """
        match e:
            case openai.RateLimitError():
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    raw = e.response.headers.get("retry-after")
                    if raw is not None:
                        try:
                            retry_after = float(raw)
                        except (ValueError, TypeError):
                            pass
                raise DobbyRateLimitError(
                    str(e), provider=self.name, retry_after=retry_after
                ) from e
            case openai.APITimeoutError():
                raise DobbyAPITimeoutError(str(e), provider=self.name) from e
            case openai.APIConnectionError():
                raise DobbyAPIConnectionError(str(e), provider=self.name) from e
            case openai.InternalServerError():
                raise DobbyInternalServerError(
                    str(e), provider=self.name, status_code=e.status_code
                ) from e
            case openai.APIStatusError():
                raise DobbyProviderError(
                    str(e), provider=self.name, status_code=e.status_code
                ) from e
            case _:
                raise DobbyProviderError(str(e), provider=self.name) from e

    @with_retries
    async def _non_stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
    ) -> StreamEndEvent:
        """Non-streaming chat completion with retry support.

        No manual auth handling at this call site — the client's callable
        `api_key` (see `_bearer_token`) refreshes transparently via the SDK's
        own request-preparation hook.

        Args:
            messages: Chat-Completions-formatted messages.
            model: Model id (per-call override or instance model).
            temperature: Sampling temperature.
            tools: Optional, already Chat-Completions-formatted tool schemas.

        Returns:
            StreamEndEvent with the complete response.
        """
        create_kwargs = self._build_kwargs(
            model=model, messages=messages, temperature=temperature, tools=tools
        )

        try:
            response = await self._client.chat.completions.create(**create_kwargs)
        except Exception as e:
            self._translate_error(e)

        message = response.choices[0].message

        parts: list[ResponsePart] = []
        if message.content:
            parts.append(TextPart(text=message.content))

        tool_calls = message.tool_calls or []
        for tool_call in tool_calls:
            parts.append(
                ToolUsePart(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    inputs=json.loads(tool_call.function.arguments),
                )
            )

        stop_reason: StopReason = (
            "tool_use" if tool_calls else _map_finish_reason(response.choices[0].finish_reason)
        )

        usage: Usage | None = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return StreamEndEvent(
            model=response.model or model,
            parts=parts,
            stop_reason=stop_reason,
            usage=usage,
        )

    @with_retries
    async def _stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming chat completion with retry support.

        Parses Chat Completions SSE delta chunks (`choices[0].delta`) into
        Dobby's discriminated StreamEvent sequence. No manual auth handling at
        this call site — same as `_non_stream_chat_completion`.

        Tool-call fragments are keyed by `index` (the only safe correlation
        key across chunks — `id` may not appear on every fragment) and merged
        idempotently: `id`/`function.name` are set-if-present on *any* chunk
        for that index, not assumed to only arrive on the first one. `usage`
        is captured via `chunk.usage is not None`, independent of whether
        `chunk.choices` is empty — real third-party (often vLLM-backed)
        OpenAI-compatible servers have shipped both deviations from native
        OpenAI's streaming conventions. See the accumulator bound constants
        above for the resource-exhaustion mitigation this implies.

        Args:
            messages: Chat-Completions-formatted messages.
            model: Model id (per-call override or instance model).
            temperature: Sampling temperature.
            tools: Optional, already Chat-Completions-formatted tool schemas.

        Yields:
            StreamEvent objects: StreamStartEvent, TextDeltaEvent,
            ToolUseEvent, StreamErrorEvent, StreamEndEvent.
        """
        create_kwargs = self._build_kwargs(
            model=model, messages=messages, temperature=temperature, tools=tools
        )
        create_kwargs["stream"] = True
        create_kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = await self._client.chat.completions.create(**create_kwargs)
        except Exception as e:
            self._translate_error(e)

        stream_started = False
        model_name: str = model
        accumulated_text: str = ""
        finish_reason: str | None = None
        usage: Usage | None = None

        # index -> {"id": str | None, "name": str | None, "arguments": str}
        tool_call_accumulator: dict[int, dict[str, Any]] = {}
        accumulated_arguments_length = 0

        # Iterate manually so mid-stream transport errors route through the same
        # unified error translation as the initial request (mirrors
        # AnthropicProvider._stream_chat_completion's manual __anext__ pattern).
        stream_iter = stream.__aiter__()
        while True:
            try:
                chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                break
            except Exception as e:
                self._translate_error(e)

            if not stream_started:
                yield StreamStartEvent(
                    id=getattr(chunk, "id", None) or f"vertexai_{model}",
                    model=getattr(chunk, "model", None) or model_name,
                )
                stream_started = True

            if getattr(chunk, "model", None):
                model_name = chunk.model

            # Checked independent of `choices` length: some OpenAI-compatible
            # (vLLM-backed) servers don't guarantee a dedicated empty-choices
            # usage chunk the way native OpenAI does.
            if chunk.usage is not None:
                usage = Usage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                accumulated_text += delta.content
                yield TextDeltaEvent(delta=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    index = tc_delta.index
                    is_new_index = index not in tool_call_accumulator

                    if is_new_index and len(tool_call_accumulator) >= _MAX_TOOL_CALL_INDICES:
                        yield StreamErrorEvent(
                            error_code="tool_call_accumulator_overflow",
                            error_message=(
                                "Exceeded max distinct tool-call indices "
                                f"({_MAX_TOOL_CALL_INDICES}) in a single stream; "
                                "stopping accumulation."
                            ),
                        )
                        return

                    fragment = ""
                    if tc_delta.function is not None and tc_delta.function.arguments:
                        fragment = tc_delta.function.arguments

                    if (
                        accumulated_arguments_length + len(fragment)
                        > _MAX_ACCUMULATED_TOOL_CALL_ARGUMENTS_LENGTH
                    ):
                        yield StreamErrorEvent(
                            error_code="tool_call_accumulator_overflow",
                            error_message=(
                                "Exceeded max accumulated tool-call arguments length "
                                f"({_MAX_ACCUMULATED_TOOL_CALL_ARGUMENTS_LENGTH} chars) "
                                "in a single stream; stopping accumulation."
                            ),
                        )
                        return

                    if is_new_index:
                        tool_call_accumulator[index] = {
                            "id": None,
                            "name": None,
                            "arguments": "",
                        }
                    acc = tool_call_accumulator[index]

                    # Idempotent set-if-present merge, not "only first chunk":
                    # real third-party servers have shipped id/name on later
                    # fragments for the same index instead of the first one.
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function is not None and tc_delta.function.name:
                        acc["name"] = tc_delta.function.name
                    if fragment:
                        acc["arguments"] += fragment
                        accumulated_arguments_length += len(fragment)

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        parts: list[ResponsePart] = []
        if accumulated_text:
            parts.append(TextPart(text=accumulated_text))

        # A tool call's arguments are only safe to treat as "fully assembled"
        # once the whole stream has ended — Chat Completions has no per-index
        # completion signal, and third-party servers may still add id/name to
        # an index on a later chunk (see the idempotent merge above).
        for acc in tool_call_accumulator.values():
            tool_event = ToolUseEvent(
                id=acc["id"] or "",
                name=acc["name"] or "",
                inputs=json.loads(acc["arguments"]) if acc["arguments"] else {},
            )
            yield tool_event
            parts.append(
                ToolUsePart(id=tool_event.id, name=tool_event.name, inputs=tool_event.inputs)
            )

        stop_reason: StopReason = (
            "tool_use" if tool_call_accumulator else _map_finish_reason(finish_reason)
        )

        yield StreamEndEvent(
            model=model_name,
            parts=parts,
            stop_reason=stop_reason,
            usage=usage,
        )
