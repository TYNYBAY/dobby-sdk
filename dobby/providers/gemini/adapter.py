"""Gemini provider for Google's Gemini models.

This module provides the GeminiProvider class for interacting with
Google's Gemini API via the google-genai SDK.
"""

import base64
from collections.abc import AsyncIterator, Iterable
from typing import Any, Literal, NoReturn, overload

from google import genai
from google.genai import (
    errors as gemini_errors,
    types as genai_types,
)

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
from .converters import to_gemini_messages

__all__ = ["GeminiProvider"]


class GeminiProvider(Provider[genai.Client]):
    """Provider for Google Gemini models using the google-genai SDK.

    Supports both the Gemini Developer API and Vertex AI backends.
    Implements streaming and non-streaming chat completions with tool support.

    Attributes:
        api_key: API key for Gemini Developer API (optional if using Vertex AI)
        vertexai: Whether to use Vertex AI backend
        project: GCP project ID (required for Vertex AI)
        location: GCP location (default: us-central1)

    Example:
        ```python
        # Gemini Developer API
        provider = GeminiProvider(model="gemini-2.5-flash", api_key="...")

        # Vertex AI
        provider = GeminiProvider(
            model="gemini-2.5-flash",
            vertexai=True,
            project="my-project",
        )
        ```
    """

    api_key: str | None
    vertexai: bool
    project: str | None
    location: str
    _model: str
    _client: genai.Client
    max_retries: int

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        vertexai: bool = False,
        project: str | None = None,
        location: str = "us-central1",
        max_retries: int = 3,
    ):
        """Initialize Gemini provider.

        Args:
            model: Model name (default: 'gemini-2.5-flash' for stable production use).
                Options: 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
                        'gemini-3-flash-preview', 'gemini-3-pro-preview'
            api_key: API key for Gemini Developer API.
                Uses GEMINI_API_KEY or GOOGLE_API_KEY env var if not provided.
            vertexai: Whether to use Vertex AI instead of Developer API.
            project: GCP project ID (required for Vertex AI).
            location: GCP location (default: us-central1).
            max_retries: Maximum retry attempts for transient errors (default: 3).
        """
        self.api_key = api_key
        self.vertexai = vertexai
        self.project = project
        self.location = location
        self._model = model
        self.max_retries = max_retries

        # Initialize client based on backend
        if vertexai:
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            self._client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        """Provider name."""
        return "gemini-vertexai" if self.vertexai else "gemini"

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
                raise DobbyProviderError(
                    str(e), provider=self.name, status_code=status
                ) from e
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
            **kwargs: Additional Gemini-specific parameters

        Returns:
            StreamEndEvent for non-streaming, AsyncIterator[StreamEvent] for streaming
        """
        # Convert messages to Gemini format
        gemini_contents = to_gemini_messages(messages)

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            # Disable automatic function calling - we handle it manually
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )

        if system_prompt:
            config.system_instruction = system_prompt

        if tools:
            config.tools = tools # type: ignore[assignment]

        if stream:
            return self._stream_chat_completion(gemini_contents, config)

        return await self._non_stream_chat_completion(gemini_contents, config)

    @with_retries
    async def _non_stream_chat_completion(
        self,
        contents: list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
    ) -> StreamEndEvent:
        """Non-streaming chat completion with retry support.

        Args:
            contents: Gemini-formatted content list.
            config: Generation configuration.

        Returns:
            StreamEndEvent with complete response.
        """
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            self._translate_error(e)

        return self._parse_response(response)

    def _parse_response(self, response: genai_types.GenerateContentResponse) -> StreamEndEvent:
        """Parse a non-streaming Gemini response into StreamEndEvent.

        Args:
            response: Gemini GenerateContentResponse object

        Returns:
            StreamEndEvent with parsed parts and usage
        """
        parts: list[ResponsePart] = []
        stop_reason: StopReason = "end_turn"

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
                                id=part.function_call.id or f"call_{part.function_call.name}",
                                name=part.function_call.name, # type: ignore[assignment]
                                inputs=dict(func_args) if func_args else {},
                                metadata={"signature": signature},
                            )
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
            model=self._model,
            parts=parts,
            stop_reason=stop_reason,
            usage=usage,
        )

    @with_retries
    async def _stream_chat_completion(
        self,
        contents: list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
    ) -> AsyncIterator[StreamEvent]:
        """Stream chat completion yielding discriminated events.

        Processes streaming events from Gemini API and yields
        typed StreamEvent objects for text and tool calls.
        Accumulates usage metadata from the stream.

        Args:
            contents: Gemini-formatted content list
            config: Generation configuration

        Yields:
            StreamEvent objects: StreamStartEvent, TextDeltaEvent,
            ToolUseEvent, StreamEndEvent
        """
        accumulated_text: str = ""
        function_calls: list[ToolUseEvent] = []
        stream_started = False
        final_usage: Usage | None = None
        stop_reason: StopReason = "end_turn"

        try:
            stream_response = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            self._translate_error(e)

        async for chunk in stream_response:
            # Emit stream start on first chunk
            if not stream_started:
                yield StreamStartEvent(
                    id=f"gemini_{self._model}",
                    model=self._model,
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
                    # Capture finish reason from the final candidate chunk
                    if candidate.finish_reason:
                        match candidate.finish_reason:
                            case "STOP":
                                stop_reason = "end_turn"
                            case "MAX_TOKENS":
                                stop_reason = "max_tokens"
                            case "SAFETY":
                                stop_reason = "content_filter"

                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                accumulated_text += part.text
                                yield TextDeltaEvent(delta=part.text)
                            elif part.function_call:
                                func_args = part.function_call.args
                                # Per https://ai.google.dev/gemini-api/docs/thought-signatures#faqs:
                                # > You can set the following dummy signatures of either "context_engineering_is_the_way_to_go"
                                # > or "skip_thought_signature_validator"
                                # Per https://cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures#using-rest-or-manual-handling:
                                # > You can set thought_signature to skip_thought_signature_validator
                                # We use "skip_thought_signature_validator" as it works for both Gemini API and Vertex AI.
                                sig_bytes = part.thought_signature or b"skip_thought_signature_validator"
                                signature = base64.b64encode(sig_bytes).decode("utf-8")

                                tool_event = ToolUseEvent(
                                    id=part.function_call.id or f"call_{part.function_call.name}",
                                    name=part.function_call.name, # type: ignore[assignment]
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
            model=self._model,
            parts=parts,
            stop_reason=stop_reason,
            usage=final_usage,
        )
