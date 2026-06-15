"""Anthropic Claude provider for Dobby SDK.

Implements the Provider interface for Anthropic's Messages API,
supporting both direct Anthropic API and Azure-hosted Claude endpoints.
"""

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any, Literal, NoReturn, overload

import anthropic
from anthropic import AsyncAnthropic, AsyncAnthropicFoundry
from anthropic.lib.foundry import AsyncAzureADTokenProvider

from ..._logging import logger
from ...types import (
    AssistantMessagePart,
    MessagePart,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningPart,
    ReasoningStartEvent,
    ResponsePart,
    StopReason,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolResultPart,
    ToolUseEvent,
    ToolUsePart,
    Usage,
    UserMessagePart,
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
from .converters import AnthropicContentBlock, content_part_to_anthropic

DEFAULT_MAX_TOKENS = 8192


class AnthropicProvider(Provider[AsyncAnthropic]):
    """Provider for Anthropic Claude and Azure-hosted Claude using Messages API.

    Supports direct Anthropic API and Azure AI Foundry deployments.
    For Azure, uses AsyncAnthropicFoundry which sends the correct "api-key"
    auth header and sets the proper base URL.

    Azure env vars (read automatically by SDK if params not passed explicitly):
        ANTHROPIC_FOUNDRY_API_KEY: Azure API key.
        ANTHROPIC_FOUNDRY_RESOURCE: Azure resource name.
        ANTHROPIC_FOUNDRY_BASE_URL: Full Azure base URL (alternative to resource).
    """

    api_key: str | None
    base_url: str | None
    _model: str
    _client: AsyncAnthropic
    _is_azure: bool
    max_retries: int

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        resource: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        max_retries: int = 3,
    ):
        """Initialize Anthropic provider.

        For direct Anthropic:
            api_key: Uses ANTHROPIC_API_KEY env var if not provided.

        For Azure AI Foundry (uses AsyncAnthropicFoundry):
            api_key: Azure key. Uses ANTHROPIC_FOUNDRY_API_KEY env var if not provided.
            resource: Azure resource name (e.g. "my-resource"). Mutually exclusive with base_url.
            base_url: Full Azure endpoint URL. Uses ANTHROPIC_FOUNDRY_BASE_URL env var if not provided.
            azure_ad_token_provider: Callable returning Azure AD bearer token.
                                     Mutually exclusive with api_key.

        Azure is detected when any of resource, azure_ad_token_provider, or an
        "azure"-containing base_url is provided.
        """
        self.api_key = api_key
        self.base_url = base_url
        self._model = model
        self.max_retries = max_retries

        self._is_azure = bool(
            resource or azure_ad_token_provider or (base_url and "azure" in base_url)
        )

        if self._is_azure:
            # resource and base_url are mutually exclusive in AsyncAnthropicFoundry.
            # Pass only whichever is set; the SDK constructs the URL from resource
            # as https://{resource}.services.ai.azure.com/anthropic/ automatically.
            foundry_kwargs: dict = {"api_key": api_key}
            if azure_ad_token_provider:
                del foundry_kwargs["api_key"]
                foundry_kwargs["azure_ad_token_provider"] = azure_ad_token_provider
            if resource:
                foundry_kwargs["resource"] = resource
            elif base_url:
                foundry_kwargs["base_url"] = base_url
            self._client = AsyncAnthropicFoundry(**foundry_kwargs)
        else:
            self._client = AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
            )

    @property
    def name(self) -> str:
        """Provider name."""
        is_azure = getattr(self, "_is_azure", bool(self.base_url and "azure" in self.base_url))
        return "azure-anthropic" if is_azure else "anthropic"

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    @property
    def client(self) -> AsyncAnthropic:
        """Authenticated client instance."""
        return self._client

    @staticmethod
    def _build_kwargs(
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        thinking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for messages.create() excluding None values.

        Args:
            model: Model name.
            messages: Anthropic-formatted messages.
            max_tokens: Maximum output tokens (required by Anthropic).
            system: Optional system prompt.
            tools: Optional tool definitions.
            temperature: Sampling temperature.
            thinking: Optional extended thinking configuration.

        Returns:
            Dictionary of kwargs to pass to messages.create().
        """
        # Anthropic requires temperature=1.0 when thinking is enabled
        if thinking is not None:
            temperature = 1.0

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system is not None:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools
        if thinking is not None:
            kwargs["thinking"] = thinking

        return kwargs

    def _translate_error(self, e: Exception) -> NoReturn:
        """Map Anthropic SDK exceptions to unified dobby errors.

        Always raises -- never returns normally.

        Args:
            e: The original Anthropic SDK exception.

        Raises:
            DobbyRateLimitError: For rate limit errors (429).
            DobbyAPIConnectionError: For connection failures.
            DobbyAPITimeoutError: For request timeouts.
            DobbyInternalServerError: For server errors (5xx).
            DobbyProviderError: For all other API errors.
        """
        match e:
            case anthropic.RateLimitError():
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
            case anthropic.APITimeoutError():
                raise DobbyAPITimeoutError(str(e), provider=self.name) from e
            case anthropic.APIConnectionError():
                raise DobbyAPIConnectionError(str(e), provider=self.name) from e
            case anthropic.InternalServerError():
                raise DobbyInternalServerError(
                    str(e), provider=self.name, status_code=e.status_code
                ) from e
            case anthropic.APIStatusError():
                raise DobbyProviderError(
                    str(e), provider=self.name, status_code=e.status_code
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
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        reasoning_effort: str | int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> StreamEndEvent: ...

    @overload
    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: Literal[True],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        reasoning_effort: str | int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]: ...

    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> StreamEndEvent | AsyncIterator[StreamEvent]:
        """Generate response from messages using Anthropic Messages API.

        Args:
            messages: Conversation history with user/assistant/tool messages.
            stream: Whether to stream response chunks.
            system_prompt: Optional system message to guide behavior.
            temperature: Controls randomness (0.0-1.0, default 0.0).
            tools: Available tools for function calling.
            reasoning_effort: Thinking budget as int (budget_tokens). Pass None to disable.
            max_tokens: Maximum number of output tokens. Defaults to 8192.
            **kwargs: Additional Anthropic-specific parameters.

        Returns:
            StreamEndEvent for non-streaming, AsyncIterator[StreamEvent] for streaming.
        """
        anthropic_messages = to_anthropic_messages(messages)

        # Build thinking config from reasoning effort (must be int budget_tokens)
        thinking = None
        if reasoning_effort is not None:
            if not isinstance(reasoning_effort, int):
                raise TypeError(
                    f"Anthropic provider requires reasoning_effort as int (budget_tokens), "
                    f"got {type(reasoning_effort).__name__}"
                )
            thinking = {"type": "enabled", "budget_tokens": reasoning_effort}

        effective_max_tokens = max_tokens or DEFAULT_MAX_TOKENS

        if stream:
            return self._stream_chat_completion(
                anthropic_messages,
                self._model,
                system=system_prompt,
                tools=tools,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                thinking=thinking,
            )

        return await self._non_stream_chat_completion(
            anthropic_messages,
            self._model,
            system=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            thinking=thinking,
        )

    @with_retries
    async def _non_stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking: dict[str, Any] | None = None,
    ) -> StreamEndEvent:
        """Non-streaming chat completion with retry support.

        Args:
            messages: Anthropic-formatted messages.
            model: Model name.
            system: Optional system prompt.
            tools: Optional tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            thinking: Optional extended thinking configuration.

        Returns:
            StreamEndEvent with complete response.
        """
        create_kwargs = self._build_kwargs(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            temperature=temperature,
            thinking=thinking,
        )

        try:
            response = await self._client.messages.create(stream=False, **create_kwargs)
        except Exception as e:
            self._translate_error(e)

        parts: list[ResponsePart] = []
        for block in response.content:
            match block.type:
                case "text":
                    parts.append(TextPart(text=block.text))
                case "thinking":
                    parts.append(
                        ReasoningPart(text=block.thinking, signature=block.signature)
                    )
                case "tool_use":
                    parts.append(
                        ToolUsePart(id=block.id, name=block.name, inputs=block.input)
                    )
                case _:
                    logger.debug(f"Unhandled content block type: {block.type}")

        # Anthropic stop_reason maps directly to Dobby StopReason
        stop_reason: StopReason = response.stop_reason or "end_turn"

        usage: Usage | None = None
        if response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cache_creation_input_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", None
                ),
                cache_read_input_tokens=getattr(
                    response.usage, "cache_read_input_tokens", None
                ),
            )

        return StreamEndEvent(
            model=response.model,
            parts=parts,
            stop_reason=stop_reason,
            usage=usage,
        )

    @with_retries
    async def _stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream chat completion yielding discriminated events.

        Processes streaming events from Anthropic Messages API and yields
        typed StreamEvent objects for text, reasoning, and tool calls.

        Args:
            messages: Anthropic-formatted messages.
            model: Model name.
            system: Optional system prompt.
            tools: Optional tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            thinking: Optional extended thinking configuration.

        Yields:
            StreamEvent objects: StreamStartEvent, TextDeltaEvent,
            ReasoningDeltaEvent, ToolUseEvent, StreamEndEvent.
        """
        create_kwargs = self._build_kwargs(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            temperature=temperature,
            thinking=thinking,
        )

        try:
            stream = await self._client.messages.create(stream=True, **create_kwargs)
        except Exception as e:
            self._translate_error(e)

        # Accumulation state
        response_id: str | None = None
        model_name: str = model
        accumulated_text: str = ""
        accumulated_reasoning: str = ""
        reasoning_signature: str | None = None
        function_calls: list[ToolUseEvent] = []

        # Current content block tracking
        current_block_type: str | None = None
        current_tool_id: str | None = None
        current_tool_name: str | None = None
        current_tool_input_json: str = ""

        # Usage tracking
        input_tokens: int = 0
        output_tokens: int = 0
        cache_creation_input_tokens: int | None = None
        cache_read_input_tokens: int | None = None
        stop_reason: StopReason = "end_turn"

        async for event in stream:
            match event.type:
                case "message_start":
                    response_id = event.message.id
                    model_name = event.message.model
                    if event.message.usage:
                        input_tokens = event.message.usage.input_tokens
                        cache_creation_input_tokens = getattr(
                            event.message.usage, "cache_creation_input_tokens", None
                        )
                        cache_read_input_tokens = getattr(
                            event.message.usage, "cache_read_input_tokens", None
                        )
                    yield StreamStartEvent(
                        id=response_id,
                        model=model_name,
                    )

                case "content_block_start":
                    current_block_type = event.content_block.type
                    if current_block_type == "thinking":
                        yield ReasoningStartEvent(type="reasoning_start")
                    elif current_block_type == "tool_use":
                        current_tool_id = event.content_block.id
                        current_tool_name = event.content_block.name
                        current_tool_input_json = ""

                case "content_block_delta":
                    match event.delta.type:
                        case "thinking_delta":
                            accumulated_reasoning += event.delta.thinking
                            yield ReasoningDeltaEvent(delta=event.delta.thinking)
                        case "text_delta":
                            accumulated_text += event.delta.text
                            yield TextDeltaEvent(delta=event.delta.text)
                        case "input_json_delta":
                            current_tool_input_json += event.delta.partial_json
                        case "signature_delta":
                            reasoning_signature = getattr(event.delta, "signature", None)

                case "content_block_stop":
                    if current_block_type == "thinking":
                        yield ReasoningEndEvent(type="reasoning_end")
                    elif current_block_type == "tool_use":
                        tool_inputs = json.loads(current_tool_input_json) if current_tool_input_json else {}
                        tool_event = ToolUseEvent(
                            id=current_tool_id or "",
                            name=current_tool_name or "",
                            inputs=tool_inputs,
                        )
                        function_calls.append(tool_event)
                        yield tool_event
                    current_block_type = None

                case "message_delta":
                    if event.delta.stop_reason:
                        stop_reason = event.delta.stop_reason
                    if event.usage:
                        output_tokens = event.usage.output_tokens

                case "message_stop":
                    parts: list[ResponsePart] = []
                    if accumulated_reasoning:
                        parts.append(
                            ReasoningPart(
                                text=accumulated_reasoning,
                                signature=reasoning_signature,
                            )
                        )
                    if accumulated_text:
                        parts.append(TextPart(text=accumulated_text))
                    for tool_event in function_calls:
                        parts.append(
                            ToolUsePart(
                                id=tool_event.id,
                                name=tool_event.name,
                                inputs=tool_event.inputs,
                            )
                        )

                    usage_data = Usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                        cache_creation_input_tokens=cache_creation_input_tokens,
                        cache_read_input_tokens=cache_read_input_tokens,
                    )

                    yield StreamEndEvent(
                        model=model_name,
                        parts=parts,
                        stop_reason=stop_reason,
                        usage=usage_data,
                    )

                case "error":
                    yield StreamErrorEvent(
                        error_code=getattr(event.error, "type", None),
                        error_message=getattr(event.error, "message", str(event.error)),
                    )


def to_anthropic_messages(messages: Iterable[MessagePart]) -> list[dict[str, Any]]:
    """Convert provider-agnostic messages to Anthropic format.

    Handles conversion of different message types and content blocks:
    - Text messages -> text content blocks
    - Multi-part messages -> content arrays with proper types
    - Tool calls -> tool_use content blocks in assistant messages
    - Tool results -> tool_result content blocks in user messages
    - Thinking/reasoning -> thinking content blocks (required for multi-turn)
    - Images -> image content blocks (base64 or URL)

    Anthropic requires strict alternation of user/assistant roles.
    Consecutive same-role messages are merged into a single message.

    Args:
        messages: Iterable of MessagePart dataclasses.

    Returns:
        List of Anthropic-formatted message dicts.
    """
    raw_messages: list[dict[str, Any]] = []

    for message in messages:
        match message:
            case AssistantMessagePart(parts=parts):
                content: list[dict[str, Any]] = []
                for part in parts:
                    match part:
                        case TextPart(text=text):
                            content.append({"type": "text", "text": text})
                        case ToolUsePart(id=tool_id, name=name, inputs=inputs):
                            content.append(
                                {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": name,
                                    "input": inputs,
                                }
                            )
                        case ReasoningPart(text=text, signature=signature):
                            block: dict[str, Any] = {
                                "type": "thinking",
                                "thinking": text,
                            }
                            if signature:
                                block["signature"] = signature
                            content.append(block)

                if content:
                    raw_messages.append({"role": "assistant", "content": content})

            case UserMessagePart(parts=parts):
                content_blocks: list[AnthropicContentBlock] = []

                for p in parts:
                    if isinstance(p, ToolResultPart):
                        tool_content: list[AnthropicContentBlock] = [
                            content_part_to_anthropic(tp) for tp in p.parts
                        ]
                        content_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": p.tool_use_id,
                                "content": tool_content,
                                "is_error": p.is_error,
                            }
                        )
                    else:
                        content_blocks.append(content_part_to_anthropic(p))

                if content_blocks:
                    raw_messages.append({"role": "user", "content": content_blocks})

    # Merge consecutive same-role messages (Anthropic requires strict alternation)
    merged: list[dict[str, Any]] = []
    for msg in raw_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)

    return merged
