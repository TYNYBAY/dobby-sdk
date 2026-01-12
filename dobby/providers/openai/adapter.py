from collections.abc import AsyncIterator, Iterable
import json
from typing import Any, Literal, overload

import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.responses import (
    ResponseFunctionCallOutputItemListParam,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
    ResponseOutputMessageParam,
    ToolParam,
)
from openai.types.shared_params.reasoning import Reasoning

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
from ..base import Provider
from .converters import OpenAIContentPart, content_part_to_openai


class OpenAIProvider(Provider[AsyncOpenAI | AsyncAzureOpenAI]):
    """Provider for OpenAI and Azure OpenAI using Responses API.

    Inherits from Provider base class and implements the chat() interface
    for OpenAI's Responses API (not Chat Completions).

    Attributes:
        api_key: API key for authentication
        base_url: Base URL for API endpoint (None for default OpenAI)
        azure_deployment_id: Azure deployment ID (auto-set when using Azure)

    Inherited from Provider:
        name: Returns "openai" or "azure-openai" based on configuration
        model: Returns the model name or Azure deployment ID
        client: Returns the AsyncOpenAI or AsyncAzureOpenAI client instance
        max_retries: Maximum retry attempts (default: 3)
    """

    api_key: str | None
    base_url: str | None
    _model: str
    azure_deployment_id: str | None
    _client: AsyncOpenAI | AsyncAzureOpenAI
    max_retries: int
    _retry_errors: tuple[type[BaseException], ...]

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        azure_deployment_id: str | None = None,
        azure_api_version: str = "2025-03-01-preview",
        max_retries: int = 3,
    ):
        """Initialize OpenAI provider.

        Args:
            model: Model name (e.g., 'gpt-4') for OpenAI. Ignored for Azure.
            api_key: API key for authentication. Uses environment variable if not provided.
            base_url: API endpoint URL. If contains 'azure', Azure client is used.
            azure_deployment_id: Azure deployment ID. Auto-set from model if using Azure.
            azure_api_version: Azure API version. Defaults to latest preview.
            max_retries: Maximum retry attempts for transient errors. Defaults to 3.

        Note:
            Azure is automatically detected if 'azure' appears in base_url.
            For Azure, the model parameter becomes the deployment ID.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.azure_deployment_id = azure_deployment_id
        self.max_retries = max_retries
        self._retry_errors = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )

        if base_url is not None and "azure" in base_url:
            self._client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                azure_deployment=azure_deployment_id,
                api_version=azure_api_version,
            )
            self._model = azure_deployment_id
        else:
            self._client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            self._model = model

    @property
    def name(self) -> str:
        """Provider name."""
        if self.base_url and "azure" in self.base_url:
            return "azure-openai"
        return "openai"

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    @property
    def client(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        """Authenticated client instance."""
        return self._client

    @staticmethod
    def _build_kwargs(
        model: str,
        input: ResponseInputParam,
        tools: list[ToolParam] | None = None,
        reasoning: Reasoning | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for responses.create() excluding None values.

        OpenAI SDK uses Omit sentinel for optional params, not None.
        This helper only includes params that have actual values.

        Args:
            model: Model name or Azure deployment ID.
            input: OpenAI-formatted input messages.
            tools: Optional tool definitions for function calling.
            reasoning: Optional reasoning configuration.
                Options for effort: "minimal", "low", "medium", "high".
                Options for summary: "auto", "concise", "detailed".

        Returns:
            Dictionary of kwargs to pass to responses.create().
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "input": input,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if reasoning is not None:
            kwargs["reasoning"] = reasoning

        # logger.debug(f"kwargs: {kwargs}")
        return kwargs

    @overload
    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: Literal[False] = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
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
        tools: list[ToolParam] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]: ...

    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] | None = None,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> StreamEndEvent | AsyncIterator[StreamEvent]:
        """Generate response from messages using Responses API.

        Converts provider-agnostic messages to OpenAI format and handles both
        streaming and non-streaming responses. Automatically processes tool calls
        and converts responses back to provider-agnostic format.

        Args:
            messages: Conversation history with user/assistant/tool messages
            stream: Whether to stream response chunks
            system_prompt: Optional system message to guide behavior
            temperature: Controls randomness (0.0-2.0, default 0.0)
            tools: Available tools for function calling
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            StreamEndEvent for non-streaming, AsyncIterator[StreamEvent] for streaming
        """
        openai_messages: ResponseInputParam = to_openai_messages(messages)
        if system_prompt is not None:
            openai_messages.insert(0, {"role": "system", "content": system_prompt})

        # Determine model to use (instance model > azure deployment)
        target_model = self._model or self.azure_deployment_id

        if stream:
            return self._stream_chat_completion(
                openai_messages, temperature, target_model, tools, reasoning_effort
            )

        # non-streaming response logic ------
        # Construct reasoning param if effort provided
        reasoning_param = None
        if reasoning_effort:
            reasoning_param = {"effort": reasoning_effort, "summary": "auto"}

        create_kwargs = self._build_kwargs(
            model=target_model,
            input=openai_messages,
            tools=tools,
            reasoning=reasoning_param,
        )
        response = await self._client.responses.create(stream=False, **create_kwargs)

        stop_reason: StopReason = "end_turn"
        parts: list[ResponsePart] = []

        if response.output:
            for output in response.output:
                match output.type:
                    case "message":
                        if output.status != "completed":
                            logger.debug(f"Unhandled output status: {output.status}")
                            continue

                        for content in output.content:
                            if content.type == "output_text":
                                parts.append(TextPart(text=content.text))
                            else:
                                logger.debug(f"Unhandled content type: {content.type}")

                    case "function_call":
                        stop_reason = "tool_use"
                        parts.append(
                            ToolUsePart(
                                id=output.id,
                                name=output.name,
                                inputs=json.loads(output.arguments),
                            )
                        )
                    case _:
                        logger.debug(f"Unhandled output type: {output.type}")

        usage: Usage | None = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return StreamEndEvent(
            model=response.model or self._model or self.azure_deployment_id,
            parts=parts,
            stop_reason=stop_reason,
            usage=usage,
        )

    @with_retries
    async def _stream_chat_completion(
        self,
        messages: ResponseInputParam,
        temperature: float,
        model: str,
        tools: list[ToolParam] | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream chat completion yielding discriminated events.

        Processes streaming events from OpenAI Responses API and yields
        typed StreamEvent objects for text, reasoning, and tool calls.

        Args:
            messages: OpenAI-formatted input messages
            temperature: Sampling temperature
            model: Model id
            tools: Optional OpenAI-formatted tool definitions
            reasoning_effort: Reasoning effort level

        Yields:
            StreamEvent objects: StreamStartEvent, TextDeltaEvent,
            ReasoningDeltaEvent, ToolCallEvent, StreamEndEvent
        """
        # Construct reasoning param if effort provided
        reasoning_param = None
        if reasoning_effort:
            reasoning_param = {"effort": reasoning_effort, "summary": "auto"}

        create_kwargs = self._build_kwargs(
            model=model,
            input=messages,
            tools=tools,
            reasoning=reasoning_param,
        )

        response_stream = await self._client.responses.create(stream=True, **create_kwargs)

        response_id: str | None = None
        model_name: str = model
        accumulated_text: str = ""
        accumulated_reasoning: str = ""
        function_calls: list[ToolUseEvent] = []

        async for event in response_stream:
            match event.type:
                case "response.created":
                    response_id = event.response.id
                    model_name = event.response.model
                    yield StreamStartEvent(
                        id=response_id,
                        model=model_name,
                    )

                case "response.output_item.added":
                    # Reasoning starts when OpenAI adds a reasoning item (content=None initially)
                    if event.item.type == "reasoning" and event.item.content is None:
                        yield ReasoningStartEvent(type="reasoning_start")

                case "response.output_text.delta":
                    delta_text = event.delta
                    accumulated_text += delta_text
                    yield TextDeltaEvent(delta=delta_text)

                case "response.reasoning_summary_text.delta":
                    accumulated_reasoning += event.delta
                    yield ReasoningDeltaEvent(delta=event.delta)

                case "response.reasoning_summary_text.done":
                    yield ReasoningEndEvent(type="reasoning_end")

                case "response.output_item.done":
                    if event.item.type == "message":
                        # Text message already handled via deltas
                        pass
                    elif event.item.type == "function_call":
                        tool_event = ToolUseEvent(
                            id=event.item.call_id,
                            name=event.item.name,
                            inputs=json.loads(event.item.arguments),
                        )
                        function_calls.append(tool_event)
                        yield tool_event

                case "response.completed":
                    # Build final parts
                    parts: list[ResponsePart] = []
                    if accumulated_reasoning:
                        parts.append(ReasoningPart(text=accumulated_reasoning))
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

                    # Determine stop reason
                    stop_reason: StopReason = "tool_use" if function_calls else "end_turn"

                    usage_data: Usage | None = None
                    if event.response and event.response.usage:
                        usage_data = Usage(
                            input_tokens=event.response.usage.input_tokens,
                            output_tokens=event.response.usage.output_tokens,
                            total_tokens=event.response.usage.total_tokens,
                        )

                    yield StreamEndEvent(
                        model=model_name,
                        parts=parts,
                        stop_reason=stop_reason,
                        usage=usage_data,
                    )

                case "response.failed":
                    error = event.response.error if event.response else None
                    yield StreamErrorEvent(
                        error_code=error.code if error else None,
                        error_message=error.message if error else "Unknown error",
                    )

                # case _:
                #     logger.debug(f"Unhandled event type: {event.type}")


def to_openai_messages(messages: Iterable[MessagePart]) -> ResponseInputParam:
    """Convert provider-agnostic messages to OpenAI format.

    Handles conversion of different message types and content blocks:
    - Text messages -> Simple content strings
    - Multi-part messages -> Content arrays with proper types
    - Tool calls -> Separate tool_calls array in assistant messages
    - Tool results -> Tool role messages with tool_call_id
    - Images -> image_url content parts (base64 or URL)

    Args:
        messages: Iterable of MessagePart dataclasses

    Returns:
        List of OpenAI-formatted message parameters
    """
    openai_messages: ResponseInputParam = []

    for message in messages:
        match message:
            case AssistantMessagePart(parts=parts):
                tool_calls: list[ResponseFunctionToolCallParam] = []
                assistant_message: ResponseOutputMessageParam = {
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "status": "completed",
                }
                for part in parts:
                    match part:
                        case TextPart(text=text):
                            assistant_message["content"].append(
                                {"type": "output_text", "text": text}
                            )
                        case ToolUsePart(id=tool_id, name=name, inputs=inputs):
                            tool_calls.append(
                                {
                                    "type": "function_call",
                                    "call_id": tool_id,
                                    "name": name,
                                    "arguments": json.dumps(inputs),
                                }
                            )
                        case ReasoningPart():
                            # Reasoning parts are not sent back to OpenAI
                            pass

                if assistant_message["content"]:
                    openai_messages.append(assistant_message)

                if tool_calls:
                    openai_messages.extend(tool_calls)

            case UserMessagePart(parts=parts):
                content_parts: list[OpenAIContentPart] = []
                tool_outputs: list[Any] = []

                for p in parts:
                    if isinstance(p, ToolResultPart):
                        # Convert ToolResultPart to function_call_output
                        output_parts: ResponseFunctionCallOutputItemListParam = [
                            content_part_to_openai(tp) for tp in p.parts
                        ]
                        # Add error prefix if tool execution failed
                        if p.is_error and output_parts:
                            output_parts.insert(
                                0, {"type": "input_text", "text": "Failed to execute tool:"}
                            )
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "output": output_parts,
                                "call_id": p.tool_use_id,
                                "status": "incomplete" if p.is_error else "completed",
                            }
                        )
                    else:
                        content_parts.append(content_part_to_openai(p))

                if content_parts:
                    openai_messages.append(
                        {"type": "message", "role": "user", "content": content_parts}
                    )
                if tool_outputs:
                    openai_messages.extend(tool_outputs)

    return openai_messages
