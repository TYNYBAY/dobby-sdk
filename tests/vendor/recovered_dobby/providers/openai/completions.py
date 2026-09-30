"""OpenAI Chat Completions API provider.

NOTE: This module uses the legacy Chat Completions API.
For new implementations, prefer adapter/OpenAIProvider which uses the Responses API.
This module is kept for backwards compatibility but may be removed in future versions.
"""

from collections.abc import AsyncIterator, Iterable
import json
import logging
from typing import Any, Literal, overload
import warnings

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_assistant_message_param import ContentArrayOfContentPart

from ..tools import BaseTool
from ..types import (
    MessagePart,
    ResponsePart,
    StopReason,
    StreamEndEvent,
    StreamEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolUsePart,
    Usage,
)

logger = logging.getLogger(__name__)

FinishReason = Choice.__annotations__["finish_reason"]


class OpenAICompletionsProvider:
    """Provider for OpenAI and Azure OpenAI chat completions (legacy API).

    NOTE: This provider uses the Chat Completions API. For new implementations,
    prefer OpenAIProvider which uses the Responses API.

    Handles both standard OpenAI API and Azure OpenAI deployments with support for:
    - Streaming and non-streaming chat completions
    - Tool/function calling
    - Image inputs
    - Automatic Azure detection based on base URL

    Attributes:
        api_key: API key for authentication
        base_url: Base URL for API endpoint (None for default OpenAI)
        model: Model name for OpenAI or deployment ID for Azure
        azure_deployment_id: Azure deployment ID (auto-set when using Azure)
        client: Async client instance (OpenAI or Azure)
    """

    api_key: str | None
    base_url: str | None
    model: str
    azure_deployment_id: str | None
    client: AsyncOpenAI | AsyncAzureOpenAI

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        azure_deployment_id: str | None = None,
        azure_api_version: str = "2024-12-01-preview",
    ):
        """Initialize OpenAI Completions provider.

        Args:
            model: Model name (e.g., 'gpt-4') for OpenAI. Ignored for Azure.
            api_key: API key for authentication. Uses environment variable if not provided.
            base_url: API endpoint URL. If contains 'azure', Azure client is used.
            azure_deployment_id: Azure deployment ID. Auto-set from model if using Azure.
            azure_api_version: Azure API version. Defaults to latest preview.

        Note:
            Azure is automatically detected if 'azure' appears in base_url.
            For Azure, the model parameter becomes the deployment ID.
        """
        warnings.warn(
            "OpenAICompletionsProvider is deprecated. Use OpenAIProvider instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.api_key = api_key
        self.base_url = base_url
        self.azure_deployment_id = azure_deployment_id

        if base_url is not None and "azure" in base_url:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                azure_deployment=azure_deployment_id,
                api_version=azure_api_version,
            )
            self.model = azure_deployment_id
        else:
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            self.model = model

    @overload
    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: Literal[False] = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[BaseTool] | None = None,
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
        tools: list[BaseTool] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]: ...

    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[BaseTool] | None = None,
        **kwargs,
    ) -> StreamEndEvent | AsyncIterator[StreamEvent]:
        """Generate chat completion from messages.

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
        openai_messages: list[ChatCompletionMessageParam] = to_openai_messages(messages)

        if system_prompt is not None:
            openai_messages.insert(0, {"role": "system", "content": system_prompt})

        # Convert tools to OpenAI format
        openai_tools: list[ChatCompletionToolParam] | None = None
        if tools:
            openai_tools = [tool.schema.to_openai_format() for tool in tools]

        if stream:
            return self._stream_chat_completion(openai_messages, temperature, openai_tools)

        # non-streaming chat logic
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            stream=False,
            tools=openai_tools,
        )
        choice = completion.choices[0]

        # Build parts from response
        parts: list[ResponsePart] = []
        stop_reason: StopReason = self._map_stop_reason(choice.finish_reason)

        if choice.message.content:
            parts.append(TextPart(type="text", text=choice.message.content))

        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                parts.append(
                    ToolUsePart(
                        type="tool_use",
                        id=tool_call.id,
                        name=tool_call.function.name,
                        inputs=json.loads(tool_call.function.arguments),
                    )
                )

        usage: Usage | None = None
        if completion.usage:
            usage = Usage(
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
            )

        return StreamEndEvent(
            model=self.model or self.azure_deployment_id,
            parts=parts,
            stop_reason=stop_reason,
            usage=usage,
        )

    async def _stream_chat_completion(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        temperature: float,
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Handle streaming chat completion responses.

        Processes streaming chunks from OpenAI, accumulating content and tool calls.
        Tool calls are collected by index and parsed when complete. Yields
        StreamEvent objects with accumulated state.

        Args:
            messages: OpenAI-formatted messages
            temperature: Sampling temperature
            tools: Optional OpenAI-formatted tool definitions

        Yields:
            StreamEvent objects for text, tool calls, and final completion
        """
        accumulated_text = ""
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        response_id: str | None = None
        model_name: str = self.model

        response_stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            tools=tools,
        )

        chunk: ChatCompletionChunk
        async for chunk in response_stream:
            if not chunk.choices:
                continue

            # Set response_id from first chunk
            if response_id is None:
                response_id = chunk.id
                model_name = chunk.model or self.model
                yield StreamStartEvent(
                    id=response_id,
                    model=model_name,
                )

            choice = chunk.choices[0]
            content_delta = choice.delta.content or ""

            if content_delta:
                accumulated_text += content_delta
                yield TextDeltaEvent(delta=content_delta)

            # Handle tool call deltas - accumulate by index
            if choice.delta.tool_calls:
                for tc_delta in choice.delta.tool_calls:
                    idx = tc_delta.index

                    if idx not in tool_calls_by_index:
                        # First chunk for this tool - has id and name
                        tool_calls_by_index[idx] = {
                            "id": tc_delta.id,
                            "name": tc_delta.function.name if tc_delta.function else None,
                            "arguments": tc_delta.function.arguments or ""
                            if tc_delta.function
                            else "",
                        }
                    else:
                        # Subsequent chunks - only append arguments
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_calls_by_index[idx]["arguments"] += tc_delta.function.arguments

            # Check for stream end
            if choice.finish_reason:
                # Build final parts
                parts: list[ResponsePart] = []

                if accumulated_text:
                    parts.append(TextPart(type="text", text=accumulated_text))

                # Add tool calls (sorted by index for consistency)
                for idx in sorted(tool_calls_by_index.keys()):
                    tool_data = tool_calls_by_index[idx]
                    if tool_data["id"] and tool_data["name"] and tool_data["arguments"]:
                        try:
                            inputs = json.loads(tool_data["arguments"])
                            tool_part = ToolUsePart(
                                type="tool_use",
                                id=tool_data["id"],
                                name=tool_data["name"],
                                inputs=inputs,
                            )
                            parts.append(tool_part)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse tool arguments: {tool_data['arguments']}"
                            )

                usage_data: Usage | None = None
                if chunk.usage:
                    usage_data = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )

                yield StreamEndEvent(
                    model=model_name,
                    parts=parts,
                    stop_reason=self._map_stop_reason(choice.finish_reason),
                    usage=usage_data,
                )

    def _map_stop_reason(self, reason: str | None) -> StopReason:
        """Map OpenAI finish reasons to provider-agnostic stop reasons.

        Args:
            reason: OpenAI finish_reason from response

        Returns:
            Normalized stop reason for cross-provider compatibility

        Mappings:
            - 'stop' -> 'end_turn': Natural conversation end
            - 'length' -> 'max_tokens': Hit token limit
            - 'content_filter' -> 'content_filter': Content policy violation
            - 'function_call'/'tool_calls' -> 'tool_use': Model invoked tools
            - None/other -> 'end_turn': Default fallback
        """
        match reason:
            case "stop":
                return "end_turn"
            case "length":
                return "max_tokens"
            case "content_filter":
                return "content_filter"
            case "function_call":
                return "tool_use"
            case "tool_calls":
                return "tool_use"
            case _:
                return "end_turn"


def to_openai_messages(messages: Iterable[MessagePart]) -> list[ChatCompletionMessageParam]:
    """Convert provider-agnostic messages to OpenAI Chat Completions format.

    Handles conversion of different message types and content blocks:
    - Text messages -> Simple content strings
    - Multi-part messages -> Content arrays with proper types
    - Tool calls -> Separate tool_calls array in assistant messages
    - Tool results -> Tool role messages with tool_call_id
    - Images -> image_url content parts (base64 or URL)

    Args:
        messages: Iterable of MessagePart dictionaries with role and parts

    Returns:
        List of OpenAI-formatted message parameters

    Note:
        - Assistant messages with tools have content and tool_calls separated
        - Tool results include error prefix when is_error=True
        - Images support both URL and base64 data URIs
    """
    openai_messages: list[ChatCompletionMessageParam] = []
    for message in messages:
        if message["role"] == "assistant":
            parts = message["parts"]
            content_parts: list[ContentArrayOfContentPart] = []
            tool_calls: list[ChatCompletionMessageToolCallParam] = []

            for part in parts:
                part_type = part["type"]
                if part_type == "text":
                    content_parts.append({"type": "text", "text": part["text"]})
                elif part_type == "tool_use":
                    tool_calls.append(
                        {
                            "type": "function",
                            "id": part["id"],
                            "function": {
                                "name": part["name"],
                                "arguments": json.dumps(part["inputs"]),
                            },
                        }
                    )
                # Reasoning parts are not sent back to OpenAI

            assistant_msg: ChatCompletionAssistantMessageParam = {"role": "assistant"}
            assistant_msg["content"] = content_parts if content_parts else None
            assistant_msg["tool_calls"] = tool_calls if tool_calls else None
            openai_messages.append(assistant_msg)

        elif message["role"] == "user":
            parts = message["parts"]
            content_parts: list[ChatCompletionContentPartParam] = []

            for part in parts:
                part_type = part["type"]
                if part_type == "text":
                    content_parts.append({"type": "text", "text": part["text"]})
                elif part_type == "image":
                    image_url = (
                        part["source"]["url"]
                        if part["source"]["type"] == "url"
                        else f"data:{part['source']['media_type']};base64,{part['source']['data']}"
                    )
                    content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                elif part_type == "document":
                    raise NotImplementedError("Documents not supported by Chat Completions API")

            if content_parts:
                openai_messages.append({"role": "user", "content": content_parts})

        elif message["role"] == "tool_result":
            # Extract text from TextPart content blocks
            text_parts = [part["text"] for part in message["parts"] if part.get("type") == "text"]
            content_text = "\n".join(text_parts)
            tool_content = (
                content_text
                if not message["is_error"]
                else f"Failed to execute tool: \n{content_text}"
            )

            openai_messages.append(
                {
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": message["tool_use_id"],
                }
            )

    return openai_messages
