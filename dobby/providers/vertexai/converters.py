"""Internal converters for Vertex AI's Chat Completions wire format.

Vertex AI's OpenAI-compatible Model-as-a-Service (MaaS) endpoint speaks the
Chat Completions API (`messages` array with `role`/`content`, `tool_calls`,
`choices[].delta`) — a structurally different payload from dobby's existing
`OpenAIProvider`, which targets the newer Responses API. See the "Key Finding
That Changes Scope From the Origin Document" section of
docs/plans/2026-07-07-001-feat-vertexai-provider-plan.md for why these
converters are a new implementation rather than a wrapper around
`dobby.providers.openai.converters`.
"""

from collections.abc import Iterable
import json
from typing import Any

from ...tools.tool import Tool
from ...types import (
    AssistantMessagePart,
    Base64ImageSource,
    DocumentPart,
    ImagePart,
    MessagePart,
    ReasoningPart,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    URLImageSource,
    UserMessagePart,
)

VertexAIContentPart = dict[str, Any]


def _text_to_vertexai(part: TextPart) -> VertexAIContentPart:
    """Convert TextPart to Chat Completions text content part."""
    return {"type": "text", "text": part.text}


def _image_to_vertexai(part: ImagePart) -> VertexAIContentPart:
    """Convert ImagePart to Chat Completions image_url content part."""
    match part.source:
        case URLImageSource(url=url):
            return {"type": "image_url", "image_url": {"url": url}}
        case Base64ImageSource(data=data, media_type=mt):
            return {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}}
    raise ValueError(f"Unknown image source type: {part.source}")


def content_part_to_vertexai(
    part: TextPart | ImagePart | DocumentPart,
) -> VertexAIContentPart:
    """Convert a text/image content part to Chat Completions format.

    Documents have no equivalent content-part type in the Chat Completions wire
    format Vertex's MaaS endpoint speaks (unlike the Responses API's
    `input_file`) — Vertex Model Garden's MaaS models are text/image chat models,
    not document-ingestion endpoints. DocumentPart is out of scope for v1; passing
    one raises rather than silently dropping or mis-encoding it.
    """
    match part:
        case TextPart():
            return _text_to_vertexai(part)
        case ImagePart():
            return _image_to_vertexai(part)
        case DocumentPart():
            raise ValueError(
                "DocumentPart is not supported by Vertex AI's Chat Completions "
                "endpoint (text/image content only)."
            )
    raise ValueError(f"Unknown content part type: {part}")


def _build_content(
    parts: list[TextPart | ImagePart | DocumentPart],
) -> str | list[VertexAIContentPart]:
    """Build a Chat Completions `content` value from content parts.

    A single text part collapses to a plain string (the common case, and what
    Chat-Completions-speaking servers expect for simple text turns); anything
    else (multiple parts, or a non-text part) becomes a list of typed content
    parts.
    """
    if len(parts) == 1 and isinstance(parts[0], TextPart):
        return parts[0].text
    return [content_part_to_vertexai(p) for p in parts]


def _tool_result_to_vertexai(part: ToolResultPart) -> dict[str, Any]:
    """Convert ToolResultPart to a Chat Completions `tool` role message.

    Chat Completions has no dedicated error flag on tool-result messages, so
    (mirroring `to_openai_messages`'s convention in
    dobby/providers/openai/adapter.py) a "Failed to execute tool:" text marker
    is prepended to the content when `is_error` is set.
    """
    content = _build_content(part.parts) if part.parts else ""

    if part.is_error:
        prefix = "Failed to execute tool:"
        if isinstance(content, str):
            content = f"{prefix} {content}" if content else prefix
        else:
            content = [{"type": "text", "text": prefix}, *content]

    return {
        "role": "tool",
        "tool_call_id": part.tool_use_id,
        "content": content,
    }


def to_vertexai_messages(messages: Iterable[MessagePart]) -> list[dict[str, Any]]:
    """Convert provider-agnostic messages to Chat Completions `messages` format.

    Handles conversion of different message types and content blocks:
    - User messages -> `{"role": "user", "content": ...}`
    - Assistant text -> `{"role": "assistant", "content": "..."}`
    - Assistant tool calls -> `{"role": "assistant", "tool_calls": [...]}`
    - Tool results -> `{"role": "tool", "tool_call_id": ..., "content": ...}`
    - ReasoningPart is dropped (Llama/Model Garden MaaS models have no
      reasoning/thinking channel), mirroring `to_openai_messages`.

    This function operates purely on `messages`; it takes no `system_prompt`
    parameter. Consistent with `OpenAIProvider.chat()` (dobby/providers/openai/
    adapter.py), which inserts the system message into the message list itself
    before calling its converter, `VertexAIProvider.chat()` (U3) is expected to
    prepend `{"role": "system", "content": system_prompt}` to this function's
    output rather than this function taking a `system_prompt` argument.

    Args:
        messages: Iterable of dobby MessagePart objects.

    Returns:
        List of Chat-Completions-formatted message dicts.
    """
    vertexai_messages: list[dict[str, Any]] = []

    for message in messages:
        match message:
            case AssistantMessagePart(parts=parts):
                text_segments: list[str] = []
                tool_calls: list[dict[str, Any]] = []

                for part in parts:
                    match part:
                        case TextPart(text=text):
                            text_segments.append(text)
                        case ToolUsePart(id=tool_id, name=name, inputs=inputs):
                            tool_calls.append(
                                {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(inputs),
                                    },
                                }
                            )
                        case ReasoningPart():
                            # Reasoning parts are not sent to Vertex's MaaS models.
                            pass

                if text_segments or tool_calls:
                    assistant_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": "".join(text_segments) or None,
                    }
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls

                    vertexai_messages.append(assistant_message)

            case UserMessagePart(parts=parts):
                content_parts: list[TextPart | ImagePart | DocumentPart] = []

                for p in parts:
                    if isinstance(p, ToolResultPart):
                        vertexai_messages.append(_tool_result_to_vertexai(p))
                    else:
                        content_parts.append(p)

                if content_parts:
                    vertexai_messages.append(
                        {"role": "user", "content": _build_content(content_parts)}
                    )

    return vertexai_messages


def to_vertexai_tool(tool: Tool) -> dict[str, Any]:
    """Convert a dobby Tool to a Chat Completions (nested) tool schema.

    Reuses `Tool.to_openai_format()`'s schema-construction logic (it already
    handles both the Pydantic-model and parameter-list code paths) and
    re-nests its flat Responses-API shape into Chat Completions' nested
    `{"type": "function", "function": {...}}` shape.

    Args:
        tool: A dobby Tool instance.

    Returns:
        Chat-Completions-formatted tool schema dict.
    """
    flat = tool.to_openai_format()

    return {
        "type": "function",
        "function": {
            "name": flat["name"],
            "description": flat["description"],
            "parameters": flat["parameters"],
        },
    }
