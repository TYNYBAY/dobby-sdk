from collections.abc import Iterable
from typing import Literal, TypedDict

from .document_part import DocumentPart
from .image_part import ImagePart
from .reasoning_part import ReasoningPart
from .text_part import TextPart
from .tool_part import ToolUsePart

# Input content parts (for user messages) - union discriminated by 'type' field
type ContentPart = TextPart | ImagePart | DocumentPart

# Output content parts (for assistant responses) - union discriminated by 'type' field
type ResponsePart = TextPart | ReasoningPart | ToolUsePart


type StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "content_filter"]
"""Reason why the model stopped generating.

- `"end_turn"`: the model reached a natural stopping point
- `"max_tokens"`: exceeded the requested `max_tokens` or the model's maximum
- `"stop_sequence"`: one of the provided custom `stop_sequences` was generated
- `"tool_use"`: the model invoked tools
- `"content_filter"`: content was omitted due to content filters
In non-streaming mode this value is always non-null. In streaming mode, 
only non-null in the last response.
"""


class UserMessagePart(TypedDict):

    role: Literal["user"]

    parts: Iterable[ContentPart]


class AssistantMessagePart(TypedDict):

    role: Literal["assistant"]

    parts: Iterable[ResponsePart]


class ToolResultMessagePart(TypedDict):

    role: Literal["tool_result"]

    tool_use_id: str

    name: str

    parts: Iterable[TextPart | ImagePart | DocumentPart]

    is_error: bool


type MessagePart = AssistantMessagePart | UserMessagePart | ToolResultMessagePart
