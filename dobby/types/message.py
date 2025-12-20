from dataclasses import dataclass, field
from typing import Literal

from .document_part import DocumentPart
from .image_part import ImagePart
from .reasoning_part import ReasoningPart
from .text_part import TextPart
from .tool_part import ToolUsePart

# Input content parts (for user messages)
type ContentPart = TextPart | ImagePart | DocumentPart

# Output content parts (for assistant responses)
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


@dataclass
class UserMessagePart:
    """A user message with content parts."""

    parts: list[ContentPart] = field(default_factory=list)

    role: Literal["user"] = "user"


@dataclass
class AssistantMessagePart:
    """An assistant message with content and tool calls."""

    parts: list[ResponsePart] = field(default_factory=list)

    role: Literal["assistant"] = "assistant"


@dataclass
class ToolResultMessagePart:
    """A tool result message."""

    tool_use_id: str

    name: str

    parts: list[TextPart | ImagePart | DocumentPart] = field(default_factory=list)

    is_error: bool = False

    role: Literal["tool_result"] = "tool_result"


type MessagePart = AssistantMessagePart | UserMessagePart | ToolResultMessagePart
