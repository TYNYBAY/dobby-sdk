from dataclasses import dataclass, field
from typing import Literal

from .document_part import DocumentPart
from .image_part import ImagePart
from .text_part import TextPart


@dataclass
class ToolResultPart:
    """A tool result message."""

    tool_use_id: str

    name: str

    parts: list[TextPart | ImagePart | DocumentPart] = field(default_factory=list)

    is_error: bool = False

    kind: Literal["tool_result"] = "tool_result"
