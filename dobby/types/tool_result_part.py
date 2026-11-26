from dataclasses import dataclass, field
from typing import Literal

from .document_part import DocumentPart
from .image_part import ImagePart
from .text_part import TextPart

# Result payloads of the executor's control-flow placeholders: a call paused for
# host approval, or cancelled. They reach history as `TextPart(text=str(...))`.
# Shared here so providers that recognise placeholders (the Gemini adapter's
# tool-call ID reuse rule) stay in step with what the executor writes. Callers
# hand out copies; these dicts must not be mutated.
_APPROVAL_REQUIRED_RESULT: dict[str, bool] = {"approval_required": True}
_CANCELLED_RESULT: dict[str, bool] = {"cancelled": True}


@dataclass
class ToolResultPart:
    """A tool result message."""

    tool_use_id: str

    name: str

    parts: list[TextPart | ImagePart | DocumentPart] = field(default_factory=list)

    is_error: bool = False

    kind: Literal["tool_result"] = "tool_result"
