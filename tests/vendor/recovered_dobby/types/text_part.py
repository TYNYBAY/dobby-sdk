from dataclasses import dataclass
from typing import Literal


@dataclass
class TextPart:
    """A text content part."""

    text: str

    kind: Literal["text"] = "text"
