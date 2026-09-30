from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ToolUsePart:
    """A tool use/call from the assistant."""

    id: str

    name: str

    inputs: dict[str, Any]
    
    metadata: dict[str, Any] | None = None

    kind: Literal["tool_use"] = "tool_use"
