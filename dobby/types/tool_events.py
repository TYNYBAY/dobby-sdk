from typing import Any, Literal, TypedDict


class ToolStreamEvent(TypedDict):
    """Event emitted by streaming tools during execution.

    Used for mid-execution streaming (e.g., document content deltas).
    """

    type: str  # e.g., "data-textDelta", "data-kind", etc.
    data: Any


class ToolResultPart(TypedDict):
    """Result from tool execution."""

    type: Literal["tool_result_event"]
    tool_use_id: str
    name: str
    result: Any
    is_error: bool


class ToolUseEndEvent(TypedDict):
    """Event when tool execution completes."""

    type: Literal["tool_use_end"]
    tool_use_id: str
    tool_name: str
