from typing import Any, Literal

from pydantic import BaseModel


class ToolStreamEvent(BaseModel):
    """Event emitted by streaming tools during execution.

    Used for mid-execution streaming (e.g., progress updates, partial results).
    """

    type: str
    data: Any


class ToolResultPart(BaseModel):
    """Result from tool execution."""

    type: Literal["tool_result_event"] = "tool_result_event"
    tool_use_id: str
    name: str
    result: Any
    is_error: bool = False


class ToolUseEndEvent(BaseModel):
    """Event when tool execution completes."""

    type: Literal["tool_use_end"] = "tool_use_end"
    tool_use_id: str
    tool_name: str
