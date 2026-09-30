from typing import Any, Literal

from pydantic import BaseModel


class ToolStreamEvent(BaseModel):
    """Event emitted by streaming tools during execution.

    Used for mid-execution streaming (e.g., progress updates, partial results).
    """

    type: str
    data: Any


class ToolResultEvent(BaseModel):
    """Result from tool execution."""

    type: Literal["tool_result_event"] = "tool_result_event"
    tool_use_id: str
    name: str
    result: Any
    is_error: bool = False
    is_terminal: bool = False
    """If True, this tool execution exits the agent loop."""


class ToolUseEndEvent(BaseModel):
    """Event when tool execution completes."""

    type: Literal["tool_use_end"] = "tool_use_end"
    tool_use_id: str
    tool_name: str


class AppliedEdit(BaseModel):
    """A single compaction edit applied to the conversation.

    Carried on :class:`ContextEditEvent` so compaction is observable, and stashes
    the replaced originals for audit/replay.
    """

    type: Literal["clear_tool_uses", "summarize"]
    cleared_tool_uses: int = 0
    """Number of tool round-trips whose payloads were cleared (trim)."""
    cleared_input_tokens_estimate: int = 0
    """Rough estimate of input tokens removed by this edit."""
    summary_text: str | None = None
    """The generated summary text, when ``type == "summarize"``."""
    replaced_originals: list[Any] | None = None
    """The pre-compaction parts this edit replaced (audit stash).

    Typed ``Any`` to avoid a parts-typing import cycle.
    """


class ContextEditEvent(BaseModel):
    """Event emitted when context compaction runs between turns."""

    type: Literal["context_edit"] = "context_edit"
    applied_edits: list[AppliedEdit]
