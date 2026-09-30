from typing import Any, Literal

from pydantic import BaseModel

from ..exceptions.error_code import ErrorCode


class ToolStreamEvent(BaseModel):
    """Event emitted by streaming tools during execution.

    Used for mid-execution streaming (e.g., progress updates, partial results).
    """

    type: str
    data: Any


class ToolErrorDetails(BaseModel):
    """Structured diagnostics for an exception raised during tool execution."""

    exception_type: str
    exception_module: str
    message: str
    traceback: str
    error_code: ErrorCode | None = None
    run_id: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    attempt: int | None = None
    max_attempts: int | None = None


class ToolResultEvent(BaseModel):
    """Result from tool execution."""

    type: Literal["tool_result_event"] = "tool_result_event"
    tool_use_id: str
    name: str
    result: Any
    is_error: bool = False
    error_details: ToolErrorDetails | None = None
    is_terminal: bool = False
    """If True, this tool execution exits the agent loop."""


class ToolUseEndEvent(BaseModel):
    """Event when tool execution completes."""

    type: Literal["tool_use_end"] = "tool_use_end"
    tool_use_id: str
    tool_name: str


class AppliedEdit(BaseModel):
    """A single compaction edit applied to the conversation.

    Records what trim or summarize changed so the executor can update its
    watermark. This is not yet emitted as a stream event.
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
