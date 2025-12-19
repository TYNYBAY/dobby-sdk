from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .message import ResponsePart, StopReason
from .tool_events import ToolResultPart, ToolStreamEvent, ToolUseEndEvent
from .usage import Usage


class StreamStartEvent(BaseModel):

    type: Literal["stream_start"] = "stream_start"

    id: str

    model: str


class TextDeltaEvent(BaseModel):

    type: Literal["text_delta"] = "text_delta"

    delta: str


class ReasoningDeltaEvent(BaseModel):

    type: Literal["reasoning_delta"] = "reasoning_delta"

    delta: str


class ReasoningStartEvent(BaseModel):
    """Event when reasoning phase begins."""

    type: Literal["reasoning_start"] = "reasoning_start"


class ReasoningEndEvent(BaseModel):
    """Event when reasoning phase completes."""

    type: Literal["reasoning_end"] = "reasoning_end"


class StreamErrorEvent(BaseModel):
    """Event when the response fails."""

    type: Literal["stream_error"] = "stream_error"

    error_code: str | None = None
    """Error code from the provider (e.g., 'rate_limit_exceeded', 'context_length_exceeded')."""

    error_message: str
    """Human-readable error description."""


class ToolUseEvent(BaseModel):
    """Event when LLM requests a tool call during streaming.
    
    Note: This is different from ToolUsePart (TypedDict) which is used in message history.
    ToolUseEvent is a Pydantic model for streaming output, ToolUsePart is for input messages.
    """

    type: Literal["tool_use"] = "tool_use"

    id: str

    name: str
    """Name of the tool to execute."""

    inputs: dict[str, Any]
    """Arguments to pass to the tool."""


class StreamEndEvent(BaseModel):

    type: Literal["stream_end"] = "stream_end"

    model: str

    parts: list[ResponsePart]

    stop_reason: StopReason

    stop_sequence: str | None = None
    """Which custom stop sequence was generated, if any.
    
    This value will be a non-null string if one of the provided custom stop sequences was generated.
    `anthropic` returns `stop_reason`, this will be `None` for other providers like openai,
    """
    
    usage: Usage | None


type StreamEvent = Annotated[
    StreamStartEvent
    | TextDeltaEvent
    | ReasoningDeltaEvent
    | ReasoningStartEvent
    | ReasoningEndEvent
    | StreamErrorEvent
    | ToolUseEvent
    | ToolStreamEvent
    | ToolResultPart
    | ToolUseEndEvent
    | StreamEndEvent,
    Field(discriminator="type"),
]
