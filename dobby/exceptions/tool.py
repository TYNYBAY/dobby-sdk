"""Tool-related exceptions."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..types.tool_events import ToolErrorDetails


class ErrorCode(StrEnum):
    """Stable codes for tool and agent errors."""

    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_INPUT_INVALID = "tool_input_invalid"
    TOOL_RETRY = "tool_retry"
    TOOL_FAILURE = "tool_failure"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    FINAL_RESULT_INVALID = "final_result_invalid"
    MODEL_RETRY_EXHAUSTED = "model_retry_exhausted"
    AGENT_ITERATION_LIMIT = "agent_iteration_limit"


_MODEL_RETRY_CODES = frozenset(
    {
        ErrorCode.TOOL_RETRY,
        ErrorCode.TOOL_NOT_FOUND,
        ErrorCode.TOOL_INPUT_INVALID,
        ErrorCode.FINAL_RESULT_INVALID,
    }
)
_TOOL_FAILURE_CODES = frozenset({ErrorCode.TOOL_FAILURE})
_EXHAUSTION_CODES = frozenset(
    {
        ErrorCode.MODEL_RETRY_EXHAUSTED,
        ErrorCode.AGENT_ITERATION_LIMIT,
    }
)


@dataclass(frozen=True, slots=True)
class ErrorDecision:
    """Model-facing disposition of a classified tool error."""

    code: ErrorCode
    model_message: str

    @property
    def retry_model(self) -> bool:
        """Whether this error permits another model correction attempt."""
        return self.code in _MODEL_RETRY_CODES


class ModelRetry(Exception):
    """Request model correction using an intentionally model-facing message."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.TOOL_RETRY,
    ) -> None:
        if code not in _MODEL_RETRY_CODES:
            raise ValueError(f"{code.value!r} is not a model-retry error code")
        self.code = code
        super().__init__(message)


class ToolFailure(Exception):
    """Report a non-retryable tool failure with a model-facing message."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.TOOL_FAILURE,
    ) -> None:
        if code not in _TOOL_FAILURE_CODES:
            raise ValueError(f"{code.value!r} is not a tool-failure error code")
        self.code = code
        super().__init__(message)


@dataclass
class ApprovalRequired(Exception):
    """Raised when a tool requires human approval before execution.

    This exception is raised during tool execution when:
    1. The tool has `requires_approval=True`
    2. The tool_call_id is not in the `approved_tool_calls` set

    The caller should catch this exception, present the tool call to the user,
    and re-run with the tool_call_id added to approved_tool_calls.

    Attributes:
        tool_call_id: Unique identifier for this tool call
        tool_name: Name of the tool that requires approval
        tool_args: Arguments that would be passed to the tool
    """

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]

    def __str__(self) -> str:
        """Describe the pending tool approval."""
        return f"Tool '{self.tool_name}' requires approval (call_id: {self.tool_call_id})"


class AgentExhaustionError(Exception):
    """Base class for host-facing agent exhaustion errors."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        attempts: int,
        last_error: ToolErrorDetails | None = None,
    ) -> None:
        if code not in _EXHAUSTION_CODES:
            raise ValueError(f"{code.value!r} is not an exhaustion error code")
        self.code = code
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(message)


class ModelRetryExhaustedError(AgentExhaustionError):
    """Raised when the model-correction budget is exhausted."""

    def __init__(
        self,
        attempts: int,
        *,
        last_error: ToolErrorDetails | None = None,
    ) -> None:
        super().__init__(
            f"Model retry budget exhausted after {attempts} attempts",
            code=ErrorCode.MODEL_RETRY_EXHAUSTED,
            attempts=attempts,
            last_error=last_error,
        )


class AgentIterationLimitError(AgentExhaustionError):
    """Raised when an agent reaches its configured iteration limit."""

    def __init__(self, attempts: int) -> None:
        super().__init__(
            f"Agent iteration limit reached after {attempts} iterations",
            code=ErrorCode.AGENT_ITERATION_LIMIT,
            attempts=attempts,
        )


def classify_tool_error(exception: BaseException) -> ErrorDecision | None:
    """Classify a tool exception for model-facing emission.

    Returns ``None`` for approval, cancellation, and other non-error control flow.
    """
    if isinstance(exception, ApprovalRequired) or not isinstance(exception, Exception):
        return None
    if isinstance(exception, ModelRetry):
        return ErrorDecision(code=exception.code, model_message=str(exception))
    if isinstance(exception, ToolFailure):
        return ErrorDecision(code=exception.code, model_message=str(exception))
    if isinstance(exception, AgentExhaustionError):
        return ErrorDecision(code=exception.code, model_message=str(exception))
    return ErrorDecision(
        code=ErrorCode.TOOL_EXECUTION_ERROR,
        model_message=str(exception),
    )


def format_model_error(decision: ErrorDecision) -> str:
    """Format a classified error as ``[code] message``."""
    return f"[{decision.code.value}] {decision.model_message}"
