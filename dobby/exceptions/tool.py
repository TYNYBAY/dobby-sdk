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


_DEFAULT_MODEL_MESSAGES = {
    ErrorCode.TOOL_NOT_FOUND: "The requested tool is not available.",
    ErrorCode.TOOL_INPUT_INVALID: "The tool arguments are invalid.",
    ErrorCode.TOOL_RETRY: "The tool call should be corrected and retried.",
    ErrorCode.TOOL_FAILURE: "The tool could not complete the request.",
    ErrorCode.TOOL_EXECUTION_ERROR: "The tool failed unexpectedly.",
    ErrorCode.FINAL_RESULT_INVALID: "The final result is invalid.",
    ErrorCode.MODEL_RETRY_EXHAUSTED: "The model correction budget was exhausted.",
    ErrorCode.AGENT_ITERATION_LIMIT: "The agent iteration limit was reached.",
}
_EXPLICIT_MODEL_MESSAGE_CODES = _MODEL_RETRY_CODES | _TOOL_FAILURE_CODES
_TRACEBACK_HEADERS = (
    "traceback (most recent call last):",
    "during handling of the above exception, another exception occurred:",
    "the above exception was the direct cause of the following exception:",
)
_MAX_MODEL_MESSAGE_LENGTH = 2000


def classify_tool_error(exception: BaseException) -> ErrorDecision | None:
    """Classify a tool exception without exposing unexpected exception details.

    Returns ``None`` for approval, cancellation, and other non-error control flow.
    Explicit tool errors are sanitized before the decision is constructed.
    """
    if isinstance(exception, ApprovalRequired) or not isinstance(exception, Exception):
        return None
    if isinstance(exception, ModelRetry):
        return _model_facing_decision(exception.code, str(exception))
    if isinstance(exception, ToolFailure):
        return _model_facing_decision(exception.code, str(exception))
    if isinstance(exception, AgentExhaustionError):
        return ErrorDecision(
            code=exception.code,
            model_message=_DEFAULT_MODEL_MESSAGES[exception.code],
        )
    return ErrorDecision(
        code=ErrorCode.TOOL_EXECUTION_ERROR,
        model_message=_DEFAULT_MODEL_MESSAGES[ErrorCode.TOOL_EXECUTION_ERROR],
    )


def _traceback_header_index(line: str) -> int:
    """Return the index of a CPython traceback header, or -1 if none."""
    normalized = line.lower()
    indexes = [normalized.find(header) for header in _TRACEBACK_HEADERS if header in normalized]
    return min(indexes) if indexes else -1


def _safe_model_message(message: str, *, fallback: str) -> str:
    """Keep model text, dropping only CPython traceback dumps."""
    safe_lines: list[str] = []
    for line in message.splitlines():
        header_at = _traceback_header_index(line)
        if header_at != -1:
            prefix = line[:header_at].rstrip()
            if prefix:
                safe_lines.append(prefix)
            break
        safe_lines.append(line)
    safe_message = "\n".join(safe_lines).strip()
    if not safe_message:
        return fallback
    return safe_message[:_MAX_MODEL_MESSAGE_LENGTH].rstrip()


def _model_facing_decision(code: ErrorCode, message: str) -> ErrorDecision:
    """Build an ErrorDecision whose model_message is already safe to expose."""
    fallback = _DEFAULT_MODEL_MESSAGES[code]
    return ErrorDecision(
        code=code,
        model_message=_safe_model_message(message, fallback=fallback),
    )


def format_model_error(decision: ErrorDecision) -> str:
    """Format a classified error for the model without host diagnostics."""
    if decision.code in _EXPLICIT_MODEL_MESSAGE_CODES:
        message = _safe_model_message(
            decision.model_message,
            fallback=_DEFAULT_MODEL_MESSAGES[decision.code],
        )
    else:
        message = _DEFAULT_MODEL_MESSAGES[decision.code]
    return f"[{decision.code.value}] {message}"
