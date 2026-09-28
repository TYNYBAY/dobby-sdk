"""Stable error codes for tool and agent failures."""

from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable codes for tool and agent errors."""

    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_INPUT_INVALID = "tool_input_invalid"
    TOOL_RETRY = "tool_retry"
    TOOL_FAILURE = "tool_failure"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    FINAL_RESULT_INVALID = "final_result_invalid"
    MODEL_RETRY_EXHAUSTED = "model_retry_exhausted"
