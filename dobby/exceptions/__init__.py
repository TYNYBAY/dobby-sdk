"""Exceptions for the dobby SDK."""

from .error_code import ErrorCode as ErrorCode
from .tool import (
    AgentExhaustionError as AgentExhaustionError,
    ApprovalRequired as ApprovalRequired,
    ErrorDecision as ErrorDecision,
    ModelRetry as ModelRetry,
    ModelRetryExhaustedError as ModelRetryExhaustedError,
    ToolFailure as ToolFailure,
    classify_tool_error as classify_tool_error,
    format_model_error as format_model_error,
)
