"""Tests for tool and agent error semantics."""

import asyncio

import pytest

from dobby.exceptions import (
    AgentExhaustionError,
    ApprovalRequired,
    ErrorCode,
    ErrorDecision,
    ModelRetry,
    ModelRetryExhaustedError,
    ToolFailure,
    classify_tool_error,
    format_model_error,
)
from dobby.providers import ProviderError, RateLimitError
from dobby.types import ToolErrorDetails


def _error_details() -> ToolErrorDetails:
    return ToolErrorDetails(
        exception_type="ValueError",
        exception_module="builtins",
        message="host diagnostic",
        traceback="Traceback (most recent call last):\nValueError: host diagnostic",
    )


def test_error_codes_are_stable_and_unique() -> None:
    assert {code.value for code in ErrorCode} == {
        "tool_not_found",
        "tool_input_invalid",
        "tool_retry",
        "tool_failure",
        "tool_execution_error",
        "final_result_invalid",
        "model_retry_exhausted",
    }
    assert len(ErrorCode) == len({code.value for code in ErrorCode})


def test_error_decision_is_immutable() -> None:
    decision = ErrorDecision(
        code=ErrorCode.TOOL_FAILURE,
        model_message="Cannot complete the operation.",
    )

    with pytest.raises(AttributeError):
        decision.code = ErrorCode.TOOL_RETRY  # type: ignore[misc]

    with pytest.raises(TypeError):
        ErrorDecision(  # type: ignore[call-arg]
            code=ErrorCode.TOOL_FAILURE,
            model_message="Cannot complete the operation.",
            retry_model=True,
        )


@pytest.mark.parametrize(
    ("code", "retry_model"),
    [
        (
            code,
            code
            in {
                ErrorCode.TOOL_RETRY,
                ErrorCode.TOOL_NOT_FOUND,
                ErrorCode.TOOL_INPUT_INVALID,
                ErrorCode.FINAL_RESULT_INVALID,
            },
        )
        for code in ErrorCode
    ],
)
def test_error_decision_derives_retry_behavior_from_code(
    code: ErrorCode,
    retry_model: bool,
) -> None:
    decision = ErrorDecision(code=code, model_message="test")

    assert decision.retry_model is retry_model


@pytest.mark.parametrize(
    ("exception", "expected_code", "retry_model"),
    [
        (ModelRetry("Use a valid account ID."), ErrorCode.TOOL_RETRY, True),
        (ToolFailure("The account is closed."), ErrorCode.TOOL_FAILURE, False),
    ],
)
def test_explicit_tool_errors_are_classified_for_the_model(
    exception: Exception,
    expected_code: ErrorCode,
    retry_model: bool,
) -> None:
    decision = classify_tool_error(exception)

    assert decision is not None
    assert decision.code is expected_code
    assert decision.model_message == str(exception)
    assert decision.retry_model is retry_model


@pytest.mark.parametrize(
    ("exception", "expected_code", "retry_model"),
    [
        (
            ModelRetry("Correct the tool arguments.", code=ErrorCode.TOOL_INPUT_INVALID),
            ErrorCode.TOOL_INPUT_INVALID,
            True,
        ),
        (
            ModelRetry("Correct the final result.", code=ErrorCode.FINAL_RESULT_INVALID),
            ErrorCode.FINAL_RESULT_INVALID,
            True,
        ),
        (
            ModelRetry("The requested tool does not exist.", code=ErrorCode.TOOL_NOT_FOUND),
            ErrorCode.TOOL_NOT_FOUND,
            True,
        ),
    ],
)
def test_context_specific_codes_preserve_retry_semantics(
    exception: Exception,
    expected_code: ErrorCode,
    retry_model: bool,
) -> None:
    decision = classify_tool_error(exception)

    assert decision is not None
    assert decision.code is expected_code
    assert decision.retry_model is retry_model


@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.TOOL_FAILURE,
        ErrorCode.TOOL_EXECUTION_ERROR,
        ErrorCode.MODEL_RETRY_EXHAUSTED,
    ],
)
def test_model_retry_rejects_non_retryable_codes(code: ErrorCode) -> None:
    with pytest.raises(ValueError, match="not a model-retry error code"):
        ModelRetry("Retry this.", code=code)


@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.TOOL_RETRY,
        ErrorCode.TOOL_NOT_FOUND,
        ErrorCode.TOOL_INPUT_INVALID,
        ErrorCode.TOOL_EXECUTION_ERROR,
        ErrorCode.FINAL_RESULT_INVALID,
        ErrorCode.MODEL_RETRY_EXHAUSTED,
    ],
)
def test_tool_failure_rejects_retry_and_host_codes(code: ErrorCode) -> None:
    with pytest.raises(ValueError, match="not a tool-failure error code"):
        ToolFailure("Do not retry this.", code=code)


def test_unexpected_tool_error_uses_exception_message() -> None:
    decision = classify_tool_error(RuntimeError("database password is secret"))

    assert decision is not None
    model_text = format_model_error(decision)

    assert decision.code is ErrorCode.TOOL_EXECUTION_ERROR
    assert decision.retry_model is False
    assert decision.model_message == "database password is secret"
    assert model_text == "[tool_execution_error] database password is secret"


@pytest.mark.parametrize(
    "provider_error",
    [
        ProviderError("bad request", provider="openai", status_code=400),
        RateLimitError("retry provider request", provider="openai"),
    ],
)
def test_provider_errors_do_not_enter_model_retry_semantics(
    provider_error: ProviderError,
) -> None:
    decision = classify_tool_error(provider_error)

    assert decision is not None
    assert decision.code is ErrorCode.TOOL_EXECUTION_ERROR
    assert decision.retry_model is False
    assert format_model_error(decision) == f"[tool_execution_error] {provider_error}"


def test_model_error_formatter_uses_only_decision_fields() -> None:
    decision = ErrorDecision(
        code=ErrorCode.FINAL_RESULT_INVALID,
        model_message="Provide the missing field.",
    )

    assert format_model_error(decision) == ("[final_result_invalid] Provide the missing field.")


@pytest.mark.parametrize(
    "message",
    [
        "Please do not include a traceback in the output",
        "ValidationError: name field is required",
        "TimeoutError: try a smaller payload",
        "File a ticket, line of business: claims",
        'Config is in File "settings.py", line 10 of the repo',
    ],
)
def test_classify_preserves_model_retry_message(message: str) -> None:
    decision = classify_tool_error(ModelRetry(message))

    assert decision is not None
    assert decision.model_message == message
    assert format_model_error(decision) == f"[tool_retry] {message}"


def test_approval_required_remains_outside_error_classification() -> None:
    approval = ApprovalRequired("call-1", "delete_record", {"record_id": "123"})

    assert classify_tool_error(approval) is None


def test_cancellation_remains_outside_error_classification() -> None:
    cancellation = asyncio.CancelledError()

    assert classify_tool_error(cancellation) is None


def test_model_retry_exhaustion_preserves_last_diagnostic() -> None:
    details = _error_details()
    error = ModelRetryExhaustedError(2, last_error=details)
    decision = classify_tool_error(error)

    assert isinstance(error, AgentExhaustionError)
    assert error.code is ErrorCode.MODEL_RETRY_EXHAUSTED
    assert error.attempts == 2
    assert error.last_error is details
    assert str(error) == "Model retry budget exhausted after 2 attempts"
    assert decision is not None
    assert decision.code is ErrorCode.MODEL_RETRY_EXHAUSTED
    assert decision.retry_model is False
    assert format_model_error(decision) == (
        "[model_retry_exhausted] Model retry budget exhausted after 2 attempts"
    )
    assert details.traceback not in format_model_error(decision)
