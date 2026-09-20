"""Tests for tool and agent error semantics."""

import asyncio

import pytest

from dobby.exceptions import (
    AgentExhaustionError,
    AgentIterationLimitError,
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
        "agent_iteration_limit",
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
        ErrorCode.AGENT_ITERATION_LIMIT,
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
        ErrorCode.AGENT_ITERATION_LIMIT,
    ],
)
def test_tool_failure_rejects_retry_and_host_codes(code: ErrorCode) -> None:
    with pytest.raises(ValueError, match="not a tool-failure error code"):
        ToolFailure("Do not retry this.", code=code)


def test_unexpected_tool_error_hides_exception_details_from_model() -> None:
    decision = classify_tool_error(RuntimeError("database password is secret"))

    assert decision is not None
    model_text = format_model_error(decision)

    assert decision.code is ErrorCode.TOOL_EXECUTION_ERROR
    assert decision.retry_model is False
    assert "database password" not in model_text
    assert "RuntimeError" not in model_text
    assert model_text == "[tool_execution_error] The tool failed unexpectedly."


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
    assert "provider" not in format_model_error(decision)


def test_model_error_formatter_uses_only_decision_fields() -> None:
    decision = ErrorDecision(
        code=ErrorCode.FINAL_RESULT_INVALID,
        model_message="Provide the missing field.",
    )

    assert format_model_error(decision) == ("[final_result_invalid] Provide the missing field.")


def test_model_error_formatter_replaces_blank_message() -> None:
    decision = ErrorDecision(
        code=ErrorCode.TOOL_FAILURE,
        model_message="  ",
    )

    assert (
        format_model_error(decision) == "[tool_failure] The tool could not complete the request."
    )


def test_classify_sanitizes_model_retry_traceback_before_formatting() -> None:
    traceback_dump = (
        "Traceback (most recent call last):\n"
        '  File "tool.py", line 10, in run\n'
        "    raise ValueError('database password is secret')\n"
        "ValueError: database password is secret"
    )
    exception = ModelRetry(
        "Use a different record ID.\n" + traceback_dump,
    )

    decision = classify_tool_error(exception)

    assert decision is not None
    assert decision.model_message == "Use a different record ID."
    assert "Traceback" not in decision.model_message
    assert "password" not in decision.model_message

    model_text = format_model_error(decision)
    assert model_text == "[tool_retry] Use a different record ID."
    assert traceback_dump not in model_text


def test_classify_replaces_traceback_only_model_retry() -> None:
    exception = ModelRetry(
        "Traceback (most recent call last):\n"
        '  File "tool.py", line 10, in run\n'
        "ValueError: database password is secret"
    )

    decision = classify_tool_error(exception)

    assert decision is not None
    assert decision.model_message == "The tool call should be corrected and retried."
    assert format_model_error(decision) == (
        "[tool_retry] The tool call should be corrected and retried."
    )


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
def test_classify_preserves_legitimate_model_retry_wording(message: str) -> None:
    decision = classify_tool_error(ModelRetry(message))

    assert decision is not None
    assert decision.model_message == message
    assert format_model_error(decision) == f"[tool_retry] {message}"


def test_model_error_formatter_bounds_explicit_message_length() -> None:
    decision = ErrorDecision(
        code=ErrorCode.TOOL_FAILURE,
        model_message="x" * 3000,
    )

    model_text = format_model_error(decision)

    assert model_text == f"[tool_failure] {'x' * 2000}"


def test_model_error_formatter_ignores_host_diagnostic_message() -> None:
    decision = ErrorDecision(
        code=ErrorCode.MODEL_RETRY_EXHAUSTED,
        model_message="Traceback and host diagnostic secret",
    )

    assert format_model_error(decision) == (
        "[model_retry_exhausted] The model correction budget was exhausted."
    )


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
    assert details.traceback not in format_model_error(decision)


def test_agent_iteration_limit_has_stable_host_metadata() -> None:
    error = AgentIterationLimitError(10)
    decision = classify_tool_error(error)

    assert isinstance(error, AgentExhaustionError)
    assert error.code is ErrorCode.AGENT_ITERATION_LIMIT
    assert error.attempts == 10
    assert error.last_error is None
    assert str(error) == "Agent iteration limit reached after 10 iterations"
    assert decision is not None
    assert decision.code is ErrorCode.AGENT_ITERATION_LIMIT
    assert decision.retry_model is False
    assert format_model_error(decision) == (
        "[agent_iteration_limit] The agent iteration limit was reached."
    )
