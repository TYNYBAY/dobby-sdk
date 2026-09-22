"""Tests for host-side tool execution retry."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from dobby import AgentExecutor
from dobby.exceptions import ApprovalRequired, ModelRetry, ToolFailure
from dobby.tools import Tool, ToolRetryPolicy
from dobby.types import StreamEndEvent, ToolResultEvent, ToolStreamEvent, ToolUsePart, Usage


def _make_mock_provider(tool_calls: list[ToolUsePart]) -> AsyncMock:
    call_count = 0

    async def mock_chat(*args, **kwargs):
        del args, kwargs
        nonlocal call_count
        call_count += 1

        async def stream():
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=tool_calls if call_count == 1 else [],
                stop_reason="tool_use" if call_count == 1 else "end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    provider.chat_calls = lambda: call_count
    return provider


async def _collect_events(tools: list[Tool], tool_calls: list[ToolUsePart], **run_kwargs):
    executor = AgentExecutor(
        provider="openai",
        llm=_make_mock_provider(tool_calls),
        tools=tools,
    )
    return [event async for event in executor.run_stream(messages=[], **run_kwargs)], executor


async def _collect_results(tools: list[Tool], tool_calls: list[ToolUsePart], **run_kwargs):
    events, executor = await _collect_events(tools, tool_calls, **run_kwargs)
    return [event for event in events if isinstance(event, ToolResultEvent)], events, executor


def test_default_policy_does_not_retry_value_error() -> None:
    calls = 0

    @dataclass
    class FailingTool(Tool):
        name = "failing"
        description = "Fail with ValueError."

        async def __call__(self) -> None:
            nonlocal calls
            calls += 1
            raise ValueError("not retryable by default")

    results, _, _ = asyncio.run(
        _collect_results(
            [FailingTool()],
            [ToolUsePart(id="call-failing", name="failing", inputs={})],
        )
    )

    assert calls == 1
    assert len(results) == 1
    assert results[0].is_error is True
    assert "not retryable by default" not in str(results[0].result)
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."
    assert FailingTool().retry_policy() == ToolRetryPolicy(
        max_retries=1,
        retryable_exceptions=(),
    )


def test_listed_retryable_exception_retries_then_succeeds() -> None:
    calls = 0

    @dataclass
    class FlakyTool(Tool):
        name = "flaky"
        description = "Fail once, then succeed."
        retryable_exceptions = (TimeoutError,)

        async def __call__(self) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("transient")
            return "ok"

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
        results, _, _ = asyncio.run(
            _collect_results(
                [FlakyTool()],
                [ToolUsePart(id="call-flaky", name="flaky", inputs={})],
            )
        )

    assert calls == 2
    assert sleep.await_count == 1
    assert len(results) == 1
    assert results[0].is_error is False
    assert results[0].result == "ok"
    assert results[0].error_details is None


def test_retry_exhaustion_uses_last_error() -> None:
    calls = 0
    errors = ("first timeout", "last timeout")

    @dataclass
    class AlwaysTimeoutTool(Tool):
        name = "timeout"
        description = "Always time out."
        retryable_exceptions = (TimeoutError,)

        async def __call__(self) -> None:
            nonlocal calls
            message = errors[min(calls, len(errors) - 1)]
            calls += 1
            raise TimeoutError(message)

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        results, _, _ = asyncio.run(
            _collect_results(
                [AlwaysTimeoutTool()],
                [ToolUsePart(id="call-timeout", name="timeout", inputs={})],
            )
        )

    assert calls == 2
    assert len(results) == 1
    assert results[0].is_error is True
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."
    assert "last timeout" not in str(results[0].result)
    assert results[0].error_details is not None
    assert results[0].error_details.exception_type == "TimeoutError"
    assert results[0].error_details.message == "last timeout"


def test_max_retries_zero_gives_exactly_one_call() -> None:
    calls = 0

    @dataclass
    class NoRetryTool(Tool):
        name = "no_retry"
        description = "Do not retry."
        max_retries = 0
        retryable_exceptions = (TimeoutError,)

        async def __call__(self) -> None:
            nonlocal calls
            calls += 1
            raise TimeoutError("still fail")

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
        results, _, _ = asyncio.run(
            _collect_results(
                [NoRetryTool()],
                [ToolUsePart(id="call-no-retry", name="no_retry", inputs={})],
            )
        )

    assert calls == 1
    assert sleep.await_count == 0
    assert results[0].is_error is True
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."
    assert "still fail" not in str(results[0].result)


def test_non_listed_exception_does_not_retry() -> None:
    calls = 0

    @dataclass
    class ListedTimeoutTool(Tool):
        name = "listed"
        description = "Retry timeouts only."
        retryable_exceptions = (TimeoutError,)

        async def __call__(self) -> None:
            nonlocal calls
            calls += 1
            raise ConnectionError("not listed")

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
        results, _, _ = asyncio.run(
            _collect_results(
                [ListedTimeoutTool()],
                [ToolUsePart(id="call-listed", name="listed", inputs={})],
            )
        )

    assert calls == 1
    assert sleep.await_count == 0
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."
    assert "not listed" not in str(results[0].result)


def test_model_retry_from_tool_body_does_not_retry() -> None:
    calls = 0

    @dataclass
    class BodyRetryTool(Tool):
        name = "body_retry"
        description = "Raise ModelRetry."
        retryable_exceptions = (TimeoutError, Exception)

        async def __call__(self) -> None:
            nonlocal calls
            calls += 1
            raise ModelRetry("correct the arguments")

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
        results, _, _ = asyncio.run(
            _collect_results(
                [BodyRetryTool()],
                [ToolUsePart(id="call-body-retry", name="body_retry", inputs={})],
            )
        )

    assert calls == 1
    assert sleep.await_count == 0
    assert results[0].is_error is True
    assert str(results[0].result) == "[tool_retry] correct the arguments"


def test_tool_failure_from_tool_body_does_not_retry() -> None:
    calls = 0

    @dataclass
    class BodyFailureTool(Tool):
        name = "body_failure"
        description = "Raise ToolFailure."
        retryable_exceptions = (Exception,)

        async def __call__(self) -> None:
            nonlocal calls
            calls += 1
            raise ToolFailure("account is closed")

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
        results, _, _ = asyncio.run(
            _collect_results(
                [BodyFailureTool()],
                [ToolUsePart(id="call-body-failure", name="body_failure", inputs={})],
            )
        )

    assert calls == 1
    assert sleep.await_count == 0
    assert str(results[0].result) == "[tool_failure] account is closed"


def test_approval_required_is_raised_and_not_retried() -> None:
    calls = 0

    @dataclass
    class ApprovalTool(Tool):
        name = "approval"
        description = "Require approval."
        requires_approval = True
        retryable_exceptions = (TimeoutError,)

        async def __call__(self) -> str:
            nonlocal calls
            calls += 1
            return "should not run"

    async def run() -> None:
        await _collect_results(
            [ApprovalTool()],
            [ToolUsePart(id="call-approval", name="approval", inputs={})],
        )

    with pytest.raises(ApprovalRequired) as exc_info:
        asyncio.run(run())

    assert calls == 0
    assert exc_info.value.tool_call_id == "call-approval"


def test_cancelled_error_propagates_and_is_not_retried() -> None:
    calls = 0

    @dataclass
    class CancelTool(Tool):
        name = "cancel"
        description = "Cancel the call."
        retryable_exceptions = (Exception,)

        async def __call__(self) -> None:
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError

    async def run() -> None:
        await _collect_results(
            [CancelTool()],
            [ToolUsePart(id="call-cancel", name="cancel", inputs={})],
        )

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run())

    assert calls == 1
    assert sleep.await_count == 0


def test_invalid_inputs_are_validated_once_and_never_invoked() -> None:
    calls = 0

    @dataclass
    class TypedTool(Tool):
        name = "typed"
        description = "Accept an integer."
        retryable_exceptions = (ValueError,)

        async def __call__(self, count: int) -> int:
            nonlocal calls
            calls += 1
            return count

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
        results, _, _ = asyncio.run(
            _collect_results(
                [TypedTool()],
                [ToolUsePart(id="call-invalid", name="typed", inputs={"count": "bad"})],
            )
        )

    assert calls == 0
    assert sleep.await_count == 0
    assert str(results[0].result).startswith("[tool_input_invalid]")


def test_retry_attempts_do_not_consume_model_correction_budget() -> None:
    calls = 0

    @dataclass
    class TimeoutTool(Tool):
        name = "timeout"
        description = "Always time out."
        retryable_exceptions = (TimeoutError,)

        async def __call__(self) -> None:
            nonlocal calls
            calls += 1
            raise TimeoutError(f"timeout {calls}")

    tool_calls = [ToolUsePart(id="call-timeout", name="timeout", inputs={})]
    provider = _make_mock_provider(tool_calls)
    executor = AgentExecutor(provider="openai", llm=provider, tools=[TimeoutTool()])

    async def run():
        return [event async for event in executor.run_stream(messages=[], max_model_corrections=0)]

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events = asyncio.run(run())

    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert calls == 2
    assert provider.chat_calls() == 2
    assert len(results) == 1
    assert results[0].is_error is True
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."
    assert "timeout 2" not in str(results[0].result)


def test_backoff_sleep_occurs_only_between_retry_attempts() -> None:
    calls = 0

    @dataclass
    class FlakyTool(Tool):
        name = "flaky"
        description = "Fail twice, then succeed."
        max_retries = 2
        retryable_exceptions = (TimeoutError,)

        async def __call__(self) -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TimeoutError("transient")
            return "ok"

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
        asyncio.run(
            _collect_results(
                [FlakyTool()],
                [ToolUsePart(id="call-flaky", name="flaky", inputs={})],
            )
        )

    assert calls == 3
    assert sleep.await_count == 2
    for awaited in sleep.await_args_list:
        delay = awaited.args[0]
        assert 0.1 <= delay <= 2.0


def test_streaming_failure_before_first_yield_retries_without_duplicate_events() -> None:
    calls = 0

    @dataclass
    class StreamingTool(Tool):
        name = "streaming"
        description = "Fail before first yield, then stream."
        stream_output = True
        retryable_exceptions = (TimeoutError,)

        async def __call__(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("before yield")
            yield ToolStreamEvent(type="progress", data="started")
            yield "done"

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        results, events, _ = asyncio.run(
            _collect_results(
                [StreamingTool()],
                [ToolUsePart(id="call-streaming", name="streaming", inputs={})],
            )
        )

    stream_events = [event for event in events if isinstance(event, ToolStreamEvent)]
    assert calls == 2
    assert [event.data for event in stream_events] == ["started"]
    assert results[0].is_error is False
    assert results[0].result == "done"


def test_streaming_yield_then_failure_does_not_retry() -> None:
    calls = 0

    @dataclass
    class StreamingTool(Tool):
        name = "streaming"
        description = "Yield then fail."
        stream_output = True
        retryable_exceptions = (TimeoutError,)

        async def __call__(self):
            nonlocal calls
            calls += 1
            yield ToolStreamEvent(type="progress", data="started")
            raise TimeoutError("after yield")

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
        results, events, _ = asyncio.run(
            _collect_results(
                [StreamingTool()],
                [ToolUsePart(id="call-streaming", name="streaming", inputs={})],
            )
        )

    stream_events = [event for event in events if isinstance(event, ToolStreamEvent)]
    assert calls == 1
    assert sleep.await_count == 0
    assert [event.data for event in stream_events] == ["started"]
    assert results[0].is_error is True
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."
    assert "after yield" not in str(results[0].result)
    assert results[0].error_details is not None
    assert results[0].error_details.exception_type == "TimeoutError"
    assert results[0].error_details.message == "after yield"
