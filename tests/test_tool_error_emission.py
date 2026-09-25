"""Tests for classified tool-error emission and control-flow assembly."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from dobby import AgentExecutor
from dobby.exceptions import ApprovalRequired, ModelRetryExhaustedError, ToolFailure
from dobby.tools import Tool
from dobby.types import (
    StreamEndEvent,
    ToolResultEvent,
    ToolResultPart,
    ToolStreamEvent,
    ToolUsePart,
    Usage,
    UserMessagePart,
)


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
    return provider


async def _collect_events(tools: list[Tool], tool_calls: list[ToolUsePart], **kwargs):
    executor = AgentExecutor(
        provider="openai",
        llm=_make_mock_provider(tool_calls),
        tools=tools,
    )
    return [event async for event in executor.run_stream(messages=[], **kwargs)], executor


async def _collect_until_exception(tools, tool_calls, exception_type, **kwargs):
    executor = AgentExecutor(
        provider="openai",
        llm=_make_mock_provider(tool_calls),
        tools=tools,
    )
    events = []
    caught = None
    with patch.object(
        executor, "_emit_tool_result", wraps=executor._emit_tool_result
    ) as emit_result:
        try:
            async for event in executor.run_stream(messages=[], **kwargs):
                events.append(event)
        except exception_type as exception:
            caught = exception
    assert caught is not None
    messages = emit_result.call_args.args[3] if emit_result.call_args else []
    return events, messages, caught


def _tool_result_parts(messages):
    return [
        part
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]


def test_unexpected_error_is_classified_and_keeps_host_diagnostics() -> None:
    @dataclass
    class BrokenTool(Tool):
        name = "broken"
        description = "Fail unexpectedly."

        async def __call__(self) -> None:
            raise ValueError("secret diagnostic")

    events, _ = asyncio.run(
        _collect_events(
            [BrokenTool()],
            [ToolUsePart(id="call-broken", name="broken", inputs={})],
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].result == "[tool_execution_error] The tool failed unexpectedly."
    assert results[0].error_details is not None
    assert results[0].error_details.message == "secret diagnostic"
    assert "secret diagnostic" in results[0].error_details.traceback


def test_tool_failure_is_classified_without_model_correction() -> None:
    @dataclass
    class ClosedTool(Tool):
        name = "closed"
        description = "Report a non-retryable failure."

        async def __call__(self) -> None:
            raise ToolFailure("account is closed")

    events, executor = asyncio.run(
        _collect_events(
            [ClosedTool()],
            [ToolUsePart(id="call-closed", name="closed", inputs={})],
            max_model_corrections=0,
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(results) == 1
    assert results[0].result == "[tool_failure] account is closed"
    assert results[0].is_error is True
    assert results[0].error_details is not None
    assert results[0].error_details.exception_type == "ToolFailure"


def test_tool_raised_agent_exhaustion_is_classified() -> None:
    @dataclass
    class ExhaustedTool(Tool):
        name = "exhausted"
        description = "Raise a host exhaustion error."

        async def __call__(self) -> None:
            raise ModelRetryExhaustedError(2)

    events, _ = asyncio.run(
        _collect_events(
            [ExhaustedTool()],
            [ToolUsePart(id="call-exhausted", name="exhausted", inputs={})],
        )
    )
    result = next(event for event in events if isinstance(event, ToolResultEvent))

    assert result.result == (
        "[model_retry_exhausted] Model retry budget exhausted after 2 attempts"
    )
    assert result.error_details is not None
    assert result.error_details.exception_type == "ModelRetryExhaustedError"


def test_phase6_exhausted_error_uses_last_exception_and_is_classified() -> None:
    calls = 0

    @dataclass
    class TimeoutTool(Tool):
        name = "timeout"
        description = "Time out twice."
        retryable_exceptions = (TimeoutError,)

        async def __call__(self) -> None:
            nonlocal calls
            calls += 1
            raise TimeoutError(f"timeout {calls}")

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events, _ = asyncio.run(
            _collect_events(
                [TimeoutTool()],
                [ToolUsePart(id="call-timeout", name="timeout", inputs={})],
                max_model_corrections=0,
            )
        )
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert calls == 2
    assert results[0].result == "[tool_execution_error] The tool failed unexpectedly."
    assert results[0].error_details is not None
    assert results[0].error_details.message == "timeout 2"


def test_parallel_sibling_keeps_success_and_classified_error_order() -> None:
    @dataclass
    class OkTool(Tool):
        name = "ok"
        description = "Succeed."

        async def __call__(self) -> str:
            return "ok"

    @dataclass
    class FailTool(Tool):
        name = "fail"
        description = "Fail."

        async def __call__(self) -> None:
            raise RuntimeError("boom")

    events, _ = asyncio.run(
        _collect_events(
            [OkTool(), FailTool()],
            [
                ToolUsePart(id="call-ok", name="ok", inputs={}),
                ToolUsePart(id="call-fail", name="fail", inputs={}),
            ],
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert [result.tool_use_id for result in results] == ["call-ok", "call-fail"]
    assert results[0].is_error is False
    assert results[0].result == "ok"
    assert results[1].is_error is True
    assert results[1].result == "[tool_execution_error] The tool failed unexpectedly."


def test_streaming_after_yield_failure_is_classified_without_retry() -> None:
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
        events, _ = asyncio.run(
            _collect_events(
                [StreamingTool()],
                [ToolUsePart(id="call-streaming", name="streaming", inputs={})],
            )
        )
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    stream_events = [event for event in events if isinstance(event, ToolStreamEvent)]

    assert calls == 1
    assert sleep.await_count == 0
    assert [event.data for event in stream_events] == ["started"]
    assert results[0].result == "[tool_execution_error] The tool failed unexpectedly."
    assert results[0].error_details is not None
    assert results[0].error_details.message == "after yield"


def test_streaming_tool_raised_cancelled_error_is_classified() -> None:
    @dataclass
    class StreamingTool(Tool):
        name = "streaming"
        description = "Raise CancelledError from the tool body."
        stream_output = True

        async def __call__(self):
            yield ToolStreamEvent(type="progress", data="started")
            raise asyncio.CancelledError

    events, _ = asyncio.run(
        _collect_events(
            [StreamingTool()],
            [ToolUsePart(id="call-streaming", name="streaming", inputs={})],
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert results[0].result == "[tool_execution_error] The tool failed unexpectedly."
    assert results[0].error_details is not None
    assert results[0].error_details.exception_type == "CancelledError"


def test_terminal_unexpected_error_is_classified_and_still_terminal() -> None:
    @dataclass
    class TerminalTool(Tool):
        name = "terminal"
        description = "Fail and stop."
        terminal = True

        async def __call__(self) -> None:
            raise RuntimeError("terminal boom")

    events, _ = asyncio.run(
        _collect_events(
            [TerminalTool()],
            [ToolUsePart(id="call-terminal", name="terminal", inputs={})],
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(results) == 1
    assert results[0].result == "[tool_execution_error] The tool failed unexpectedly."
    assert results[0].is_terminal is True
    assert results[0].error_details is not None
    assert results[0].error_details.message == "terminal boom"


def test_streaming_then_later_terminal_assemble_control_flow() -> None:
    @dataclass
    class StreamingTool(Tool):
        name = "streaming"
        description = "Require approval."
        requires_approval = True
        stream_output = True

        async def __call__(self):
            yield ToolStreamEvent(type="progress", data="started")

    @dataclass
    class TerminalTool(Tool):
        name = "terminal"
        description = "Should not run."
        terminal = True

        async def __call__(self) -> str:
            return "done"

    events, messages, error = asyncio.run(
        _collect_until_exception(
            [StreamingTool(), TerminalTool()],
            [
                ToolUsePart(id="call-streaming", name="streaming", inputs={}),
                ToolUsePart(id="call-terminal", name="terminal", inputs={}),
            ],
            ApprovalRequired,
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    parts = _tool_result_parts(messages)

    assert isinstance(error, ApprovalRequired)
    assert [result.tool_use_id for result in results] == ["call-streaming", "call-terminal"]
    assert [part.tool_use_id for part in parts] == ["call-streaming", "call-terminal"]
    assert all(result.result == {"approval_required": True} for result in results)
    assert all(result.is_error is True for result in results)
    assert all(result.error_details is None for result in results)


@pytest.mark.parametrize("later_kind", ["streaming", "terminal"])
def test_regular_approval_assembles_later_unexecuted_tools(later_kind: str) -> None:
    later_ran = False

    @dataclass
    class RegularTool(Tool):
        name = "regular"
        description = "Host control flow."
        requires_approval = True

        async def __call__(self) -> str:
            return "should not run"

    @dataclass
    class StreamingLaterTool(Tool):
        name = "streaming"
        description = "Must not run after regular control flow."
        stream_output = True

        async def __call__(self):
            nonlocal later_ran
            later_ran = True
            yield "should not run"

    @dataclass
    class TerminalLaterTool(Tool):
        name = "terminal"
        description = "Must not run after regular control flow."
        terminal = True

        async def __call__(self) -> str:
            nonlocal later_ran
            later_ran = True
            return "should not run"

    later_tool = StreamingLaterTool() if later_kind == "streaming" else TerminalLaterTool()
    events, messages, error = asyncio.run(
        _collect_until_exception(
            [RegularTool(), later_tool],
            [
                ToolUsePart(id="call-regular", name="regular", inputs={}),
                ToolUsePart(id=f"call-{later_kind}", name=later_kind, inputs={}),
            ],
            ApprovalRequired,
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    parts = _tool_result_parts(messages)

    assert isinstance(error, ApprovalRequired)
    assert later_ran is False
    assert [result.tool_use_id for result in results] == [
        "call-regular",
        f"call-{later_kind}",
    ]
    assert [part.tool_use_id for part in parts] == [
        "call-regular",
        f"call-{later_kind}",
    ]
    assert [result.result for result in results] == [
        {"approval_required": True},
        {"approval_required": True},
    ]
    assert all(result.is_error is True for result in results)
    assert all(result.error_details is None for result in results)
    assert all("[tool_execution_error]" not in part.parts[0].text for part in parts)
    assert all("[tool_retry]" not in part.parts[0].text for part in parts)


def test_streaming_approval_is_unclassified() -> None:
    @dataclass
    class StreamingTool(Tool):
        name = "streaming"
        description = "Host control flow."
        requires_approval = True
        stream_output = True

        async def __call__(self):
            yield ToolStreamEvent(type="progress", data="started")

    events, messages, error = asyncio.run(
        _collect_until_exception(
            [StreamingTool()],
            [ToolUsePart(id="call-streaming", name="streaming", inputs={})],
            ApprovalRequired,
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    parts = _tool_result_parts(messages)

    assert isinstance(error, ApprovalRequired)
    assert len(results) == 1
    assert results[0].result == {"approval_required": True}
    assert results[0].error_details is None
    assert "[tool_execution_error]" not in parts[0].parts[0].text
    assert "[tool_retry]" not in parts[0].parts[0].text
