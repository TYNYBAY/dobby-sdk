"""Tests for parallel tool execution in AgentExecutor."""

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Annotated
from unittest.mock import AsyncMock, patch

import pytest

from dobby import AgentExecutor
from dobby.exceptions import ApprovalRequired
from dobby.tools import Tool
from dobby.types import (
    AssistantMessagePart,
    StreamEndEvent,
    ToolResultEvent,
    ToolResultPart,
    ToolStreamEvent,
    ToolUsePart,
    Usage,
    UserMessagePart,
)

# --- Helpers ---


def _make_mock_provider(tool_calls: list[ToolUsePart]):
    """Create a mock provider that returns the given tool calls once, then stops."""
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        async def stream():
            if call_count == 1:
                yield StreamEndEvent(
                    type="stream_end",
                    model="mock",
                    parts=tool_calls,
                    stop_reason="tool_use",
                    usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                )
            else:
                yield StreamEndEvent(
                    type="stream_end",
                    model="mock",
                    parts=[],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    return provider


async def _collect_results(executor, **kwargs):
    """Run executor.run_stream and collect ToolResultEvents."""
    results = []
    async for event in executor.run_stream(messages=[], system_prompt=None, **kwargs):
        if isinstance(event, ToolResultEvent):
            results.append(event)
    return results


async def _collect_results_and_emitted_messages(executor):
    """Collect tool results and the messages built while emitting them."""
    with patch.object(
        executor, "_emit_tool_result", wraps=executor._emit_tool_result
    ) as emit_result:
        results = await _collect_results(executor)

    assert emit_result.call_args is not None
    return results, emit_result.call_args.args[3]


async def _collect_batch_until_control_flow(executor, exception_type):
    """Collect an assembled batch and the host control-flow exception."""
    results = []
    caught = None
    with patch.object(
        executor, "_emit_tool_result", wraps=executor._emit_tool_result
    ) as emit_result:
        try:
            async for event in executor.run_stream(messages=[], system_prompt=None):
                if isinstance(event, ToolResultEvent):
                    results.append(event)
        except exception_type as exception:
            caught = exception

    assert caught is not None
    assert emit_result.call_args is not None
    return results, emit_result.call_args.args[3], caught


def _tool_result_parts(messages):
    """Return model-facing tool results in conversation order."""
    return [
        part
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]


def _tool_use_ids(messages):
    """Return assembled assistant tool-call IDs in conversation order."""
    return [
        part.id
        for message in messages
        if isinstance(message, AssistantMessagePart)
        for part in message.parts
        if isinstance(part, ToolUsePart)
    ]


def _assert_unsuccessful_control_flow(event, part, *, approval: bool) -> None:
    """Control-flow entries must be unsuccessful and unclassified."""
    expected = {"approval_required": True} if approval else {"cancelled": True}
    assert event.is_error is True
    assert event.error_details is None
    assert event.result == expected
    assert part.is_error is True
    assert part.tool_use_id == event.tool_use_id
    assert str(expected) in part.parts[0].text
    assert "[tool_execution_error]" not in part.parts[0].text
    assert "[tool_failure]" not in part.parts[0].text
    assert "[tool_retry]" not in part.parts[0].text


# --- Tool definitions ---


@dataclass
class SlowToolA(Tool):
    name = "slow_tool_a"
    description = "A tool that sleeps for 1 second."

    async def __call__(self, label: Annotated[str, "Label"]) -> dict[str, str]:
        await asyncio.sleep(1)
        return {"label": label, "done": "true"}


@dataclass
class SlowToolB(Tool):
    name = "slow_tool_b"
    description = "A tool that sleeps for 1 second."

    async def __call__(self, label: Annotated[str, "Label"]) -> dict[str, str]:
        await asyncio.sleep(1)
        return {"label": label, "done": "true"}


@dataclass
class SlowToolC(Tool):
    name = "slow_tool_c"
    description = "A tool that sleeps for 1 second."

    async def __call__(self, label: Annotated[str, "Label"]) -> dict[str, str]:
        await asyncio.sleep(1)
        return {"label": label, "done": "true"}


@dataclass
class SyncTool(Tool):
    name = "sync_tool"
    description = "A synchronous tool."

    def __call__(self, value: Annotated[str, "A value"]) -> dict[str, str]:
        return {"value": value}


# --- Tests ---


class TestToolSequentialAttribute:
    """Test cases for Tool.sequential class attribute."""

    def test_sequential_default_false(self) -> None:
        assert SlowToolA.sequential is False

    def test_sequential_true(self) -> None:
        @dataclass
        class SeqTool(Tool):
            name = "seq"
            description = "Sequential"
            sequential = True

            def __call__(self) -> dict:
                return {}

        assert SeqTool.sequential is True

    def test_sequential_on_instance(self) -> None:
        @dataclass
        class SeqTool(Tool):
            name = "seq"
            description = "Sequential"
            sequential = True

            def __call__(self) -> dict:
                return {}

        assert SeqTool().sequential is True


class TestParallelExecution:
    """Test that multiple non-streaming tools run concurrently."""

    def test_parallel_is_faster_than_sequential(self) -> None:
        """3 tools sleeping 1s each should complete in ~1s when parallel."""

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="slow_tool_a", inputs={"label": "a"}),
                ToolUsePart(id="tc2", name="slow_tool_b", inputs={"label": "b"}),
                ToolUsePart(id="tc3", name="slow_tool_c", inputs={"label": "c"}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[SlowToolA(), SlowToolB(), SlowToolC()],
            )
            start = time.monotonic()
            results = await _collect_results(executor)
            elapsed = time.monotonic() - start
            return results, elapsed

        results, elapsed = asyncio.run(run())
        assert len(results) == 3
        assert elapsed < 2.0, f"Took {elapsed:.2f}s, expected <2s for parallel execution"

    def test_results_returned_for_all_tools(self) -> None:
        """All tool results are yielded with correct data."""

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="slow_tool_a", inputs={"label": "first"}),
                ToolUsePart(id="tc2", name="slow_tool_b", inputs={"label": "second"}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[SlowToolA(), SlowToolB()],
            )
            return await _collect_results(executor)

        results = asyncio.run(run())
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"slow_tool_a", "slow_tool_b"}
        for r in results:
            assert r.is_error is False

    def test_single_tool_call_no_gather(self) -> None:
        """A single tool call still works (sequential path, no unnecessary gather)."""

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="sync_tool", inputs={"value": "hello"}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[SyncTool()],
            )
            return await _collect_results(executor)

        results = asyncio.run(run())
        assert len(results) == 1
        assert results[0].result == {"value": "hello"}

    def test_successful_regular_tool_has_no_error_details(self) -> None:
        """Successful tool results leave error_details unset."""

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="sync_tool", inputs={"value": "hello"}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[SyncTool()],
            )
            return await _collect_results(executor)

        results = asyncio.run(run())
        assert len(results) == 1
        assert results[0].is_error is False
        assert results[0].error_details is None


class TestSequentialFallback:
    """Test that sequential=True forces sequential execution."""

    def test_sequential_flag_forces_sequential(self) -> None:
        """When any tool has sequential=True, all run sequentially (~3s)."""

        @dataclass
        class SeqA(Tool):
            name = "seq_a"
            description = "Sequential A"
            sequential = True

            async def __call__(self, label: Annotated[str, "L"]) -> dict:
                await asyncio.sleep(1)
                return {"label": label}

        @dataclass
        class SeqB(Tool):
            name = "seq_b"
            description = "Sequential B"
            sequential = True

            async def __call__(self, label: Annotated[str, "L"]) -> dict:
                await asyncio.sleep(1)
                return {"label": label}

        @dataclass
        class SeqC(Tool):
            name = "seq_c"
            description = "Sequential C"
            sequential = True

            async def __call__(self, label: Annotated[str, "L"]) -> dict:
                await asyncio.sleep(1)
                return {"label": label}

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="seq_a", inputs={"label": "a"}),
                ToolUsePart(id="tc2", name="seq_b", inputs={"label": "b"}),
                ToolUsePart(id="tc3", name="seq_c", inputs={"label": "c"}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[SeqA(), SeqB(), SeqC()],
            )
            start = time.monotonic()
            results = await _collect_results(executor)
            elapsed = time.monotonic() - start
            return results, elapsed

        results, elapsed = asyncio.run(run())
        assert len(results) == 3
        assert elapsed >= 2.5, f"Took {elapsed:.2f}s, expected >=2.5s for sequential execution"


class TestToolErrorHandling:
    """Test that errors in parallel tools are handled correctly."""

    def test_regular_tool_error_keeps_host_diagnostics_and_exception_message(self) -> None:
        """Host traceback stays on error_details; the model gets the exception text."""

        @dataclass
        class FailingTool(Tool):
            name = "failing_tool"
            description = "Always fails"

            async def __call__(self) -> dict:
                raise ValueError("diagnostic failure")

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="failing_tool", inputs={}),
            ]
            provider = _make_mock_provider(tool_calls)
            received_messages = []
            original_chat = provider.chat

            async def recording_chat(messages, *args, **kwargs):
                received_messages.append(list(messages))
                return await original_chat(messages, *args, **kwargs)

            provider.chat = recording_chat
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[FailingTool()],
            )
            results = await _collect_results(executor)
            return results[0], received_messages[-1]

        result, messages = asyncio.run(run())

        assert result.is_error is True
        assert result.error_details is not None
        assert result.error_details.exception_type == "ValueError"
        assert result.error_details.exception_module == "builtins"
        assert result.error_details.message == "diagnostic failure"
        assert "ValueError" in result.error_details.traceback
        assert "test_parallel_tools.py" in result.error_details.traceback
        assert "in __call__" in result.error_details.traceback

        tool_result_parts = [
            part
            for message in messages
            if isinstance(message, UserMessagePart)
            for part in message.parts
            if isinstance(part, ToolResultPart)
        ]
        assert len(tool_result_parts) == 1
        model_text = tool_result_parts[0].parts[0].text
        assert model_text == "[tool_execution_error] diagnostic failure"
        assert result.error_details.traceback not in model_text
        assert "Traceback (most recent call last)" not in model_text

    def test_streaming_tool_error_keeps_host_diagnostics_and_exception_message(self) -> None:
        """Streaming exceptions retain host diagnostics and emit the exception text."""

        @dataclass
        class FailingStreamingTool(Tool):
            name = "failing_streaming_tool"
            description = "Streams once and then fails"
            stream_output = True

            async def __call__(self):
                yield ToolStreamEvent(type="progress", data="started")
                raise ValueError("streaming diagnostic failure")

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="failing_streaming_tool", inputs={}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[FailingStreamingTool()],
            )
            return await _collect_results_and_emitted_messages(executor)

        results, messages = asyncio.run(run())
        result = results[0]

        assert result.is_error is True
        assert result.error_details is not None
        assert result.error_details.exception_type == "ValueError"
        assert result.error_details.exception_module == "builtins"
        assert result.error_details.message == "streaming diagnostic failure"
        assert "ValueError: streaming diagnostic failure" in result.error_details.traceback
        assert "test_parallel_tools.py" in result.error_details.traceback

        tool_result_parts = [
            part
            for message in messages
            if isinstance(message, UserMessagePart)
            for part in message.parts
            if isinstance(part, ToolResultPart)
        ]
        assert len(tool_result_parts) == 1
        model_text = tool_result_parts[0].parts[0].text
        assert model_text == "[tool_execution_error] streaming diagnostic failure"
        assert result.error_details.traceback not in model_text
        assert "Traceback (most recent call last)" not in model_text

    def test_terminal_tool_error_keeps_host_diagnostics_and_exception_message(self) -> None:
        """Terminal exceptions retain host diagnostics and emit the exception text."""

        @dataclass
        class FailingTerminalTool(Tool):
            name = "failing_terminal_tool"
            description = "Always fails and terminates"
            terminal = True

            async def __call__(self) -> dict:
                raise RuntimeError("terminal diagnostic failure")

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="failing_terminal_tool", inputs={}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[FailingTerminalTool()],
            )
            return await _collect_results_and_emitted_messages(executor)

        results, messages = asyncio.run(run())
        result = results[0]

        assert result.is_error is True
        assert result.error_details is not None
        assert result.error_details.exception_type == "RuntimeError"
        assert result.error_details.exception_module == "builtins"
        assert result.error_details.message == "terminal diagnostic failure"
        assert "RuntimeError: terminal diagnostic failure" in result.error_details.traceback
        assert "test_parallel_tools.py" in result.error_details.traceback

        tool_result_parts = [
            part
            for message in messages
            if isinstance(message, UserMessagePart)
            for part in message.parts
            if isinstance(part, ToolResultPart)
        ]
        assert len(tool_result_parts) == 1
        model_text = tool_result_parts[0].parts[0].text
        assert model_text == "[tool_execution_error] terminal diagnostic failure"
        assert result.error_details.traceback not in model_text
        assert "Traceback (most recent call last)" not in model_text

    def test_streaming_tool_approval_required_propagates(self) -> None:
        """Streaming tool approval remains control flow rather than a tool error."""

        @dataclass
        class ApprovalStreamingTool(Tool):
            name = "approval_streaming_tool"
            description = "Requires approval before streaming"
            requires_approval = True
            stream_output = True

            async def __call__(self):
                yield ToolStreamEvent(type="progress", data="started")

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="approval_streaming_tool", inputs={}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[ApprovalStreamingTool()],
            )
            return await _collect_batch_until_control_flow(executor, ApprovalRequired)

        results, messages, exception = asyncio.run(run())
        result_parts = _tool_result_parts(messages)

        assert exception.tool_call_id == "tc1"
        assert exception.tool_name == "approval_streaming_tool"
        assert [result.tool_use_id for result in results] == ["tc1"]
        _assert_unsuccessful_control_flow(results[0], result_parts[0], approval=True)

    def test_terminal_tool_approval_required_propagates(self) -> None:
        """Terminal tool approval remains control flow rather than a tool error."""

        @dataclass
        class ApprovalTerminalTool(Tool):
            name = "approval_terminal_tool"
            description = "Requires approval before terminating"
            requires_approval = True
            terminal = True

            async def __call__(self) -> dict:
                return {"status": "done"}

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="approval_terminal_tool", inputs={}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[ApprovalTerminalTool()],
            )
            return await _collect_batch_until_control_flow(executor, ApprovalRequired)

        results, messages, exception = asyncio.run(run())
        result_parts = _tool_result_parts(messages)

        assert exception.tool_call_id == "tc1"
        assert exception.tool_name == "approval_terminal_tool"
        assert [result.tool_use_id for result in results] == ["tc1"]
        _assert_unsuccessful_control_flow(results[0], result_parts[0], approval=True)

    def test_error_in_one_tool_doesnt_break_others(self) -> None:
        """If one parallel tool fails, others still return results."""

        @dataclass
        class FailingTool(Tool):
            name = "failing_tool"
            description = "Always fails"

            async def __call__(self) -> dict:
                raise ValueError("intentional failure")

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="sync_tool", inputs={"value": "ok"}),
                ToolUsePart(id="tc2", name="failing_tool", inputs={}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[SyncTool(), FailingTool()],
            )
            return await _collect_results_and_emitted_messages(executor)

        results, messages = asyncio.run(run())
        result_parts = _tool_result_parts(messages)
        assert len(results) == 2
        assert [result.tool_use_id for result in results] == ["tc1", "tc2"]
        assert _tool_use_ids(messages) == ["tc1", "tc2"]
        assert [part.tool_use_id for part in result_parts] == ["tc1", "tc2"]
        ok_result = next(r for r in results if r.name == "sync_tool")
        err_result = next(r for r in results if r.name == "failing_tool")
        assert ok_result.is_error is False
        assert result_parts[0].is_error is False
        assert err_result.is_error is True
        assert result_parts[1].is_error is True
        assert str(err_result.result) == "[tool_execution_error] intentional failure"

    def test_retrying_tool_does_not_change_sibling_result_order(self) -> None:
        """A retrying parallel tool does not reorder or re-run its sibling."""
        retry_calls = 0
        sibling_calls = 0

        @dataclass
        class RetryingTool(Tool):
            name = "retrying_tool"
            description = "Fail once, then succeed"
            retryable_exceptions = (TimeoutError,)

            async def __call__(self) -> dict:
                nonlocal retry_calls
                retry_calls += 1
                if retry_calls == 1:
                    raise TimeoutError("transient")
                return {"status": "recovered"}

        @dataclass
        class SiblingTool(Tool):
            name = "sibling_tool"
            description = "Succeed once"

            async def __call__(self) -> dict:
                nonlocal sibling_calls
                sibling_calls += 1
                return {"status": "ok"}

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="retrying_tool", inputs={}),
                ToolUsePart(id="tc2", name="sibling_tool", inputs={}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[RetryingTool(), SiblingTool()],
            )
            return await _collect_results_and_emitted_messages(executor)

        with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
            results, messages = asyncio.run(run())

        result_parts = _tool_result_parts(messages)
        assert retry_calls == 2
        assert sibling_calls == 1
        assert sleep.await_count == 1
        assert [result.tool_use_id for result in results] == ["tc1", "tc2"]
        assert _tool_use_ids(messages) == ["tc1", "tc2"]
        assert [part.tool_use_id for part in result_parts] == ["tc1", "tc2"]
        assert results[0].is_error is False
        assert results[1].is_error is False
        assert results[0].result == {"status": "recovered"}
        assert results[1].result == {"status": "ok"}

    def test_regular_tool_error_logs_exception_with_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unexpected tool exceptions are logged with traceback via logger.exception."""

        @dataclass
        class FailingTool(Tool):
            name = "failing_tool"
            description = "Always fails"

            async def __call__(self) -> dict:
                raise ValueError("logged failure")

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="failing_tool", inputs={}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[FailingTool()],
            )
            return await _collect_results(executor)

        with caplog.at_level(logging.ERROR, logger="dobby"):
            asyncio.run(run())

        records = [record for record in caplog.records if record.exc_info]
        assert records
        record = records[0]
        assert record.exc_info is not None
        exc_type, exc, _tb = record.exc_info
        assert exc_type is ValueError
        assert str(exc) == "logged failure"
        assert "ValueError" in caplog.text
        assert "logged failure" in caplog.text
        assert "Traceback (most recent call last)" in caplog.text

    @pytest.mark.parametrize(
        "tool_order",
        [
            ("approval_tool", "sync_tool"),
            ("sync_tool", "approval_tool"),
        ],
        ids=["approval-first", "approval-last"],
    )
    def test_parallel_approval_assembles_sibling_results_before_propagating(
        self, tool_order: tuple[str, str]
    ) -> None:
        """Approval remains host control flow after ordered results are assembled."""
        sibling_ran = False

        @dataclass
        class ApprovalTool(Tool):
            name = "approval_tool"
            description = "Requires approval"
            requires_approval = True

            async def __call__(self) -> dict:
                return {"approved": True}

        @dataclass
        class SiblingTool(Tool):
            name = "sync_tool"
            description = "Completes normally"

            async def __call__(self, value: Annotated[str, "A value"]) -> dict[str, str]:
                nonlocal sibling_ran
                sibling_ran = True
                return {"value": value}

        async def run():
            tool_calls = [
                ToolUsePart(
                    id=f"tc{index}",
                    name=name,
                    inputs={} if name == "approval_tool" else {"value": "ok"},
                )
                for index, name in enumerate(tool_order, start=1)
            ]
            executor = AgentExecutor(
                provider="openai",
                llm=_make_mock_provider(tool_calls),
                tools=[SiblingTool(), ApprovalTool()],
            )
            return await _collect_batch_until_control_flow(executor, ApprovalRequired)

        results, messages, exception = asyncio.run(run())
        result_parts = _tool_result_parts(messages)
        approval_index = tool_order.index("approval_tool")

        assert exception.tool_call_id == f"tc{approval_index + 1}"
        assert sibling_ran is True
        assert [result.tool_use_id for result in results] == ["tc1", "tc2"]
        assert _tool_use_ids(messages) == ["tc1", "tc2"]
        assert [part.tool_use_id for part in result_parts] == ["tc1", "tc2"]
        _assert_unsuccessful_control_flow(
            results[approval_index], result_parts[approval_index], approval=True
        )
        sibling_index = 1 - approval_index
        assert results[sibling_index].is_error is False
        assert results[sibling_index].result == {"value": "ok"}
        assert result_parts[sibling_index].is_error is False

    @pytest.mark.parametrize(
        "tool_order",
        [
            ("cancelling_tool", "completing_tool"),
            ("completing_tool", "cancelling_tool"),
        ],
        ids=["cancellation-first", "cancellation-last"],
    )
    def test_parallel_cancellation_assembles_results_in_call_order(
        self, tool_order: tuple[str, str]
    ) -> None:
        """Cancellation propagates only after every ordered result is assembled."""
        completed = False

        @dataclass
        class CancellingTool(Tool):
            name = "cancelling_tool"
            description = "Cancels during execution"

            async def __call__(self) -> dict:
                raise asyncio.CancelledError

        @dataclass
        class CompletingTool(Tool):
            name = "completing_tool"
            description = "Completes normally"

            async def __call__(self) -> dict:
                nonlocal completed
                await asyncio.sleep(0)
                completed = True
                return {"status": "done"}

        async def run():
            tool_calls = [
                ToolUsePart(id=f"tc{index}", name=name, inputs={})
                for index, name in enumerate(tool_order, start=1)
            ]
            executor = AgentExecutor(
                provider="openai",
                llm=_make_mock_provider(tool_calls),
                tools=[CancellingTool(), CompletingTool()],
            )
            return await _collect_batch_until_control_flow(executor, asyncio.CancelledError)

        results, messages, exception = asyncio.run(run())
        result_parts = _tool_result_parts(messages)
        cancel_index = tool_order.index("cancelling_tool")

        assert isinstance(exception, asyncio.CancelledError)
        assert completed is True
        assert [result.tool_use_id for result in results] == ["tc1", "tc2"]
        assert _tool_use_ids(messages) == ["tc1", "tc2"]
        assert [part.tool_use_id for part in result_parts] == ["tc1", "tc2"]
        _assert_unsuccessful_control_flow(
            results[cancel_index], result_parts[cancel_index], approval=False
        )
        sibling_index = 1 - cancel_index
        assert results[sibling_index].is_error is False
        assert results[sibling_index].result == {"status": "done"}
        assert result_parts[sibling_index].is_error is False

    @pytest.mark.parametrize("control_flow", ["approval", "cancellation"])
    def test_sequential_control_flow_assembles_all_unexecuted_calls(
        self,
        control_flow: str,
    ) -> None:
        """Sequential control flow emits placeholders without invoking later tools."""
        execution_order = []

        @dataclass
        class SuccessfulSequentialTool(Tool):
            name = "successful_sequential_tool"
            description = "Completes before approval"
            sequential = True

            async def __call__(self) -> dict:
                execution_order.append(self.name)
                return {"status": "done"}

        @dataclass
        class ControlFlowSequentialTool(Tool):
            name = "control_flow_sequential_tool"
            description = "Stops execution after an earlier call"
            requires_approval = control_flow == "approval"
            sequential = True

            async def __call__(self) -> dict:
                execution_order.append(self.name)
                if control_flow == "cancellation":
                    raise asyncio.CancelledError
                return {"should_not_run": True}

        @dataclass
        class LaterSequentialTool(Tool):
            name = "later_sequential_tool"
            description = "Must not run after approval"
            sequential = True

            async def __call__(self) -> dict:
                execution_order.append(self.name)
                return {"late": True}

        @dataclass
        class LaterStreamingTool(Tool):
            name = "later_streaming_tool"
            description = "Must not stream after control flow"
            stream_output = True

            async def __call__(self):
                execution_order.append(self.name)
                yield {"late": True}

        @dataclass
        class LaterTerminalTool(Tool):
            name = "later_terminal_tool"
            description = "Must not run after control flow"
            terminal = True

            async def __call__(self) -> dict:
                execution_order.append(self.name)
                return {"late": True}

        async def run():
            tool_calls = [
                ToolUsePart(id="tc1", name="successful_sequential_tool", inputs={}),
                ToolUsePart(id="tc2", name="control_flow_sequential_tool", inputs={}),
                ToolUsePart(id="tc3", name="later_sequential_tool", inputs={}),
                ToolUsePart(id="tc4", name="later_streaming_tool", inputs={}),
                ToolUsePart(id="tc5", name="later_terminal_tool", inputs={}),
            ]
            executor = AgentExecutor(
                provider="openai",
                llm=_make_mock_provider(tool_calls),
                tools=[
                    SuccessfulSequentialTool(),
                    ControlFlowSequentialTool(),
                    LaterSequentialTool(),
                    LaterStreamingTool(),
                    LaterTerminalTool(),
                ],
            )
            expected = ApprovalRequired if control_flow == "approval" else asyncio.CancelledError
            return await _collect_batch_until_control_flow(executor, expected)

        results, messages, exception = asyncio.run(run())
        result_parts = _tool_result_parts(messages)

        if control_flow == "approval":
            assert exception.tool_call_id == "tc2"
        expected_execution_order = ["successful_sequential_tool"]
        if control_flow == "cancellation":
            expected_execution_order.append("control_flow_sequential_tool")
        assert execution_order == expected_execution_order
        assert [result.tool_use_id for result in results] == ["tc1", "tc2", "tc3", "tc4", "tc5"]
        assert _tool_use_ids(messages) == ["tc1", "tc2", "tc3", "tc4", "tc5"]
        assert [part.tool_use_id for part in result_parts] == [
            "tc1",
            "tc2",
            "tc3",
            "tc4",
            "tc5",
        ]
        assert results[0].is_error is False
        assert results[0].result == {"status": "done"}
        assert result_parts[0].is_error is False
        for result, result_part in zip(results[1:], result_parts[1:], strict=True):
            _assert_unsuccessful_control_flow(
                result,
                result_part,
                approval=control_flow == "approval",
            )
