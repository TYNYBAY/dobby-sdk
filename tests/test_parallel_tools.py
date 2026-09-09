"""Tests for parallel tool execution in AgentExecutor."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Annotated
from unittest.mock import AsyncMock, patch

import pytest

from dobby import AgentExecutor
from dobby.exceptions import ApprovalRequired
from dobby.tools import Tool
from dobby.types import (
    StreamEndEvent,
    ToolResultEvent,
    ToolResultPart,
    ToolStreamEvent,
    ToolUsePart,
    UserMessagePart,
    Usage,
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
        assert elapsed >= 2.5, (
            f"Took {elapsed:.2f}s, expected >=2.5s for sequential execution"
        )


class TestToolErrorHandling:
    """Test that errors in parallel tools are handled correctly."""

    def test_regular_tool_error_has_diagnostics_but_model_message_does_not(self) -> None:
        """Structured diagnostics stay out of the model-facing tool result."""

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
        assert "diagnostic failure" in model_text
        assert result.error_details.traceback not in model_text
        assert "Traceback (most recent call last)" not in model_text

    def test_streaming_tool_error_has_diagnostics_but_model_message_does_not(self) -> None:
        """Streaming exceptions retain diagnostics outside the model message."""

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
        assert "streaming diagnostic failure" in model_text
        assert result.error_details.traceback not in model_text
        assert "Traceback (most recent call last)" not in model_text

    def test_terminal_tool_error_has_diagnostics_but_model_message_does_not(self) -> None:
        """Terminal exceptions retain diagnostics outside the model message."""

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
        assert "terminal diagnostic failure" in model_text
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
            return await _collect_results(executor)

        with pytest.raises(ApprovalRequired) as exc_info:
            asyncio.run(run())

        assert exc_info.value.tool_call_id == "tc1"
        assert exc_info.value.tool_name == "approval_streaming_tool"

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
            return await _collect_results(executor)

        with pytest.raises(ApprovalRequired) as exc_info:
            asyncio.run(run())

        assert exc_info.value.tool_call_id == "tc1"
        assert exc_info.value.tool_name == "approval_terminal_tool"

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
            return await _collect_results(executor)

        results = asyncio.run(run())
        assert len(results) == 2
        ok_result = next(r for r in results if r.name == "sync_tool")
        err_result = next(r for r in results if r.name == "failing_tool")
        assert ok_result.is_error is False
        assert err_result.is_error is True
        assert "intentional failure" in str(err_result.result)

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

    def test_parallel_cancellation_propagates_without_tool_result(self) -> None:
        """Cancellation remains control flow instead of becoming a tool error."""
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
                ToolUsePart(id="tc1", name="cancelling_tool", inputs={}),
                ToolUsePart(id="tc2", name="completing_tool", inputs={}),
            ]
            provider = _make_mock_provider(tool_calls)
            executor = AgentExecutor(
                provider="openai",
                llm=provider,
                tools=[CancellingTool(), CompletingTool()],
            )

            with patch.object(executor, "_emit_tool_result") as emit_result:
                with pytest.raises(asyncio.CancelledError):
                    await _collect_results(executor)
                return emit_result.call_count

        emit_count = asyncio.run(run())

        assert completed is True
        assert emit_count == 0
