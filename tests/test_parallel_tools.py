"""Tests for parallel tool execution in AgentExecutor."""

import asyncio
import time
from dataclasses import dataclass
from typing import Annotated
from unittest.mock import AsyncMock

from dobby import AgentExecutor
from dobby.tools import Tool
from dobby.types import (
    StreamEndEvent,
    ToolResultEvent,
    ToolUsePart,
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
