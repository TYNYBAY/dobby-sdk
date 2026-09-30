"""Production ContextEditEvent emission and compact_context routing."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dobby import AgentExecutor
from dobby.context import ContextPolicy
from dobby.exceptions import ApprovalRequired
from dobby.tools import CompactContextTool, Tool
from dobby.types import (
    AssistantMessagePart,
    ContextEditEvent,
    StreamEndEvent,
    TextPart,
    ToolResultEvent,
    ToolResultPart,
    ToolStreamEvent,
    ToolUsePart,
    Usage,
    UserMessagePart,
)


def _usage(input_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _policy(*, keep_last_n: int = 1, mode: str = "summarize") -> ContextPolicy:
    return ContextPolicy(
        context_window=128_000,
        trigger_pct=0.8,
        keep_last_n=keep_last_n,
        mode=mode,  # type: ignore[arg-type]
    )


def _history(pairs: int = 3) -> list[Any]:
    messages: list[Any] = [UserMessagePart(parts=[TextPart(text="question")])]
    for index in range(pairs):
        call_id = f"h{index}"
        messages.append(
            AssistantMessagePart(parts=[ToolUsePart(id=call_id, name="search", inputs={})])
        )
        messages.append(
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id=call_id,
                        name="search",
                        parts=[TextPart(text=f"payload-{index}")],
                    )
                ]
            )
        )
    return messages


def _compact_call(call_id: str = "compact-1", *, keep_last_n: int | None = 1) -> ToolUsePart:
    inputs: dict[str, Any] = {"instructions": "keep ids"}
    if keep_last_n is not None:
        inputs["keep_last_n"] = keep_last_n
    return ToolUsePart(id=call_id, name="compact_context", inputs=inputs)


class _Scripted:
    def __init__(self, turns: list[tuple[list[Any], Usage | None]]) -> None:
        self.turns = turns
        self.agent_calls: list[list[Any]] = []
        self.summarize_calls: list[list[Any]] = []
        self._index = 0

    async def chat(self, messages: list[Any], **kwargs: Any) -> Any:
        if kwargs.get("stream") is False:
            self.summarize_calls.append(list(messages))
            return StreamEndEvent(
                model="summarizer",
                parts=[TextPart(text="digest-kept")],
                stop_reason="end_turn",
                usage=_usage(0),
            )
        self.agent_calls.append(list(messages))
        index = min(self._index, len(self.turns) - 1)
        self._index += 1
        parts, usage = self.turns[index]

        async def stream() -> Any:
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=usage,
            )

        return stream()


def _edits(events: list[Any]) -> list[ContextEditEvent]:
    return [event for event in events if isinstance(event, ContextEditEvent)]


def _results(events: list[Any]) -> list[ToolResultEvent]:
    return [event for event in events if isinstance(event, ToolResultEvent)]


def _drive(
    turns: list[tuple[list[Any], Usage | None]],
    messages: list[Any],
    tools: list[Tool],
    *,
    policy: ContextPolicy | None,
    **run_kwargs: Any,
) -> tuple[list[Any], _Scripted]:
    provider = _Scripted(turns)
    executor = AgentExecutor(
        "openai",
        provider,  # type: ignore[arg-type]
        tools=tools,
        context_policy=policy,
    )

    async def run() -> list[Any]:
        collected: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=len(turns), **run_kwargs):
            collected.append(event)
        return collected

    return asyncio.run(run()), provider


@dataclass
class _CountingCompact(CompactContextTool):
    def __call__(  # type: ignore[override]
        self,
        instructions: str,
        keep_last_n: int | None = None,
    ) -> dict[str, str]:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        return super().__call__(instructions, keep_last_n)


@dataclass
class _EchoTool(Tool):
    name = "echo"
    description = "Echo."

    def __call__(self) -> str:
        return "echo"


@dataclass
class _TypedTool(Tool):
    name = "typed"
    description = "Accept an integer."

    async def __call__(self, count: int) -> int:
        return count


def test_compact_tool_emits_summary_event_and_preserves_caller() -> None:
    caller = _history()
    snapshot = copy.deepcopy(caller)
    tool = _CountingCompact()
    events, provider = _drive(
        [([_compact_call()], _usage(0)), ([], _usage(0))],
        caller,
        [tool],
        policy=_policy(keep_last_n=1),
    )
    edits = _edits(events)
    assert tool._calls == 1  # type: ignore[attr-defined]
    assert len(provider.summarize_calls) == 1
    assert len(edits) == 1
    assert edits[0].type == "context_edit"
    assert edits[0].applied_edits[0].type == "summarize"
    assert edits[0].applied_edits[0].summary_text == "digest-kept"
    assert _results(events)[0].result["status"] == "context_compacted"
    assert any("<summary>digest-kept</summary>" in str(message) for message in provider.agent_calls[1])
    assert caller == snapshot


def test_compact_tool_without_policy_returns_result_only() -> None:
    tool = _CountingCompact()
    events, provider = _drive(
        [([_compact_call()], _usage(0)), ([], _usage(0))],
        _history(),
        [tool],
        policy=None,
    )
    assert tool._calls == 1  # type: ignore[attr-defined]
    assert provider.summarize_calls == []
    assert _edits(events) == []
    assert _results(events)[0].result["status"] == "context_compacted"
    assert _results(events)[0].is_error is False


def test_automatic_trim_and_compact_tool_do_not_both_compact() -> None:
    """An automatic trim latches the turn, so compact_context does not summarize."""
    policy = _policy(keep_last_n=1, mode="trim")
    trigger = policy.trigger_tokens
    events, provider = _drive(
        [
            ([ToolUsePart(id="n1", name="echo", inputs={})], _usage(trigger)),
            ([_compact_call()], _usage(trigger)),
            ([], _usage(trigger)),
        ],
        _history(pairs=2),
        [_EchoTool(), _CountingCompact()],
        policy=policy,
    )
    edits = _edits(events)
    assert provider.summarize_calls == []
    assert edits
    assert all(edit.applied_edits[0].type == "clear_tool_uses" for edit in edits)
    assert any(result.name == "compact_context" for result in _results(events))


def test_invalid_compact_call_is_a_model_correction() -> None:
    tool = _CountingCompact()
    events, provider = _drive(
        [
            (
                [ToolUsePart(id="bad", name="compact_context", inputs={"keep_last_n": "nope"})],
                _usage(0),
            ),
            ([], _usage(0)),
        ],
        _history(),
        [tool],
        policy=_policy(),
    )
    assert getattr(tool, "_calls", 0) == 0
    assert provider.summarize_calls == []
    assert _edits(events) == []
    assert str(_results(events)[0].result).startswith("[tool_input_invalid]")
    assert _results(events)[0].is_error is True


def test_parallel_correction_skips_compact_tool() -> None:
    tool = _CountingCompact()
    events, _provider = _drive(
        [
            (
                [
                    ToolUsePart(id="bad", name="typed", inputs={"count": "bad"}),
                    _compact_call(),
                ],
                _usage(0),
            ),
            ([], _usage(0)),
        ],
        [],
        [_TypedTool(), tool],
        policy=_policy(),
    )
    results = _results(events)
    assert getattr(tool, "_calls", 0) == 0
    assert str(results[0].result).startswith("[tool_input_invalid]")
    assert results[1].name == "compact_context"
    assert results[1].result == {"skipped": True, "reason": "model_correction"}
    assert _edits(events) == []


def test_host_retry_runs_before_a_single_summarize() -> None:
    calls = 0

    @dataclass
    class _FlakyCompact(Tool):
        name = "compact_context"
        description = "Fail once, then compact."
        edits_context = True
        retryable_exceptions = (TimeoutError,)

        def __call__(self, instructions: str, keep_last_n: int | None = None) -> dict[str, str]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("transient")
            return {"status": "context_compacted", "detail": "ok"}

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events, provider = _drive(
            [([_compact_call()], _usage(0)), ([], _usage(0))],
            _history(),
            [_FlakyCompact()],
            policy=_policy(keep_last_n=1),
        )
    assert calls == 2
    assert len(_results(events)) == 1
    assert _results(events)[0].is_error is False
    assert len(provider.summarize_calls) == 1
    assert len(_edits(events)) == 1
    assert _edits(events)[0].applied_edits[0].type == "summarize"


def test_approval_raises_before_compact_and_skips_terminal() -> None:
    terminal_calls = 0

    @dataclass
    class _ApprovalCompact(CompactContextTool):
        requires_approval = True

    @dataclass
    class _Finish(Tool):
        name = "finish"
        description = "Stop."
        terminal = True

        def __call__(self) -> str:
            nonlocal terminal_calls
            terminal_calls += 1
            return "done"

    provider = _Scripted(
        [
            (
                [_compact_call(), ToolUsePart(id="end", name="finish", inputs={})],
                _usage(0),
            )
        ]
    )
    executor = AgentExecutor(
        "openai",
        provider,  # type: ignore[arg-type]
        tools=[_ApprovalCompact(), _Finish()],
        context_policy=_policy(),
    )

    async def run() -> None:
        async for _event in executor.run_stream([]):
            pass

    with pytest.raises(ApprovalRequired) as caught:
        asyncio.run(run())
    assert caught.value.tool_name == "compact_context"
    assert terminal_calls == 0
    assert provider.summarize_calls == []


def test_host_cancellation_skips_terminal_after_compact_tool() -> None:
    started = asyncio.Event()
    terminal_calls = 0

    @dataclass
    class _BlockingCompact(Tool):
        name = "compact_context"
        description = "Block until cancelled."
        edits_context = True

        async def __call__(
            self,
            instructions: str,
            keep_last_n: int | None = None,
        ) -> dict[str, str]:
            started.set()
            await asyncio.Event().wait()
            return {"status": "context_compacted"}

    @dataclass
    class _Finish(Tool):
        name = "finish"
        description = "Stop."
        terminal = True

        def __call__(self) -> str:
            nonlocal terminal_calls
            terminal_calls += 1
            return "done"

    provider = _Scripted(
        [
            (
                [_compact_call(), ToolUsePart(id="end", name="finish", inputs={})],
                _usage(0),
            )
        ]
    )
    executor = AgentExecutor(
        "openai",
        provider,  # type: ignore[arg-type]
        tools=[_BlockingCompact(), _Finish()],
        context_policy=_policy(),
    )

    async def run() -> None:
        async def consume() -> None:
            async for _event in executor.run_stream([]):
                pass

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert terminal_calls == 0


def test_tool_raised_cancellation_is_a_structured_error() -> None:
    @dataclass
    class _CancelCompact(Tool):
        name = "compact_context"
        description = "Cancel from the body."
        edits_context = True

        def __call__(self, instructions: str, keep_last_n: int | None = None) -> dict[str, str]:
            raise asyncio.CancelledError

    events, provider = _drive(
        [([_compact_call()], _usage(0)), ([], _usage(0))],
        [],
        [_CancelCompact()],
        policy=None,
    )
    result = _results(events)[0]
    assert result.is_error is True
    assert str(result.result) == "[tool_execution_error] The tool failed unexpectedly."
    assert result.error_details is not None
    assert result.error_details.exception_type == "CancelledError"
    assert provider.summarize_calls == []
    assert _edits(events) == []


def test_compact_tool_runs_after_parallel_and_streaming_before_terminal() -> None:
    order: list[str] = []

    @dataclass
    class _Parallel(Tool):
        name = "echo"
        description = "Parallel."

        def __call__(self) -> str:
            order.append("parallel")
            return "echo"

    @dataclass
    class _Streamer(Tool):
        name = "streamer"
        description = "Stream."
        stream_output = True

        async def __call__(self):  # type: ignore[no-untyped-def]
            order.append("stream")
            yield ToolStreamEvent(type="progress", data="x")
            yield "streamed"

    @dataclass
    class _OrderedCompact(CompactContextTool):
        def __call__(  # type: ignore[override]
            self,
            instructions: str,
            keep_last_n: int | None = None,
        ) -> dict[str, str]:
            order.append("compact")
            return super().__call__(instructions, keep_last_n)

    @dataclass
    class _Finish(Tool):
        name = "finish"
        description = "Stop."
        terminal = True

        def __call__(self) -> str:
            order.append("terminal")
            return "done"

    events, _provider = _drive(
        [
            (
                [
                    ToolUsePart(id="p", name="echo", inputs={}),
                    ToolUsePart(id="s", name="streamer", inputs={}),
                    _compact_call(keep_last_n=5),
                    ToolUsePart(id="end", name="finish", inputs={}),
                ],
                _usage(0),
            )
        ],
        [],
        [_Parallel(), _Streamer(), _OrderedCompact(), _Finish()],
        policy=_policy(keep_last_n=5),
    )
    assert order == ["parallel", "stream", "compact", "terminal"]
    names = [event.name for event in _results(events)]
    assert names == ["echo", "streamer", "compact_context", "finish"]
    assert _results(events)[-1].is_terminal is True
    stream_at = next(i for i, event in enumerate(events) if isinstance(event, ToolStreamEvent))
    compact_at = next(i for i, event in enumerate(events) if getattr(event, "name", None) == "compact_context")
    assert stream_at < compact_at
