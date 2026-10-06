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
from dobby.providers.base import ProviderError
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


def _compact_call(call_id: str = "compact-1", *, keep_last_n: Any = 1) -> ToolUsePart:
    inputs: dict[str, Any] = {"instructions": "keep ids"}
    if keep_last_n is not None:
        inputs["keep_last_n"] = keep_last_n
    return ToolUsePart(id=call_id, name="compact_context", inputs=inputs)


class _Scripted:
    name = "scripted"

    def __init__(
        self,
        turns: list[tuple[list[Any], Usage | None]],
        *,
        summary_text: str = "digest-kept",
        summary_error: Exception | None = None,
    ) -> None:
        self.turns = turns
        self.summary_text = summary_text
        self.summary_error = summary_error
        self.agent_calls: list[list[Any]] = []
        self.summarize_calls: list[list[Any]] = []
        self.summarize_prompts: list[Any] = []
        self._index = 0

    async def chat(self, messages: list[Any], **kwargs: Any) -> Any:
        if kwargs.get("stream") is False:
            self.summarize_calls.append(list(messages))
            self.summarize_prompts.append(kwargs.get("system_prompt"))
            if self.summary_error is not None:
                raise self.summary_error
            return StreamEndEvent(
                model="summarizer",
                parts=[TextPart(text=self.summary_text)],
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
    summary_text: str = "digest-kept",
    summary_error: Exception | None = None,
    **run_kwargs: Any,
) -> tuple[list[Any], _Scripted]:
    provider = _Scripted(turns, summary_text=summary_text, summary_error=summary_error)
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
    assert any(
        "<summary>digest-kept</summary>" in str(message) for message in provider.agent_calls[1]
    )
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
    assert _results(events)[0].result == {
        "status": "context_unchanged",
        "reason": "no_policy",
    }
    assert _results(events)[0].is_error is False


def test_compact_tool_reports_nothing_to_compact_when_keep_window_covers_history() -> None:
    events, provider = _drive(
        [([_compact_call(keep_last_n=5)], _usage(0)), ([], _usage(0))],
        _history(pairs=3),
        [CompactContextTool()],
        policy=_policy(keep_last_n=5),
    )
    assert provider.summarize_calls == []
    assert _edits(events) == []
    assert _results(events)[0].is_error is False
    assert _results(events)[0].result == {
        "status": "context_unchanged",
        "reason": "nothing_to_compact",
    }


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
    compact_results = [result for result in _results(events) if result.name == "compact_context"]
    assert compact_results[0].result == {
        "status": "context_unchanged",
        "reason": "already_compacted",
    }


def test_negative_keep_last_n_is_a_model_correction() -> None:
    """``keep_last_n=-1`` fails input validation and never reaches ``ContextPolicy``."""
    tool = CompactContextTool()
    events, provider = _drive(
        [
            (
                [
                    ToolUsePart(
                        id="bad",
                        name="compact_context",
                        inputs={"instructions": "keep ids", "keep_last_n": -1},
                    )
                ],
                _usage(0),
            ),
            ([], _usage(0)),
        ],
        _history(),
        [tool],
        policy=_policy(),
    )
    assert provider.summarize_calls == []
    assert _edits(events) == []
    assert str(_results(events)[0].result).startswith("[tool_input_invalid]")
    assert _results(events)[0].is_error is True


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


def test_numeric_string_keep_last_n_override_is_validated() -> None:
    """``keep_last_n="2"`` is coerced for the policy and does not raise TypeError."""
    events, provider = _drive(
        [([_compact_call(keep_last_n="2")], _usage(0)), ([], _usage(0))],
        _history(pairs=3),
        [CompactContextTool()],
        policy=_policy(keep_last_n=1),
    )
    assert _results(events)[0].is_error is False
    assert _results(events)[0].result["status"] == "context_compacted"
    assert len(_edits(events)) == 1
    assert len(provider.summarize_calls) == 1
    summarized = str(provider.summarize_calls[0])
    assert "payload-0" in summarized
    assert "payload-1" in summarized
    assert "payload-2" not in summarized
    assert "payload-2" in str(provider.agent_calls[1])


def test_float_keep_last_n_from_custom_tool_falls_back_to_policy() -> None:
    """A custom edits_context tool may accept ``1.5``; the policy keeps its integer default."""

    @dataclass
    class _FloatKeepCompact(Tool):
        name = "compact_context"
        description = "Accept a float keep_last_n."
        edits_context = True

        def __call__(self, instructions: str, keep_last_n: float | None = None) -> dict[str, str]:
            return {"status": "context_compacted", "detail": "ok"}

    events, provider = _drive(
        [([_compact_call(keep_last_n=1.5)], _usage(0)), ([], _usage(0))],
        _history(pairs=3),
        [_FloatKeepCompact()],
        policy=_policy(keep_last_n=2),
    )
    assert _results(events)[0].is_error is False
    assert _results(events)[0].result["status"] == "context_compacted"
    assert len(_edits(events)) == 1
    summarized = str(provider.summarize_calls[0])
    assert "payload-0" in summarized
    assert "payload-1" in summarized
    assert "payload-2" not in summarized
    assert "payload-2" in str(provider.agent_calls[1])


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


def test_non_retryable_edits_context_failure_does_not_summarize() -> None:
    """``is_error`` without ``retry_model`` records the failure and skips summarize."""

    @dataclass
    class _FailingCompact(Tool):
        name = "compact_context"
        description = "Fail without asking the model to retry."
        edits_context = True

        def __call__(self, instructions: str, keep_last_n: int | None = None) -> dict[str, str]:
            raise RuntimeError("compact failed")

    caller = _history()
    snapshot = copy.deepcopy(caller)
    events, provider = _drive(
        [([_compact_call()], _usage(0)), ([], _usage(0))],
        caller,
        [_FailingCompact()],
        policy=_policy(keep_last_n=1),
    )
    results = _results(events)
    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].name == "compact_context"
    assert str(results[0].result).startswith("[tool_execution_error]")
    assert provider.summarize_calls == []
    assert _edits(events) == []
    assert caller == snapshot


def test_blank_compact_digest_watermarks_until_context_grows() -> None:
    """A whitespace digest from compact_context leaves history unchanged and watermarks.

    The same combined basis does not summarize again. Growth does. A ProviderError
    from that summarizer aborts without a ContextEditEvent or write-back.
    """

    @dataclass
    class _HugeTool(Tool):
        name = "huge"
        description = "Return a large blob."

        def __call__(self) -> dict[str, str]:
            return {"blob": "H" * ((128_000 + 10) * 4)}

    policy = _policy(keep_last_n=1, mode="summarize")
    trigger = policy.trigger_tokens
    caller = _history()
    snapshot = copy.deepcopy(caller)
    events, provider = _drive(
        [
            ([_compact_call()], _usage(trigger)),
            ([ToolUsePart(id="e1", name="echo", inputs={})], _usage(trigger)),
            ([ToolUsePart(id="h1", name="huge", inputs={})], _usage(trigger)),
            ([], _usage(trigger)),
        ],
        caller,
        [CompactContextTool(), _EchoTool(), _HugeTool()],
        policy=policy,
        summary_text="  \n\t ",
    )
    # Compact blanks and watermarks. Later turns at that basis do not summarize.
    # The huge result raises the basis, so summarize runs once more. That call's
    # span keeps the newest pair, so the blob itself is not what the summarizer sees.
    assert len(provider.summarize_calls) == 2
    assert "keep ids" in provider.summarize_prompts[0]
    assert "payload-0" in str(provider.summarize_calls[0])
    assert "tool call echo" not in str(provider.summarize_calls[0])
    assert "Additional instructions" not in (provider.summarize_prompts[1] or "")
    assert "tool call echo" in str(provider.summarize_calls[1])
    assert "tool call huge" not in str(provider.summarize_calls[1])
    assert _edits(events) == []
    compact_results = [result for result in _results(events) if result.name == "compact_context"]
    assert len(compact_results) == 1
    assert compact_results[0].is_error is False
    assert compact_results[0].result == {
        "status": "context_unchanged",
        "reason": "empty_summary",
    }
    assert caller == snapshot
    assert all("<summary>" not in str(message) for message in provider.agent_calls)
    assert "payload-0" in str(provider.agent_calls[-1])

    error_caller = _history()
    error_snapshot = copy.deepcopy(error_caller)
    error_provider = _Scripted(
        [([_compact_call()], _usage(trigger)), ([], _usage(trigger))],
        summary_error=ProviderError("summarizer down"),
    )
    executor = AgentExecutor(
        "openai",
        error_provider,  # type: ignore[arg-type]
        tools=[CompactContextTool()],
        context_policy=policy,
    )
    error_events: list[Any] = []

    async def run() -> None:
        async for event in executor.run_stream(error_caller, max_iterations=2):
            error_events.append(event)

    with pytest.raises(ProviderError, match="summarizer down"):
        asyncio.run(run())

    assert len(error_provider.summarize_calls) == 1
    assert "keep ids" in error_provider.summarize_prompts[0]
    assert len(error_provider.agent_calls) == 1
    assert _edits(error_events) == []
    assert _results(error_events)[0].name == "compact_context"
    assert _results(error_events)[0].is_error is True
    assert str(_results(error_events)[0].result).startswith("[tool_execution_error]")
    assert error_caller == error_snapshot


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
    compact_at = next(
        i for i, event in enumerate(events) if getattr(event, "name", None) == "compact_context"
    )
    assert stream_at < compact_at
