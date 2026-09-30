# ruff: noqa: E402
"""Phase 17: recovery after compaction-related failures (deterministic mocks)."""

# isort: off
from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from dobby import AgentExecutor as CurrentAgentExecutor
from dobby.tools import Tool as CurrentTool
from dobby.types import StreamEndEvent as CurrentStreamEndEvent
from dobby.types import TextPart as CurrentTextPart
from dobby.types import ToolResultEvent
from dobby.types import ToolResultPart as CurrentToolResultPart
from dobby.types import ToolUsePart as CurrentToolUsePart
from dobby.types import Usage as CurrentUsage

from tests.compaction_subject import load_recovered

load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context.edit import _find_tool_pairs
from recovered_dobby.providers.base import APITimeoutError, ProviderError
from recovered_dobby.tools import Tool
from recovered_dobby.types import (
    AssistantMessagePart,
    StreamEndEvent,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart,
)

# isort: on

_WINDOW = 100_000
_TRIGGER = int(0.8 * _WINDOW)
_KEEP = "[Tool result cleared to save context.]"


class _NoopTool(Tool):
    name = "noop"
    description = "Ack."

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


class _CountingNoop(Tool):
    name = "noop"
    description = "Counted noop."

    def __call__(self) -> dict[str, str]:
        self._calls = getattr(self, "_calls", 0) + 1  # type: ignore[attr-defined]
        return {"ok": "yes", "calls": str(self._calls)}


@dataclass
class _FailTool(CurrentTool):
    name = "fail_tool"
    description = "Always fails."

    def __call__(self) -> dict[str, str]:
        self._calls = getattr(self, "_calls", 0) + 1  # type: ignore[attr-defined]
        raise RuntimeError("tool boom")


def _usage(tokens: int) -> Usage:
    return Usage(input_tokens=tokens, output_tokens=0, total_tokens=tokens)


def _history(pairs: int = 4, text: str = "payload") -> list[Any]:
    messages: list[Any] = [UserMessagePart(parts=[TextPart(text="task")])]
    for index in range(pairs):
        call_id = f"h{index}"
        messages.extend(
            [
                AssistantMessagePart(parts=[ToolUsePart(id=call_id, name="search", inputs={"i": index})]),
                UserMessagePart(
                    parts=[
                        ToolResultPart(
                            tool_use_id=call_id,
                            name="search",
                            parts=[TextPart(text=f"{text}-{index}")],
                        )
                    ]
                ),
            ]
        )
    return messages


def _fingerprint(messages: list[Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, TextPart):
                rows.append((message.role, part.text))
            elif isinstance(part, ToolResultPart):
                rows.append(
                    (
                        message.role,
                        "".join(p.text for p in part.parts if isinstance(p, TextPart)),
                    )
                )
    return rows


def _summary_count(messages: list[Any]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, TextPart) and part.text.startswith("<summary>")
    )


def _assert_caller_unchanged(before: list[Any], after: list[Any]) -> None:
    assert _fingerprint(before) == _fingerprint(after)
    assert len(before) == len(after)


def _structure_valid(messages: list[Any]) -> bool:
    pairs = _find_tool_pairs(messages)
    for use_idx, result_idx in pairs:
        use_msg = messages[use_idx]
        result_msg = messages[result_idx]
        if not isinstance(use_msg, AssistantMessagePart) or not isinstance(result_msg, UserMessagePart):
            return False
        uses = [p for p in use_msg.parts if isinstance(p, ToolUsePart)]
        results = [p for p in result_msg.parts if isinstance(p, ToolResultPart)]
        if len(uses) != 1 or not results or uses[0].id != results[0].tool_use_id:
            return False
    return True


def _llm_agent_then_summarize(
    *,
    summarize_exc: BaseException | None = None,
    summarize_end: StreamEndEvent | None = None,
    stream_false_returns_stream: bool = False,
    captured: list[list[Any]],
    summary_attempts: list[int],
    agent_usage: int = _TRIGGER,
) -> AsyncMock:
    agent_calls = 0

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal agent_calls
        if kwargs.get("stream", True) is False:
            summary_attempts.append(len(messages))
            if summarize_exc is not None:
                raise summarize_exc
            if stream_false_returns_stream:

                async def broken_stream() -> AsyncIterator[StreamEndEvent]:
                    raise ProviderError("streaming failure", provider="mock")
                    yield StreamEndEvent(  # pragma: no cover
                        model="mock",
                        parts=[],
                        stop_reason="end_turn",
                        usage=_usage(0),
                    )

                return broken_stream()
            assert summarize_end is not None
            return summarize_end
        agent_calls += 1
        captured.append(list(messages))

        async def stream() -> AsyncIterator[StreamEndEvent]:
            yield StreamEndEvent(
                model="mock",
                parts=[_tool_call(f"agent-{agent_calls}")],
                stop_reason="tool_use",
                usage=_usage(agent_usage),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    return provider


def _tool_call(call_id: str) -> ToolUsePart:
    return ToolUsePart(id=call_id, name="noop", inputs={})


def _scripted_recovered(
    turns: list[tuple[list[Any], Usage | None]],
    captured: list[list[Any]],
) -> AsyncMock:
    call_count = 0

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        captured.append(list(messages))
        idx = min(call_count, len(turns) - 1)
        parts, usage = turns[idx]
        call_count += 1

        async def stream() -> AsyncIterator[StreamEndEvent]:
            yield StreamEndEvent(
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=usage or _usage(_TRIGGER),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    return provider


def _drive_recovered(
    policy: ContextPolicy,
    turns: list[tuple[list[Any], Usage | None]],
    seed: list[Any],
    *,
    tools: list[Tool] | None = None,
) -> tuple[list[Any], list[list[Any]]]:
    captured: list[list[Any]] = []
    executor = AgentExecutor(
        "openai",
        _scripted_recovered(turns, captured),
        tools=tools or [_NoopTool()],
        context_policy=policy,
    )

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(seed, max_iterations=len(turns) + 2):
            events.append(event)
        return events

    return asyncio.run(run()), captured


def test_recovery_after_summarization_failure_then_healthy_turn() -> None:
    """Timeout on summarize leaves caller history intact; next run compacts successfully."""
    messages = _history()
    snapshot = copy.deepcopy(messages)
    captured: list[list[Any]] = []
    summary_attempts: list[int] = []
    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=1, mode="summarize")
    noop = _CountingNoop()
    llm = _llm_agent_then_summarize(
        summarize_exc=APITimeoutError("timeout", provider="mock"),
        captured=captured,
        summary_attempts=summary_attempts,
    )
    executor = AgentExecutor("openai", llm, tools=[noop], context_policy=policy)

    async def fail_run() -> None:
        async for _event in executor.run_stream(messages, max_iterations=2):
            pass

    with pytest.raises(APITimeoutError):
        asyncio.run(fail_run())

    _assert_caller_unchanged(snapshot, messages)
    assert noop._calls == 1  # type: ignore[attr-defined]

    ok_end = StreamEndEvent(
        model="mock",
        parts=[TextPart(text="recovered digest")],
        stop_reason="end_turn",
        usage=_usage(0),
    )
    executor.llm = _llm_agent_then_summarize(
        summarize_end=ok_end,
        captured=captured,
        summary_attempts=summary_attempts,
    )

    async def recover_run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=2):
            events.append(event)
        return events

    events = asyncio.run(recover_run())
    edits = [event for event in events if isinstance(event, ContextEditEvent)]
    assert len(edits) == 1
    assert edits[0].applied_edits[0].summary_text == "recovered digest"
    assert noop._calls >= 2  # type: ignore[attr-defined]
    calls_after_fail = 1
    assert noop._calls > calls_after_fail  # type: ignore[attr-defined]
    _assert_caller_unchanged(snapshot, messages)
    assert any("recovered digest" in text for _, text in _fingerprint(captured[-1]))


def test_recovery_after_tool_failure_then_compaction() -> None:
    """Tool error row is captured once; later trim compaction keeps paired ids."""
    fail = _FailTool()
    results, live = _run_current_batch(
        [CurrentToolUsePart(id="bad", name="fail_tool", inputs={})],
        [fail],
    )
    assert results[0].is_error is True
    assert fail._calls == 1  # type: ignore[attr-defined]

    history = _project_messages(live)
    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=1, mode="trim")
    turns = [
        ([ToolUsePart(id="n1", name="noop", inputs={})], _usage(_TRIGGER)),
        ([], _usage(_TRIGGER)),
    ]
    events, captured = _drive_recovered(policy, turns, history, tools=[_NoopTool()])
    assert fail._calls == 1  # type: ignore[attr-defined]
    assert any(isinstance(event, ContextEditEvent) for event in events)
    assert _structure_valid(captured[-1])
    assert _KEEP in "\n".join(t for _, t in _fingerprint(captured[-1]))


@dataclass
class _AlwaysFailTool(CurrentTool):
    name = "always_fail"
    description = "Fails every time."
    max_retries: ClassVar[int] = 1

    def __call__(self) -> dict[str, str]:
        self._calls = getattr(self, "_calls", 0) + 1  # type: ignore[attr-defined]
        raise RuntimeError("retry me")


def test_recovery_after_retry_exhaustion_then_compaction() -> None:
    """Host retry exhaustion leaves error history; compaction does not re-invoke fail tool."""
    always_fail = _AlwaysFailTool()
    holder: dict[str, list[Any]] = {}

    async def mock_chat(*args: Any, **kwargs: Any) -> Any:
        async def stream() -> AsyncIterator[CurrentStreamEndEvent]:
            yield CurrentStreamEndEvent(
                model="mock",
                parts=[CurrentToolUsePart(id="x1", name="always_fail", inputs={})],
                stop_reason="tool_use",
                usage=CurrentUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    executor = CurrentAgentExecutor("openai", provider, tools=[always_fail])
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def run_fail() -> list[Any]:
        collected: list[Any] = []
        async for event in executor.run_stream([], max_iterations=2):
            collected.append(event)
        return collected

    fail_events = asyncio.run(run_fail())
    assert always_fail._calls == 2  # type: ignore[attr-defined]
    tool_results = [event for event in fail_events if isinstance(event, ToolResultEvent)]
    assert tool_results and tool_results[-1].is_error is True
    history = _project_messages(holder["messages"])

    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=1, mode="trim")
    turns = [
        ([ToolUsePart(id="warm", name="noop", inputs={})], _usage(_TRIGGER - 1)),
        ([ToolUsePart(id="n1", name="noop", inputs={})], _usage(_TRIGGER)),
        ([], _usage(_TRIGGER)),
    ]
    events, captured = _drive_recovered(policy, turns, history, tools=[_NoopTool()])
    assert always_fail._calls == 2  # type: ignore[attr-defined]
    assert any(isinstance(event, ContextEditEvent) for event in events)
    assert _structure_valid(captured[-1])


def _run_current_batch(
    tool_calls: list[CurrentToolUsePart],
    tools: list[CurrentTool],
) -> tuple[list[ToolResultEvent], list[Any]]:
    call_count = 0

    async def mock_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        parts = tool_calls if call_count == 1 else []

        async def stream() -> AsyncIterator[CurrentStreamEndEvent]:
            yield CurrentStreamEndEvent(
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=CurrentUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    executor = CurrentAgentExecutor("openai", provider, tools=tools)
    holder: dict[str, list[Any]] = {}
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def run() -> list[ToolResultEvent]:
        results: list[ToolResultEvent] = []
        async for event in executor.run_stream([], max_iterations=3):
            if isinstance(event, ToolResultEvent):
                results.append(event)
        return results

    results = asyncio.run(run())
    return results, holder.get("messages", [])


def _project_messages(messages: list[Any]) -> list[Any]:
    projected: list[Any] = []
    for message in messages:
        if message.role == "assistant":
            parts = [
                ToolUsePart(id=part.id, name=part.name, inputs=dict(part.inputs))
                for part in message.parts
                if isinstance(part, CurrentToolUsePart)
            ]
            projected.append(AssistantMessagePart(parts=parts))
        else:
            parts = []
            for part in message.parts:
                if isinstance(part, CurrentToolResultPart):
                    parts.append(
                        ToolResultPart(
                            tool_use_id=part.tool_use_id,
                            name=part.name,
                            parts=[
                                TextPart(text=p.text)
                                for p in part.parts
                                if isinstance(p, CurrentTextPart)
                            ],
                            is_error=part.is_error,
                        )
                    )
            projected.append(UserMessagePart(parts=parts))
    return projected


def test_recovery_after_streaming_failure_then_compaction() -> None:
    """Streaming tool errors once; compaction afterward does not re-run the streamer."""
    stream_fail = _StreamFailAfterYield()

    async def mock_chat(*args: Any, **kwargs: Any) -> Any:
        async def stream() -> AsyncIterator[CurrentStreamEndEvent]:
            yield CurrentStreamEndEvent(
                model="mock",
                parts=[
                    CurrentToolUsePart(id="s1", name="stream_fail", inputs={"label": "KEEP-ME"})
                ],
                stop_reason="tool_use",
                usage=CurrentUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    executor = CurrentAgentExecutor("openai", provider, tools=[stream_fail])
    holder: dict[str, list[Any]] = {}
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream([], max_iterations=1):
            events.append(event)
        return events

    events = asyncio.run(run())
    assert stream_fail._calls == 1  # type: ignore[attr-defined]
    tool_results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert tool_results and tool_results[0].is_error is True
    history = _project_messages(holder["messages"])

    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=2, mode="trim")
    turns = [
        ([ToolUsePart(id="n1", name="noop", inputs={})], _usage(_TRIGGER - 1)),
        ([ToolUsePart(id="n2", name="noop", inputs={})], _usage(_TRIGGER)),
        ([], _usage(_TRIGGER)),
    ]
    events2, captured = _drive_recovered(policy, turns, history, tools=[_NoopTool()])
    assert stream_fail._calls == 1  # type: ignore[attr-defined]
    assert any(isinstance(event, ContextEditEvent) for event in events2)
    assert _structure_valid(captured[-1])


@dataclass
class _StreamFailAfterYield(CurrentTool):
    name = "stream_fail"
    description = "Yield then fail."
    stream_output: bool = True

    async def __call__(self, label: str = "") -> Any:
        self._calls = getattr(self, "_calls", 0) + 1  # type: ignore[attr-defined]
        yield {"chunk": label}
        raise RuntimeError("stream failed")


def test_recovery_after_malformed_summarize_stream_then_healthy_turn() -> None:
    """Misconfigured summarizer stream raises; caller history intact; later summarize succeeds."""
    messages = _history()
    snapshot = copy.deepcopy(messages)
    captured: list[list[Any]] = []
    summary_attempts: list[int] = []
    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=1, mode="summarize")
    llm = _llm_agent_then_summarize(
        stream_false_returns_stream=True,
        captured=captured,
        summary_attempts=summary_attempts,
        summarize_end=StreamEndEvent(
            model="mock",
            parts=[TextPart(text="unused")],
            stop_reason="end_turn",
            usage=_usage(0),
        ),
    )
    executor = AgentExecutor("openai", llm, tools=[_NoopTool()], context_policy=policy)

    async def fail_run() -> None:
        async for _event in executor.run_stream(messages, max_iterations=2):
            pass

    with pytest.raises((ProviderError, AttributeError, TypeError)):
        asyncio.run(fail_run())

    _assert_caller_unchanged(snapshot, messages)
    assert _summary_count(messages) == 0

    executor.llm = _llm_agent_then_summarize(
        summarize_end=StreamEndEvent(
            model="mock",
            parts=[TextPart(text="post-malformed digest")],
            stop_reason="end_turn",
            usage=_usage(0),
        ),
        captured=captured,
        summary_attempts=summary_attempts,
    )
    events = asyncio.run(_collect_events(executor, messages, max_iterations=2))
    assert len([e for e in events if isinstance(e, ContextEditEvent)]) == 1
    _assert_caller_unchanged(snapshot, messages)


def test_recovery_after_empty_summary_then_healthy_turn() -> None:
    """Empty digest in one run is a known defect within that run; caller list can compact on retry."""
    messages = _history()
    snapshot = copy.deepcopy(messages)
    captured: list[list[Any]] = []
    summary_attempts: list[int] = []
    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=1, mode="summarize")
    empty_end = StreamEndEvent(
        model="mock",
        parts=[TextPart(text="   ")],
        stop_reason="end_turn",
        usage=_usage(0),
    )
    llm = _llm_agent_then_summarize(
        summarize_end=empty_end,
        captured=captured,
        summary_attempts=summary_attempts,
    )
    noop = _CountingNoop()
    executor = AgentExecutor("openai", llm, tools=[noop], context_policy=policy)

    events1 = asyncio.run(_collect_events(executor, messages, max_iterations=2))
    edits1 = [event for event in events1 if isinstance(event, ContextEditEvent)]
    _assert_caller_unchanged(snapshot, messages)
    calls_after_empty = noop._calls  # type: ignore[attr-defined]
    assert calls_after_empty >= 1

    if edits1:
        assert edits1[0].applied_edits[0].summary_text.strip() == ""

    executor.llm = _llm_agent_then_summarize(
        summarize_end=StreamEndEvent(
            model="mock",
            parts=[TextPart(text="healthy-after-empty")],
            stop_reason="end_turn",
            usage=_usage(0),
        ),
        captured=captured,
        summary_attempts=summary_attempts,
    )
    events2 = asyncio.run(_collect_events(executor, messages, max_iterations=2))
    edits2 = [event for event in events2 if isinstance(event, ContextEditEvent)]
    assert len(edits2) == 1
    assert edits2[0].applied_edits[0].summary_text == "healthy-after-empty"
    assert noop._calls > calls_after_empty  # type: ignore[attr-defined]
    _assert_caller_unchanged(snapshot, messages)
    assert any("healthy-after-empty" in text for _, text in _fingerprint(captured[-1]))


async def _collect_events(
    executor: AgentExecutor,
    messages: list[Any],
    *,
    max_iterations: int,
) -> list[Any]:
    events: list[Any] = []
    async for event in executor.run_stream(messages, max_iterations=max_iterations):
        events.append(event)
    return events


def test_compaction_failure_then_healthy_turn_no_duplicate_tools() -> None:
    """Summarize failure then success in separate runs: noop invoked once per successful agent turn."""
    messages = _history(pairs=3)
    snapshot = copy.deepcopy(messages)
    captured: list[list[Any]] = []
    summary_attempts: list[int] = []
    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=1, mode="summarize")
    noop = _CountingNoop()

    llm_fail = _llm_agent_then_summarize(
        summarize_exc=ProviderError("compaction failed", provider="mock"),
        captured=captured,
        summary_attempts=summary_attempts,
    )
    executor = AgentExecutor("openai", llm_fail, tools=[noop], context_policy=policy)

    with pytest.raises(ProviderError):
        asyncio.run(_collect_events(executor, messages, max_iterations=2))

    _assert_caller_unchanged(snapshot, messages)
    calls_after_fail = noop._calls  # type: ignore[attr-defined]
    assert calls_after_fail >= 1

    executor.llm = _llm_agent_then_summarize(
        summarize_end=StreamEndEvent(
            model="mock",
            parts=[TextPart(text="usable-after-failure")],
            stop_reason="end_turn",
            usage=_usage(0),
        ),
        captured=captured,
        summary_attempts=summary_attempts,
    )
    events = asyncio.run(_collect_events(executor, messages, max_iterations=2))
    assert len([e for e in events if isinstance(e, ContextEditEvent)]) == 1
    assert noop._calls > calls_after_fail  # type: ignore[attr-defined]
    _assert_caller_unchanged(snapshot, messages)


def test_recovery_after_model_correction_rows_then_trim() -> None:
    """Invalid tool corrected on current executor; recovered trim compacts without re-invoking body."""
    typed = _TypedTool()
    call_count = 0

    async def mock_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        parts = (
            [CurrentToolUsePart(id="v1", name="typed_tool", inputs={"n": "bad"})]
            if call_count == 1
            else [CurrentToolUsePart(id="v2", name="typed_tool", inputs={"n": 7})]
            if call_count == 2
            else []
        )

        async def stream() -> AsyncIterator[CurrentStreamEndEvent]:
            yield CurrentStreamEndEvent(
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=CurrentUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    executor = CurrentAgentExecutor("openai", provider, tools=[typed])
    holder: dict[str, list[Any]] = {}
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def run() -> None:
        async for _event in executor.run_stream([], max_model_corrections=2, max_iterations=4):
            pass

    asyncio.run(run())
    assert typed._calls == 1  # type: ignore[attr-defined]
    history = _project_messages(holder["messages"])
    assert len(_find_tool_pairs(history)) >= 2

    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=1, mode="trim")
    turns = [
        ([ToolUsePart(id="c1", name="noop", inputs={})], _usage(_TRIGGER)),
        ([], _usage(_TRIGGER)),
    ]
    events, captured = _drive_recovered(policy, turns, history, tools=[_NoopTool()])
    assert typed._calls == 1  # type: ignore[attr-defined]
    assert any(isinstance(event, ContextEditEvent) for event in events)
    assert _structure_valid(captured[-1])


@dataclass
class _TypedTool(CurrentTool):
    name = "typed_tool"
    description = "Typed body."

    def __call__(self, n: int) -> dict[str, int]:
        self._calls = getattr(self, "_calls", 0) + 1  # type: ignore[attr-defined]
        return {"n": n}
