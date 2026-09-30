# ruff: noqa: E402
"""Phase 11: compaction interaction with model correction (``max_model_corrections``).

Model correction runs on the current ``dobby`` executor. Compaction uses the
recovered ``dobby-compaction-94b5a8f`` package. Scripted providers only.
"""

# isort: off
from __future__ import annotations

import asyncio
import copy
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from dobby import AgentExecutor as CurrentAgentExecutor
from dobby.exceptions import ModelRetry
from dobby.tools import Tool as CurrentTool
from dobby.types import StreamEndEvent as CurrentStreamEndEvent
from dobby.types import TextPart as CurrentTextPart
from dobby.types import ToolResultEvent
from dobby.types import ToolResultPart as CurrentToolResultPart
from dobby.types import ToolUsePart as CurrentToolUsePart
from dobby.types import Usage as CurrentUsage
from dobby.types import UserMessagePart as CurrentUserMessagePart


def _recovered_root() -> Path:
    repo = Path(__file__).resolve().parents[1]
    pointer = repo / ".git" / "worktrees" / "dobby-compaction-94b5a8f" / "gitdir"
    return Path(pointer.read_text(encoding="utf-8").strip()).parent


def _load_recovered() -> Any:
    name = "recovered_dobby"
    if name in sys.modules:
        return sys.modules[name]
    root = _recovered_root()
    init = root / "dobby" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name,
        init,
        submodule_search_locations=[str(root / "dobby")],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load recovered compaction package from {init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context import edit_context, summarize_context
from recovered_dobby.types import (
    AssistantMessagePart,
    StreamEndEvent,
    TextPart,
    ToolResultEvent as RecoveredToolResultEvent,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart,
)

# isort: on

_PLACEHOLDER = "[Tool result cleared to save context.]"
_FACT_RE = re.compile(r"\[DOBBY-FACT:(?P<category>[a-z]+):(?P<key>[^=\]]+)=(?P<value>[^\]]+)\]")


class ScriptedProvider:
    """One scripted ``StreamEndEvent`` per ``chat`` call."""

    def __init__(self, turns: list[list[CurrentToolUsePart]]) -> None:
        self.turns = turns
        self.calls: list[list[Any]] = []

    async def chat(self, messages: list[Any], **kwargs: Any) -> Any:
        del kwargs
        self.calls.append(list(messages))
        idx = len(self.calls) - 1
        parts = self.turns[idx] if idx < len(self.turns) else []

        async def stream() -> Any:
            yield CurrentStreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=CurrentUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()


def _usage(input_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _trim_policy(*, keep_last_n: int = 2, mode: str = "trim") -> ContextPolicy:
    return ContextPolicy(
        context_window=128_000,
        trigger_pct=0.8,
        keep_last_n=keep_last_n,
        mode=mode,
    )


def _pair(
    call_id: str,
    text: str,
    *,
    name: str = "typed",
    inputs: dict[str, Any] | None = None,
    is_error: bool = False,
) -> tuple[AssistantMessagePart, UserMessagePart]:
    return (
        AssistantMessagePart(
            parts=[
                ToolUsePart(
                    id=call_id,
                    name=name,
                    inputs={} if inputs is None else inputs,
                )
            ]
        ),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id=call_id,
                    name=name,
                    parts=[TextPart(text=text)],
                    is_error=is_error,
                )
            ]
        ),
    )


def _result_parts(messages: list[Any]) -> list[ToolResultPart]:
    return [
        part
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]


def _use_parts(messages: list[Any]) -> list[ToolUsePart]:
    return [
        part
        for message in messages
        if isinstance(message, AssistantMessagePart)
        for part in message.parts
        if isinstance(part, ToolUsePart)
    ]


def _result_texts(messages: list[Any]) -> list[str]:
    texts: list[str] = []
    for part in _result_parts(messages):
        texts.append("".join(piece.text for piece in part.parts if isinstance(piece, TextPart)))
    return texts


def _context_blob(messages: list[Any]) -> str:
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, TextPart):
                chunks.append(part.text)
            elif isinstance(part, ToolResultPart):
                chunks.extend(
                    piece.text for piece in part.parts if isinstance(piece, TextPart)
                )
    return "\n".join(chunks)


def _project_current_messages(messages: list[Any]) -> list[Any]:
    projected: list[Any] = []
    for message in messages:
        if message.role == "assistant":
            parts: list[Any] = []
            for part in message.parts:
                if isinstance(part, CurrentToolUsePart):
                    parts.append(
                        ToolUsePart(
                            id=part.id,
                            name=part.name,
                            inputs=dict(part.inputs),
                        )
                    )
                elif isinstance(part, CurrentTextPart):
                    parts.append(TextPart(text=part.text))
            projected.append(AssistantMessagePart(parts=parts))
        else:
            parts = []
            for part in message.parts:
                if isinstance(part, CurrentToolResultPart):
                    inner = [
                        TextPart(text=piece.text)
                        for piece in part.parts
                        if isinstance(piece, CurrentTextPart)
                    ]
                    parts.append(
                        ToolResultPart(
                            tool_use_id=part.tool_use_id,
                            name=part.name,
                            parts=inner,
                            is_error=part.is_error,
                        )
                    )
                elif isinstance(part, CurrentTextPart):
                    parts.append(TextPart(text=part.text))
            projected.append(UserMessagePart(parts=parts))
    return projected


def _current_tool_results(messages: list[Any]) -> list[CurrentToolResultPart]:
    return [
        part
        for message in messages
        if isinstance(message, CurrentUserMessagePart)
        for part in message.parts
        if isinstance(part, CurrentToolResultPart)
    ]


def _scripted_recovered(
    turns: list[tuple[list[Any], Usage | None]],
    captured: list[list[Any]],
) -> Any:
    call_count = 0

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        idx = min(call_count, len(turns) - 1)
        call_count += 1
        captured.append(list(messages))
        parts, usage = turns[idx]

        async def stream() -> Any:
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=usage,
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    provider.name = "scripted"
    return provider


class _NoopTool(CurrentTool):
    name = "noop"
    description = "Ack."
    edits_context: ClassVar[bool] = False

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


@dataclass
class _TypedTool(CurrentTool):
    name = "typed"
    description = "Accept an integer count."
    edits_context: ClassVar[bool] = False

    async def __call__(self, count: int) -> int:
        invocations = getattr(self, "_invocations", 0)
        self._invocations = invocations + 1  # type: ignore[attr-defined]
        return count


@dataclass
class _FailingTool(CurrentTool):
    name = "failing"
    description = "Fail during execution."
    edits_context: ClassVar[bool] = False

    async def __call__(self) -> None:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        raise ValueError("execution failure")


@dataclass
class _BodyRetryTool(CurrentTool):
    name = "body_retry"
    description = "Request model correction from the tool body."
    edits_context: ClassVar[bool] = False

    async def __call__(self) -> None:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        raise ModelRetry("fix the payload")


def _run_current(
    provider: ScriptedProvider,
    tools: list[CurrentTool],
    messages: list[Any] | None = None,
    **run_kwargs: Any,
) -> tuple[list[Any], list[Any], list[list[Any]]]:
    executor = CurrentAgentExecutor("openai", provider, tools=tools)
    holder: dict[str, list[Any]] = {"messages": list(messages or [])}

    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def collect() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(holder["messages"], **run_kwargs):
            events.append(event)
        return events

    events = asyncio.run(collect())
    return events, holder["messages"], provider.calls


def _drive_recovered(
    policy: ContextPolicy,
    turns: list[tuple[list[Any], Usage | None]],
    messages: list[Any],
    *,
    tools: list[CurrentTool] | None = None,
) -> tuple[list[Any], list[list[Any]], list[Any]]:
    captured: list[list[Any]] = []
    snapshots: list[list[Any]] = []
    llm = _scripted_recovered(turns, captured)
    executor = AgentExecutor(
        "openai",
        llm,
        tools=tools or [_NoopTool()],
        context_policy=policy,
    )
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        snapshots.append(copy.deepcopy(args[3]))
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=len(turns)):
            events.append(event)
        return events

    events = asyncio.run(run())
    working = snapshots[-1] if snapshots else list(messages)
    return events, captured, working


def _fact_preserving_summarizer() -> AsyncMock:
    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", True) is False:
            span_text = _context_blob(messages)
            markers = " ".join(m.group(0) for m in _FACT_RE.finditer(span_text))
            digest = markers or "empty-span"
            return StreamEndEvent(
                type="stream_end",
                model="mock-summarizer",
                parts=[TextPart(text=digest)],
                stop_reason="end_turn",
                usage=_usage(0),
            )
        raise AssertionError("unexpected streaming chat in summarize test")

    provider = AsyncMock()
    provider.chat = mock_chat
    provider.name = "deterministic-summarizer"
    return provider


def test_correction_before_compaction_preserves_tool_use_ids() -> None:
    """Invalid input → correction → success, then auto-trim keeps ids and results."""
    typed = _TypedTool()
    provider = ScriptedProvider(
        [
            [CurrentToolUsePart(id="call-v1", name="typed", inputs={"count": "bad"})],
            [CurrentToolUsePart(id="call-v2", name="typed", inputs={"count": 2})],
            [],
        ]
    )
    events, live_messages, _calls = _run_current(
        provider, [typed], max_model_corrections=3
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert typed._invocations == 1  # type: ignore[attr-defined]
    assert [event.tool_use_id for event in results] == ["call-v1", "call-v2"]
    assert str(results[0].result).startswith("[tool_input_invalid]")
    assert results[1].result == 2

    history = _project_current_messages(live_messages)
    assert [part.tool_use_id for part in _result_parts(history)] == ["call-v1", "call-v2"]

    policy = _trim_policy()
    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="warm", name="noop", inputs={})], _usage(trigger)),
        ([ToolUsePart(id="after", name="noop", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    compact_events, captured, _working = _drive_recovered(policy, turns, history)
    edits = [event for event in compact_events if isinstance(event, ContextEditEvent)]
    assert len(edits) >= 1
    send = captured[1]
    ids = [part.tool_use_id for part in _result_parts(send)]
    assert "call-v2" in ids
    assert "2" in _result_texts(send)
    assert getattr(typed, "_invocations", 0) == 1


def test_correction_after_compaction_sees_trimmed_context() -> None:
    """After auto-trim, a later correction run still emits ``[tool_input_invalid]`` feedback."""
    policy = _trim_policy(keep_last_n=2)
    filler: list[Any] = []
    for index in range(5):
        filler.extend(_pair(f"f{index}", f"filler-{index}", name="noop"))

    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="warm", name="noop", inputs={})], _usage(trigger)),
        ([ToolUsePart(id="post-warm", name="noop", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    _events, captured, working = _drive_recovered(policy, turns, filler)
    assert len(captured) >= 2
    trimmed_send = captured[1]
    assert _PLACEHOLDER in _context_blob(trimmed_send)
    assert "filler-0" not in _context_blob(trimmed_send)

    typed = _TypedTool()
    provider = ScriptedProvider(
        [
            [CurrentToolUsePart(id="post-c1", name="typed", inputs={"count": "bad"})],
            [],
        ]
    )
    run_events, _final_messages, model_calls = _run_current(
        provider, [typed], messages=working, max_model_corrections=3
    )
    tool_results = [event for event in run_events if isinstance(event, ToolResultEvent)]
    assert len(model_calls) == 2
    assert len(tool_results) == 1
    assert getattr(typed, "_invocations", 0) == 0
    assert str(tool_results[0].result).startswith("[tool_input_invalid]")
    assert tool_results[0].tool_use_id == "post-c1"


def test_multiple_correction_rounds_trim_by_pair() -> None:
    """Three correction rows plus success; keep_last_n=2 clears the oldest pair."""
    history: list[Any] = []
    history.extend(
        _pair(
            "r1",
            "[tool_input_invalid] count must be int",
            inputs={"count": "bad-1"},
            is_error=True,
        )
    )
    history.extend(
        _pair(
            "r2",
            "[tool_input_invalid] count must be int",
            inputs={"count": "bad-2"},
            is_error=True,
        )
    )
    history.extend(
        _pair(
            "r3",
            "[tool_retry] fix constraint",
            name="body_retry",
            inputs={},
            is_error=True,
        )
    )
    history.extend(_pair("ok", "7", inputs={"count": 7}, is_error=False))

    edited, applied = edit_context(history, _trim_policy(keep_last_n=2))
    assert applied is not None
    assert applied.cleared_tool_uses == 2
    ids = [part.tool_use_id for part in _result_parts(edited)]
    assert ids == ["r1", "r2", "r3", "ok"]
    texts = _result_texts(edited)
    assert texts[0] == _PLACEHOLDER
    assert texts[1] == _PLACEHOLDER
    assert "[tool_retry]" in texts[2]
    assert texts[3] == "7"


def test_failed_execution_then_correction_then_success() -> None:
    """Execution error does not consume budget; later body correction then typed success."""
    failing = _FailingTool()
    body = _BodyRetryTool()
    typed = _TypedTool()
    provider = ScriptedProvider(
        [
            [CurrentToolUsePart(id="exec-fail", name="failing", inputs={})],
            [CurrentToolUsePart(id="body", name="body_retry", inputs={})],
            [CurrentToolUsePart(id="ok", name="typed", inputs={"count": 5})],
            [],
        ]
    )
    events, messages, _ = _run_current(
        provider,
        [failing, body, typed],
        max_model_corrections=3,
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert failing._calls == 1  # type: ignore[attr-defined]
    assert body._calls == 1  # type: ignore[attr-defined]
    assert typed._invocations == 1  # type: ignore[attr-defined]
    assert [event.tool_use_id for event in results] == ["exec-fail", "body", "ok"]
    assert "[tool_execution_error]" in str(results[0].result)
    assert str(results[1].result) == "[tool_retry] fix the payload"

    projected = _project_current_messages(messages)
    edited, applied = edit_context(projected, _trim_policy(keep_last_n=1))
    assert applied is not None
    blob = _context_blob(edited)
    assert _PLACEHOLDER in blob
    assert "[tool_execution_error]" not in blob
    assert "5" in blob


def test_correction_history_summarize_preserves_required_marker() -> None:
    marker = "[DOBBY-FACT:constraint:correction_hint=use_int_count]"
    history: list[Any] = []
    history.extend(
        _pair(
            "c1",
            f"[tool_input_invalid] bad type\n{marker}",
            inputs={"count": "x"},
            is_error=True,
        )
    )
    history.extend(_pair("c2", "99", inputs={"count": 99}))
    policy = _trim_policy(keep_last_n=1, mode="summarize")
    applied = asyncio.run(summarize_context(history, policy, _fact_preserving_summarizer()))
    assert applied is not None
    assert marker in applied.summary_text
    assert "99" in _result_texts(history)


def test_parallel_tools_model_correction_trim_keeps_ids() -> None:
    """Parallel batch: unknown tool (correction) + successful sibling; trim keeps pairing."""
    typed = _TypedTool()
    provider = ScriptedProvider(
        [
            [
                CurrentToolUsePart(id="unknown", name="missing", inputs={}),
                CurrentToolUsePart(id="sibling", name="typed", inputs={"count": 1}),
            ],
            [CurrentToolUsePart(id="unknown-2", name="missing", inputs={})],
            [],
        ]
    )
    events, live_messages, _ = _run_current(provider, [typed], max_model_corrections=2)
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert typed._invocations == 1  # type: ignore[attr-defined]
    assert [event.tool_use_id for event in results] == ["unknown", "sibling", "unknown-2"]

    projected = _project_current_messages(live_messages)
    edited, applied = edit_context(projected, _trim_policy(keep_last_n=2))
    assert applied is not None
    ids = [part.tool_use_id for part in _result_parts(edited)]
    assert ids == ["unknown", "sibling", "unknown-2"]
    use_ids = [part.id for part in _use_parts(edited)]
    assert use_ids == ids


def test_compaction_does_not_reexecute_corrected_tools() -> None:
    """Tool bodies run only during correction run, not on later compaction turns."""
    typed = _TypedTool()
    provider = ScriptedProvider(
        [
            [CurrentToolUsePart(id="c1", name="typed", inputs={"count": "bad"})],
            [CurrentToolUsePart(id="c2", name="typed", inputs={"count": 3})],
            [],
        ]
    )
    _events, live_messages, _ = _run_current(provider, [typed], max_model_corrections=2)
    assert typed._invocations == 1  # type: ignore[attr-defined]
    history = _project_current_messages(live_messages)

    policy = _trim_policy()
    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="n1", name="noop", inputs={})], _usage(trigger)),
        ([ToolUsePart(id="n2", name="noop", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    compact_events, _captured, _ = _drive_recovered(policy, turns, history)
    assert any(isinstance(event, ContextEditEvent) for event in compact_events)
    assert typed._invocations == 1  # type: ignore[attr-defined]


def test_trimmed_correction_feedback_not_in_later_model_context() -> None:
    """Oracle: cleared ``[tool_retry]`` text is absent from the post-trim model payload."""
    critical = "[tool_retry] must keep account ACME-99"
    history: list[Any] = []
    for index in range(4):
        history.extend(_pair(f"old-{index}", f"old-payload-{index}", name="noop"))
    history.extend(
        _pair(
            "corr",
            critical,
            name="body_retry",
            inputs={},
            is_error=True,
        )
    )
    history.extend(_pair("keep", "final-ok", name="noop"))

    policy = _trim_policy(keep_last_n=2)
    trigger = policy.trigger_tokens
    _events, captured, working = _drive_recovered(
        policy,
        [
            ([ToolUsePart(id="w", name="noop", inputs={})], _usage(trigger)),
            ([ToolUsePart(id="z", name="noop", inputs={})], _usage(trigger)),
            ([], _usage(trigger)),
        ],
        history,
    )
    assert len(captured) >= 2
    assert critical not in _context_blob(captured[1])
    assert "final-ok" in _context_blob(captured[1])

    edited, _ = edit_context(working, policy)
    assert critical not in _context_blob(edited)


def test_recovered_executor_does_not_validate_for_model_correction() -> None:
    """Witness: recovered executor runs tool body with invalid args (no ``[tool_input_invalid]``)."""
    typed = _TypedTool()
    policy = _trim_policy()
    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="only", name="typed", inputs={"count": "bad"})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, _captured, _working = _drive_recovered(
        policy, turns, [], tools=[typed]
    )
    results = [event for event in events if isinstance(event, RecoveredToolResultEvent)]
    assert len(results) == 1
    assert typed._invocations == 1  # type: ignore[attr-defined]
    assert "[tool_input_invalid]" not in str(results[0].result)
