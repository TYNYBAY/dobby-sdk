# ruff: noqa: E402
"""Phase 20: determinism and reproducibility (deterministic mocks only)."""

# isort: off
from __future__ import annotations

import asyncio
import copy
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

_REPEAT_RUNS = 5
_SCENARIO_TURNS = 30
_PLACEHOLDER = "[Tool result cleared to save context.]"
_FACT_RE = re.compile(r"\[DOBBY-FACT:(?P<category>[a-z]+):(?P<key>[^=\]]+)=(?P<value>[^\]]+)\]")


def _load_reduction_module() -> Any:
    path = Path(__file__).with_name("test_context_compaction_reduction.py")
    name = "_reduction_harness"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reduction harness from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_REDUCTION = _load_reduction_module()
ReductionMode = _REDUCTION.ReductionMode
_seed_messages = _REDUCTION._seed_messages
_scripted_llm = _REDUCTION._scripted_llm
_seed_registry = _REDUCTION._seed_registry
_NoopTool = _REDUCTION._NoopTool
_WINDOW = _REDUCTION._WINDOW
ContextEditEvent = _REDUCTION.ContextEditEvent
ContextPolicy = _REDUCTION.ContextPolicy
AgentExecutor = _REDUCTION.AgentExecutor
UserMessagePart = _REDUCTION.UserMessagePart
TextPart = _REDUCTION.TextPart
ToolResultPart = _REDUCTION.ToolResultPart
ToolUsePart = _REDUCTION.ToolUsePart
AssistantMessagePart = _REDUCTION.AssistantMessagePart
StreamEndEvent = _REDUCTION.StreamEndEvent
_find_tool_pairs = _REDUCTION._find_tool_pairs

from recovered_dobby.context import edit_context, summarize_context
from recovered_dobby.tools import Tool

# isort: on


@dataclass(frozen=True)
class CompactionTrace:
    trigger_decisions: tuple[tuple[int, int, bool], ...]
    structure: tuple[Any, ...]
    summary_indices: tuple[int, ...]
    facts_retained: frozenset[str]
    facts_cleared: frozenset[str]
    pairings: tuple[tuple[str, str], ...]
    edit_sequence: tuple[tuple[Any, ...], ...]
    send_len_at_edits: tuple[int, ...]
    final_fingerprint: tuple[tuple[str, str], ...]


def _fingerprint(messages: list[Any]) -> tuple[tuple[str, str], ...]:
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
    return tuple(rows)


def _facts_in_blob(messages: list[Any]) -> frozenset[str]:
    blob_parts: list[str] = []
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, TextPart):
                blob_parts.append(part.text)
            elif isinstance(part, ToolResultPart):
                blob_parts.append(
                    "".join(p.text for p in part.parts if isinstance(p, TextPart))
                )
    blob = "\n".join(blob_parts)
    return frozenset(
        f"{m.group('category')}:{m.group('key')}={m.group('value')}"
        for m in _FACT_RE.finditer(blob)
    )


def _introduced_facts(turns: int) -> frozenset[str]:
    reg = _seed_registry()
    keys = [
        f"{k.split(':')[0]}:{k.split(':')[1]}={v}"
        for k, v in reg.items()
    ]
    keys.extend(f"id:turn-{i}=T-{i:04d}" for i in range(turns))
    return frozenset(keys)


def _structure_signature(messages: list[Any]) -> tuple[Any, ...]:
    rows: list[Any] = []
    for index, message in enumerate(messages):
        role = message.role
        for part in message.parts:
            if isinstance(part, ToolUsePart):
                rows.append(("use", index, part.id, part.name))
            elif isinstance(part, ToolResultPart):
                text = "".join(p.text for p in part.parts if isinstance(p, TextPart))
                kind = "ph" if text == _PLACEHOLDER else "body"
                rows.append(("result", index, part.tool_use_id, kind, len(text)))
            elif isinstance(part, TextPart):
                kind = "summary" if part.text.startswith("<summary>") else "text"
                rows.append((kind, index, len(part.text)))
    return tuple(rows)


def _summary_indices(messages: list[Any]) -> tuple[int, ...]:
    return tuple(
        index
        for index, message in enumerate(messages)
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, TextPart) and part.text.startswith("<summary>")
    )


def _pairings(messages: list[Any]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for use_idx, result_idx in _find_tool_pairs(messages):
        use_msg = messages[use_idx]
        result_msg = messages[result_idx]
        use_id = next(p.id for p in use_msg.parts if isinstance(p, ToolUsePart))
        result_id = next(
            p.tool_use_id for p in result_msg.parts if isinstance(p, ToolResultPart)
        )
        pairs.append((use_id, result_id))
    return tuple(pairs)


def _serialize_edits(events: list[Any]) -> tuple[tuple[Any, ...], ...]:
    sig: list[tuple[Any, ...]] = []
    for event in events:
        if not isinstance(event, ContextEditEvent):
            continue
        for applied in event.applied_edits:
            sig.append(
                (
                    applied.type,
                    getattr(applied, "cleared_tool_uses", None),
                    getattr(applied, "summary_text", None),
                )
            )
    return tuple(sig)


def _deterministic_summarizer() -> AsyncMock:
    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream") is False:
            span = " ".join(
                part.text
                for message in messages
                if isinstance(message, UserMessagePart)
                for part in message.parts
                if isinstance(part, TextPart)
            )
            markers = " ".join(m.group(0) for m in _FACT_RE.finditer(span))
            wrapped = f"<summary>{markers or 'empty'}</summary>"
            return StreamEndEvent(
                model="deterministic-summarizer",
                parts=[TextPart(text=wrapped)],
                stop_reason="end_turn",
                usage=_REDUCTION._usage(0),
            )
        raise AssertionError("unexpected streaming summarizer call")

    provider = AsyncMock()
    provider.chat = mock_chat
    provider.name = "deterministic-summarizer"
    return provider


async def _capture_trace(
    mode: Literal["trim", "summarize"],
    turns: int,
) -> CompactionTrace:
    summary_calls: list[dict[str, Any]] = []
    captured: list[list[Any]] = []
    usage_log: list[int] = []
    llm = _scripted_llm(turns=turns, summary_calls=summary_calls, captured_sends=captured)
    original_chat = llm.chat

    async def logging_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream") is False:
            return await original_chat(messages, *args, **kwargs)
        idx = len(usage_log)
        result = await original_chat(messages, *args, **kwargs)
        usage_log.append(_REDUCTION._usage_tokens_for_turn(idx, turns=turns))
        return result

    llm.chat = logging_chat

    policy = ContextPolicy(
        context_window=_WINDOW,
        trigger_pct=0.8,
        keep_last_n=2,
        mode=mode,
    )
    executor = AgentExecutor("openai", llm, tools=[_NoopTool()], context_policy=policy)

    events: list[Any] = []
    send_len_at_edits: list[int] = []

    async for event in executor.run_stream(_seed_messages(), max_iterations=turns + 3):
        events.append(event)
        if isinstance(event, ContextEditEvent):
            send = captured[-1] if captured else _seed_messages()
            send_len_at_edits.append(len(send))

    edits = [event for event in events if isinstance(event, ContextEditEvent)]
    if mode == "trim":
        trigger_decisions = tuple(
            (call_idx, usage_log[call_idx], call_idx > 0 and call_idx <= len(edits))
            for call_idx in range(len(usage_log))
        )
    else:
        trigger_decisions = tuple(
            (call_idx, usage_log[call_idx], call_idx > 0 and call_idx <= len(edits))
            for call_idx in range(len(usage_log))
        )

    final_send = captured[-1] if captured else _seed_messages()
    retained = _facts_in_blob(final_send)
    introduced = _introduced_facts(turns)

    return CompactionTrace(
        trigger_decisions=tuple(trigger_decisions),
        structure=_structure_signature(final_send),
        summary_indices=_summary_indices(final_send),
        facts_retained=retained,
        facts_cleared=introduced - retained,
        pairings=_pairings(final_send),
        edit_sequence=_serialize_edits(events),
        send_len_at_edits=tuple(send_len_at_edits),
        final_fingerprint=_fingerprint(final_send),
    )


def _assert_all_traces_equal(traces: list[CompactionTrace]) -> None:
    baseline = traces[0]
    for index, trace in enumerate(traces[1:], start=1):
        assert trace == baseline, f"run {index} diverged from run 0"


@pytest.mark.parametrize("mode", ("trim", "summarize"))
def test_repeated_executor_runs_produce_identical_traces(mode: Literal["trim", "summarize"]) -> None:
    """Identical scenario repeated N times yields the same compaction trace."""
    traces = [asyncio.run(_capture_trace(mode, _SCENARIO_TURNS)) for _ in range(_REPEAT_RUNS)]
    _assert_all_traces_equal(traces)


def test_concurrent_isolated_runs_match_sequential() -> None:
    """Concurrent asyncio tasks with separate executors do not cross-contaminate."""

    async def run_all() -> tuple[CompactionTrace, list[CompactionTrace]]:
        sequential = await _capture_trace("trim", 20)
        concurrent = list(await asyncio.gather(*[_capture_trace("trim", 20) for _ in range(4)]))
        return sequential, concurrent

    sequential, concurrent = asyncio.run(run_all())
    for trace in concurrent:
        assert trace == sequential


def test_edit_context_repeated_trim_is_stable() -> None:
    """Direct trim on deep-copied history is stable across invocations."""
    history = asyncio.run(_grow_history(15))
    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=2, mode="trim")
    signatures: list[tuple[Any, ...]] = []
    for _ in range(_REPEAT_RUNS):
        edited, applied = edit_context(copy.deepcopy(history), policy)
        signatures.append(
            (
                _structure_signature(edited),
                _pairings(edited),
                None if applied is None else (applied.type, applied.cleared_tool_uses),
            )
        )
    assert signatures.count(signatures[0]) == len(signatures)


@pytest.mark.parametrize("mode", ("trim", "summarize"))
def test_repeated_runs_identical_edit_event_sequence(mode: Literal["trim", "summarize"]) -> None:
    traces = [asyncio.run(_capture_trace(mode, 25)) for _ in range(3)]
    sequences = [t.edit_sequence for t in traces]
    assert sequences.count(sequences[0]) == len(sequences)
    assert traces[0].pairings == traces[1].pairings


def test_summarize_context_repeated_with_deterministic_summarizer() -> None:
    """summarize_context with a marker-echo summarizer is reproducible."""
    history = asyncio.run(_grow_history(12))
    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=2, mode="summarize")
    llm = _deterministic_summarizer()

    async def run_many() -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
        for _ in range(_REPEAT_RUNS):
            working = copy.deepcopy(history)
            applied = await summarize_context(working, policy, llm)
            out.append(
                (
                    _summary_indices(working),
                    applied.type if applied else None,
                    applied.cleared_tool_uses if applied else None,
                    _facts_in_blob(working),
                )
            )
        return out

    signatures = asyncio.run(run_many())
    assert signatures.count(signatures[0]) == len(signatures)


async def _grow_history(turns: int) -> list[Any]:
    summary_calls: list[dict[str, Any]] = []
    captured: list[list[Any]] = []
    llm = _scripted_llm(turns=turns, summary_calls=summary_calls, captured_sends=captured)
    executor = AgentExecutor("openai", llm, tools=[_NoopTool()], context_policy=None)
    async for _ in executor.run_stream(_seed_messages(), max_iterations=turns + 2):
        pass
    return copy.deepcopy(captured[-1])


class _SleepEcho(Tool):
    name = "echo"
    description = "Echo with configurable async delay."
    delays: list[float] = [0.0, 0.0, 0.0]

    async def __call__(self, label: str = "0") -> dict[str, str]:
        rank = int(label)
        await asyncio.sleep(self.delays[rank])
        return {"label": label}


async def _parallel_order_trace(delays: list[float]) -> CompactionTrace:
    echo = _SleepEcho()
    echo.delays = list(delays)
    batch = [
        ToolUsePart(id=f"p-{i}", name="echo", inputs={"label": str(i)})
        for i in range(3)
    ]
    trigger = int(0.8 * _WINDOW)
    turns_script = [
        ([ToolUsePart(id="warm", name="noop", inputs={"turn": 0, "fact": ""})], trigger),
        (batch, trigger),
        ([], trigger),
    ]
    captured: list[list[Any]] = []
    call_count = 0

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        idx = min(call_count, len(turns_script) - 1)
        call_count += 1
        captured.append(list(messages))
        parts, usage_tokens = turns_script[idx]

        async def stream() -> Any:
            yield StreamEndEvent(
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=_REDUCTION._usage(usage_tokens),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    policy = ContextPolicy(context_window=_WINDOW, trigger_pct=0.8, keep_last_n=2, mode="trim")
    executor = AgentExecutor(
        "openai",
        provider,
        tools=[echo, _NoopTool()],
        context_policy=policy,
    )
    filler = list(_seed_messages())
    events: list[Any] = []
    async for event in executor.run_stream(filler, max_iterations=4):
        events.append(event)
    final_send = captured[-1]
    return CompactionTrace(
        trigger_decisions=(),
        structure=_structure_signature(final_send),
        summary_indices=_summary_indices(final_send),
        facts_retained=_facts_in_blob(final_send),
        facts_cleared=frozenset(),
        pairings=_pairings(final_send),
        edit_sequence=_serialize_edits(events),
        send_len_at_edits=(),
        final_fingerprint=_fingerprint(final_send),
    )


def test_parallel_tool_completion_order_does_not_change_compaction_trace() -> None:
    """Different async completion ordering yields the same model-visible trim outcome."""
    forward = asyncio.run(_parallel_order_trace([0.001, 0.002, 0.003]))
    reverse = asyncio.run(_parallel_order_trace([0.003, 0.002, 0.001]))
    assert forward.structure == reverse.structure
    assert forward.pairings == reverse.pairings
    assert forward.final_fingerprint == reverse.final_fingerprint
    assert forward.edit_sequence == reverse.edit_sequence
