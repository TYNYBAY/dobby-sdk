# ruff: noqa: E402
"""Phase 19: latency and cost — none vs trim vs summarize (deterministic mocks)."""

# isort: off
from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_LATENCY_HORIZONS = (20, 50, 100)


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
_MODES: tuple[ReductionMode, ...] = _REDUCTION._MODES
_seed_messages = _REDUCTION._seed_messages
_scripted_llm = _REDUCTION._scripted_llm
_message_chars = _REDUCTION._message_chars
from recovered_dobby.context._tokens import estimate_input_tokens as _estimate_input_tokens

ContextEditEvent = _REDUCTION.ContextEditEvent
ContextPolicy = _REDUCTION.ContextPolicy
AgentExecutor = _REDUCTION.AgentExecutor
_NoopTool = _REDUCTION._NoopTool
_WINDOW = _REDUCTION._WINDOW
from recovered_dobby.context import edit_context, summarize_context

# isort: on


@dataclass
class LatencyCostReport:
    turns: int
    mode: ReductionMode
    total_runtime_s: float
    compaction_execution_s: float
    edit_context_s: float
    summarize_context_s: float
    summarizer_mock_s: float
    compaction_events: int
    summarizer_calls: int
    summarizer_input_chars: int
    summarizer_input_tokens_est: int
    message_count_final: int
    estimated_chars_final: int
    estimated_tokens_final: int
    tokens_vs_none_ratio: float
    overhead_vs_none_s: float
    overhead_per_turn_ms: float
    compaction_ms_per_event_mean: float
    per_compaction_ms: list[float] = field(default_factory=list)
    cumulative_samples: list[tuple[int, float, int, int]] = field(default_factory=list)


@dataclass
class _TimingState:
    edit_context_s: float = 0.0
    summarize_context_s: float = 0.0
    per_compaction_ms: list[float] = field(default_factory=list)
    cumulative_compaction_s: float = 0.0


def _span_chars(messages: list[Any]) -> int:
    total = 0
    from recovered_dobby.types import TextPart, UserMessagePart

    for message in messages:
        if isinstance(message, UserMessagePart):
            for part in message.parts:
                if isinstance(part, TextPart):
                    total += len(part.text)
    return total


def _wrap_edit_context(state: _TimingState) -> Callable[..., Any]:
    def timed_edit_context(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return edit_context(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            state.edit_context_s += elapsed
            state.cumulative_compaction_s += elapsed
            state.per_compaction_ms.append(elapsed * 1000.0)

    return timed_edit_context


def _wrap_summarize_context(state: _TimingState) -> Callable[..., Any]:
    async def timed_summarize_context(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return await summarize_context(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            state.summarize_context_s += elapsed
            state.cumulative_compaction_s += elapsed
            state.per_compaction_ms.append(elapsed * 1000.0)

    return timed_summarize_context


async def _run_latency_cost(
    turns: int,
    mode: ReductionMode,
    *,
    none_baseline_tokens: int | None = None,
    keep_last_n: int = 2,
) -> LatencyCostReport:
    summary_calls: list[dict[str, Any]] = []
    captured: list[list[Any]] = []
    llm = _scripted_llm(turns=turns, summary_calls=summary_calls, captured_sends=captured)

    summarizer_mock_total = 0.0
    original_chat = llm.chat

    async def timed_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal summarizer_mock_total
        if kwargs.get("stream") is False:
            t0 = time.perf_counter()
            try:
                result = await original_chat(messages, *args, **kwargs)
            finally:
                summarizer_mock_total += time.perf_counter() - t0
            if summary_calls:
                summary_calls[-1]["span_chars"] = _span_chars(messages)
                summary_calls[-1]["span_tokens_est"] = _estimate_input_tokens(messages)
            return result
        return await original_chat(messages, *args, **kwargs)

    llm.chat = timed_chat

    policy: ContextPolicy | None
    if mode == "none":
        policy = None
    else:
        policy = ContextPolicy(
            context_window=_WINDOW,
            trigger_pct=0.8,
            keep_last_n=keep_last_n,
            mode=mode,
        )

    executor = AgentExecutor("openai", llm, tools=[_NoopTool()], context_policy=policy)
    state = _TimingState()
    cumulative_samples: list[tuple[int, float, int, int]] = []
    compaction_index = 0

    timed_edit = _wrap_edit_context(state)
    timed_summarize = _wrap_summarize_context(state)

    run_t0 = time.perf_counter()
    with (
        patch("recovered_dobby.executor.edit_context", timed_edit),
        patch("recovered_dobby.executor.summarize_context", timed_summarize),
    ):
        async for event in executor.run_stream(_seed_messages(), max_iterations=turns + 3):
            if isinstance(event, ContextEditEvent):
                compaction_index += 1
                send = captured[-1] if captured else _seed_messages()
                cumulative_samples.append(
                    (
                        compaction_index,
                        state.cumulative_compaction_s,
                        len(send),
                        _estimate_input_tokens(send),
                    )
                )
    total_runtime_s = time.perf_counter() - run_t0

    final_send = captured[-1] if captured else _seed_messages()
    tokens_final = _estimate_input_tokens(final_send)
    chars_final = _message_chars(final_send)
    edits = compaction_index

    span_chars = sum(c.get("span_chars", 0) for c in summary_calls)
    span_tokens = sum(c.get("span_tokens_est", 0) for c in summary_calls)

    compaction_execution_s = state.edit_context_s + state.summarize_context_s
    if none_baseline_tokens is None or none_baseline_tokens <= 0:
        ratio = 1.0
    else:
        ratio = tokens_final / none_baseline_tokens

    overhead_vs_none_s = 0.0
    overhead_per_turn_ms = 0.0

    return LatencyCostReport(
        turns=turns,
        mode=mode,
        total_runtime_s=total_runtime_s,
        compaction_execution_s=compaction_execution_s,
        edit_context_s=state.edit_context_s,
        summarize_context_s=state.summarize_context_s,
        summarizer_mock_s=summarizer_mock_total,
        compaction_events=edits,
        summarizer_calls=len(summary_calls),
        summarizer_input_chars=span_chars,
        summarizer_input_tokens_est=span_tokens,
        message_count_final=len(final_send),
        estimated_chars_final=chars_final,
        estimated_tokens_final=tokens_final,
        tokens_vs_none_ratio=ratio,
        overhead_vs_none_s=overhead_vs_none_s,
        overhead_per_turn_ms=overhead_per_turn_ms,
        compaction_ms_per_event_mean=(
            (compaction_execution_s * 1000.0 / edits) if edits else 0.0
        ),
        per_compaction_ms=list(state.per_compaction_ms),
        cumulative_samples=cumulative_samples,
    )


async def _run_all_with_none_baseline(turns: int) -> dict[ReductionMode, LatencyCostReport]:
    none_report = await _run_latency_cost(turns, "none")
    baseline_tokens = none_report.estimated_tokens_final
    baseline_runtime = none_report.total_runtime_s
    out: dict[ReductionMode, LatencyCostReport] = {"none": none_report}
    for mode in ("trim", "summarize"):
        report = await _run_latency_cost(turns, mode, none_baseline_tokens=baseline_tokens)
        report.overhead_vs_none_s = report.total_runtime_s - baseline_runtime
        report.overhead_per_turn_ms = (
            (report.overhead_vs_none_s / turns) * 1000.0 if turns else 0.0
        )
        out[mode] = report
    none_report.tokens_vs_none_ratio = 1.0
    return out


@pytest.mark.parametrize(
    ("mode", "turns"),
    [(m, t) for m in _MODES for t in _LATENCY_HORIZONS],
    ids=lambda x: f"{x}" if isinstance(x, str) else f"{x}-turns",
)
def test_latency_cost_mode_horizon(mode: ReductionMode, turns: int) -> None:
    """Compaction timing, summarizer workload, and context size at shared horizons."""
    report = asyncio.run(_run_latency_cost(turns, mode))

    assert report.total_runtime_s > 0
    assert report.estimated_tokens_final > 0
    _assert_mode_oracle(report, mode, turns)


def _assert_mode_oracle(report: LatencyCostReport, mode: ReductionMode, turns: int) -> None:
    if mode == "none":
        assert report.compaction_events == 0
        assert report.compaction_execution_s < 0.001
        assert report.summarizer_calls == 0
        assert report.summarizer_input_chars == 0
    elif mode == "trim":
        assert report.compaction_events == turns
        assert report.compaction_execution_s > 0
        assert report.summarizer_calls == 0
        assert report.edit_context_s > 0
        assert report.summarize_context_s < 0.001
    else:
        assert report.compaction_events >= turns - 2
        assert report.summarizer_calls == report.compaction_events
        assert report.summarize_context_s > 0
        assert report.summarizer_input_chars > 0
        assert report.summarizer_input_tokens_est > 0


@pytest.mark.parametrize("turns", (50, 100), ids=lambda n: f"overhead-{n}-turns")
def test_latency_cost_cumulative_overhead_grows_with_turns(turns: int) -> None:
    """Trim/summarize pay more total runtime vs none as turns increase."""
    small = asyncio.run(_run_all_with_none_baseline(20))
    large = asyncio.run(_run_all_with_none_baseline(turns))

    for mode in ("trim", "summarize"):
        assert large[mode].overhead_vs_none_s >= small[mode].overhead_vs_none_s
        assert large[mode].compaction_events >= small[mode].compaction_events
        if large[mode].cumulative_samples and small[mode].cumulative_samples:
            assert (
                large[mode].cumulative_samples[-1][1]
                >= small[mode].cumulative_samples[-1][1]
            )


def test_latency_cost_context_reduction_vs_none_at_50() -> None:
    """Trim and summarize reduce final context size relative to no compaction."""
    reports = asyncio.run(_run_all_with_none_baseline(50))
    none = reports["none"]
    assert reports["trim"].tokens_vs_none_ratio < 1.0
    assert reports["summarize"].tokens_vs_none_ratio < 1.0
    assert reports["trim"].message_count_final == none.message_count_final
    assert reports["summarize"].message_count_final < none.message_count_final


def test_latency_cost_summarizer_workload_tracks_span_growth() -> None:
    """Summarizer input workload grows with horizon (more history per span)."""
    r20 = asyncio.run(_run_latency_cost(20, "summarize"))
    r100 = asyncio.run(_run_latency_cost(100, "summarize"))
    assert r100.summarizer_input_chars > r20.summarizer_input_chars
    assert r100.summarizer_input_tokens_est > r20.summarizer_input_tokens_est
    assert r100.compaction_execution_s > r20.compaction_execution_s
