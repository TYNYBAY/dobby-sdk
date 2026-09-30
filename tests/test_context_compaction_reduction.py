# ruff: noqa: E402
"""Phase 18: reduction comparison — none vs trim vs summarize (deterministic mocks)."""

# isort: off
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

ReductionMode = Literal["none", "trim", "summarize"]

from tests.compaction_subject import load_recovered

load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context._tokens import estimate_input_tokens
from recovered_dobby.context.edit import _find_tool_pairs
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

_HORIZONS = (20, 50, 100)
_MODES: tuple[ReductionMode, ...] = ("none", "trim", "summarize")
_WINDOW = 128_000
_TRIGGER = int(0.8 * _WINDOW)
_PLACEHOLDER = "[Tool result cleared to save context.]"
_FACT_RE = re.compile(r"\[DOBBY-FACT:(?P<category>[a-z]+):(?P<key>[^=\]]+)=(?P<value>[^\]]+)\]")


@dataclass
class ReductionReport:
    turns: int
    mode: ReductionMode
    message_count_final: int
    estimated_chars_final: int
    estimated_tokens_final: int
    compaction_events: int
    summarizer_calls: int
    summary_turns: int
    placeholder_results: int
    tool_pairs_final: int
    facts_introduced: int
    facts_retained: int
    recent_tool_results_retained: int
    facts_required_latest: bool
    structurally_valid: bool
    context_usable: bool
    model_calls: int
    samples: list[tuple[int, int, int]] = field(default_factory=list)


def _marker(category: str, key: str, value: str) -> str:
    return f"[DOBBY-FACT:{category}:{key}={value}]"


def _seed_registry() -> dict[str, str]:
    return {
        "id:root": "ROOT-9000",
        "number:baseline": "314159",
        "decision:policy": "approve tier-2 only",
    }


def _pair(index: int, text: str) -> tuple[AssistantMessagePart, UserMessagePart]:
    call_id = f"seed-{index}"
    return (
        AssistantMessagePart(parts=[ToolUsePart(id=call_id, name="record", inputs={"i": index})]),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id=call_id,
                    name="record",
                    parts=[TextPart(text=text)],
                )
            ]
        ),
    )


def _seed_messages() -> list[Any]:
    reg = _seed_registry()
    payload = " ".join(_marker(k.split(":")[0], k.split(":")[1], v) for k, v in reg.items())
    return [
        UserMessagePart(parts=[TextPart(text="reduction-comparison task")]),
        *_pair(0, payload),
        *_pair(1, "seed-bridge"),
    ]


def _message_chars(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        if isinstance(message, UserMessagePart):
            for part in message.parts:
                if isinstance(part, TextPart):
                    total += len(part.text)
                elif isinstance(part, ToolResultPart):
                    for inner in part.parts:
                        if isinstance(inner, TextPart):
                            total += len(inner.text)
        elif isinstance(message, AssistantMessagePart):
            for part in message.parts:
                if isinstance(part, ToolUsePart):
                    total += len(part.name) + len(str(part.inputs))
    return total


def _facts_in_messages(messages: list[Any]) -> dict[str, str]:
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
    out: dict[str, str] = {}
    for match in _FACT_RE.finditer(blob):
        key = f"{match.group('category')}:{match.group('key')}"
        out[key] = match.group("value")
    return out


def _summary_turn_count(messages: list[Any]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, TextPart) and part.text.startswith("<summary>")
    )


def _placeholder_count(messages: list[Any]) -> int:
    n = 0
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, ToolResultPart):
                for inner in part.parts:
                    if isinstance(inner, TextPart) and inner.text == _PLACEHOLDER:
                        n += 1
    return n


def _tool_result_is_placeholder(message: UserMessagePart) -> bool:
    for part in message.parts:
        if isinstance(part, ToolResultPart):
            for inner in part.parts:
                if isinstance(inner, TextPart) and inner.text == _PLACEHOLDER:
                    return True
    return False


def _recent_tool_results_retained(messages: list[Any], *, keep_last_n: int) -> int:
    pairs = _find_tool_pairs(messages)
    tail = pairs[-keep_last_n:] if len(pairs) > keep_last_n else pairs
    retained = 0
    for _use_idx, result_idx in tail:
        result_msg = messages[result_idx]
        if isinstance(result_msg, UserMessagePart) and not _tool_result_is_placeholder(result_msg):
            retained += 1
    return retained


def _validate_structure(messages: list[Any]) -> bool:
    pairs = _find_tool_pairs(messages)
    for use_idx, result_idx in pairs:
        use_msg = messages[use_idx]
        result_msg = messages[result_idx]
        if not isinstance(use_msg, AssistantMessagePart) or not isinstance(result_msg, UserMessagePart):
            return False
        use_parts = [p for p in use_msg.parts if isinstance(p, ToolUsePart)]
        result_parts = [p for p in result_msg.parts if isinstance(p, ToolResultPart)]
        if len(use_parts) != 1 or not result_parts:
            return False
        if use_parts[0].id != result_parts[0].tool_use_id:
            return False
    return True


def _context_usable(
    *,
    mode: ReductionMode,
    structurally_valid: bool,
    latest_fact_ok: bool,
    facts_retained: int,
    facts_introduced: int,
    compaction_events: int,
    turns: int,
) -> bool:
    if not structurally_valid or not latest_fact_ok:
        return False
    if mode == "none":
        return compaction_events == 0 and facts_retained == facts_introduced
    if mode == "trim":
        return compaction_events == turns and facts_retained >= 1
    return compaction_events >= turns - 2 and facts_retained >= facts_introduced


class _NoopTool(Tool):
    name = "noop"
    description = "Ack; echoes turn fact payload when provided."

    def __call__(self, turn: int = 0, fact: str = "") -> dict[str, str]:
        return {"turn": str(turn), "payload": fact or "ok"}


def _usage(input_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _extract_markers(text: str) -> str:
    return " ".join(m.group(0) for m in _FACT_RE.finditer(text))


def _usage_tokens_for_turn(turn_index: int, *, turns: int) -> int:
    """Identical usage sequence for every reduction mode."""
    if turn_index < turns:
        return _TRIGGER + turn_index * 512
    return _TRIGGER + turns * 512


def _scripted_llm(
    *,
    turns: int,
    summary_calls: list[dict[str, Any]],
    captured_sends: list[list[Any]],
) -> Any:
    model_call = 0

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal model_call
        if kwargs.get("stream") is False:
            span_text = " ".join(
                part.text
                for message in messages
                if isinstance(message, UserMessagePart)
                for part in message.parts
                if isinstance(part, TextPart)
            )
            summary_calls.append({"span_text": span_text})
            digest = _extract_markers(span_text) or "empty-span"
            wrapped = f"<summary>{digest}</summary>" if digest != "empty-span" else "<summary></summary>"
            return StreamEndEvent(
                model="mock-summarizer",
                parts=[TextPart(text=wrapped)],
                stop_reason="end_turn",
                usage=_usage(0),
            )

        idx = model_call
        model_call += 1
        captured_sends.append(list(messages))

        if idx < turns:
            turn_fact = _marker("id", f"turn-{idx}", f"T-{idx:04d}")
            parts = [
                ToolUsePart(
                    id=f"call-{idx}",
                    name="noop",
                    inputs={"turn": idx, "fact": turn_fact},
                )
            ]
            usage_tokens = _usage_tokens_for_turn(idx, turns=turns)
        else:
            parts = []
            usage_tokens = _usage_tokens_for_turn(turns, turns=turns)

        async def stream() -> Any:
            yield StreamEndEvent(
                model="mock-agent",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=_usage(usage_tokens),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    provider.name = "reduction-mock"
    return provider


async def _run_reduction(
    turns: int,
    mode: ReductionMode,
    *,
    keep_last_n: int = 2,
) -> tuple[list[Any], ReductionReport]:
    summary_calls: list[dict[str, Any]] = []
    captured: list[list[Any]] = []
    llm = _scripted_llm(turns=turns, summary_calls=summary_calls, captured_sends=captured)

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

    events: list[Any] = []
    samples: list[tuple[int, int, int]] = []

    async for event in executor.run_stream(_seed_messages(), max_iterations=turns + 3):
        events.append(event)
        if isinstance(event, ContextEditEvent):
            send = captured[-1] if captured else _seed_messages()
            mc = len(send)
            if len(samples) < 12 and (not samples or mc != samples[-1][1]):
                samples.append((len(events), mc, estimate_input_tokens(send)))

    final_send = captured[-1] if captured else _seed_messages()
    edits = [e for e in events if isinstance(e, ContextEditEvent)]
    facts = _facts_in_messages(final_send)
    introduced = len(_seed_registry()) + turns
    latest_key = f"id:turn-{turns - 1}"
    latest_ok = latest_key in facts if turns > 0 else True
    struct_ok = _validate_structure(final_send)

    report = ReductionReport(
        turns=turns,
        mode=mode,
        message_count_final=len(final_send),
        estimated_chars_final=_message_chars(final_send),
        estimated_tokens_final=estimate_input_tokens(final_send),
        compaction_events=len(edits),
        summarizer_calls=len(summary_calls),
        summary_turns=_summary_turn_count(final_send),
        placeholder_results=_placeholder_count(final_send),
        tool_pairs_final=len(_find_tool_pairs(final_send)),
        facts_introduced=introduced,
        facts_retained=len(facts),
        recent_tool_results_retained=_recent_tool_results_retained(
            final_send, keep_last_n=keep_last_n
        ),
        facts_required_latest=latest_ok,
        structurally_valid=struct_ok,
        context_usable=_context_usable(
            mode=mode,
            structurally_valid=struct_ok,
            latest_fact_ok=latest_ok,
            facts_retained=len(facts),
            facts_introduced=introduced,
            compaction_events=len(edits),
            turns=turns,
        ),
        model_calls=len(captured),
        samples=samples,
    )
    return events, report


async def _run_all_modes(turns: int) -> dict[ReductionMode, ReductionReport]:
    out: dict[ReductionMode, ReductionReport] = {}
    for mode in _MODES:
        _events, report = await _run_reduction(turns, mode, keep_last_n=2)
        out[mode] = report
    return out


@pytest.mark.parametrize(
    ("mode", "turns"),
    [(m, t) for m in _MODES for t in _HORIZONS],
    ids=lambda x: f"{x}" if isinstance(x, str) else f"{x}-turns",
)
def test_reduction_mode_horizon(mode: ReductionMode, turns: int) -> None:
    """Each mode on the shared scenario: metrics and usability oracles."""
    _events, report = asyncio.run(_run_reduction(turns, mode, keep_last_n=2))

    assert report.structurally_valid
    assert report.facts_required_latest
    assert report.context_usable
    assert report.model_calls == turns + 1

    expected_rows = 5 + 2 * turns
    if mode == "none":
        assert report.compaction_events == 0
        assert report.summarizer_calls == 0
        assert report.summary_turns == 0
        assert report.placeholder_results == 0
        assert report.message_count_final == expected_rows
        assert report.facts_retained == report.facts_introduced
        assert report.recent_tool_results_retained == min(2 + turns, 2)
    elif mode == "trim":
        assert report.compaction_events == turns
        assert report.summarizer_calls == 0
        assert report.message_count_final == expected_rows
        assert report.placeholder_results >= turns
        assert report.facts_retained <= len(_seed_registry()) + 2
        assert report.recent_tool_results_retained == 2
        assert report.estimated_tokens_final < report.estimated_chars_final // 3
    else:
        assert report.summary_turns == report.compaction_events
        assert report.summarizer_calls == report.compaction_events
        assert report.compaction_events >= turns - 2
        assert report.facts_retained >= report.facts_introduced
        assert report.message_count_final <= turns + 8
        assert report.recent_tool_results_retained <= 2


@pytest.mark.parametrize("turns", (50,), ids=lambda n: f"compare-{n}-turns")
def test_reduction_modes_directly_comparable(turns: int) -> None:
    """Same inputs: none retains everything; trim/summarize reduce size with different tradeoffs."""
    reports = asyncio.run(_run_all_modes(turns))
    none = reports["none"]
    trim = reports["trim"]
    summarize = reports["summarize"]

    assert none.model_calls == trim.model_calls == summarize.model_calls
    assert none.compaction_events == 0
    assert trim.compaction_events == turns
    assert summarize.compaction_events >= turns - 2

    assert none.facts_retained == none.facts_introduced
    assert summarize.facts_retained >= none.facts_introduced
    assert trim.facts_retained < none.facts_retained

    assert trim.estimated_tokens_final < none.estimated_tokens_final
    assert summarize.estimated_tokens_final < none.estimated_tokens_final
    assert summarize.message_count_final < none.message_count_final
    assert none.message_count_final == trim.message_count_final

    assert none.recent_tool_results_retained == 2
    assert trim.recent_tool_results_retained == 2
    assert summarize.recent_tool_results_retained <= 2

    assert none.context_usable and trim.context_usable and summarize.context_usable


def test_reduction_none_identical_send_growth_without_edits() -> None:
    """Without a policy, context rows grow linearly and no compaction events fire."""
    _events, report = asyncio.run(_run_reduction(20, "none"))
    assert report.compaction_events == 0
    assert report.message_count_final == 5 + 2 * 20
    assert report.tool_pairs_final == 2 + 20
    assert report.placeholder_results == 0
