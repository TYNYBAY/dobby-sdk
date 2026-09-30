# ruff: noqa: E402
"""Phase 16: long-horizon compaction (deterministic mocks, no live providers)."""

# isort: off
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

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

_HORIZONS = (20, 50, 100, 250, 500)
_WINDOW = 128_000
_TRIGGER = int(0.8 * _WINDOW)
_PLACEHOLDER = "[Tool result cleared to save context.]"
_FACT_RE = re.compile(r"\[DOBBY-FACT:(?P<category>[a-z]+):(?P<key>[^=\]]+)=(?P<value>[^\]]+)\]")


@dataclass
class LongHorizonReport:
    turns: int
    mode: Literal["trim", "summarize"]
    message_count_final: int
    max_message_count: int
    estimated_tokens_final: int
    compaction_events: int
    summarizer_calls: int
    summary_turns: int
    placeholder_results: int
    tool_pairs_final: int
    facts_introduced: int
    facts_retained: int
    facts_required_latest: bool
    structurally_valid: bool
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
        UserMessagePart(parts=[TextPart(text="long-horizon task")]),
        *_pair(0, payload),
        *_pair(1, "seed-bridge"),
    ]


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


class _NoopTool(Tool):
    name = "noop"
    description = "Ack; echoes turn fact payload when provided."

    def __call__(self, turn: int = 0, fact: str = "") -> dict[str, str]:
        return {"turn": str(turn), "payload": fact or "ok"}


def _usage(input_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _extract_markers(text: str) -> str:
    return " ".join(m.group(0) for m in _FACT_RE.finditer(text))


def _scripted_llm(
    *,
    turns: int,
    mode: Literal["trim", "summarize"],
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
            if mode == "trim":
                usage_tokens = _TRIGGER
            else:
                usage_tokens = _TRIGGER + idx * 512
        else:
            parts = []
            usage_tokens = _TRIGGER + turns * 512

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
    provider.name = "long-horizon-mock"
    return provider


async def _run_horizon(
    turns: int,
    mode: Literal["trim", "summarize"],
    *,
    keep_last_n: int = 2,
) -> tuple[list[Any], LongHorizonReport]:
    policy = ContextPolicy(
        context_window=_WINDOW,
        trigger_pct=0.8,
        keep_last_n=keep_last_n,
        mode=mode,
    )
    summary_calls: list[dict[str, Any]] = []
    captured: list[list[Any]] = []
    llm = _scripted_llm(
        turns=turns,
        mode=mode,
        summary_calls=summary_calls,
        captured_sends=captured,
    )
    executor = AgentExecutor("openai", llm, tools=[_NoopTool()], context_policy=policy)

    events: list[Any] = []
    max_messages = 0
    samples: list[tuple[int, int, int]] = []

    async for event in executor.run_stream(_seed_messages(), max_iterations=turns + 3):
        events.append(event)
        if isinstance(event, ContextEditEvent):
            send = captured[-1] if captured else _seed_messages()
            mc = len(send)
            max_messages = max(max_messages, mc)
            if len(samples) < 12 and (not samples or mc != samples[-1][1]):
                samples.append((len(events), mc, estimate_input_tokens(send)))

    final_send = captured[-1] if captured else _seed_messages()
    mc_final = len(final_send)
    max_messages = max(max_messages, mc_final)

    edits = [e for e in events if isinstance(e, ContextEditEvent)]
    facts = _facts_in_messages(final_send)
    introduced = len(_seed_registry()) + turns
    latest_key = f"id:turn-{turns - 1}"
    latest_ok = latest_key in facts if turns > 0 else True

    report = LongHorizonReport(
        turns=turns,
        mode=mode,
        message_count_final=mc_final,
        max_message_count=max_messages,
        estimated_tokens_final=estimate_input_tokens(final_send),
        compaction_events=len(edits),
        summarizer_calls=len(summary_calls),
        summary_turns=_summary_turn_count(final_send),
        placeholder_results=_placeholder_count(final_send),
        tool_pairs_final=len(_find_tool_pairs(final_send)),
        facts_introduced=introduced,
        facts_retained=len(facts),
        facts_required_latest=latest_ok,
        structurally_valid=_validate_structure(final_send),
        model_calls=len(captured),
        samples=samples,
    )
    return events, report


@pytest.mark.parametrize("turns", _HORIZONS, ids=lambda n: f"summarize-{n}-turns")
def test_long_horizon_summarize_retains_facts_and_structure(turns: int) -> None:
    """Fact-preserving summarizer mock: bounded pairs, all markers retained, valid structure."""
    _events, report = asyncio.run(_run_horizon(turns, "summarize", keep_last_n=2))

    assert report.structurally_valid
    assert report.tool_pairs_final <= 2 + 1  # keep_last_n pairs + possible in-flight
    assert report.summary_turns == report.compaction_events
    assert report.summarizer_calls == report.compaction_events
    assert report.compaction_events >= turns - 2
    assert report.facts_retained >= len(_seed_registry()) + turns
    assert report.facts_required_latest
    assert report.max_message_count <= turns + 8  # summaries + kept pairs (not linear in turns)


@pytest.mark.parametrize("turns", _HORIZONS, ids=lambda n: f"trim-{n}-turns")
def test_long_horizon_trim_bounds_context_and_placeholders(turns: int) -> None:
    """Trim mode: row count grows (transient trim); payloads collapse to placeholders."""
    _events, report = asyncio.run(_run_horizon(turns, "trim", keep_last_n=2))

    assert report.structurally_valid
    assert report.facts_required_latest
    # Trim clears result text only — message pairs remain in the send list (Phase 4 oracle).
    expected_rows = 5 + 2 * turns  # preamble + 2 seed pairs + one pair per agent turn
    assert report.message_count_final == expected_rows
    assert report.tool_pairs_final == 2 + turns
    assert report.placeholder_results >= turns
    assert report.compaction_events == turns
    assert report.summarizer_calls == 0
    assert report.summary_turns == 0
    # Placeholders cap token growth vs retaining full fact payloads on every row.
    assert report.estimated_tokens_final < turns * 80
    assert report.facts_retained <= len(_seed_registry()) + 2


def test_long_horizon_summarize_500_compaction_count_matches_edits() -> None:
    """At max horizon, compaction events must track summarizer invocations 1:1."""
    events, report = asyncio.run(_run_horizon(500, "summarize"))
    edits = sum(isinstance(e, ContextEditEvent) for e in events)
    assert edits == report.compaction_events == report.summarizer_calls
    assert report.summary_turns == edits
