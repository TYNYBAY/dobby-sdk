# ruff: noqa: E402
"""Phase 8: repeated summarization / compaction cycles (deterministic mocks)."""

# isort: off
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.compaction_subject import load_recovered

load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context import summarize_context
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

_FACT_RE = re.compile(r"\[DOBBY-FACT:(?P<category>[a-z]+):(?P<key>[^=\]]+)=(?P<value>[^\]]+)\]")

_CYCLE_COUNTS = (1, 2, 5, 10, 21)


@dataclass(frozen=True)
class CycleReport:
    cycles: int
    summarize_calls: int
    summary_messages: int
    tool_pairs: int
    facts_expected: dict[str, str]
    facts_observed: dict[str, str]
    hallucinated_keys: frozenset[str]
    drifted_keys: frozenset[str]
    structurally_valid: bool


def _use(call_id: str, *, pair: int) -> ToolUsePart:
    return ToolUsePart(id=call_id, name="record", inputs={"pair": pair})


def _result(call_id: str, text: str) -> ToolResultPart:
    return ToolResultPart(
        tool_use_id=call_id,
        name="record",
        parts=[TextPart(text=text)],
    )


def _pair(index: int, text: str) -> tuple[AssistantMessagePart, UserMessagePart]:
    call_id = f"call-{index}"
    return (
        AssistantMessagePart(parts=[_use(call_id, pair=index)]),
        UserMessagePart(parts=[_result(call_id, text)]),
    )


def _marker(category: str, key: str, value: str) -> str:
    return f"[DOBBY-FACT:{category}:{key}={value}]"


def _seed_registry() -> dict[str, str]:
    return {
        "id:root": "ROOT-9000",
        "number:baseline": "314159",
        "decision:policy": "approve tier-2 only",
        "constraint:cap": "max_batch=500",
        "relationship:graph": "node-A parent_of node-B",
    }


def _seed_messages(registry: dict[str, str]) -> list[Any]:
    payload = "\n".join(
        _marker(cat_key.split(":")[0], cat_key.split(":")[1], val)
        for cat_key, val in registry.items()
    )
    messages: list[Any] = [
        UserMessagePart(parts=[TextPart(text="workflow question")]),
        *_pair(0, payload),
        *_pair(1, "filler-bridge"),
    ]
    return messages


def _facts_in_context(messages: list[Any]) -> dict[str, str]:
    blob = "\n".join(
        part.text
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, TextPart)
    )
    blob += "\n" + "\n".join(
        "".join(p.text for p in part.parts if isinstance(p, TextPart))
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    )
    observed: dict[str, str] = {}
    for match in _FACT_RE.finditer(blob):
        key = f"{match.group('category')}:{match.group('key')}"
        observed[key] = match.group("value")
    return observed


def _summary_message_count(messages: list[Any]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, TextPart) and part.text.startswith("<summary>")
    )


def _latest_summary_inner(messages: list[Any]) -> str:
    latest = ""
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, TextPart) and part.text.startswith("<summary>"):
                latest = part.text
    if not latest:
        return ""
    return latest.removeprefix("<summary>").removesuffix("</summary>")


def _facts_in_text(text: str) -> dict[str, str]:
    observed: dict[str, str] = {}
    for match in _FACT_RE.finditer(text):
        key = f"{match.group('category')}:{match.group('key')}"
        observed[key] = match.group("value")
    return observed


def _validate_structure(messages: list[Any]) -> bool:
    pairs = _find_tool_pairs(messages)
    used_indices = {idx for pair in pairs for idx in pair}
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
    for index, message in enumerate(messages):
        if index in used_indices:
            continue
        if isinstance(message, UserMessagePart) and any(
            isinstance(p, TextPart) and p.text.startswith("<summary>") for p in message.parts
        ):
            continue
        if isinstance(message, UserMessagePart) and any(
            isinstance(p, TextPart) and not p.text.startswith("<summary>") for p in message.parts
        ):
            continue
        if isinstance(message, AssistantMessagePart) and any(
            isinstance(p, ToolUsePart) for p in message.parts
        ):
            # in-flight tool call without result
            if index + 1 >= len(messages) or not _find_tool_pairs(messages[index : index + 2]):
                if index == len(messages) - 1:
                    continue
            return False
    return True


def _extract_markers_from_span_text(span_text: str) -> str:
    return " ".join(m.group(0) for m in _FACT_RE.finditer(span_text))


def _llm_fact_preserving(summary_calls: list[dict[str, Any]]) -> AsyncMock:
    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", True) is False:
            span_text = " ".join(
                part.text
                for message in messages
                if isinstance(message, UserMessagePart)
                for part in message.parts
                if isinstance(part, TextPart)
            )
            summary_calls.append({"span_text": span_text})
            digest = _extract_markers_from_span_text(span_text) or "empty-span"
            return StreamEndEvent(
                type="stream_end",
                model="mock-summarizer",
                parts=[TextPart(text=digest)],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )
        raise AssertionError("unexpected streaming call")

    provider = AsyncMock()
    provider.chat = mock_chat
    return provider


def _llm_last_marker_only(summary_calls: list[dict[str, Any]]) -> AsyncMock:
    """Lossy mock: keeps only the last DOBBY-FACT marker in the span (simulates degradation)."""

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", True) is False:
            span_text = " ".join(
                part.text
                for message in messages
                if isinstance(message, UserMessagePart)
                for part in message.parts
                if isinstance(part, TextPart)
            )
            summary_calls.append({"span_text": span_text})
            markers = [m.group(0) for m in _FACT_RE.finditer(span_text)]
            digest = markers[-1] if markers else "empty-span"
            return StreamEndEvent(
                type="stream_end",
                model="mock-lossy",
                parts=[TextPart(text=digest)],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )
        raise AssertionError("unexpected streaming call")

    provider = AsyncMock()
    provider.chat = mock_chat
    return provider


async def _run_cycles(
    cycles: int,
    llm: Any,
    summary_calls: list[dict[str, Any]],
    *,
    keep_last_n: int = 1,
) -> tuple[list[Any], CycleReport]:
    registry = _seed_registry()
    messages = _seed_messages(registry)
    policy = ContextPolicy(keep_last_n=keep_last_n, mode="summarize")
    next_pair_index = 2
    successful = 0
    for cycle in range(cycles):
        key = f"cycle-{cycle}"
        registry[f"id:{key}"] = f"CYC-{cycle:03d}"
        payload = _marker("id", key, registry[f"id:{key}"])
        messages.extend(_pair(next_pair_index, payload))
        next_pair_index += 1
        applied = await summarize_context(messages, policy, llm)
        assert applied is not None, f"summarize no-op on cycle {cycle}"
        successful += 1

    observed = _facts_in_context(messages)
    expected = dict(registry)
    hallucinated = frozenset(observed.keys() - expected.keys())
    drifted = frozenset(
        key for key, value in observed.items() if key in expected and expected[key] != value
    )
    report = CycleReport(
        cycles=cycles,
        summarize_calls=successful,
        summary_messages=_summary_message_count(messages),
        tool_pairs=len(_find_tool_pairs(messages)),
        facts_expected=expected,
        facts_observed=observed,
        hallucinated_keys=hallucinated,
        drifted_keys=drifted,
        structurally_valid=_validate_structure(messages),
    )
    assert len(summary_calls) == cycles
    return messages, report


class _NoopTool(Tool):
    name = "noop"
    description = "Ack."

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


def _tool_call(call_id: str) -> ToolUsePart:
    return ToolUsePart(id=call_id, name="noop", inputs={})


def _history() -> list[Any]:
    registry = _seed_registry()
    return _seed_messages(registry)


def _usage(tokens: int) -> Usage:
    return Usage(input_tokens=tokens, output_tokens=0, total_tokens=tokens)


def _executor_provider_for_cycles(
    compactions: int,
    captured: list[list[Any]],
    summary_calls: list[dict[str, Any]],
) -> Any:
    """Agent turns with monotonically increasing usage so each turn can summarize again."""
    call_count = 0
    base = 200_000

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        if kwargs.get("stream", True) is False:
            span_text = " ".join(
                part.text
                for message in messages
                if isinstance(message, UserMessagePart)
                for part in message.parts
                if isinstance(part, TextPart)
            )
            summary_calls.append({"span_text": span_text})
            digest = _extract_markers_from_span_text(span_text) or "empty-span"
            return StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=[TextPart(text=digest)],
                stop_reason="end_turn",
                usage=_usage(0),
            )
        idx = min(call_count, compactions)
        call_count += 1
        captured.append(list(messages))
        parts = [_tool_call(f"t{idx}")] if idx < compactions else []
        usage = _usage(base + idx * 50_000)

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
    return provider


@pytest.mark.parametrize("cycles", _CYCLE_COUNTS, ids=lambda n: f"repeated-cycles-{n}")
def test_repeated_summarize_preserves_facts_without_drift(cycles: int) -> None:
    """Fact-preserving mock: no hallucination, no value drift, structure stays valid."""
    summary_calls: list[dict[str, Any]] = []
    llm = _llm_fact_preserving(summary_calls)
    _messages, report = asyncio.run(_run_cycles(cycles, llm, summary_calls))

    assert report.structurally_valid, f"invalid structure after {cycles} cycles"
    assert report.summary_messages == cycles
    assert report.tool_pairs == 1
    assert report.summarize_calls == cycles
    assert not report.hallucinated_keys, f"hallucinated keys: {report.hallucinated_keys}"
    assert not report.drifted_keys, f"drifted values: {report.drifted_keys}"
    assert set(report.facts_expected.keys()).issubset(set(report.facts_observed.keys()))
    for key, expected in report.facts_expected.items():
        assert report.facts_observed[key] == expected
    assert "DOBBY-FACT:id:root" in summary_calls[0]["span_text"]
    if cycles > 1:
        assert not any("DOBBY-FACT:id:root" in call["span_text"] for call in summary_calls[1:])


@pytest.mark.parametrize("cycles", (5, 10, 21), ids=lambda n: f"lossy-cycles-{n}")
def test_lossy_repeated_summarize_shows_summary_of_summary_degradation(cycles: int) -> None:
    """Lossy mock drops older markers each cycle — documents drift, not a plumbing defect."""
    summary_calls: list[dict[str, Any]] = []
    llm = _llm_last_marker_only(summary_calls)
    messages, report = asyncio.run(_run_cycles(cycles, llm, summary_calls))

    assert report.structurally_valid
    assert report.summary_messages == cycles
    latest_facts = _facts_in_text(_latest_summary_inner(messages))
    assert len(latest_facts) <= 1
    assert "id:root" not in latest_facts
    assert any(key.startswith("id:cycle-") for key in latest_facts)
    assert "id:root" not in report.facts_observed
    assert "number:baseline" not in report.facts_observed
    assert len(report.facts_observed) < len(report.facts_expected)


def test_context_usable_after_many_cycles_oracle() -> None:
    """After 21 cycles, a perfect agent can still read every canonical fact from context."""
    summary_calls: list[dict[str, Any]] = []
    llm = _llm_fact_preserving(summary_calls)
    _messages, report = asyncio.run(_run_cycles(21, llm, summary_calls))
    assert all(key in report.facts_observed for key in report.facts_expected)


@pytest.mark.parametrize("compactions", (1, 2, 5, 10, 21), ids=lambda n: f"executor-compactions-{n}")
def test_executor_repeated_summarize_with_increasing_tokens(compactions: int) -> None:
    """Executor path: ``compactions`` summarize events when usage increases each turn."""
    policy = ContextPolicy(context_window=100_000, trigger_pct=0.8, keep_last_n=1, mode="summarize")
    captured: list[list[Any]] = []
    summary_calls: list[dict[str, Any]] = []
    provider = _executor_provider_for_cycles(compactions, captured, summary_calls)
    executor = AgentExecutor("openai", provider, tools=[_NoopTool()], context_policy=policy)

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(
            _history(),
            max_iterations=compactions + 1,
        ):
            events.append(event)
        return events

    events = asyncio.run(run())
    edits = [event for event in events if isinstance(event, ContextEditEvent)]
    assert len(edits) == compactions
    assert len(summary_calls) == compactions
    if compactions > 0:
        last_send = captured[-1]
        observed = _facts_in_context(last_send)
        assert "id:root" in observed or compactions == 1
