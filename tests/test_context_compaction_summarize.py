# ruff: noqa: E402
"""Phase 6: summarization tests for recovered ``summarize_context``.

Uses deterministic mock summarizers (no live providers). Facts use
``[DOBBY-FACT:category:key=value]`` markers in tool-result text.
"""

# isort: off
from __future__ import annotations

import asyncio
import copy
import re
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.compaction_subject import load_recovered

load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context import SUMMARIZE_PROMPT, summarize_context
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

_ARCHIVE_CATEGORIES = (
    "id",
    "number",
    "decision",
    "constraint",
    "negative",
)

_ARCHIVE_VALUES: dict[str, str] = {
    "id": "ORD-9001",
    "number": "250000",
    "decision": "reject wire transfer",
    "constraint": "max_single_transfer=10000",
    "negative": "do_not_retry_upstream=true",
}


@dataclass(frozen=True)
class FactSpec:
    pair_index: int
    category: str
    key: str
    value: str

    def marker(self) -> str:
        return f"[DOBBY-FACT:{self.category}:{self.key}={self.value}]"


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


def _history_from_facts(
    facts: list[FactSpec],
    *,
    filler_pairs: int,
    preamble: str | None = None,
) -> list[Any]:
    max_index = max((f.pair_index for f in facts), default=-1)
    total = max(max_index, filler_pairs - 1) + 1
    by_index: dict[int, list[FactSpec]] = {}
    for fact in facts:
        by_index.setdefault(fact.pair_index, []).append(fact)
    messages: list[Any] = []
    if preamble is not None:
        messages.append(UserMessagePart(parts=[TextPart(text=preamble)]))
    for index in range(total):
        if index in by_index:
            payload = "\n".join(f.marker() for f in by_index[index])
            messages.extend(_pair(index, payload))
        else:
            messages.extend(_pair(index, f"filler-{index}"))
    return messages


def _tool_result_texts(messages: list[Any]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, ToolResultPart):
                texts.append("".join(p.text for p in part.parts if isinstance(p, TextPart)))
            elif isinstance(part, TextPart):
                texts.append(part.text)
    return texts


def _context_blob(messages: list[Any]) -> str:
    return "\n".join(_tool_result_texts(messages))


def _fact_keys_in_blob(blob: str) -> set[str]:
    return {f"{m.group('category')}:{m.group('key')}" for m in _FACT_RE.finditer(blob)}


def _summary_messages(messages: list[Any]) -> list[str]:
    return [
        part.text
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, TextPart) and part.text.startswith("<summary>")
    ]


def _extract_markers_from_span_text(span_text: str) -> str:
    markers = [m.group(0) for m in _FACT_RE.finditer(span_text)]
    return " ".join(markers)


def _fact_preserving_llm(
    *,
    summary_calls: list[dict[str, Any]] | None = None,
    fixed_digest: str | None = None,
) -> AsyncMock:
    """Non-streaming: echo all DOBBY-FACT markers found in the span (deterministic)."""

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", True) is False:
            record = {
                "messages": copy.deepcopy(messages),
                "system_prompt": kwargs.get("system_prompt"),
            }
            if summary_calls is not None:
                summary_calls.append(record)
            span_text = " ".join(
                part.text
                for message in messages
                if isinstance(message, UserMessagePart)
                for part in message.parts
                if isinstance(part, TextPart)
            )
            if fixed_digest is not None:
                digest = fixed_digest
            else:
                digest = _extract_markers_from_span_text(span_text) or "empty-span"
            return StreamEndEvent(
                type="stream_end",
                model="mock-summarizer",
                parts=[TextPart(text=digest)],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )
        raise AssertionError("unexpected streaming call in summarize unit test")

    provider = AsyncMock()
    provider.chat = mock_chat
    provider.name = "deterministic-summarizer"
    return provider


async def _summarize(
    messages: list[Any],
    keep_last_n: int,
    llm: Any,
    *,
    extra_instructions: str | None = None,
) -> tuple[list[Any], Any]:
    policy = ContextPolicy(keep_last_n=keep_last_n, mode="summarize")
    applied = await summarize_context(
        messages,
        policy,
        llm,
        extra_instructions=extra_instructions,
    )
    return messages, applied


def test_summarize_replaces_old_span_with_summary_message() -> None:
    """Clearable tool pairs become one ``<summary>`` user turn."""
    messages = _history_from_facts([], filler_pairs=5)
    before_len = len(messages)
    llm = _fact_preserving_llm()

    async def run() -> Any:
        return await _summarize(messages, 2, llm)

    working, applied = asyncio.run(run())
    assert applied is not None
    assert applied.type == "summarize"
    assert applied.cleared_tool_uses == 3
    assert len(working) == before_len - 6 + 1
    summaries = _summary_messages(working)
    assert len(summaries) == 1
    assert summaries[0].startswith("<summary>")
    assert summaries[0].endswith("</summary>")


def test_summarize_preserves_recent_tool_pairs_verbatim() -> None:
    """Pairs newer than ``keep_last_n`` stay out of the summarized span."""
    recent = FactSpec(4, "id", "recent", "KEEP-ME")
    messages = _history_from_facts([recent], filler_pairs=5)
    llm = _fact_preserving_llm()

    async def run() -> None:
        return await _summarize(messages, 2, llm)

    working, applied = asyncio.run(run())
    assert applied is not None
    texts = _tool_result_texts(working)
    assert recent.marker() in texts[-1]
    assert "filler-3" in texts[-2]
    assert all("filler-0" not in text and "filler-1" not in text for text in texts[:2])


def test_summarize_does_not_mutate_stashed_part_objects() -> None:
    """Replaced message parts keep their original text; only list slots change."""
    fact = FactSpec(0, "id", "stash", "OBJ-1")
    messages = _history_from_facts([fact], filler_pairs=4)
    cleared_result = messages[1].parts[0]
    assert isinstance(cleared_result, ToolResultPart)
    inner = cleared_result.parts[0]
    assert isinstance(inner, TextPart)
    original_text = inner.text
    llm = _fact_preserving_llm()

    async def run() -> Any:
        return await _summarize(messages, 1, llm)

    working, applied = asyncio.run(run())
    assert applied is not None
    assert applied.replaced_originals is not None
    stashed = applied.replaced_originals[1]
    assert isinstance(stashed, UserMessagePart)
    stashed_text = stashed.parts[0].parts[0].text
    assert stashed_text == original_text == fact.marker()
    assert inner.text == original_text
    assert len(working) < 4 + 2


def test_summarize_mutates_working_list_in_place() -> None:
    """Write-back replaces a slice of the live list passed in."""
    messages = _history_from_facts([], filler_pairs=4)
    same = messages
    llm = _fact_preserving_llm()

    async def run() -> None:
        return await _summarize(same, 1, llm)

    asyncio.run(run())
    assert same is messages
    assert len(messages) == 1 + 2 * 1


def test_summarize_noop_when_nothing_older_than_keep_window() -> None:
    messages = _history_from_facts([], filler_pairs=2)
    llm = _fact_preserving_llm()

    async def run() -> Any:
        return await _summarize(messages, 2, llm)

    working, applied = asyncio.run(run())
    assert applied is None
    assert len(working) == 4


@pytest.mark.parametrize("category", _ARCHIVE_CATEGORIES, ids=lambda c: f"archive-{c}")
def test_deterministic_summary_retains_old_span_markers(category: str) -> None:
    """Ideal summarizer mock copies archived facts from the summarized span."""
    value = _ARCHIVE_VALUES[category]
    fact = FactSpec(0, category, "primary", value)
    messages = _history_from_facts([fact], filler_pairs=5)
    llm = _fact_preserving_llm()

    async def run() -> Any:
        return await _summarize(messages, 2, llm)

    working, applied = asyncio.run(run())
    assert applied is not None
    blob = _context_blob(working)
    assert f"{category}:primary" in _fact_keys_in_blob(blob)
    assert value in blob


def test_relationship_and_conditional_facts_in_summary() -> None:
    """Relationships and conditional markers in the old span appear in summary text."""
    rel = FactSpec(0, "relationship", "link", "ORD-1 owned_by C-9")
    cond = FactSpec(1, "conditional", "refund", "allowed_if_ticket=TCK-88")
    messages = _history_from_facts([rel, cond], filler_pairs=4)
    llm = _fact_preserving_llm()

    async def run() -> Any:
        return await _summarize(messages, 1, llm)

    working, applied = asyncio.run(run())
    assert applied is not None
    keys = _fact_keys_in_blob(_context_blob(working))
    assert "relationship:link" in keys
    assert "conditional:refund" in keys


def test_summary_usable_as_replacement_context_for_later_turn() -> None:
    """Facts only in the summarized region remain reachable via ``<summary>``."""
    archived = FactSpec(0, "decision", "pick", "use vendor B")
    recent = FactSpec(4, "number", "qty", "12")
    messages = _history_from_facts([archived, recent], filler_pairs=5)
    llm = _fact_preserving_llm()

    async def run() -> Any:
        return await _summarize(messages, 2, llm)

    working, applied = asyncio.run(run())
    assert applied is not None
    blob = _context_blob(working)
    keys = _fact_keys_in_blob(blob)
    assert "decision:pick" in keys
    assert "number:qty" in keys
    assert "use vendor B" in blob
    assert recent.marker() in blob


def test_compare_original_vs_post_summary_fact_coverage() -> None:
    """Report loss when summarizer omits markers; full coverage when mock preserves them."""
    facts = [
        FactSpec(0, "id", "a", "A-1"),
        FactSpec(1, "number", "b", "42"),
        FactSpec(4, "constraint", "c", "cap=99"),
    ]
    messages = _history_from_facts(facts, filler_pairs=5)
    original_keys = _fact_keys_in_blob(_context_blob(messages))

    async def run_good() -> set[str]:
        good_msgs = _history_from_facts(facts, filler_pairs=5)
        await _summarize(good_msgs, 2, _fact_preserving_llm())
        return _fact_keys_in_blob(_context_blob(good_msgs))

    async def run_bad() -> set[str]:
        bad_msgs = _history_from_facts(facts, filler_pairs=5)
        await _summarize(
            bad_msgs,
            2,
            _fact_preserving_llm(fixed_digest="unrelated chatter"),
        )
        return _fact_keys_in_blob(_context_blob(bad_msgs))

    after_good = asyncio.run(run_good())
    after_bad = asyncio.run(run_bad())

    assert original_keys == {"id:a", "number:b", "constraint:c"}
    assert after_good == original_keys
    assert after_bad == {"constraint:c"}
    assert "id:a" not in after_bad and "number:b" not in after_bad


def test_imperfect_summarizer_causes_information_loss() -> None:
    """Generic digest drops archived facts; kept pair facts still verbatim."""
    archived = FactSpec(0, "id", "lost", "GONE-1")
    kept = FactSpec(3, "id", "kept", "HERE-9")
    messages = _history_from_facts([archived, kept], filler_pairs=4)
    llm = _fact_preserving_llm(fixed_digest="summary with no markers")

    async def run() -> Any:
        return await _summarize(messages, 1, llm)

    working, applied = asyncio.run(run())
    assert applied is not None
    keys = _fact_keys_in_blob(_context_blob(working))
    assert "id:lost" not in keys
    assert "id:kept" in keys


def test_preamble_outside_tool_pairs_is_not_summarized() -> None:
    """User preamble before tool history survives summarize unchanged."""
    messages = _history_from_facts([], filler_pairs=4, preamble="user question")
    preamble_msg = messages[0]
    llm = _fact_preserving_llm()

    async def run() -> Any:
        return await _summarize(messages, 1, llm)

    working, _applied = asyncio.run(run())
    assert working[0] is preamble_msg


def test_extra_instructions_forwarded_to_summarizer() -> None:
    summary_calls: list[dict[str, Any]] = []
    messages = _history_from_facts([], filler_pairs=4)
    llm = _fact_preserving_llm(summary_calls=summary_calls)

    async def run() -> None:
        return await _summarize(
            messages,
            1,
            llm,
            extra_instructions="Keep ticket IDs",
        )

    asyncio.run(run())
    assert summary_calls
    prompt = summary_calls[0]["system_prompt"]
    assert SUMMARIZE_PROMPT in prompt
    assert "Additional instructions: Keep ticket IDs" in prompt


def test_inflight_tool_use_stays_outside_summarized_span() -> None:
    """Trailing tool call without a result is not part of the summarized span."""
    messages = _history_from_facts([], filler_pairs=4)
    inflight = _use("call-open", pair=99)
    messages.append(AssistantMessagePart(parts=[inflight]))
    llm = _fact_preserving_llm()

    async def run() -> Any:
        return await _summarize(messages, 1, llm)

    working, applied = asyncio.run(run())
    assert applied is not None
    assert applied.cleared_tool_uses == 3
    assert _use_parts(working)[-1].id == "call-open"


def _use_parts(messages: list[Any]) -> list[ToolUsePart]:
    return [
        part
        for message in messages
        if isinstance(message, AssistantMessagePart)
        for part in message.parts
        if isinstance(part, ToolUsePart)
    ]


class _NoopTool(Tool):
    name = "noop"
    description = "Ack."

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


def _usage(tokens: int) -> Usage:
    return Usage(input_tokens=tokens, output_tokens=0, total_tokens=tokens)


def _tool_call(call_id: str) -> ToolUsePart:
    return ToolUsePart(id=call_id, name="noop", inputs={})


def _executor_summarizing_provider(
    turns: list[tuple[list[Any], Usage | None]],
    captured: list[list[Any]],
    summary_calls: list[dict[str, Any]],
) -> Any:
    call_count = 0

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        if kwargs.get("stream", True) is False:
            summary_calls.append(
                {"messages": copy.deepcopy(messages), "system_prompt": kwargs.get("system_prompt")}
            )
            span_text = " ".join(
                part.text
                for message in messages
                if isinstance(message, UserMessagePart)
                for part in message.parts
                if isinstance(part, TextPart)
            )
            digest = _extract_markers_from_span_text(span_text)
            return StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=[TextPart(text=digest or "empty")],
                stop_reason="end_turn",
                usage=_usage(0),
            )
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


def test_executor_summarize_write_back_visible_on_next_model_call() -> None:
    """Executor summarize mode injects ``<summary>`` before the next agent turn."""
    archived = FactSpec(0, "constraint", "rule", "never delete prod")
    history = _history_from_facts([archived], filler_pairs=4)
    policy = ContextPolicy(context_window=128_000, trigger_pct=0.8, keep_last_n=1, mode="summarize")
    trigger = policy.trigger_tokens
    captured: list[list[Any]] = []
    summary_calls: list[dict[str, Any]] = []
    turns = [
        ([_tool_call("t1")], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    provider = _executor_summarizing_provider(turns, captured, summary_calls)
    executor = AgentExecutor(
        "openai",
        provider,
        tools=[_NoopTool()],
        context_policy=policy,
    )

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(history, max_iterations=2):
            events.append(event)
        return events

    events = asyncio.run(run())
    edits = [e for e in events if isinstance(e, ContextEditEvent)]
    assert len(edits) == 1
    assert len(captured) >= 2
    second_call_blob = _context_blob(captured[1])
    assert "<summary>" in second_call_blob
    assert "constraint:rule" in _fact_keys_in_blob(second_call_blob)
    assert len(summary_calls) == 1
