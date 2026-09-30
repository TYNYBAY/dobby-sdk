# ruff: noqa: E402
"""Phase 5: information-loss tests for recovered trim (``edit_context``).

Facts are embedded in tool-result text with explicit markers. Tests apply trim,
then check whether a later turn's model-visible context still contains each
required marker. Scripted providers only; no live LLM calls.
"""

# isort: off
from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from unittest.mock import AsyncMock

import pytest


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
from recovered_dobby.context import edit_context
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

_PLACEHOLDER = "[Tool result cleared to save context.]"
_FACT_RE = re.compile(r"\[DOBBY-FACT:(?P<category>[a-z]+):(?P<key>[^=\]]+)=(?P<value>[^\]]+)\]")

_CATEGORIES = (
    "id",
    "number",
    "name",
    "date",
    "url",
    "config",
    "permission",
    "constraint",
    "error",
    "decision",
    "calculation",
    "relationship",
)

_SAMPLE_VALUES: dict[str, str] = {
    "id": "INV-28491",
    "number": "1847293",
    "name": "Asha Mehta",
    "date": "2026-03-15",
    "url": "https://api.example.com/v2/orders/28491",
    "config": "max_retries=7",
    "permission": "billing:refund:approve",
    "constraint": "NEVER disable audit logging",
    "error": "HTTP 503 upstream timeout after 30s",
    "decision": "approved partial refund 42.50 USD",
    "calculation": "subtotal=100.00 tax=8.25 total=108.25",
    "relationship": "order-28491 belongs_to customer-c-991",
}


@dataclass(frozen=True)
class FactSpec:
    """One labeled fact stored in a specific tool-result pair index."""

    pair_index: int
    category: str
    key: str
    value: str
    is_error: bool = False

    def marker(self) -> str:
        return f"[DOBBY-FACT:{self.category}:{self.key}={self.value}]"


@dataclass(frozen=True)
class RetentionOutcome:
    """Measured retention for one scenario."""

    facts_introduced: int
    facts_retained: int
    facts_discarded: int
    facts_required_later: tuple[str, ...]
    availability: dict[str, bool]
    agent_outcome: str | None = None

    @property
    def all_required_available(self) -> bool:
        return all(self.availability.values())


def _policy(keep_last_n: int, **overrides: Any) -> ContextPolicy:
    return ContextPolicy(keep_last_n=keep_last_n, **overrides)


def _use(call_id: str, *, name: str = "record", inputs: dict[str, Any] | None = None) -> ToolUsePart:
    return ToolUsePart(id=call_id, name=name, inputs={} if inputs is None else inputs)


def _result(
    call_id: str,
    text: str,
    *,
    name: str = "record",
    is_error: bool = False,
) -> ToolResultPart:
    return ToolResultPart(
        tool_use_id=call_id,
        name=name,
        parts=[TextPart(text=text)],
        is_error=is_error,
    )


def _pair(
    pair_index: int,
    payload: str,
    *,
    is_error: bool = False,
    tool_name: str = "record",
) -> tuple[AssistantMessagePart, UserMessagePart]:
    call_id = f"call-{pair_index}"
    return (
        AssistantMessagePart(
            parts=[
                _use(
                    call_id,
                    name=tool_name,
                    inputs={"pair": pair_index, "query": f"step-{pair_index}"},
                )
            ]
        ),
        UserMessagePart(parts=[_result(call_id, payload, name=tool_name, is_error=is_error)]),
    )


def _history_from_facts(facts: list[FactSpec], *, filler_pairs: int = 0) -> list[Any]:
    """Build A..N pairs; multiple facts may share one result payload."""
    max_index = max((f.pair_index for f in facts), default=-1)
    total = max(max_index, filler_pairs - 1) + 1
    messages: list[Any] = []
    facts_by_index: dict[int, list[FactSpec]] = {}
    for fact in facts:
        facts_by_index.setdefault(fact.pair_index, []).append(fact)
    for index in range(total):
        if index in facts_by_index:
            group = facts_by_index[index]
            payload = "\n".join(f.marker() for f in group)
            messages.extend(
                _pair(
                    index,
                    payload,
                    is_error=any(f.is_error for f in group),
                )
            )
        else:
            messages.extend(_pair(index, f"filler-payload-{index}"))
    return messages


def _result_texts(messages: list[Any]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, ToolResultPart):
                texts.append("".join(p.text for p in part.parts if isinstance(p, TextPart)))
    return texts


def _context_blob(messages: list[Any]) -> str:
    """All model-visible tool-result text after trim (what the LLM reads)."""
    return "\n".join(_result_texts(messages))


def _facts_in_blob(blob: str) -> set[str]:
    found: set[str] = set()
    for match in _FACT_RE.finditer(blob):
        found.add(f"{match.group('category')}:{match.group('key')}")
    return found


def _fact_key(category: str, key: str) -> str:
    return f"{category}:{key}"


def _measure(
    messages: list[Any],
    facts: list[FactSpec],
    *,
    keep_last_n: int,
    required: list[FactSpec],
) -> RetentionOutcome:
    edited, _applied = edit_context(messages, _policy(keep_last_n))
    blob = _context_blob(edited)
    present = _facts_in_blob(blob)
    all_keys = [_fact_key(f.category, f.key) for f in facts]
    retained = sum(1 for key in all_keys if key in present)
    required_keys = [_fact_key(f.category, f.key) for f in required]
    availability = {key: key in present for key in required_keys}
    return RetentionOutcome(
        facts_introduced=len(facts),
        facts_retained=retained,
        facts_discarded=len(facts) - retained,
        facts_required_later=tuple(required_keys),
        availability=availability,
    )


def _assert_outcome(
    outcome: RetentionOutcome,
    *,
    expect_available: dict[str, bool],
    scenario: str,
) -> None:
    for key, expect in expect_available.items():
        got = outcome.availability[key]
        assert got is expect, (
            f"{scenario} required={key} available={got} expected={expect} "
            f"introduced={outcome.facts_introduced} retained={outcome.facts_retained} "
            f"discarded={outcome.facts_discarded}"
        )


@pytest.mark.parametrize("category", _CATEGORIES, ids=lambda c: f"fact-category-{c}")
def test_trimmed_fact_not_in_later_context(category: str) -> None:
    """Facts in cleared pairs are absent from the trimmed view (expected loss)."""
    value = _SAMPLE_VALUES[category]
    facts = [FactSpec(0, category, "primary", value)]
    filler = _history_from_facts(facts, filler_pairs=5)
    required = facts
    outcome = _measure(filler, facts, keep_last_n=2, required=required)
    _assert_outcome(
        outcome,
        expect_available={_fact_key(category, "primary"): False},
        scenario="trimmed-then-required",
    )
    assert outcome.facts_discarded == 1
    assert outcome.facts_retained == 0


@pytest.mark.parametrize("category", _CATEGORIES, ids=lambda c: f"fact-category-{c}")
def test_retained_fact_still_in_later_context(category: str) -> None:
    """Facts in kept pairs remain available after trim."""
    value = _SAMPLE_VALUES[category]
    facts = [FactSpec(4, category, "primary", value)]
    messages = _history_from_facts(facts, filler_pairs=5)
    outcome = _measure(messages, facts, keep_last_n=2, required=facts)
    _assert_outcome(
        outcome,
        expect_available={_fact_key(category, "primary"): True},
        scenario="retained-then-required",
    )
    assert outcome.facts_retained == 1
    assert outcome.facts_discarded == 0


def test_split_required_facts_across_trimmed_and_retained() -> None:
    """Later work needs markers from both a cleared pair and a kept pair."""
    id_fact = FactSpec(0, "id", "order", "ORD-7001")
    name_fact = FactSpec(4, "name", "owner", "Jordan Lee")
    rel_fact = FactSpec(4, "relationship", "link", "ORD-7001 owned_by Jordan Lee")
    facts = [id_fact, name_fact, rel_fact]
    messages = _history_from_facts(facts, filler_pairs=5)
    required = [id_fact, name_fact, rel_fact]
    outcome = _measure(messages, facts, keep_last_n=2, required=required)
    _assert_outcome(
        outcome,
        expect_available={
            _fact_key("id", "order"): False,
            _fact_key("name", "owner"): True,
            _fact_key("relationship", "link"): True,
        },
        scenario="split-across-window",
    )
    assert outcome.all_required_available is False


def test_early_critical_constraint_required_much_later() -> None:
    """Constraint introduced on pair 0 is gone when only the last two pairs stay."""
    constraint = FactSpec(0, "constraint", "policy", _SAMPLE_VALUES["constraint"])
    late_need = FactSpec(0, "constraint", "policy", _SAMPLE_VALUES["constraint"])
    messages = _history_from_facts([constraint], filler_pairs=8)
    outcome = _measure(messages, [constraint], keep_last_n=2, required=[late_need])
    _assert_outcome(
        outcome,
        expect_available={_fact_key("constraint", "policy"): False},
        scenario="early-constraint-late-need",
    )


def test_failed_operation_error_fact_trimmed_then_required() -> None:
    """Error payload in a cleared pair is not visible later (expected loss)."""
    err = FactSpec(
        1,
        "error",
        "upstream",
        _SAMPLE_VALUES["error"],
        is_error=True,
    )
    messages = _history_from_facts([err], filler_pairs=5)
    outcome = _measure(messages, [err], keep_last_n=2, required=[err])
    _assert_outcome(
        outcome,
        expect_available={_fact_key("error", "upstream"): False},
        scenario="error-trimmed-then-required",
    )
    edited, _ = edit_context(messages, _policy(2))
    result_parts = [
        p
        for m in edited
        if isinstance(m, UserMessagePart)
        for p in m.parts
        if isinstance(p, ToolResultPart) and p.tool_use_id == "call-1"
    ]
    assert len(result_parts) == 1
    assert result_parts[0].is_error is True
    assert _result_texts(edited)[1] == _PLACEHOLDER


def test_tool_call_inputs_do_not_substitute_for_cleared_result_facts() -> None:
    """Trim clears result text; tool-call inputs are unchanged but lack result facts."""
    secret = FactSpec(0, "config", "api_key", "sk-live-abcdef")
    messages = _history_from_facts([secret], filler_pairs=5)
    edited, applied = edit_context(messages, _policy(2))
    assert applied is not None
    blob = _context_blob(edited)
    assert _fact_key("config", "api_key") not in _facts_in_blob(blob)
    use_parts = [
        p
        for m in edited
        if isinstance(m, AssistantMessagePart)
        for p in m.parts
        if isinstance(p, ToolUsePart) and p.id == "call-0"
    ]
    assert use_parts[0].inputs["query"] == "step-0"
    assert "sk-live" not in json.dumps(use_parts[0].inputs)


class _RecordTool(Tool):
    name = "record"
    description = "Record a labeled fact blob."

    def __call__(self, label: Annotated[str, "Step label"]) -> dict[str, str]:
        return {"label": label}


class _VerifyTool(Tool):
    name = "verify"
    description = "Echo verification payload from the model."

    def __call__(self, required_keys: Annotated[list[str], "Fact keys"]) -> dict[str, Any]:
        return {"required_keys": required_keys, "status": "ok"}


def _usage(tokens: int) -> Usage:
    return Usage(input_tokens=tokens, output_tokens=0, total_tokens=tokens)


def _scripted_executor_turns(
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


def _drive(
    policy: ContextPolicy,
    turns: list[tuple[list[Any], Usage | None]],
    messages: list[Any],
    tools: list[Tool],
) -> tuple[list[Any], list[list[Any]]]:
    captured: list[list[Any]] = []
    executor = AgentExecutor(
        "openai",
        _scripted_executor_turns(turns, captured),
        tools=tools,
        context_policy=policy,
    )

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=len(turns)):
            events.append(event)
        return events

    events = asyncio.run(run())
    return events, captured


def _oracle_agent_can_satisfy_verify(captured_messages: list[Any], required: list[FactSpec]) -> bool:
    """Deterministic stand-in for a perfect agent: all required markers are visible."""
    blob = _context_blob(captured_messages)
    present = _facts_in_blob(blob)
    return all(_fact_key(f.category, f.key) in present for f in required)


def test_executor_trimmed_view_blocks_verify_for_cleared_fact() -> None:
    """After auto-trim, the next model call cannot see facts from cleared pairs."""
    constraint = FactSpec(0, "constraint", "global", _SAMPLE_VALUES["constraint"])
    history = _history_from_facts([constraint], filler_pairs=5)
    policy = ContextPolicy(context_window=128_000, trigger_pct=0.8, keep_last_n=2, mode="trim")
    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="t1", name="record", inputs={"label": "s1"})], _usage(trigger)),
        (
            [
                ToolUsePart(
                    id="t2",
                    name="verify",
                    inputs={"required_keys": [_fact_key(constraint.category, constraint.key)]},
                )
            ],
            _usage(trigger),
        ),
        ([], _usage(trigger)),
    ]
    events, captured = _drive(policy, turns, history, [_RecordTool(), _VerifyTool()])
    edits = [e for e in events if isinstance(e, ContextEditEvent)]
    assert len(edits) >= 1
    assert len(captured) >= 2
    verify_call = captured[1]
    assert _oracle_agent_can_satisfy_verify(verify_call, [constraint]) is False


def test_executor_retained_fact_allows_deterministic_verify() -> None:
    """Verification succeeds when every required marker sits in the keep window."""
    calc = FactSpec(2, "calculation", "total", _SAMPLE_VALUES["calculation"])
    history = _history_from_facts([calc], filler_pairs=3)
    policy = ContextPolicy(context_window=128_000, trigger_pct=0.8, keep_last_n=2, mode="trim")
    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="t1", name="record", inputs={"label": "s1"})], _usage(trigger)),
        (
            [
                ToolUsePart(
                    id="t2",
                    name="verify",
                    inputs={"required_keys": [_fact_key(calc.category, calc.key)]},
                )
            ],
            _usage(trigger),
        ),
        ([], _usage(trigger)),
    ]
    events, captured = _drive(policy, turns, history, [_RecordTool(), _VerifyTool()])
    assert len(captured) >= 2
    second_call = captured[1]
    assert _oracle_agent_can_satisfy_verify(second_call, [calc]) is True
    assert any(isinstance(e, ContextEditEvent) for e in events)
