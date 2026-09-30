# ruff: noqa: E402
"""Phase 9: token accounting and compaction trigger behavior (recovered executor)."""

# isort: off
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.compaction_subject import load_recovered

load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context._tokens import estimate_input_tokens
from recovered_dobby.executor import _compaction_triggered
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


class _NoopTool(Tool):
    name = "noop"
    description = "Ack."

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


def _huge_tool(chars: int) -> Tool:
    class _HugeTool(Tool):
        name = "huge"
        description = "Return a large blob."

        def __call__(self) -> dict[str, str]:
            return {"blob": "H" * chars}

    return _HugeTool()


def _usage(input_tokens: int | None) -> Usage | None:
    if input_tokens is None:
        return None
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _tool_call(call_id: str, *, name: str = "noop") -> ToolUsePart:
    return ToolUsePart(id=call_id, name=name, inputs={})


def _history(pairs: int = 2, text: str = "history-payload") -> list[Any]:
    messages: list[Any] = [UserMessagePart(parts=[TextPart(text="question")])]
    for index in range(pairs):
        call_id = f"h{index}"
        messages.append(
            AssistantMessagePart(
                parts=[ToolUsePart(id=call_id, name="search", inputs={"q": call_id})]
            )
        )
        messages.append(
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id=call_id,
                        name="search",
                        parts=[TextPart(text=f"{text}-{index}")],
                    )
                ]
            )
        )
    return messages


def _scripted(turns: list[tuple[list[Any], Usage | None]], captured: list[list[Any]]) -> Any:
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
    *,
    tools: list[Tool] | None = None,
    provider: Any | None = None,
    captured: list[list[Any]] | None = None,
) -> tuple[list[Any], list[list[Any]]]:
    recorded: list[list[Any]] = [] if captured is None else captured
    llm = provider if provider is not None else _scripted(turns, recorded)
    executor = AgentExecutor(
        "openai",
        llm,
        tools=tools or [_NoopTool()],
        context_policy=policy,
    )

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=len(turns)):
            events.append(event)
        return events

    return asyncio.run(run()), recorded


def _edits(events: list[Any]) -> list[ContextEditEvent]:
    return [event for event in events if isinstance(event, ContextEditEvent)]


def _trim_applied(messages: list[Any]) -> bool:
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if not isinstance(part, ToolResultPart):
                continue
            for inner in part.parts:
                if isinstance(inner, TextPart) and inner.text == _PLACEHOLDER:
                    return True
    return False


def _policy(
    window: int = 128_000,
    pct: float = 0.8,
    *,
    keep_last_n: int = 1,
    mode: str = "trim",
) -> ContextPolicy:
    return ContextPolicy(
        context_window=window,
        trigger_pct=pct,
        keep_last_n=keep_last_n,
        mode=mode,
    )


@pytest.mark.parametrize(
    ("last_tokens", "watermark", "expect"),
    [
        (None, None, False),
        (0, None, False),
        (102_399, None, False),
        (102_400, None, True),
        (999_999_999, None, True),
        (102_400, 102_400, False),
        (102_401, 102_400, True),
    ],
)
def test_compaction_triggered_respects_usage_and_watermark(
    last_tokens: int | None,
    watermark: int | None,
    expect: bool,
) -> None:
    policy = _policy()
    assert _compaction_triggered(last_tokens, policy, watermark) is expect


def test_usage_present_compacts_when_previous_turn_at_threshold() -> None:
    policy = _policy()
    trigger = policy.trigger_tokens
    turns = [
        ([_tool_call("t1")], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, captured = _drive(policy, turns, _history())
    assert len(captured) == 2
    assert len(_edits(events)) == 1
    assert _trim_applied(captured[1])


def test_usage_missing_on_first_turn_uses_character_estimate() -> None:
    """When ``usage`` is absent and no prior count exists, executor uses ``estimate_input_tokens``."""
    policy = _policy()
    trigger = policy.trigger_tokens
    blob = "X" * (trigger * 4)
    messages = _history(pairs=2, text=blob)
    assert estimate_input_tokens(messages) >= trigger
    turns = [
        ([_tool_call("t1")], None),
        ([], _usage(0)),
    ]
    events, captured = _drive(policy, turns, messages)
    assert len(_edits(events)) == 1
    assert _trim_applied(captured[1])


def test_usage_zero_does_not_trigger_compaction() -> None:
    policy = _policy()
    turns = [
        ([_tool_call("t1")], _usage(0)),
        ([], _usage(0)),
    ]
    events, captured = _drive(policy, turns, _history())
    assert _edits(events) == []
    assert not _trim_applied(captured[1])


def test_usage_huge_triggers_compaction() -> None:
    policy = _policy()
    turns = [
        ([_tool_call("t1")], _usage(10**9)),
        ([], _usage(10**9)),
    ]
    events, _captured = _drive(policy, turns, _history())
    assert len(_edits(events)) == 1


def test_over_reported_usage_triggers_even_when_messages_are_small() -> None:
    """Reported usage is trusted over on-wire size (may compact while char estimate is low)."""
    policy = _policy()
    trigger = policy.trigger_tokens
    assert estimate_input_tokens([UserMessagePart(parts=[TextPart(text="tiny")])]) < trigger
    turns = [
        ([_tool_call("t1")], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, captured = _drive(policy, turns, _history(pairs=2))
    assert len(_edits(events)) == 1
    assert _trim_applied(captured[1])
    assert estimate_input_tokens(captured[1]) < trigger


def test_under_reported_usage_delays_compaction_until_threshold_crosses() -> None:
    """Low reported usage prevents trigger even as messages grow (late / missing compaction)."""
    policy = _policy(window=128_000, pct=0.8)
    trigger = policy.trigger_tokens
    below = trigger - 1
    turns = [
        ([_tool_call("t1")], _usage(below)),
        ([_tool_call("t2")], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, captured = _drive(policy, turns, _history())
    assert not _trim_applied(captured[1])
    assert len(_edits(events)) == 1
    assert _trim_applied(captured[2])


def test_usage_changes_between_turns_trim_refires_while_at_threshold() -> None:
    """Trim mode has no watermark: every turn at/above threshold compacts again."""
    policy = _policy()
    t0 = policy.trigger_tokens
    turns = [
        ([_tool_call("a")], _usage(t0)),
        ([_tool_call("b")], _usage(t0)),
        ([_tool_call("c")], _usage(t0 + 50_000)),
        ([], _usage(t0 + 50_000)),
    ]
    events, captured = _drive(policy, turns, _history())
    assert len(_edits(events)) == 3
    assert _trim_applied(captured[1])
    assert _trim_applied(captured[2])
    assert _trim_applied(captured[3])


def test_summarize_watermark_requires_usage_increase_to_retrigger() -> None:
    policy = _policy(mode="summarize")
    t0 = policy.trigger_tokens
    summary_calls: list[list[Any]] = []
    turns = [
        ([_tool_call("a")], _usage(t0)),
        ([_tool_call("b")], _usage(t0)),
        ([_tool_call("c")], _usage(t0 + 1)),
        ([], _usage(t0 + 1)),
    ]
    captured: list[list[Any]] = []
    call_count = 0

    async def agent_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        if kwargs.get("stream", True) is False:
            summary_calls.append(list(messages))
            return StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=[TextPart(text="DIGEST")],
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
    provider.chat = agent_chat
    events, _cap = _drive(policy, turns, _history(), provider=provider, captured=captured)
    assert len(_edits(events)) == 2
    assert len(summary_calls) == 2


def test_usage_disappears_keeps_last_count_without_re_estimating() -> None:
    """After usage was present, ``usage=None`` keeps the prior count (no char re-estimate)."""
    policy = _policy()
    trigger = policy.trigger_tokens
    turns = [
        ([_tool_call("t1")], _usage(trigger)),
        ([], None),
    ]
    events, captured = _drive(policy, turns, _history())
    assert len(_edits(events)) == 1
    assert _trim_applied(captured[1])
    assert estimate_input_tokens(captured[1]) < trigger


def test_character_estimate_is_chars_div_four() -> None:
    messages = [UserMessagePart(parts=[TextPart(text="abcd")])]
    assert estimate_input_tokens(messages) == 1


@pytest.mark.parametrize(
    ("window", "previous"),
    [
        pytest.param(128_000, 102_399, id="normal-window-128k"),
        pytest.param(1_000_000, 799_999, id="large-window-1m"),
    ],
)
def test_huge_tool_can_exceed_window_before_compaction_runs(window: int, previous: int) -> None:
    """Oracle: overflow request must not be sent without compaction when over window."""
    policy = _policy(window=window, pct=0.8, keep_last_n=0)
    assert previous < policy.trigger_tokens
    tool = _huge_tool((window + 10) * 4)
    messages = [UserMessagePart(parts=[TextPart(text="hi")])]
    turns = [
        ([_tool_call("c1", name="huge")], _usage(previous)),
        ([], _usage(previous)),
    ]
    events, captured = _drive(policy, turns, messages, tools=[tool])
    second_estimate = estimate_input_tokens(captured[1])
    compacted = _trim_applied(captured[1]) or bool(_edits(events))
    assert second_estimate <= window or compacted, (
        f"estimate={second_estimate} window={window} trigger={policy.trigger_tokens} "
        f"previous_usage={previous} compacted={compacted}"
    )
