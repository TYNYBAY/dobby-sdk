"""Production executor compaction hook (trim/summarize, no compact-tool surface)."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dobby import AgentExecutor
from dobby.context import ContextPolicy
from dobby.context._tokens import compaction_trigger_basis, estimate_input_tokens
from dobby.context.summarize import summarize_context
from dobby.executor import _compaction_triggered
from dobby.providers.base import ProviderError
from dobby.providers.gemini.converters import to_gemini_messages
from dobby.tools import Tool
from dobby.types import (
    AssistantMessagePart,
    ContextEditEvent,
    StreamEndEvent,
    TextPart,
    ToolResultEvent,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart,
)

_PLACEHOLDER = "[Tool result cleared to save context.]"


def _usage(input_tokens: int | None) -> Usage | None:
    if input_tokens is None:
        return None
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _policy(
    *,
    keep_last_n: int = 1,
    mode: str = "trim",
    window: int = 128_000,
    pct: float = 0.8,
) -> ContextPolicy:
    return ContextPolicy(
        context_window=window,
        trigger_pct=pct,
        keep_last_n=keep_last_n,
        mode=mode,  # type: ignore[arg-type]
    )


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


def _result_texts(messages: list[Any]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, ToolResultPart):
                texts.extend(inner.text for inner in part.parts if isinstance(inner, TextPart))
            elif isinstance(part, TextPart):
                texts.append(part.text)
    return texts


class _Scripted:
    """One agent turn per streaming chat; ``stream=False`` is the summarizer."""

    name = "scripted"

    def __init__(
        self,
        turns: list[tuple[list[Any], Usage | None]],
        *,
        summary_text: str = "digest-kept",
        summary_error: Exception | None = None,
    ) -> None:
        self.turns = turns
        self.summary_text = summary_text
        self.summary_error = summary_error
        self.agent_calls: list[list[Any]] = []
        self.summarize_calls: list[list[Any]] = []
        self._index = 0

    async def chat(self, messages: list[Any], **kwargs: Any) -> Any:
        if kwargs.get("stream") is False:
            self.summarize_calls.append(list(messages))
            if self.summary_error is not None:
                raise self.summary_error
            return StreamEndEvent(
                model="summarizer",
                parts=[TextPart(text=self.summary_text)],
                stop_reason="end_turn",
                usage=_usage(0),
            )
        self.agent_calls.append(list(messages))
        index = min(self._index, len(self.turns) - 1)
        self._index += 1
        parts, usage = self.turns[index]

        async def stream() -> Any:
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=usage,
            )

        return stream()


@dataclass
class _NoopTool(Tool):
    name = "noop"
    description = "Ack."

    def __call__(self) -> dict[str, str]:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        return {"ok": "yes"}


@dataclass
class _HugeTool(Tool):
    name = "huge"
    description = "Return a large blob."

    def __call__(self) -> dict[str, str]:
        return {"blob": "H" * ((128_000 + 10) * 4)}


@dataclass
class _TypedTool(Tool):
    name = "typed"
    description = "Accept an integer."

    async def __call__(self, count: int) -> int:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        return count


@dataclass
class _FlakyTool(Tool):
    name = "flaky"
    description = "Fail once, then succeed."
    retryable_exceptions = (TimeoutError,)

    async def __call__(self) -> str:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        if calls == 0:
            raise TimeoutError("transient")
        return "recovered"


def _drive(
    policy: ContextPolicy | None,
    turns: list[tuple[list[Any], Usage | None]],
    messages: list[Any],
    *,
    tools: list[Tool] | None = None,
    summary_text: str = "digest-kept",
    summary_error: Exception | None = None,
) -> tuple[list[Any], _Scripted]:
    provider = _Scripted(turns, summary_text=summary_text, summary_error=summary_error)
    executor = AgentExecutor(
        "openai",
        provider,  # type: ignore[arg-type]
        tools=tools or [_NoopTool()],
        context_policy=policy,
    )

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=len(turns)):
            events.append(event)
        return events

    return asyncio.run(run()), provider


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


def test_trigger_uses_outgoing_estimate_after_huge_tool_result() -> None:
    policy = _policy()
    trigger = policy.trigger_tokens
    small = [UserMessagePart(parts=[TextPart(text="hi")])]
    assert compaction_trigger_basis(trigger - 1, small) == trigger - 1
    assert _compaction_triggered(trigger - 1, policy, None, outgoing_messages=small) is False
    outgoing = [
        UserMessagePart(parts=[TextPart(text="hi")]),
        AssistantMessagePart(parts=[ToolUsePart(id="c1", name="huge", inputs={})]),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id="c1",
                    name="huge",
                    parts=[TextPart(text="H" * ((policy.context_window + 10) * 4))],
                )
            ]
        ),
    ]
    assert _compaction_triggered(trigger - 1, policy, None, outgoing_messages=outgoing) is True


def test_trim_runs_before_next_model_call_and_preserves_caller() -> None:
    """Previous-turn usage at the threshold trims the next send only."""
    policy = _policy(keep_last_n=1)
    trigger = policy.trigger_tokens
    caller = _history()
    snapshot = copy.deepcopy(caller)
    noop = _NoopTool()
    events, provider = _drive(
        policy,
        [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(trigger)),
            ([], _usage(trigger)),
        ],
        caller,
        tools=[noop],
    )
    assert len(provider.agent_calls) == 2
    assert provider.agent_calls[0] == snapshot
    assert not _trim_applied(provider.agent_calls[0])
    assert _trim_applied(provider.agent_calls[1])
    sent_texts = _result_texts(provider.agent_calls[1])
    assert sent_texts.count(_PLACEHOLDER) == 2
    assert "{'ok': 'yes'}" in sent_texts
    assert "history-payload-0" not in sent_texts
    assert "history-payload-1" not in sent_texts
    assert caller == snapshot
    assert noop._calls == 1  # type: ignore[attr-defined]
    edits = [event for event in events if isinstance(event, ContextEditEvent)]
    assert len(edits) == 1
    assert edits[0].applied_edits[0].type == "clear_tool_uses"


def test_usage_zero_does_not_trim() -> None:
    policy = _policy()
    _events, provider = _drive(
        policy,
        [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(0)),
            ([], _usage(0)),
        ],
        _history(),
    )
    assert len(provider.agent_calls) == 2
    assert not _trim_applied(provider.agent_calls[1])


def test_missing_usage_falls_back_to_character_estimate() -> None:
    policy = _policy(keep_last_n=1)
    trigger = policy.trigger_tokens
    caller = _history(pairs=2, text="X" * (trigger * 4))
    assert estimate_input_tokens(caller) >= trigger
    _events, provider = _drive(
        policy,
        [
            ([ToolUsePart(id="t1", name="noop", inputs={})], None),
            ([], _usage(0)),
        ],
        caller,
    )
    assert _trim_applied(provider.agent_calls[1])


def test_huge_tool_result_compacts_before_second_send() -> None:
    """Reported usage just under the trigger still compacts after a huge result."""
    policy = _policy(keep_last_n=0)
    previous = policy.trigger_tokens - 1
    caller = [UserMessagePart(parts=[TextPart(text="hi")])]
    snapshot = copy.deepcopy(caller)
    _events, provider = _drive(
        policy,
        [
            ([ToolUsePart(id="c1", name="huge", inputs={})], _usage(previous)),
            ([], _usage(previous)),
        ],
        caller,
        tools=[_HugeTool()],
    )
    assert _trim_applied(provider.agent_calls[1])
    assert estimate_input_tokens(provider.agent_calls[1]) <= policy.context_window
    assert caller == snapshot


def test_blank_summary_does_not_retry_until_context_grows() -> None:
    """A whitespace digest watermarks the current basis and retries only after growth."""
    policy = _policy(keep_last_n=1, mode="summarize")
    trigger = policy.trigger_tokens
    caller = _history(pairs=3)
    snapshot = copy.deepcopy(caller)
    events, provider = _drive(
        policy,
        [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(trigger)),
            ([ToolUsePart(id="t2", name="noop", inputs={})], _usage(trigger)),
            ([ToolUsePart(id="t3", name="huge", inputs={})], _usage(trigger)),
            ([], _usage(trigger)),
        ],
        caller,
        tools=[_NoopTool(), _HugeTool()],
        summary_text="  \n\t ",
    )
    # Turn 2 blanks and watermarks. Turn 3 is the same basis. The huge result
    # makes turn 4's basis larger, so the summarizer runs once more.
    assert len(provider.summarize_calls) == 2
    assert not any(isinstance(event, ContextEditEvent) for event in events)
    assert caller == snapshot
    assert all("<summary>" not in text for text in _result_texts(provider.agent_calls))


def test_summarize_provider_error_does_not_advance_watermark() -> None:
    """A summarizer ProviderError aborts the run and is not a completed attempt."""
    policy = _policy(keep_last_n=1, mode="summarize")
    trigger = policy.trigger_tokens
    caller = _history(pairs=3)
    snapshot = copy.deepcopy(caller)
    provider = _Scripted(
        [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(trigger)),
            ([ToolUsePart(id="t2", name="noop", inputs={})], _usage(trigger)),
            ([], _usage(trigger)),
        ],
        summary_error=ProviderError("summarizer down"),
    )
    executor = AgentExecutor(
        "openai",
        provider,  # type: ignore[arg-type]
        tools=[_NoopTool()],
        context_policy=policy,
    )
    events: list[Any] = []

    async def run() -> None:
        async for event in executor.run_stream(caller, max_iterations=3):
            events.append(event)

    with pytest.raises(ProviderError, match="summarizer down"):
        asyncio.run(run())

    assert len(provider.summarize_calls) == 1
    assert len(provider.agent_calls) == 1
    assert not any(isinstance(event, ContextEditEvent) for event in events)
    assert caller == snapshot


def test_summarize_writes_back_once_and_preserves_caller() -> None:
    """Summarize mutates the working copy, watermarks, and leaves the caller list intact."""
    policy = _policy(keep_last_n=1, mode="summarize")
    trigger = policy.trigger_tokens
    caller = _history(pairs=3)
    snapshot = copy.deepcopy(caller)
    _events, provider = _drive(
        policy,
        [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(trigger)),
            ([ToolUsePart(id="t2", name="noop", inputs={})], _usage(trigger)),
            ([], _usage(trigger)),
        ],
        caller,
        tools=[_NoopTool()],
    )
    assert len(provider.summarize_calls) == 1
    sent = _result_texts(provider.agent_calls[1])
    assert any("<summary>digest-kept</summary>" in text for text in sent)
    assert "<summary>" not in "".join(_result_texts(provider.agent_calls[0]))
    assert caller == snapshot


def test_model_correction_still_validates_when_policy_is_set() -> None:
    """Invalid args stay a correction row; the tool body does not run."""
    policy = _policy(keep_last_n=1)
    typed = _TypedTool()
    _events, provider = _drive(
        policy,
        [
            ([ToolUsePart(id="bad", name="typed", inputs={"count": "bad"})], _usage(0)),
            ([ToolUsePart(id="ok", name="typed", inputs={"count": 3})], _usage(0)),
            ([], _usage(0)),
        ],
        [],
        tools=[typed],
    )
    assert getattr(typed, "_calls", 0) == 1
    first_retry = _result_texts(provider.agent_calls[1])
    assert any(text.startswith("[tool_input_invalid]") for text in first_retry)
    assert not _trim_applied(provider.agent_calls[1])


def test_host_retry_still_collapses_to_one_result_when_policy_is_set() -> None:
    policy = _policy()
    flaky = _FlakyTool()
    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events, provider = _drive(
            policy,
            [
                ([ToolUsePart(id="f1", name="flaky", inputs={})], _usage(0)),
                ([], _usage(0)),
            ],
            [],
            tools=[flaky],
        )
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert flaky._calls == 2  # type: ignore[attr-defined]
    assert len(results) == 1
    assert results[0].result == "recovered"
    assert results[0].is_error is False
    assert not _trim_applied(provider.agent_calls[1])


def test_host_retry_sleep_is_not_invoked_by_trim() -> None:
    """Compaction does not enter the tool-retry path."""
    policy = _policy()
    trigger = policy.trigger_tokens
    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock) as sleep:
        _events, provider = _drive(
            policy,
            [
                ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(trigger)),
                ([], _usage(trigger)),
            ],
            _history(),
            tools=[_NoopTool()],
        )
    sleep.assert_not_called()
    assert _trim_applied(provider.agent_calls[1])


def test_summarize_after_user_preamble_alternates_gemini_roles() -> None:
    """A leading user message plus a summary user turn must not reach Gemini as two user roles."""
    messages: list[Any] = [
        UserMessagePart(parts=[TextPart(text="Find the account.")]),
        AssistantMessagePart(parts=[ToolUsePart(id="old", name="search", inputs={"q": "old"})]),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id="old",
                    name="search",
                    parts=[TextPart(text="stale ledger")],
                )
            ]
        ),
        AssistantMessagePart(parts=[ToolUsePart(id="new", name="search", inputs={"q": "new"})]),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id="new",
                    name="search",
                    parts=[TextPart(text="current balance")],
                )
            ]
        ),
    ]
    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value=StreamEndEvent(
            model="summarizer",
            parts=[TextPart(text="account lookup was stale")],
            stop_reason="end_turn",
            usage=_usage(0),
        )
    )

    applied = asyncio.run(
        summarize_context(messages, _policy(keep_last_n=1, mode="summarize"), llm)
    )
    assert applied is not None
    assert [message.role for message in messages][:2] == ["user", "user"]

    contents = to_gemini_messages(messages)
    roles = [content.role for content in contents]
    assert roles == ["user", "model", "user"]
    assert all(roles[index] != roles[index + 1] for index in range(len(roles) - 1))
    opening = "".join(getattr(part, "text", None) or "" for part in contents[0].parts or [])
    assert "Find the account." in opening
    assert "<summary>account lookup was stale</summary>" in opening
    assert contents[1].parts[0].function_call is not None
    assert contents[2].parts[0].function_response is not None
