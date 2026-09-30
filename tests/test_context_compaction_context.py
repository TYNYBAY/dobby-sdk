"""Direct unit tests for production ``dobby.context`` (not the recovered snapshot)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from pydantic import ValidationError
import pytest

from dobby.context import SUMMARIZE_PROMPT, ContextPolicy, edit_context, summarize_context
from dobby.types import (
    AssistantMessagePart,
    StreamEndEvent,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    UserMessagePart,
)

_PLACEHOLDER = "[Tool result cleared to save context.]"
_LABELS = ("A", "B", "C", "D", "E")


def _pair(label: str) -> tuple[AssistantMessagePart, UserMessagePart]:
    call_id = f"id-{label}"
    return (
        AssistantMessagePart(
            parts=[ToolUsePart(id=call_id, name="search", inputs={"label": label})]
        ),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id=call_id,
                    name="search",
                    parts=[TextPart(text=f"result-{label}")],
                )
            ]
        ),
    )


def _history(labels: tuple[str, ...] = _LABELS) -> list[Any]:
    messages: list[Any] = []
    for label in labels:
        use, result = _pair(label)
        messages.extend((use, result))
    return messages


def _result_texts(messages: list[Any]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, ToolResultPart):
                texts.extend(inner.text for inner in part.parts if isinstance(inner, TextPart))
    return texts


def _result_ids(messages: list[Any]) -> list[str]:
    ids: list[str] = []
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, ToolResultPart):
                ids.append(part.tool_use_id)
    return ids


def _use_ids(messages: list[Any]) -> list[str]:
    ids: list[str] = []
    for message in messages:
        if not isinstance(message, AssistantMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, ToolUsePart):
                ids.append(part.id)
    return ids


def test_trigger_tokens_uses_decimal_ceiling() -> None:
    """``trigger_tokens`` is ``ceil(pct * window)``, not ``int(float product)``."""
    policy = ContextPolicy(context_window=128_000, trigger_pct=0.8)
    assert policy.trigger_tokens == 102_400

    # 0.8 * 9 = 7.2; int() truncates to 7, decimal ceiling is 8.
    small = ContextPolicy(context_window=9, trigger_pct=0.8)
    assert int(0.8 * 9) == 7
    assert small.trigger_tokens == 8


def test_context_policy_validation_bounds() -> None:
    with pytest.raises(ValidationError):
        ContextPolicy(trigger_pct=0.49)
    accepted = ContextPolicy(trigger_pct=0.5)
    assert accepted.trigger_pct == 0.5

    with pytest.raises(ValidationError):
        ContextPolicy(keep_last_n=-1)
    zero = ContextPolicy(keep_last_n=0)
    assert zero.keep_last_n == 0


def test_edit_context_keep_last_two_of_five_preserves_pairing_and_identity() -> None:
    messages = _history()
    original_ids = [id(message) for message in messages]
    original_texts = _result_texts(messages)
    policy = ContextPolicy(keep_last_n=2, mode="trim")

    edited, applied = edit_context(messages, policy)

    assert applied is not None
    assert applied.type == "clear_tool_uses"
    assert applied.cleared_tool_uses == 3
    assert _result_texts(edited) == [
        _PLACEHOLDER,
        _PLACEHOLDER,
        _PLACEHOLDER,
        "result-D",
        "result-E",
    ]
    assert _use_ids(edited) == [f"id-{label}" for label in _LABELS]
    assert _result_ids(edited) == [f"id-{label}" for label in _LABELS]

    assert edited is not messages
    assert [id(message) for message in messages] == original_ids
    assert _result_texts(messages) == original_texts
    # Tool-use messages and the kept last two results are reused by identity.
    for index in (0, 2, 4, 6, 7, 8, 9):
        assert edited[index] is messages[index]
    for index in (1, 3, 5):
        assert edited[index] is not messages[index]


def test_edit_context_inflight_tool_use_is_not_a_completed_pair() -> None:
    messages = _history()
    inflight = ToolUsePart(id="id-F", name="search", inputs={"label": "F"})
    messages.append(AssistantMessagePart(parts=[inflight]))
    policy = ContextPolicy(keep_last_n=2, mode="trim")

    edited, applied = edit_context(messages, policy)

    assert applied is not None
    assert applied.cleared_tool_uses == 3
    assert _result_texts(edited) == [
        _PLACEHOLDER,
        _PLACEHOLDER,
        _PLACEHOLDER,
        "result-D",
        "result-E",
    ]
    assert _result_ids(edited) == [f"id-{label}" for label in _LABELS]
    assert "id-F" not in _result_ids(edited)
    last_use = edited[-1]
    assert isinstance(last_use, AssistantMessagePart)
    assert last_use.parts[0] is inflight
    assert last_use is messages[-1]


def test_edit_context_noop_when_pairs_within_keep_window() -> None:
    messages = _history(("D", "E"))
    policy = ContextPolicy(keep_last_n=2, mode="trim")

    edited, applied = edit_context(messages, policy)

    assert applied is None
    assert edited is messages
    assert _result_texts(messages) == ["result-D", "result-E"]


def test_summarize_context_forwards_extra_instructions() -> None:
    messages = _history(("A", "B", "C"))
    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value=StreamEndEvent(
            model="summarizer",
            parts=[TextPart(text="digest")],
            stop_reason="end_turn",
            usage=None,
        )
    )
    policy = ContextPolicy(keep_last_n=1, mode="summarize")

    applied = asyncio.run(
        summarize_context(
            messages,
            policy,
            llm,
            extra_instructions="keep ticket IDs",
        )
    )

    assert applied is not None
    llm.chat.assert_awaited_once()
    kwargs = llm.chat.await_args.kwargs
    assert kwargs["stream"] is False
    assert SUMMARIZE_PROMPT in kwargs["system_prompt"]
    assert "Additional instructions: keep ticket IDs" in kwargs["system_prompt"]


def test_summarize_context_empty_digest_is_noop() -> None:
    messages = _history(("A", "B", "C"))
    snapshot = list(messages)
    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value=StreamEndEvent(
            model="summarizer",
            parts=[TextPart(text="   ")],
            stop_reason="end_turn",
            usage=None,
        )
    )
    policy = ContextPolicy(keep_last_n=1, mode="summarize")

    applied = asyncio.run(summarize_context(messages, policy, llm))

    assert applied is None
    assert messages == snapshot
    assert [id(message) for message in messages] == [id(message) for message in snapshot]


def test_summarize_context_noop_does_not_call_chat() -> None:
    messages = _history(("D", "E"))
    llm = AsyncMock()
    llm.chat = AsyncMock()
    policy = ContextPolicy(keep_last_n=2, mode="summarize")

    applied = asyncio.run(summarize_context(messages, policy, llm))

    assert applied is None
    llm.chat.assert_not_called()
    assert _result_texts(messages) == ["result-D", "result-E"]
