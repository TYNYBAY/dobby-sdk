"""Direct unit tests for production ``dobby.context`` (not the recovered snapshot)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from pydantic import ValidationError
import pytest

from dobby.context import SUMMARIZE_PROMPT, ContextPolicy, edit_context, summarize_context
from dobby.context.edit import _find_tool_pairs
from dobby.providers.gemini.converters import to_gemini_messages
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


def _pair_ids(messages: list[Any]) -> list[tuple[str, str]]:
    paired: list[tuple[str, str]] = []
    for use_index, result_index in _find_tool_pairs(messages):
        use = next(part for part in messages[use_index].parts if isinstance(part, ToolUsePart))
        result = next(
            part for part in messages[result_index].parts if isinstance(part, ToolResultPart)
        )
        paired.append((use.id, result.tool_use_id))
    return paired


def test_find_tool_pairs_keeps_adjacent_executor_pairs() -> None:
    messages = _history(("A", "B"))
    assert _find_tool_pairs(messages) == [(0, 1), (2, 3)]
    assert _pair_ids(messages) == [("id-A", "id-A"), ("id-B", "id-B")]


def test_find_tool_pairs_matches_interleaved_ids() -> None:
    use_a, result_a = _pair("A")
    use_b, result_b = _pair("B")
    messages = [use_a, use_b, result_a, result_b]
    assert _pair_ids(messages) == [("id-A", "id-A"), ("id-B", "id-B")]


def test_find_tool_pairs_does_not_pair_unrelated_ids() -> None:
    use_a, _result_a = _pair("A")
    _use_b, result_b = _pair("B")
    messages = [use_a, result_b]
    assert _find_tool_pairs(messages) == []


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


def test_summarize_preserves_messages_between_clearable_pairs() -> None:
    """User instructions and assistant text between old pairs are not swallowed."""
    preamble = UserMessagePart(parts=[TextPart(text="user question")])
    never_delete = UserMessagePart(parts=[TextPart(text="NEVER-DELETE")])
    assistant_text = AssistantMessagePart(parts=[TextPart(text="assistant note")])
    use_a, result_a = _pair("A")
    use_b, result_b = _pair("B")
    use_c, result_c = _pair("C")
    messages: list[Any] = [
        preamble,
        use_a,
        result_a,
        never_delete,
        assistant_text,
        use_b,
        result_b,
        use_c,
        result_c,
    ]
    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value=StreamEndEvent(
            model="summarizer",
            parts=[TextPart(text="digest-AB")],
            stop_reason="end_turn",
            usage=None,
        )
    )
    policy = ContextPolicy(keep_last_n=1, mode="summarize")

    applied = asyncio.run(summarize_context(messages, policy, llm))

    assert applied is not None
    assert applied.cleared_tool_uses == 2
    assert messages[0] is preamble
    assert messages[1] is not never_delete
    assert isinstance(messages[1], UserMessagePart)
    assert messages[1].parts[0].text == "<summary>digest-AB</summary>"
    assert messages[2] is never_delete
    assert messages[3] is assistant_text
    assert messages[4] is use_c
    assert messages[5] is result_c
    assert len(messages) == 6
    assert use_a not in messages
    assert result_a not in messages
    assert use_b not in messages
    assert result_b not in messages
    summaries = [
        message
        for message in messages
        if isinstance(message, UserMessagePart)
        and any(
            isinstance(part, TextPart) and part.text.startswith("<summary>")
            for part in message.parts
        )
    ]
    assert len(summaries) == 1


def _gemini_roles(contents: list[Any]) -> list[str]:
    return [content.role for content in contents]


def _gemini_texts(content: Any) -> list[str]:
    return [part.text for part in content.parts or [] if getattr(part, "text", None)]


def test_gemini_merges_three_adjacent_plain_user_messages() -> None:
    contents = to_gemini_messages(
        [
            UserMessagePart(parts=[TextPart(text="one")]),
            UserMessagePart(parts=[TextPart(text="two")]),
            UserMessagePart(parts=[TextPart(text="three")]),
        ]
    )
    assert _gemini_roles(contents) == ["user"]
    assert _gemini_texts(contents[0]) == ["one", "two", "three"]


def test_gemini_plain_user_before_tool_result_stays_separate() -> None:
    _use, result = _pair("A")
    contents = to_gemini_messages(
        [
            UserMessagePart(parts=[TextPart(text="before")]),
            result,
        ]
    )
    assert _gemini_roles(contents) == ["user", "user"]
    assert _gemini_texts(contents[0]) == ["before"]
    assert contents[1].parts[0].function_response is not None
    assert _gemini_texts(contents[1]) == []


def test_gemini_plain_user_after_tool_result_stays_separate() -> None:
    _use, result = _pair("A")
    contents = to_gemini_messages(
        [
            result,
            UserMessagePart(parts=[TextPart(text="after")]),
        ]
    )
    assert _gemini_roles(contents) == ["user", "user"]
    assert contents[0].parts[0].function_response is not None
    assert _gemini_texts(contents[0]) == []
    assert _gemini_texts(contents[1]) == ["after"]


def test_gemini_model_turn_breaks_plain_user_merge() -> None:
    contents = to_gemini_messages(
        [
            UserMessagePart(parts=[TextPart(text="one")]),
            UserMessagePart(parts=[TextPart(text="two")]),
            AssistantMessagePart(parts=[TextPart(text="mid")]),
            UserMessagePart(parts=[TextPart(text="three")]),
        ]
    )
    assert _gemini_roles(contents) == ["user", "model", "user"]
    assert _gemini_texts(contents[0]) == ["one", "two"]
    assert _gemini_texts(contents[1]) == ["mid"]
    assert _gemini_texts(contents[2]) == ["three"]


def test_gemini_alternating_user_model_history_is_unchanged() -> None:
    contents = to_gemini_messages(
        [
            UserMessagePart(parts=[TextPart(text="hi")]),
            AssistantMessagePart(parts=[TextPart(text="hello")]),
            UserMessagePart(parts=[TextPart(text="again")]),
            AssistantMessagePart(parts=[TextPart(text="ok")]),
        ]
    )
    assert _gemini_roles(contents) == ["user", "model", "user", "model"]
    assert [_gemini_texts(content) for content in contents] == [
        ["hi"],
        ["hello"],
        ["again"],
        ["ok"],
    ]


def test_gemini_merges_assistant_text_before_kept_tool_use() -> None:
    """Summary + assistant text + tool-use must not reach Gemini as two model roles."""
    use, result = _pair("C")
    contents = to_gemini_messages(
        [
            UserMessagePart(parts=[TextPart(text="user question")]),
            UserMessagePart(parts=[TextPart(text="<summary>digest-kept</summary>")]),
            AssistantMessagePart(parts=[TextPart(text="assistant note")]),
            use,
            result,
        ]
    )
    assert _gemini_roles(contents) == ["user", "model", "user"]
    assert _gemini_texts(contents[0]) == ["user question", "<summary>digest-kept</summary>"]
    assert _gemini_texts(contents[1]) == ["assistant note"]
    assert contents[1].parts[1].function_call is not None
    assert contents[1].parts[1].function_call.name == "search"
    assert contents[2].parts[0].function_response is not None
    assert contents[2].parts[0].function_response.name == "search"
    assert contents[2].parts[0].function_response.response == {"text": "result-C"}
