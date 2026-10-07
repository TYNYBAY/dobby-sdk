"""Summarize must not split a kept tool pair that shares a message with a cleared one."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from dobby.context import ContextPolicy, summarize_context
from dobby.providers import (
    to_anthropic_messages,
    to_gemini_messages,
    to_openai_messages,
    to_vertexai_messages,
)
from dobby.types import (
    AssistantMessagePart,
    StreamEndEvent,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    UserMessagePart,
)

_DIGEST = "digest-kept"


def _use_part(call_id: str) -> ToolUsePart:
    return ToolUsePart(id=call_id, name="search", inputs={"q": call_id})


def _result_part(call_id: str, text: str) -> ToolResultPart:
    return ToolResultPart(
        tool_use_id=call_id,
        name="search",
        parts=[TextPart(text=text)],
    )


def _use_message(*call_ids: str, text: str | None = None) -> AssistantMessagePart:
    parts: list[Any] = [TextPart(text=text)] if text is not None else []
    parts.extend(_use_part(call_id) for call_id in call_ids)
    return AssistantMessagePart(parts=parts)


def _result_message(*payloads: tuple[str, str], text: str | None = None) -> UserMessagePart:
    parts: list[Any] = [TextPart(text=text)] if text is not None else []
    parts.extend(_result_part(call_id, body) for call_id, body in payloads)
    return UserMessagePart(parts=parts)


def _llm() -> AsyncMock:
    llm = AsyncMock()

    async def chat(messages: list[Any], **kwargs: Any) -> StreamEndEvent:
        return StreamEndEvent(
            model="summarizer",
            parts=[TextPart(text=_DIGEST)],
            stop_reason="end_turn",
            usage=None,
        )

    llm.chat = chat
    return llm


def _summarize(messages: list[Any]) -> Any:
    return asyncio.run(
        summarize_context(messages, ContextPolicy(keep_last_n=1, mode="summarize"), _llm())
    )


def _use_ids(messages: list[Any]) -> list[str]:
    ids: list[str] = []
    for message in messages:
        if isinstance(message, AssistantMessagePart):
            ids.extend(part.id for part in message.parts if isinstance(part, ToolUsePart))
    return ids


def _result_ids(messages: list[Any]) -> list[str]:
    ids: list[str] = []
    for message in messages:
        if isinstance(message, UserMessagePart):
            ids.extend(
                part.tool_use_id for part in message.parts if isinstance(part, ToolResultPart)
            )
    return ids


def _result_text(messages: list[Any], call_id: str) -> str:
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, ToolResultPart) and part.tool_use_id == call_id:
                return "".join(inner.text for inner in part.parts if isinstance(inner, TextPart))
    return ""


def _summary_indexes(messages: list[Any]) -> list[int]:
    found: list[int] = []
    for index, message in enumerate(messages):
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if isinstance(part, TextPart) and part.text == f"<summary>{_DIGEST}</summary>":
                found.append(index)
    return found


def _assert_kept_pair(messages: list[Any], call_id: str, result_text: str) -> None:
    assert _use_ids(messages) == [call_id]
    assert _result_ids(messages) == [call_id]
    assert _result_text(messages, call_id) == result_text
    summaries = _summary_indexes(messages)
    assert len(summaries) == 1
    use_at = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message, AssistantMessagePart)
        and any(isinstance(part, ToolUsePart) and part.id == call_id for part in message.parts)
    )
    result_at = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message, UserMessagePart)
        and any(
            isinstance(part, ToolResultPart) and part.tool_use_id == call_id
            for part in message.parts
        )
    )
    for summary_at in summaries:
        assert not (use_at < summary_at < result_at)


def _conversion_problems(messages: list[Any]) -> list[str]:
    problems: list[str] = []
    try:
        openai_items = to_openai_messages(messages)
    except Exception as exc:
        problems.append(f"openai raised {type(exc).__name__}: {exc}")
    else:
        call_ids = {
            item["call_id"] for item in openai_items if item.get("type") == "function_call"
        }
        output_ids = {
            item["call_id"] for item in openai_items if item.get("type") == "function_call_output"
        }
        if call_ids != output_ids:
            problems.append(f"openai unpaired calls={call_ids} outputs={output_ids}")
        if _DIGEST not in str(openai_items):
            problems.append("openai dropped the summary")

    try:
        anthropic_messages = to_anthropic_messages(messages)
    except Exception as exc:
        problems.append(f"anthropic raised {type(exc).__name__}: {exc}")
    else:
        roles = [message["role"] for message in anthropic_messages]
        if any(roles[index] == roles[index + 1] for index in range(len(roles) - 1)):
            problems.append(f"anthropic roles do not alternate: {roles}")
        uses = [
            block["id"]
            for message in anthropic_messages
            for block in message["content"]
            if block.get("type") == "tool_use"
        ]
        results = [
            block["tool_use_id"]
            for message in anthropic_messages
            for block in message["content"]
            if block.get("type") == "tool_result"
        ]
        if uses != results:
            problems.append(f"anthropic unpaired uses={uses} results={results}")
        if _DIGEST not in str(anthropic_messages):
            problems.append("anthropic dropped the summary")

    try:
        gemini_contents = to_gemini_messages(messages)
    except Exception as exc:
        problems.append(f"gemini raised {type(exc).__name__}: {exc}")
    else:
        roles = [content.role for content in gemini_contents]
        if any(roles[index] == roles[index + 1] for index in range(len(roles) - 1)):
            problems.append(f"gemini roles do not alternate: {roles}")
        calls: list[str] = []
        responses: list[str] = []
        for index, content in enumerate(gemini_contents):
            for part in content.parts or []:
                call = getattr(part, "function_call", None)
                response = getattr(part, "function_response", None)
                if call is not None:
                    calls.append(call.name)
                    if index + 1 >= len(gemini_contents) or not any(
                        getattr(inner, "function_response", None) is not None
                        for inner in gemini_contents[index + 1].parts or []
                    ):
                        problems.append(
                            "gemini function call is not followed by a function response"
                        )
                if response is not None:
                    responses.append(response.name)
        if calls != responses:
            problems.append(f"gemini unpaired calls={calls} responses={responses}")
        texts = [
            part.text
            for content in gemini_contents
            for part in content.parts or []
            if getattr(part, "text", None)
        ]
        if _DIGEST not in "\n".join(texts):
            problems.append("gemini dropped the summary")

    try:
        vertex_messages = to_vertexai_messages(messages)
    except Exception as exc:
        problems.append(f"vertex raised {type(exc).__name__}: {exc}")
    else:
        call_ids = [
            call["id"] for message in vertex_messages for call in message.get("tool_calls") or []
        ]
        result_ids = [
            message["tool_call_id"] for message in vertex_messages if message["role"] == "tool"
        ]
        if call_ids != result_ids:
            problems.append(f"vertex unpaired calls={call_ids} results={result_ids}")
        if _DIGEST not in str(vertex_messages):
            problems.append("vertex dropped the summary")
    return problems


def test_shared_results_keep_last_n_one_leaves_kept_pair() -> None:
    messages: list[Any] = [
        _use_message("A"),
        _use_message("B"),
        _result_message(("A", "result-A"), ("B", "result-B")),
    ]
    applied = _summarize(messages)
    assert applied is not None
    assert applied.cleared_tool_uses == 1
    _assert_kept_pair(messages, "B", "result-B")
    assert "result-A" not in _result_text(messages, "B")


def test_shared_uses_keep_last_n_one_leaves_kept_pair() -> None:
    messages: list[Any] = [
        _use_message("A", "B"),
        _result_message(("A", "result-A")),
        _result_message(("B", "result-B")),
    ]
    applied = _summarize(messages)
    assert applied is not None
    assert applied.cleared_tool_uses == 1
    _assert_kept_pair(messages, "B", "result-B")


def test_nested_interleave_keep_last_n_one_does_not_split_kept_pair() -> None:
    messages: list[Any] = [
        _use_message("A"),
        _use_message("B"),
        _result_message(("B", "result-B")),
        _result_message(("A", "result-A")),
    ]
    applied = _summarize(messages)
    assert applied is not None
    assert applied.cleared_tool_uses == 1
    _assert_kept_pair(messages, "A", "result-A")
    assert "result-B" not in "".join(_result_text(messages, "A"))


def test_mixed_text_and_tool_parts_are_preserved_on_kept_pair() -> None:
    messages: list[Any] = [
        _use_message("A", "B", text="KEEP-ASSISTANT"),
        _result_message(("A", "result-A"), ("B", "result-B"), text="KEEP-USER"),
    ]
    applied = _summarize(messages)
    assert applied is not None
    assert applied.cleared_tool_uses == 1
    _assert_kept_pair(messages, "B", "result-B")
    assistant = next(message for message in messages if isinstance(message, AssistantMessagePart))
    user = next(
        message
        for message in messages
        if isinstance(message, UserMessagePart)
        and any(isinstance(part, ToolResultPart) for part in message.parts)
    )
    assert any(
        isinstance(part, TextPart) and part.text == "KEEP-ASSISTANT" for part in assistant.parts
    )
    assert any(isinstance(part, TextPart) and part.text == "KEEP-USER" for part in user.parts)
    assert [part.id for part in assistant.parts if isinstance(part, ToolUsePart)] == ["B"]
    assert [part.tool_use_id for part in user.parts if isinstance(part, ToolResultPart)] == ["B"]


def test_shared_and_nested_histories_convert_after_summarize() -> None:
    histories = [
        [
            _use_message("A"),
            _use_message("B"),
            _result_message(("A", "result-A"), ("B", "result-B")),
        ],
        [
            _use_message("A", "B"),
            _result_message(("A", "result-A")),
            _result_message(("B", "result-B")),
        ],
        [
            _use_message("A"),
            _use_message("B"),
            _result_message(("B", "result-B")),
            _result_message(("A", "result-A")),
        ],
        [
            _use_message("A", "B", text="KEEP-ASSISTANT"),
            _result_message(("A", "result-A"), ("B", "result-B")),
        ],
    ]
    problems: list[str] = []
    for history in histories:
        _summarize(history)
        problems.extend(_conversion_problems(history))
    assert problems == []
