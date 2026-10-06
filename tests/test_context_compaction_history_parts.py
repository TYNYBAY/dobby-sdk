"""Regressions for summary accumulation, non-text parts, and cleared_tool_uses."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

from dobby.context import ContextPolicy, edit_context, summarize_context
from dobby.context._tokens import estimate_input_tokens, part_to_text
from dobby.types import (
    AssistantMessagePart,
    Base64ImageSource,
    Base64PDFSource,
    ContextEditEvent,
    DocumentPart,
    FileDocumentSource,
    ImagePart,
    PlainTextSource,
    ReasoningPart,
    StreamEndEvent,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    URLImageSource,
    URLSource,
    UserMessagePart,
)

_BLOB = "A" * 40_000


def _use(call_id: str) -> AssistantMessagePart:
    return AssistantMessagePart(
        parts=[ToolUsePart(id=call_id, name="search", inputs={"q": call_id})]
    )


def _result(call_id: str, text: str = "payload") -> UserMessagePart:
    return UserMessagePart(
        parts=[
            ToolResultPart(
                tool_use_id=call_id,
                name="search",
                parts=[TextPart(text=text)],
            )
        ]
    )


def _pair(call_id: str, text: str = "payload") -> tuple[AssistantMessagePart, UserMessagePart]:
    return _use(call_id), _result(call_id, text)


def _summary_messages(messages: list[Any]) -> list[UserMessagePart]:
    found: list[UserMessagePart] = []
    for message in messages:
        if not isinstance(message, UserMessagePart) or len(message.parts) != 1:
            continue
        part = message.parts[0]
        if isinstance(part, TextPart) and part.text.startswith("<summary>"):
            found.append(message)
    return found


def _llm(reply_for: Any) -> AsyncMock:
    llm = AsyncMock()
    spans: list[str] = []

    async def chat(messages: list[Any], **kwargs: Any) -> StreamEndEvent:
        span = messages[0].parts[0].text
        spans.append(span)
        text = reply_for(span) if callable(reply_for) else reply_for
        return StreamEndEvent(
            model="summarizer",
            parts=[TextPart(text=text)],
            stop_reason="end_turn",
            usage=None,
        )

    llm.chat = chat
    llm.spans = spans
    return llm


def test_repeated_summarize_folds_prior_summary_and_keeps_its_facts() -> None:
    """A second summarize must absorb the earlier digest instead of stacking another turn."""
    use_a, result_a = _pair("A", "FACT-OLD")
    note = UserMessagePart(parts=[TextPart(text="NEVER-DELETE")])
    use_b, result_b = _pair("B", "result-B")
    use_c, result_c = _pair("C", "result-C")
    preamble = UserMessagePart(parts=[TextPart(text="user question")])
    messages: list[Any] = [
        preamble,
        use_a,
        result_a,
        note,
        use_b,
        result_b,
        use_c,
        result_c,
    ]

    def reply(span: str) -> str:
        if "<summary>" not in span:
            return "FACT-OLD"
        assert "FACT-OLD" in span
        assert "NEVER-DELETE" not in span
        return "FACT-OLD plus result-C"

    llm = _llm(reply)
    policy = ContextPolicy(keep_last_n=1, mode="summarize")
    first = asyncio.run(summarize_context(messages, policy, llm))
    assert first is not None
    assert len(_summary_messages(messages)) == 1
    assert any(message is note for message in messages)

    use_d, result_d = _pair("D", "result-D")
    messages.extend((use_d, result_d))
    second = asyncio.run(summarize_context(messages, policy, llm))

    assert second is not None
    summaries = _summary_messages(messages)
    assert len(summaries) == 1
    assert summaries[0].parts[0].text == "<summary>FACT-OLD plus result-C</summary>"
    assert "FACT-OLD" in llm.spans[1]
    assert any(message is preamble for message in messages)
    assert any(message is note for message in messages)
    assert use_d in messages and result_d in messages


def test_summary_after_kept_tail_is_not_folded() -> None:
    use_a, result_a = _pair("A")
    use_b, result_b = _pair("B")
    trailing = UserMessagePart(parts=[TextPart(text="<summary>keep-me</summary>")])
    messages: list[Any] = [use_a, result_a, use_b, result_b, trailing]
    llm = _llm("digest")
    applied = asyncio.run(
        summarize_context(messages, ContextPolicy(keep_last_n=1, mode="summarize"), llm)
    )
    assert applied is not None
    assert trailing in messages
    assert len(_summary_messages(messages)) == 2
    assert "keep-me" not in llm.spans[0]


def test_user_text_mentioning_summary_is_not_removed() -> None:
    use_a, result_a = _pair("A")
    mention = UserMessagePart(parts=[TextPart(text="see <summary>notes</summary> later")])
    use_b, result_b = _pair("B")
    use_c, result_c = _pair("C")
    messages: list[Any] = [use_a, result_a, mention, use_b, result_b, use_c, result_c]
    llm = _llm("digest")
    asyncio.run(summarize_context(messages, ContextPolicy(keep_last_n=1, mode="summarize"), llm))
    assert mention in messages
    assert "notes" not in llm.spans[0]


def test_media_and_reasoning_payloads_are_not_flattened() -> None:
    image = ImagePart(source=Base64ImageSource(data=_BLOB, media_type="image/png"))
    data_url = ImagePart(source=URLImageSource(url=f"data:image/png;base64,{_BLOB}"))
    linked = ImagePart(source=URLImageSource(url="https://example.com/a.png"))
    pdf = DocumentPart(
        source=Base64PDFSource(data=_BLOB, media_type="application/pdf"),
        filename="report.pdf",
    )
    notes = DocumentPart(
        source=PlainTextSource(data="ticket TCK-1"),
        filename="notes.txt",
    )
    remote = DocumentPart(
        source=URLSource(url="https://example.com/notes.txt"),
        filename="notes.txt",
    )
    stored = DocumentPart(
        source=FileDocumentSource(file_id="file_123"),
        filename="notes.txt",
    )
    reasoning = ReasoningPart(text="decided to retry", signature=_BLOB)
    redacted = ReasoningPart(text=_BLOB, signature=_BLOB, redacted=True)
    messages = [
        UserMessagePart(parts=[image, data_url, linked, pdf, notes, remote, stored]),
        AssistantMessagePart(parts=[reasoning, redacted, TextPart(text="ok")]),
    ]

    flat = " ".join(
        part_to_text(part, labeled=True) for message in messages for part in message.parts
    )
    unlabeled = " ".join(part_to_text(part) for message in messages for part in message.parts)

    assert _BLOB not in flat
    assert _BLOB not in unlabeled
    assert "[image image/png]" in flat
    assert "https://example.com/a.png" in flat
    assert "[document report.pdf]" in flat
    assert "ticket TCK-1" in flat
    assert "https://example.com/notes.txt" in flat
    assert "file_123" in flat
    assert "decided to retry" in flat
    assert "[redacted reasoning]" in flat
    assert "signature" not in flat
    assert estimate_input_tokens(messages) < 200
    assert image.source.data == _BLOB
    assert pdf.source.data == _BLOB
    assert redacted.text == _BLOB
    assert reasoning.signature == _BLOB


def test_summarize_span_omits_image_bytes_but_keeps_the_part() -> None:
    image = ImagePart(source=Base64ImageSource(data=_BLOB, media_type="image/png"))
    use = _use("A")
    result = UserMessagePart(
        parts=[
            ToolResultPart(
                tool_use_id="A",
                name="search",
                parts=[TextPart(text="caption"), image],
            )
        ]
    )
    use_b, result_b = _pair("B")
    messages: list[Any] = [use, result, use_b, result_b]
    llm = _llm("digest")
    applied = asyncio.run(
        summarize_context(messages, ContextPolicy(keep_last_n=0, mode="summarize"), llm)
    )
    assert applied is not None
    assert _BLOB not in llm.spans[0]
    assert "[image image/png]" in llm.spans[0]
    assert "caption" in llm.spans[0]
    stashed = applied.replaced_originals
    assert stashed is not None
    assert any(
        image in part.parts
        for message in stashed
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    )
    assert image.source.data == _BLOB


def test_cleared_tool_uses_matches_results_actually_cleared() -> None:
    """Shared result messages count every payload removed, on both edit paths."""
    shared: list[Any] = [
        _use("A"),
        _use("B"),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id="A",
                    name="search",
                    parts=[TextPart(text="result-A")],
                ),
                ToolResultPart(
                    tool_use_id="B",
                    name="search",
                    parts=[TextPart(text="result-B")],
                ),
            ]
        ),
    ]
    policy = ContextPolicy(keep_last_n=1, mode="summarize")
    trimmed, trim_edit = edit_context(shared, policy)
    assert trim_edit is not None
    assert trim_edit.cleared_tool_uses == 2
    assert trimmed is not shared

    summarized = list(shared)
    llm = _llm("digest")
    summary_edit = asyncio.run(summarize_context(summarized, policy, llm))
    assert summary_edit is not None
    assert summary_edit.cleared_tool_uses == trim_edit.cleared_tool_uses == 2

    separate: list[Any] = []
    for label in ("A", "B", "C"):
        separate.extend(_pair(label))
    separate_policy = ContextPolicy(keep_last_n=1, mode="trim")
    _trimmed, separate_trim = edit_context(separate, separate_policy)
    separate_summary = asyncio.run(
        summarize_context(
            list(separate),
            ContextPolicy(keep_last_n=1, mode="summarize"),
            _llm("digest"),
        )
    )
    assert separate_trim is not None and separate_summary is not None
    assert separate_trim.cleared_tool_uses == separate_summary.cleared_tool_uses == 2


def test_replaced_originals_are_json_serializable() -> None:
    """Hosts can log ContextEditEvent.replaced_originals without a custom encoder."""
    image = ImagePart(source=Base64ImageSource(data="abc", media_type="image/png"))
    use, result = _pair("A", "payload")
    result.parts.append(image)
    messages: list[Any] = [use, result, *_pair("B")]
    applied = asyncio.run(
        summarize_context(messages, ContextPolicy(keep_last_n=0, mode="summarize"), _llm("digest"))
    )
    assert applied is not None
    event = ContextEditEvent(applied_edits=[applied])
    payload = json.loads(event.model_dump_json())
    assert payload["applied_edits"][0]["replaced_originals"]
    json.dumps(event.model_dump())
    json.dumps(event.model_dump(mode="json"))
