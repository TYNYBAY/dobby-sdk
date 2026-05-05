"""Extended Anthropic provider tests: cache tokens, multi-tool, documents, streaming errors, executor."""

import asyncio
import base64
from dataclasses import dataclass
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest

from dobby.providers.base import RateLimitError
from dobby.types import (
    DocumentPart,
    ImagePart,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningPart,
    ReasoningStartEvent,
    StreamEndEvent,
    StreamErrorEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolUseEvent,
    ToolUsePart,
    URLImageSource,
    UserMessagePart,
)
from dobby.types.document_part import Base64PDFSource, PlainTextSource, URLSource

# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------


def _make_provider():
    from dobby.providers.anthropic.adapter import AnthropicProvider

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.api_key = "test"
    provider.base_url = None
    provider._model = "claude-sonnet-4-6"
    provider.max_retries = 1
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    provider._client = client
    return provider


def _fake_response(
    blocks,
    stop_reason="end_turn",
    model="claude-sonnet-4-6",
    input_tokens=10,
    output_tokens=20,
    cache_creation_input_tokens=None,
    cache_read_input_tokens=None,
):
    resp = MagicMock()
    resp.content = blocks
    resp.stop_reason = stop_reason
    resp.model = model
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation_input_tokens
    usage.cache_read_input_tokens = cache_read_input_tokens
    resp.usage = usage
    return resp


def _text_block(text: str):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_cb(tool_id: str, tool_name: str) -> MagicMock:
    """Build a content_block mock for tool_use blocks.

    MagicMock(name=...) sets the mock's display name, not .name attribute,
    so we must assign .name explicitly after construction.
    """
    cb = MagicMock()
    cb.type = "tool_use"
    cb.id = tool_id
    cb.name = tool_name
    return cb


def _tool_use_block(tool_id: str, name: str, inputs: dict):
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = inputs
    return b


def _thinking_block(text: str, signature: str = "sig-test"):
    b = MagicMock()
    b.type = "thinking"
    b.thinking = text
    b.signature = signature
    return b


def _event(type_, **fields):
    ev = MagicMock()
    ev.type = type_
    for k, v in fields.items():
        setattr(ev, k, v)
    return ev


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _drive_stream(provider, events, **chat_kwargs):
    provider._client.messages.create.return_value = _AsyncIter(events)

    async def run():
        out = []
        stream = await provider.chat(
            messages=[UserMessagePart(parts=[TextPart(text="test")])],
            stream=True,
            **chat_kwargs,
        )
        async for ev in stream:
            out.append(ev)
        return out

    return asyncio.run(run())


def _minimal_pdf_bytes(title: str = "Claim Document", body: str = "Sample claim content") -> bytes:
    """Generate a minimal valid PDF using raw bytes — no external dependencies."""
    title_s = title.replace("(", "\\(").replace(")", "\\)")
    body_s = body.replace("(", "\\(").replace(")", "\\)")
    stream_src = f"BT /F1 14 Tf 50 750 Td ({title_s}) Tj 0 -25 Td /F1 10 Tf ({body_s}) Tj ET"
    stream_b = stream_src.encode("latin-1")

    objs: list[bytes] = [
        b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n",
        b"2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n",
        b"3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>\nendobj\n",
        f"4 0 obj\n<</Length {len(stream_b)}>>\nstream\n".encode() + stream_b + b"\nendstream\nendobj\n",
        b"5 0 obj\n<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>\nendobj\n",
    ]

    header = b"%PDF-1.4\n"
    body_bytes = b""
    offsets: list[int] = []
    pos = len(header)
    for obj in objs:
        offsets.append(pos)
        body_bytes += obj
        pos += len(obj)

    xref_pos = pos
    xref = "xref\n0 6\n0000000000 65535 f \n" + "".join(
        f"{o:010d} 00000 n \n" for o in offsets
    )
    trailer = f"trailer\n<</Size 6/Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF\n"

    return header + body_bytes + xref.encode() + trailer.encode()


def _pdf_b64(title: str = "Claim", body: str = "Content") -> str:
    return base64.b64encode(_minimal_pdf_bytes(title, body)).decode()


# ---------------------------------------------------------------------------
# Cache token tests
# ---------------------------------------------------------------------------


class TestAnthropicCacheTokens:
    """Verify cache token fields are passed through Usage correctly."""

    def test_non_streaming_cache_creation_tokens(self):
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[_text_block("ok")],
            cache_creation_input_tokens=500,
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])])
        )

        assert isinstance(result, StreamEndEvent)
        assert result.usage is not None
        assert result.usage.cache_creation_input_tokens == 500

    def test_non_streaming_cache_read_tokens(self):
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[_text_block("ok")],
            cache_read_input_tokens=1200,
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])])
        )

        assert result.usage is not None
        assert result.usage.cache_read_input_tokens == 1200

    def test_non_streaming_both_cache_fields(self):
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[_text_block("ok")],
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=700,
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])])
        )

        u = result.usage
        assert u is not None
        assert u.input_tokens == 100
        assert u.output_tokens == 50
        assert u.total_tokens == 150
        assert u.cache_creation_input_tokens == 300
        assert u.cache_read_input_tokens == 700

    def test_non_streaming_null_cache_fields(self):
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[_text_block("ok")],
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])])
        )

        u = result.usage
        assert u is not None
        assert u.cache_creation_input_tokens is None
        assert u.cache_read_input_tokens is None

    def test_streaming_cache_tokens_from_message_start(self):
        provider = _make_provider()

        msg_start = _event(
            "message_start",
            message=MagicMock(
                id="msg_c",
                model="claude-sonnet-4-6",
                usage=MagicMock(
                    input_tokens=80,
                    cache_creation_input_tokens=400,
                    cache_read_input_tokens=None,
                ),
            ),
        )
        msg_delta = _event(
            "message_delta",
            delta=MagicMock(stop_reason="end_turn"),
            usage=MagicMock(output_tokens=15),
        )
        msg_stop = _event("message_stop")

        events = _drive_stream(provider, [msg_start, msg_delta, msg_stop])

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        assert end.usage is not None
        assert end.usage.input_tokens == 80
        assert end.usage.output_tokens == 15
        assert end.usage.cache_creation_input_tokens == 400
        assert end.usage.cache_read_input_tokens is None

    def test_streaming_cache_read_tokens_non_zero(self):
        provider = _make_provider()

        msg_start = _event(
            "message_start",
            message=MagicMock(
                id="msg_cr",
                model="claude-sonnet-4-6",
                usage=MagicMock(
                    input_tokens=50,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=2000,
                ),
            ),
        )
        msg_delta = _event(
            "message_delta",
            delta=MagicMock(stop_reason="end_turn"),
            usage=MagicMock(output_tokens=10),
        )
        msg_stop = _event("message_stop")

        events = _drive_stream(provider, [msg_start, msg_delta, msg_stop])

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        assert end.usage.cache_read_input_tokens == 2000
        assert end.usage.cache_creation_input_tokens == 0


# ---------------------------------------------------------------------------
# Multi-tool call tests
# ---------------------------------------------------------------------------


class TestAnthropicMultiToolCall:
    """Provider correctly handles multiple tool calls in one LLM response."""

    def test_non_streaming_two_tool_calls(self):
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[
                _tool_use_block("c1", "search", {"q": "alpha"}),
                _tool_use_block("c2", "lookup", {"id": "42"}),
            ],
            stop_reason="tool_use",
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="go")])])
        )

        tool_parts = [p for p in result.parts if isinstance(p, ToolUsePart)]
        assert len(tool_parts) == 2
        assert tool_parts[0].id == "c1"
        assert tool_parts[0].name == "search"
        assert tool_parts[1].id == "c2"
        assert tool_parts[1].name == "lookup"

    def test_non_streaming_three_tool_calls_order_preserved(self):
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[
                _tool_use_block("t1", "tool_a", {"x": 1}),
                _tool_use_block("t2", "tool_b", {"x": 2}),
                _tool_use_block("t3", "tool_c", {"x": 3}),
            ],
            stop_reason="tool_use",
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="go")])])
        )

        tool_parts = [p for p in result.parts if isinstance(p, ToolUsePart)]
        assert [p.name for p in tool_parts] == ["tool_a", "tool_b", "tool_c"]
        assert [p.inputs["x"] for p in tool_parts] == [1, 2, 3]

    def test_non_streaming_text_plus_two_tools(self):
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[
                _text_block("Let me run both."),
                _tool_use_block("c1", "fetch", {"url": "a"}),
                _tool_use_block("c2", "fetch", {"url": "b"}),
            ],
            stop_reason="tool_use",
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="go")])])
        )

        text_parts = [p for p in result.parts if isinstance(p, TextPart)]
        tool_parts = [p for p in result.parts if isinstance(p, ToolUsePart)]
        assert len(text_parts) == 1
        assert text_parts[0].text == "Let me run both."
        assert len(tool_parts) == 2
        assert tool_parts[0].inputs == {"url": "a"}
        assert tool_parts[1].inputs == {"url": "b"}

    def test_streaming_two_tool_calls_assembled(self):
        provider = _make_provider()

        msg_start = _event("message_start", message=MagicMock(id="m", model="m", usage=None))
        # First tool
        t1_start = _event("content_block_start", content_block=_tool_cb("c1", "alpha"))
        t1_p1 = _event("content_block_delta", delta=MagicMock(type="input_json_delta", partial_json='{"k":'))
        t1_p2 = _event("content_block_delta", delta=MagicMock(type="input_json_delta", partial_json='"v1"}'))
        t1_stop = _event("content_block_stop")
        # Second tool
        t2_start = _event("content_block_start", content_block=_tool_cb("c2", "beta"))
        t2_p1 = _event("content_block_delta", delta=MagicMock(type="input_json_delta", partial_json='{"k":'))
        t2_p2 = _event("content_block_delta", delta=MagicMock(type="input_json_delta", partial_json='"v2"}'))
        t2_stop = _event("content_block_stop")

        msg_delta = _event("message_delta", delta=MagicMock(stop_reason="tool_use"), usage=MagicMock(output_tokens=5))
        msg_stop = _event("message_stop")

        events = _drive_stream(
            provider,
            [msg_start, t1_start, t1_p1, t1_p2, t1_stop, t2_start, t2_p1, t2_p2, t2_stop, msg_delta, msg_stop],
        )

        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert len(tool_events) == 2
        assert tool_events[0].id == "c1"
        assert tool_events[0].inputs == {"k": "v1"}
        assert tool_events[1].id == "c2"
        assert tool_events[1].inputs == {"k": "v2"}

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        tool_parts = [p for p in end.parts if isinstance(p, ToolUsePart)]
        assert len(tool_parts) == 2
        assert [p.name for p in tool_parts] == ["alpha", "beta"]

    def test_streaming_empty_tool_input(self):
        """Tool with no inputs (empty JSON) should not crash."""
        provider = _make_provider()

        msg_start = _event("message_start", message=MagicMock(id="m", model="m", usage=None))
        t_start = _event("content_block_start", content_block=_tool_cb("c1", "ping"))
        # No input_json_delta events (empty input)
        t_stop = _event("content_block_stop")
        msg_delta = _event("message_delta", delta=MagicMock(stop_reason="tool_use"), usage=MagicMock(output_tokens=1))
        msg_stop = _event("message_stop")

        events = _drive_stream(provider, [msg_start, t_start, t_stop, msg_delta, msg_stop])

        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].inputs == {}


# ---------------------------------------------------------------------------
# Document / PDF handling tests
# ---------------------------------------------------------------------------


class TestAnthropicDocumentHandling:
    """PDF and document content parts convert correctly for Anthropic."""

    def test_pdf_base64_in_user_message_conversion(self):
        from dobby.providers.anthropic.adapter import to_anthropic_messages

        pdf_data = _pdf_b64("Test Claim", "Claim amount: $5000")
        messages = [
            UserMessagePart(
                parts=[
                    DocumentPart(
                        source=Base64PDFSource(data=pdf_data, media_type="application/pdf"),
                        filename="claim.pdf",
                    )
                ]
            )
        ]
        result = to_anthropic_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        doc_block = result[0]["content"][0]
        assert doc_block["type"] == "document"
        assert doc_block["source"]["type"] == "base64"
        assert doc_block["source"]["media_type"] == "application/pdf"
        assert doc_block["source"]["data"] == pdf_data

    def test_pdf_url_in_user_message_conversion(self):
        from dobby.providers.anthropic.adapter import to_anthropic_messages

        messages = [
            UserMessagePart(
                parts=[
                    DocumentPart(
                        source=URLSource(url="https://example.com/claim.pdf"),
                        filename="claim.pdf",
                    )
                ]
            )
        ]
        result = to_anthropic_messages(messages)

        doc_block = result[0]["content"][0]
        assert doc_block["type"] == "document"
        assert doc_block["source"]["type"] == "url"
        assert doc_block["source"]["url"] == "https://example.com/claim.pdf"

    def test_text_and_pdf_in_same_message(self):
        from dobby.providers.anthropic.adapter import to_anthropic_messages

        pdf_data = _pdf_b64()
        messages = [
            UserMessagePart(
                parts=[
                    TextPart(text="Please review this claim:"),
                    DocumentPart(
                        source=Base64PDFSource(data=pdf_data, media_type="application/pdf"),
                        filename="claim.pdf",
                    ),
                ]
            )
        ]
        result = to_anthropic_messages(messages)

        content = result[0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Please review this claim:"}
        assert content[1]["type"] == "document"

    def test_multiple_pdfs_in_message(self):
        from dobby.providers.anthropic.adapter import to_anthropic_messages

        claim_pdf = _pdf_b64("Claim Form", "Loss amount: $10,000")
        evidence_pdf = _pdf_b64("Evidence", "Photos of damage")
        messages = [
            UserMessagePart(
                parts=[
                    DocumentPart(
                        source=Base64PDFSource(data=claim_pdf, media_type="application/pdf"),
                        filename="claim_form.pdf",
                    ),
                    DocumentPart(
                        source=Base64PDFSource(data=evidence_pdf, media_type="application/pdf"),
                        filename="evidence.pdf",
                    ),
                ]
            )
        ]
        result = to_anthropic_messages(messages)

        content = result[0]["content"]
        assert len(content) == 2
        assert all(b["type"] == "document" for b in content)
        assert content[0]["source"]["data"] == claim_pdf
        assert content[1]["source"]["data"] == evidence_pdf

    def test_plain_text_document_falls_back_to_text_block(self):
        from dobby.providers.anthropic.adapter import to_anthropic_messages

        messages = [
            UserMessagePart(
                parts=[
                    DocumentPart(
                        source=PlainTextSource(data="Policy number: POL-123456"),
                        filename="policy.txt",
                    )
                ]
            )
        ]
        result = to_anthropic_messages(messages)

        block = result[0]["content"][0]
        assert block["type"] == "text"
        assert "POL-123456" in block["text"]

    def test_pdf_passed_to_provider_non_streaming(self):
        """End-to-end: PDF in UserMessagePart flows correctly to messages.create kwargs."""
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[_text_block("Claim approved for $5000.")]
        )

        pdf_data = _pdf_b64("Liability Claim", "Claimant: John Doe. Loss: $5000.")

        async def run():
            return await provider.chat(
                messages=[
                    UserMessagePart(
                        parts=[
                            TextPart(text="Underwrite this claim:"),
                            DocumentPart(
                                source=Base64PDFSource(data=pdf_data, media_type="application/pdf"),
                                filename="liability_claim.pdf",
                            ),
                        ]
                    )
                ]
            )

        result = asyncio.run(run())

        assert isinstance(result, StreamEndEvent)
        assert "5000" in result.parts[0].text

        call_kwargs = provider._client.messages.create.call_args.kwargs
        sent_messages = call_kwargs["messages"]
        assert sent_messages[0]["role"] == "user"
        content = sent_messages[0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "document"
        assert content[1]["source"]["data"] == pdf_data


# ---------------------------------------------------------------------------
# Streaming error event tests
# ---------------------------------------------------------------------------


class TestAnthropicStreamingErrors:
    """StreamErrorEvent is emitted when Anthropic sends an error event."""

    def test_stream_error_event_is_yielded(self):
        provider = _make_provider()

        msg_start = _event("message_start", message=MagicMock(id="m", model="m", usage=None))
        err_event = _event(
            "error",
            error=MagicMock(type="overloaded_error", message="Server overloaded"),
        )

        events = _drive_stream(provider, [msg_start, err_event])

        error_events = [e for e in events if isinstance(e, StreamErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error_code == "overloaded_error"
        assert "overloaded" in error_events[0].error_message.lower()

    def test_stream_error_event_has_code_and_message(self):
        provider = _make_provider()

        msg_start = _event("message_start", message=MagicMock(id="m", model="m", usage=None))
        err_event = _event(
            "error",
            error=MagicMock(type="rate_limit_exceeded", message="Too many requests"),
        )
        msg_stop = _event("message_stop")

        events = _drive_stream(provider, [msg_start, err_event, msg_stop])

        error_events = [e for e in events if isinstance(e, StreamErrorEvent)]
        assert error_events[0].error_code == "rate_limit_exceeded"
        assert "Too many requests" in error_events[0].error_message

    def test_stream_error_does_not_prevent_message_stop(self):
        """A StreamErrorEvent mid-stream should not prevent message_stop from firing."""
        provider = _make_provider()

        msg_start = _event("message_start", message=MagicMock(id="m", model="m", usage=None))
        err_event = _event("error", error=MagicMock(type="timeout", message="Request timed out"))
        msg_delta = _event("message_delta", delta=MagicMock(stop_reason="end_turn"), usage=MagicMock(output_tokens=0))
        msg_stop = _event("message_stop")

        events = _drive_stream(provider, [msg_start, err_event, msg_delta, msg_stop])

        assert any(isinstance(e, StreamErrorEvent) for e in events)
        assert any(isinstance(e, StreamEndEvent) for e in events)


# ---------------------------------------------------------------------------
# Mixed streaming content tests
# ---------------------------------------------------------------------------


class TestAnthropicMixedStreamContent:
    """Streaming responses combining thinking, text, and tool calls."""

    def test_streaming_thinking_then_text(self):
        provider = _make_provider()

        msg_start = _event("message_start", message=MagicMock(id="m", model="m", usage=None))
        think_start = _event("content_block_start", content_block=MagicMock(type="thinking"))
        think_delta = _event("content_block_delta", delta=MagicMock(type="thinking_delta", thinking="Analyzing..."))
        sig_delta = _event("content_block_delta", delta=MagicMock(type="signature_delta", signature="sig-mix"))
        think_stop = _event("content_block_stop")
        text_start = _event("content_block_start", content_block=MagicMock(type="text"))
        text_delta = _event("content_block_delta", delta=MagicMock(type="text_delta", text="Result: approved"))
        text_stop = _event("content_block_stop")
        msg_delta = _event("message_delta", delta=MagicMock(stop_reason="end_turn"), usage=MagicMock(output_tokens=20))
        msg_stop = _event("message_stop")

        events = _drive_stream(
            provider,
            [msg_start, think_start, think_delta, sig_delta, think_stop,
             text_start, text_delta, text_stop, msg_delta, msg_stop],
            reasoning_effort=5000,
        )

        assert any(isinstance(e, ReasoningStartEvent) for e in events)
        assert any(isinstance(e, ReasoningEndEvent) for e in events)
        assert any(isinstance(e, TextDeltaEvent) for e in events)

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        reasoning_parts = [p for p in end.parts if isinstance(p, ReasoningPart)]
        text_parts = [p for p in end.parts if isinstance(p, TextPart)]
        assert reasoning_parts[0].text == "Analyzing..."
        assert reasoning_parts[0].signature == "sig-mix"
        assert text_parts[0].text == "Result: approved"

    def test_streaming_thinking_text_and_tool(self):
        """All three block types in one stream — full mixed response."""
        provider = _make_provider()

        msg_start = _event("message_start", message=MagicMock(id="m", model="m", usage=None))
        # Thinking block
        think_start = _event("content_block_start", content_block=MagicMock(type="thinking"))
        think_delta = _event("content_block_delta", delta=MagicMock(type="thinking_delta", thinking="Need data."))
        sig_delta = _event("content_block_delta", delta=MagicMock(type="signature_delta", signature="s1"))
        think_stop = _event("content_block_stop")
        # Text block
        text_start = _event("content_block_start", content_block=MagicMock(type="text"))
        text_delta = _event("content_block_delta", delta=MagicMock(type="text_delta", text="Let me fetch."))
        text_stop = _event("content_block_stop")
        # Tool block
        tool_start = _event("content_block_start", content_block=_tool_cb("cx", "fetch"))
        tool_delta = _event("content_block_delta", delta=MagicMock(type="input_json_delta", partial_json='{"id":"99"}'))
        tool_stop = _event("content_block_stop")

        msg_delta = _event("message_delta", delta=MagicMock(stop_reason="tool_use"), usage=MagicMock(output_tokens=30))
        msg_stop = _event("message_stop")

        events = _drive_stream(
            provider,
            [msg_start, think_start, think_delta, sig_delta, think_stop,
             text_start, text_delta, text_stop, tool_start, tool_delta, tool_stop,
             msg_delta, msg_stop],
            reasoning_effort=8000,
        )

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        assert any(isinstance(p, ReasoningPart) for p in end.parts)
        assert any(isinstance(p, TextPart) for p in end.parts)
        assert any(isinstance(p, ToolUsePart) for p in end.parts)

        tool_parts = [p for p in end.parts if isinstance(p, ToolUsePart)]
        assert tool_parts[0].inputs == {"id": "99"}

    def test_text_accumulates_across_multiple_deltas(self):
        provider = _make_provider()

        msg_start = _event("message_start", message=MagicMock(id="m", model="m", usage=None))
        cb_start = _event("content_block_start", content_block=MagicMock(type="text"))
        chunks = [
            _event("content_block_delta", delta=MagicMock(type="text_delta", text=chunk))
            for chunk in ["The ", "claim ", "is ", "valid."]
        ]
        cb_stop = _event("content_block_stop")
        msg_delta = _event("message_delta", delta=MagicMock(stop_reason="end_turn"), usage=MagicMock(output_tokens=4))
        msg_stop = _event("message_stop")

        events = _drive_stream(provider, [msg_start, cb_start] + chunks + [cb_stop, msg_delta, msg_stop])

        text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert "".join(d.delta for d in text_deltas) == "The claim is valid."

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        text_parts = [p for p in end.parts if isinstance(p, TextPart)]
        assert text_parts[0].text == "The claim is valid."


# ---------------------------------------------------------------------------
# AgentExecutor + Anthropic integration tests
# ---------------------------------------------------------------------------


@dataclass
class _LookupTool:
    """Tool for testing executor integration."""

    name = "lookup"
    description = "Look up a policy by ID"

    async def __call__(
        self,
        policy_id: Annotated[str, "The policy ID"],
    ) -> str:
        return f"Policy {policy_id}: Active, coverage $50,000"


# Make it a proper Tool subclass
from dobby.tools import Tool as _ToolBase


@dataclass
class LookupPolicyTool(_ToolBase):
    name = "lookup_policy"
    description = "Look up a policy by ID and return coverage details"

    async def __call__(
        self,
        policy_id: Annotated[str, "The policy ID to look up"],
    ) -> str:
        return f"Policy {policy_id}: Active, $50,000 coverage"


class TestAgentExecutorAnthropicIntegration:
    """AgentExecutor wired to Anthropic provider: schema generation and agentic loop."""

    def _make_executor(self, tools=None):
        from dobby.executor import AgentExecutor

        provider = _make_provider()
        executor = AgentExecutor(
            provider="anthropic",
            llm=provider,
            tools=tools or [],
        )
        return executor, provider

    def test_anthropic_tool_schema_has_input_schema_key(self):
        executor, _ = self._make_executor(tools=[LookupPolicyTool()])

        schemas = executor.get_tools_schema()

        assert len(schemas) == 1
        schema = schemas[0]
        assert "input_schema" in schema
        assert schema["name"] == "lookup_policy"
        assert "description" in schema

    def test_anthropic_tool_schema_required_fields(self):
        executor, _ = self._make_executor(tools=[LookupPolicyTool()])

        schema = executor.get_tools_schema()[0]
        input_schema = schema["input_schema"]
        assert input_schema["type"] == "object"
        assert "policy_id" in input_schema["properties"]
        assert "policy_id" in input_schema.get("required", [])

    def test_anthropic_tool_schema_not_openai_format(self):
        """Anthropic format must NOT use {'type': 'function', ...} wrapper."""
        executor, _ = self._make_executor(tools=[LookupPolicyTool()])

        schema = executor.get_tools_schema()[0]
        assert schema.get("type") != "function"
        assert "input_schema" in schema

    def test_executor_run_stream_no_tools_yields_stream_end(self):
        """With no tool calls, executor completes in one iteration."""
        executor, provider = self._make_executor()

        msg_start = _event("message_start", message=MagicMock(id="r1", model="m", usage=None))
        cb_start = _event("content_block_start", content_block=MagicMock(type="text"))
        text_delta = _event("content_block_delta", delta=MagicMock(type="text_delta", text="Done."))
        cb_stop = _event("content_block_stop")
        msg_delta = _event("message_delta", delta=MagicMock(stop_reason="end_turn"), usage=MagicMock(output_tokens=1))
        msg_stop = _event("message_stop")

        provider._client.messages.create.return_value = _AsyncIter(
            [msg_start, cb_start, text_delta, cb_stop, msg_delta, msg_stop]
        )

        async def run():
            events = []
            async for ev in executor.run_stream(
                messages=[UserMessagePart(parts=[TextPart(text="summarize")])],
            ):
                events.append(ev)
            return events

        events = asyncio.run(run())
        assert any(isinstance(e, StreamEndEvent) for e in events)
        end = next(e for e in events if isinstance(e, StreamEndEvent))
        assert end.stop_reason == "end_turn"

    def test_executor_run_stream_one_tool_call_then_final_response(self):
        """Executor invokes the tool, appends result, then gets final text response."""
        executor, provider = self._make_executor(tools=[LookupPolicyTool()])

        # First LLM turn: requests tool call
        def _tool_stream_events():
            msg_start = _event("message_start", message=MagicMock(id="r1", model="m", usage=None))
            t_start = _event("content_block_start", content_block=_tool_cb("tc1", "lookup_policy"))
            t_delta = _event("content_block_delta", delta=MagicMock(type="input_json_delta", partial_json='{"policy_id":"POL-001"}'))
            t_stop = _event("content_block_stop")
            msg_delta = _event("message_delta", delta=MagicMock(stop_reason="tool_use"), usage=MagicMock(output_tokens=5))
            msg_stop = _event("message_stop")
            return [msg_start, t_start, t_delta, t_stop, msg_delta, msg_stop]

        # Second LLM turn: final answer
        def _final_stream_events():
            msg_start = _event("message_start", message=MagicMock(id="r2", model="m", usage=None))
            cb_start = _event("content_block_start", content_block=MagicMock(type="text"))
            text_d = _event("content_block_delta", delta=MagicMock(type="text_delta", text="Policy POL-001 is active."))
            cb_stop = _event("content_block_stop")
            msg_delta = _event("message_delta", delta=MagicMock(stop_reason="end_turn"), usage=MagicMock(output_tokens=8))
            msg_stop = _event("message_stop")
            return [msg_start, cb_start, text_d, cb_stop, msg_delta, msg_stop]

        call_count = 0

        async def _create_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _AsyncIter(_tool_stream_events())
            return _AsyncIter(_final_stream_events())

        provider._client.messages.create.side_effect = _create_side_effect

        async def run():
            events = []
            async for ev in executor.run_stream(
                messages=[UserMessagePart(parts=[TextPart(text="Check policy POL-001")])],
            ):
                events.append(ev)
            return events

        events = asyncio.run(run())

        # Should have had two LLM calls
        assert call_count == 2

        # The final StreamEndEvent should contain the text response
        end_events = [e for e in events if isinstance(e, StreamEndEvent)]
        final_end = end_events[-1]
        text_parts = [p for p in final_end.parts if isinstance(p, TextPart)]
        assert text_parts[0].text == "Policy POL-001 is active."

    def test_executor_tool_schema_cached_on_second_call(self):
        """get_tools_schema() returns identical object on repeated calls (cached)."""
        executor, _ = self._make_executor(tools=[LookupPolicyTool()])

        schemas1 = executor.get_tools_schema()
        schemas2 = executor.get_tools_schema()
        assert schemas1 is schemas2


# ---------------------------------------------------------------------------
# Usage edge-case tests
# ---------------------------------------------------------------------------


class TestAnthropicUsageEdgeCases:
    """Edge cases in Usage construction and token math."""

    def test_usage_total_tokens_is_sum(self):
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[_text_block("ok")],
            input_tokens=123,
            output_tokens=456,
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])])
        )

        assert result.usage.total_tokens == 579

    def test_streaming_usage_total_tokens(self):
        provider = _make_provider()

        msg_start = _event(
            "message_start",
            message=MagicMock(id="m", model="m", usage=MagicMock(
                input_tokens=300,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            )),
        )
        msg_delta = _event("message_delta", delta=MagicMock(stop_reason="end_turn"), usage=MagicMock(output_tokens=150))
        msg_stop = _event("message_stop")

        events = _drive_stream(provider, [msg_start, msg_delta, msg_stop])
        end = next(e for e in events if isinstance(e, StreamEndEvent))
        assert end.usage.total_tokens == 450

    def test_non_streaming_no_usage_field(self):
        """If response.usage is falsy, StreamEndEvent.usage should be None."""
        provider = _make_provider()
        resp = MagicMock()
        resp.content = [_text_block("ok")]
        resp.stop_reason = "end_turn"
        resp.model = "m"
        resp.usage = None
        provider._client.messages.create.return_value = resp

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])])
        )

        assert result.usage is None

    def test_zero_output_tokens(self):
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[_text_block("")],
            input_tokens=50,
            output_tokens=0,
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])])
        )

        assert result.usage.output_tokens == 0
        assert result.usage.total_tokens == 50


# ---------------------------------------------------------------------------
# Build kwargs extended tests
# ---------------------------------------------------------------------------


class TestAnthropicBuildKwargsExtended:
    """Additional _build_kwargs coverage."""

    def test_custom_max_tokens_passed_through(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-6",
            messages=[],
            max_tokens=1024,
        )
        assert kwargs["max_tokens"] == 1024

    def test_no_system_key_when_none(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-6",
            messages=[],
            system=None,
        )
        assert "system" not in kwargs

    def test_no_tools_key_when_none(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-6",
            messages=[],
            tools=None,
        )
        assert "tools" not in kwargs

    def test_no_thinking_key_when_none(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-6",
            messages=[],
            thinking=None,
        )
        assert "thinking" not in kwargs

    def test_temperature_unchanged_without_thinking(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-6",
            messages=[],
            temperature=0.7,
        )
        assert kwargs["temperature"] == 0.7

    def test_tools_key_present_when_provided(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        tool_schema = [{"name": "search", "description": "web search", "input_schema": {}}]
        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-6",
            messages=[],
            tools=tool_schema,
        )
        assert kwargs["tools"] == tool_schema

    def test_non_streaming_max_tokens_override_end_to_end(self):
        """Custom max_tokens flows from chat() all the way to messages.create kwargs."""
        provider = _make_provider()
        provider._client.messages.create.return_value = _fake_response(blocks=[_text_block("ok")])

        asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                max_tokens=512,
            )
        )

        kwargs = provider._client.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 512

    def test_streaming_max_tokens_override_end_to_end(self):
        """Custom max_tokens flows through streaming path."""
        provider = _make_provider()

        msg_start = _event("message_start", message=MagicMock(id="m", model="m", usage=None))
        msg_stop = _event("message_stop")
        _drive_stream(provider, [msg_start, msg_stop], max_tokens=256)

        kwargs = provider._client.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 256


# ---------------------------------------------------------------------------
# Tool result content tests
# ---------------------------------------------------------------------------


class TestAnthropicToolResultContent:
    """Tool results containing various content types convert correctly."""

    def test_tool_result_with_image_content(self):
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart, ToolResultPart

        messages = [
            UserMessagePart(parts=[TextPart(text="describe image")]),
            AssistantMessagePart(
                parts=[ToolUsePart(id="c1", name="capture_image", inputs={})]
            ),
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id="c1",
                        name="capture_image",
                        parts=[
                            ImagePart(source=URLImageSource(url="https://example.com/img.jpg")),
                        ],
                    )
                ]
            ),
        ]

        result = to_anthropic_messages(messages)

        assert result[2]["role"] == "user"
        tool_result_block = result[2]["content"][0]
        assert tool_result_block["type"] == "tool_result"
        assert tool_result_block["tool_use_id"] == "c1"
        # Image content should be in tool_result's content list
        img_content = tool_result_block["content"]
        assert img_content[0]["type"] == "image"

    def test_tool_result_with_text_and_pdf(self):
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart, ToolResultPart

        pdf_data = _pdf_b64("Result", "Extraction complete")
        messages = [
            UserMessagePart(parts=[TextPart(text="extract")]),
            AssistantMessagePart(
                parts=[ToolUsePart(id="c1", name="extract_pdf", inputs={"url": "x"})]
            ),
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id="c1",
                        name="extract_pdf",
                        parts=[
                            TextPart(text="Extracted successfully"),
                            DocumentPart(
                                source=Base64PDFSource(data=pdf_data, media_type="application/pdf"),
                                filename="result.pdf",
                            ),
                        ],
                    )
                ]
            ),
        ]

        result = to_anthropic_messages(messages)

        tool_result_block = result[2]["content"][0]
        assert tool_result_block["type"] == "tool_result"
        content = tool_result_block["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "document"

    def test_tool_result_is_error_false_by_default(self):
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import ToolResultPart

        messages = [
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id="c1",
                        name="search",
                        parts=[TextPart(text="found")],
                    )
                ]
            )
        ]
        result = to_anthropic_messages(messages)

        assert result[0]["content"][0]["is_error"] is False


# ---------------------------------------------------------------------------
# Provider initialization tests
# ---------------------------------------------------------------------------


class TestAnthropicProviderInit:
    """Provider construction and property tests."""

    def test_default_provider_model_property(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        p = AnthropicProvider.__new__(AnthropicProvider)
        p._model = "claude-sonnet-4-6"
        assert p.model == "claude-sonnet-4-6"

    def test_azure_base_url_sets_azure_name(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        p = AnthropicProvider.__new__(AnthropicProvider)
        p.base_url = "https://my-resource.azure.com/openai"
        assert p.name == "azure-anthropic"

    def test_non_azure_base_url_keeps_anthropic_name(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        p = AnthropicProvider.__new__(AnthropicProvider)
        p.base_url = "https://my-proxy.example.com"
        assert p.name == "anthropic"

    def test_none_base_url_is_anthropic(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        p = AnthropicProvider.__new__(AnthropicProvider)
        p.base_url = None
        assert p.name == "anthropic"

    def test_stop_reason_defaults_to_end_turn_when_none(self):
        """If Anthropic returns stop_reason=None, we default to 'end_turn'."""
        provider = _make_provider()
        resp = MagicMock()
        resp.content = [_text_block("ok")]
        resp.stop_reason = None
        resp.model = "m"
        resp.usage = MagicMock(
            input_tokens=1, output_tokens=1,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        )
        provider._client.messages.create.return_value = resp

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])])
        )

        assert result.stop_reason == "end_turn"
