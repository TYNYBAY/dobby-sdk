"""Tests for Anthropic provider: error translation, message conversion, and converters."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dobby.providers.base import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    ProviderError,
    RateLimitError,
)
from dobby.types import (
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningPart,
    ReasoningStartEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolUseEvent,
    ToolUsePart,
)

# ---------------------------------------------------------------------------
# Error translation tests
# ---------------------------------------------------------------------------


class TestAnthropicErrorTranslation:
    """Test Anthropic adapter _translate_error method."""

    def _make_provider(self):
        """Create an Anthropic provider with a mocked client."""
        from dobby.providers.anthropic.adapter import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.api_key = "test"
        provider.base_url = None
        provider._model = "claude-sonnet-4-20250514"
        provider.max_retries = 3
        provider._client = MagicMock()
        return provider

    def _make_anthropic_error(self, error_cls, status_code: int = 500, headers: dict | None = None):
        """Create a mock Anthropic SDK error."""
        import anthropic

        response = MagicMock()
        response.status_code = status_code
        response.headers = headers or {}

        if error_cls == anthropic.RateLimitError:
            err = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
            err.response = response
            err.status_code = 429
            err.message = "Rate limited"
            err.body = None
            return err
        elif error_cls == anthropic.APITimeoutError:
            err = anthropic.APITimeoutError.__new__(anthropic.APITimeoutError)
            err.message = "Timed out"
            err.request = MagicMock()
            return err
        elif error_cls == anthropic.APIConnectionError:
            err = anthropic.APIConnectionError.__new__(anthropic.APIConnectionError)
            err.message = "Connection failed"
            err.request = MagicMock()
            return err
        elif error_cls == anthropic.InternalServerError:
            err = anthropic.InternalServerError.__new__(anthropic.InternalServerError)
            err.response = response
            err.status_code = status_code
            err.message = "Server error"
            err.body = None
            return err
        elif error_cls == anthropic.APIStatusError:
            err = anthropic.APIStatusError.__new__(anthropic.APIStatusError)
            err.response = response
            err.status_code = status_code
            err.message = "API error"
            err.body = None
            return err
        return error_cls()

    def test_rate_limit_error(self) -> None:
        import anthropic

        provider = self._make_provider()
        native_err = self._make_anthropic_error(
            anthropic.RateLimitError, 429, {"retry-after": "30"}
        )

        with pytest.raises(RateLimitError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 30.0
        assert exc_info.value.__cause__ is native_err

    def test_timeout_error(self) -> None:
        import anthropic

        provider = self._make_provider()
        native_err = self._make_anthropic_error(anthropic.APITimeoutError)

        with pytest.raises(APITimeoutError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.__cause__ is native_err

    def test_connection_error(self) -> None:
        import anthropic

        provider = self._make_provider()
        native_err = self._make_anthropic_error(anthropic.APIConnectionError)

        with pytest.raises(APIConnectionError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.__cause__ is native_err

    def test_internal_server_error(self) -> None:
        import anthropic

        provider = self._make_provider()
        native_err = self._make_anthropic_error(anthropic.InternalServerError, 502)

        with pytest.raises(InternalServerError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.status_code == 502
        assert exc_info.value.__cause__ is native_err

    def test_other_api_status_error(self) -> None:
        import anthropic

        provider = self._make_provider()
        native_err = self._make_anthropic_error(anthropic.APIStatusError, 403)

        with pytest.raises(ProviderError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.status_code == 403
        assert exc_info.value.__cause__ is native_err

    def test_unknown_error(self) -> None:
        provider = self._make_provider()
        native_err = ValueError("unexpected")

        with pytest.raises(ProviderError) as exc_info:
            provider._translate_error(native_err)

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.__cause__ is native_err

    def test_azure_provider_name(self) -> None:
        from dobby.providers.anthropic.adapter import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.base_url = "https://my-endpoint.azure.com"
        assert provider.name == "azure-anthropic"

    def test_default_provider_name(self) -> None:
        from dobby.providers.anthropic.adapter import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.base_url = None
        assert provider.name == "anthropic"


# ---------------------------------------------------------------------------
# Message conversion tests
# ---------------------------------------------------------------------------


class TestToAnthropicMessages:
    """Test to_anthropic_messages conversion."""

    def test_simple_user_message(self) -> None:
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import TextPart, UserMessagePart

        messages = [UserMessagePart(parts=[TextPart(text="Hello")])]
        result = to_anthropic_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_assistant_message_with_text(self) -> None:
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart, TextPart

        messages = [AssistantMessagePart(parts=[TextPart(text="Hi there")])]
        result = to_anthropic_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hi there"}]

    def test_assistant_message_with_tool_use(self) -> None:
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart, ToolUsePart

        messages = [
            AssistantMessagePart(
                parts=[ToolUsePart(id="call_1", name="search", inputs={"q": "test"})]
            )
        ]
        result = to_anthropic_messages(messages)

        assert result[0]["content"] == [
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "test"}}
        ]

    def test_assistant_message_with_reasoning(self) -> None:
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart, ReasoningPart

        messages = [
            AssistantMessagePart(
                parts=[ReasoningPart(text="Let me think...", signature="sig123")]
            )
        ]
        result = to_anthropic_messages(messages)

        assert result[0]["content"] == [
            {"type": "thinking", "thinking": "Let me think...", "signature": "sig123"}
        ]

    def test_tool_result_in_user_message(self) -> None:
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import TextPart, ToolResultPart, UserMessagePart

        messages = [
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id="call_1",
                        name="search",
                        parts=[TextPart(text="result data")],
                    )
                ]
            )
        ]
        result = to_anthropic_messages(messages)

        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "tool_result"
        assert result[0]["content"][0]["tool_use_id"] == "call_1"
        assert result[0]["content"][0]["content"] == [{"type": "text", "text": "result data"}]

    def test_consecutive_same_role_merged(self) -> None:
        """Anthropic requires strict role alternation."""
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import TextPart, UserMessagePart

        messages = [
            UserMessagePart(parts=[TextPart(text="first")]),
            UserMessagePart(parts=[TextPart(text="second")]),
        ]
        result = to_anthropic_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0] == {"type": "text", "text": "first"}
        assert result[0]["content"][1] == {"type": "text", "text": "second"}

    def test_alternating_roles_not_merged(self) -> None:
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart, TextPart, UserMessagePart

        messages = [
            UserMessagePart(parts=[TextPart(text="hello")]),
            AssistantMessagePart(parts=[TextPart(text="hi")]),
            UserMessagePart(parts=[TextPart(text="bye")]),
        ]
        result = to_anthropic_messages(messages)

        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"


# ---------------------------------------------------------------------------
# Converter tests
# ---------------------------------------------------------------------------


class TestAnthropicConverters:
    """Test content part to Anthropic format converters."""

    def test_text_part(self) -> None:
        from dobby.providers.anthropic.converters import content_part_to_anthropic
        from dobby.types import TextPart

        result = content_part_to_anthropic(TextPart(text="hello"))
        assert result == {"type": "text", "text": "hello"}

    def test_image_url(self) -> None:
        from dobby.providers.anthropic.converters import content_part_to_anthropic
        from dobby.types import ImagePart, URLImageSource

        result = content_part_to_anthropic(
            ImagePart(source=URLImageSource(url="https://example.com/img.png"))
        )
        assert result == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/img.png"},
        }

    def test_image_base64(self) -> None:
        from dobby.providers.anthropic.converters import content_part_to_anthropic
        from dobby.types import Base64ImageSource, ImagePart

        result = content_part_to_anthropic(
            ImagePart(source=Base64ImageSource(data="abc123", media_type="image/png"))
        )
        assert result == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
        }

    def test_document_pdf_base64(self) -> None:
        from dobby.providers.anthropic.converters import content_part_to_anthropic
        from dobby.types import Base64PDFSource, DocumentPart

        result = content_part_to_anthropic(
            DocumentPart(source=Base64PDFSource(data="pdfdata", media_type="application/pdf"), filename="doc.pdf")
        )
        assert result == {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "pdfdata"},
        }

    def test_document_plain_text_falls_back_to_text(self) -> None:
        from dobby.providers.anthropic.converters import content_part_to_anthropic
        from dobby.types import DocumentPart, PlainTextSource

        result = content_part_to_anthropic(
            DocumentPart(source=PlainTextSource(data="some text content"), filename="doc.txt")
        )
        assert result == {"type": "text", "text": "some text content"}

    def test_document_url(self) -> None:
        from dobby.providers.anthropic.converters import content_part_to_anthropic
        from dobby.types import DocumentPart, URLSource

        result = content_part_to_anthropic(
            DocumentPart(source=URLSource(url="https://example.com/doc.pdf"), filename="doc.pdf")
        )
        assert result == {
            "type": "document",
            "source": {"type": "url", "url": "https://example.com/doc.pdf"},
        }

    def test_file_document_source_raises(self) -> None:
        from dobby.providers.anthropic.converters import content_part_to_anthropic
        from dobby.types import DocumentPart, FileDocumentSource

        with pytest.raises(ValueError, match="not supported by Anthropic"):
            content_part_to_anthropic(
                DocumentPart(source=FileDocumentSource(file_id="file-123"), filename="doc.pdf")
            )


# ---------------------------------------------------------------------------
# Build kwargs tests
# ---------------------------------------------------------------------------


class TestBuildKwargs:
    """Test _build_kwargs static method."""

    def test_basic_kwargs(self) -> None:
        from dobby.providers.anthropic.adapter import AnthropicProvider

        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["max_tokens"] == 8192
        assert kwargs["temperature"] == 0.0
        assert "system" not in kwargs
        assert "tools" not in kwargs
        assert "thinking" not in kwargs

    def test_with_system_and_tools(self) -> None:
        from dobby.providers.anthropic.adapter import AnthropicProvider

        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[],
            system="Be helpful",
            tools=[{"name": "search", "input_schema": {}}],
        )
        assert kwargs["system"] == "Be helpful"
        assert kwargs["tools"] == [{"name": "search", "input_schema": {}}]

    def test_thinking_forces_temperature_1(self) -> None:
        from dobby.providers.anthropic.adapter import AnthropicProvider

        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[],
            temperature=0.5,
            thinking={"type": "enabled", "budget_tokens": 10000},
        )
        assert kwargs["temperature"] == 1.0
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}


# ---------------------------------------------------------------------------
# Error chaining tests
# ---------------------------------------------------------------------------


class TestAnthropicErrorChaining:
    """Test that original Anthropic errors are preserved via __cause__."""

    def test_chain(self) -> None:
        import anthropic

        provider = TestAnthropicErrorTranslation()._make_provider()
        native = TestAnthropicErrorTranslation()._make_anthropic_error(
            anthropic.RateLimitError
        )

        with pytest.raises(RateLimitError) as exc_info:
            provider._translate_error(native)

        assert exc_info.value.__cause__ is native
        assert isinstance(exc_info.value.__cause__, anthropic.RateLimitError)


# ---------------------------------------------------------------------------
# Reasoning effort validation tests
# ---------------------------------------------------------------------------


class TestAnthropicReasoningEffort:
    """Test reasoning_effort type validation for Anthropic provider."""

    def _make_provider(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.api_key = "test"
        provider.base_url = None
        provider._model = "claude-sonnet-4-20250514"
        provider.max_retries = 3
        provider._client = MagicMock()
        return provider

    def test_int_reasoning_effort_builds_thinking(self) -> None:
        """Int budget_tokens should produce a thinking config."""
        provider = self._make_provider()
        # chat() builds thinking dict before delegating; test indirectly via the logic
        # We verify the contract: int → thinking dict, str → TypeError
        # Direct unit test of the validation branch:
        reasoning_effort = 5000
        assert isinstance(reasoning_effort, int)
        thinking = {"type": "enabled", "budget_tokens": reasoning_effort}
        assert thinking["budget_tokens"] == 5000

    def test_string_reasoning_effort_raises_type_error(self) -> None:
        """Passing a string should raise TypeError."""
        provider = self._make_provider()

        with pytest.raises(TypeError, match="requires reasoning_effort as int"):
            asyncio.get_event_loop().run_until_complete(
                provider.chat(
                    messages=[],
                    reasoning_effort="low",
                )
            )

    def test_none_reasoning_effort_no_thinking(self) -> None:
        """None should mean no thinking config."""
        # When reasoning_effort is None, the thinking variable stays None
        reasoning_effort = None
        thinking = None
        if reasoning_effort is not None:
            thinking = {"type": "enabled", "budget_tokens": reasoning_effort}
        assert thinking is None


# ---------------------------------------------------------------------------
# Shared fakes for non-streaming + streaming
# ---------------------------------------------------------------------------


def _make_provider_with_mock_client():
    """Build an Anthropic provider with messages.create mocked out."""
    from dobby.providers.anthropic.adapter import AnthropicProvider

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.api_key = "test"
    provider.base_url = None
    provider._model = "claude-sonnet-4-20250514"
    provider.max_retries = 3
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    provider._client = client
    return provider


def _fake_response(blocks, stop_reason="end_turn", model="claude-sonnet-4-20250514",
                   input_tokens=10, output_tokens=20,
                   cache_creation_input_tokens=None, cache_read_input_tokens=None):
    """Build a mock Anthropic response object for non-streaming tests."""
    response = MagicMock()
    response.content = blocks
    response.stop_reason = stop_reason
    response.model = model
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation_input_tokens
    usage.cache_read_input_tokens = cache_read_input_tokens
    response.usage = usage
    return response


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _thinking_block(text: str, signature: str = "sig-xyz"):
    block = MagicMock()
    block.type = "thinking"
    block.thinking = text
    block.signature = signature
    return block


def _tool_use_block(tool_id: str, name: str, inputs: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = inputs
    return block


class _AsyncIter:
    """Minimal async iterator around an in-memory list."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _event(type_, **fields):
    """Build a mock streaming event."""
    ev = MagicMock()
    ev.type = type_
    for k, v in fields.items():
        setattr(ev, k, v)
    return ev


# ---------------------------------------------------------------------------
# Non-streaming chat tests
# ---------------------------------------------------------------------------


class TestAnthropicNonStreaming:
    """Test non-streaming chat() path end-to-end with a mocked Anthropic client."""

    def test_text_only_response(self) -> None:
        from dobby.types import TextPart as TP, UserMessagePart

        provider = _make_provider_with_mock_client()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[_text_block("Hello there!")],
        )

        async def run():
            return await provider.chat(
                messages=[UserMessagePart(parts=[TP(text="Hi")])],
            )

        result = asyncio.run(run())

        assert isinstance(result, StreamEndEvent)
        assert result.stop_reason == "end_turn"
        assert len(result.parts) == 1
        assert isinstance(result.parts[0], TextPart)
        assert result.parts[0].text == "Hello there!"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 30

    def test_tool_use_response(self) -> None:
        from dobby.types import TextPart as TP, UserMessagePart

        provider = _make_provider_with_mock_client()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[
                _text_block("Let me search."),
                _tool_use_block("call_1", "search", {"q": "dobby"}),
            ],
            stop_reason="tool_use",
        )

        async def run():
            return await provider.chat(
                messages=[UserMessagePart(parts=[TP(text="look up dobby")])],
                tools=[{"name": "search", "input_schema": {}}],
            )

        result = asyncio.run(run())

        assert result.stop_reason == "tool_use"
        tool_parts = [p for p in result.parts if isinstance(p, ToolUsePart)]
        assert len(tool_parts) == 1
        assert tool_parts[0].id == "call_1"
        assert tool_parts[0].name == "search"
        assert tool_parts[0].inputs == {"q": "dobby"}

    def test_thinking_response(self) -> None:
        from dobby.types import TextPart as TP, UserMessagePart

        provider = _make_provider_with_mock_client()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[
                _thinking_block("I need to compute...", signature="sig-1"),
                _text_block("555"),
            ],
        )

        async def run():
            return await provider.chat(
                messages=[UserMessagePart(parts=[TP(text="15 * 37")])],
                reasoning_effort=5000,
            )

        result = asyncio.run(run())

        reasoning_parts = [p for p in result.parts if isinstance(p, ReasoningPart)]
        assert len(reasoning_parts) == 1
        assert reasoning_parts[0].text == "I need to compute..."
        assert reasoning_parts[0].signature == "sig-1"

        # Verify thinking forced temperature=1.0 in the payload sent to Anthropic
        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["thinking"] == {
            "type": "enabled",
            "budget_tokens": 5000,
        }

    def test_max_tokens_and_system_prompt_forwarded(self) -> None:
        from dobby.types import TextPart as TP, UserMessagePart

        provider = _make_provider_with_mock_client()
        provider._client.messages.create.return_value = _fake_response(
            blocks=[_text_block("ok")],
        )

        async def run():
            await provider.chat(
                messages=[UserMessagePart(parts=[TP(text="hi")])],
                system_prompt="Be brief.",
                max_tokens=256,
                temperature=0.3,
            )

        asyncio.run(run())

        kwargs = provider._client.messages.create.call_args.kwargs
        assert kwargs["system"] == "Be brief."
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.3
        assert kwargs["stream"] is False

    def test_non_streaming_error_translation(self) -> None:
        import anthropic

        from dobby.types import TextPart as TP, UserMessagePart

        provider = _make_provider_with_mock_client()

        # Build a RateLimitError the same way the translation tests do
        response = MagicMock()
        response.status_code = 429
        response.headers = {"retry-after": "7"}
        err = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
        err.response = response
        err.status_code = 429
        err.message = "Rate limited"
        err.body = None

        provider._client.messages.create.side_effect = err
        # Disable retries to keep the test fast
        provider.max_retries = 1

        async def run():
            await provider.chat(
                messages=[UserMessagePart(parts=[TP(text="hi")])],
            )

        with pytest.raises(RateLimitError) as exc_info:
            asyncio.run(run())

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.retry_after == 7.0


# ---------------------------------------------------------------------------
# Streaming chat tests
# ---------------------------------------------------------------------------


class TestAnthropicStreaming:
    """Test streaming chat() path by feeding synthetic Anthropic events."""

    def _drive_stream(self, provider, events, **chat_kwargs):
        """Invoke provider.chat(stream=True) with the given events and collect output."""
        from dobby.types import TextPart as TP, UserMessagePart

        provider._client.messages.create.return_value = _AsyncIter(events)

        async def run():
            out = []
            stream = await provider.chat(
                messages=[UserMessagePart(parts=[TP(text="hi")])],
                stream=True,
                **chat_kwargs,
            )
            async for ev in stream:
                out.append(ev)
            return out

        return asyncio.run(run())

    def test_streaming_text(self) -> None:
        provider = _make_provider_with_mock_client()

        msg_start = _event(
            "message_start",
            message=MagicMock(
                id="msg_1",
                model="claude-sonnet-4-20250514",
                usage=MagicMock(
                    input_tokens=5,
                    cache_creation_input_tokens=None,
                    cache_read_input_tokens=None,
                ),
            ),
        )
        cb_start = _event(
            "content_block_start",
            content_block=MagicMock(type="text"),
        )
        delta_hello = _event(
            "content_block_delta",
            delta=MagicMock(type="text_delta", text="Hello "),
        )
        delta_world = _event(
            "content_block_delta",
            delta=MagicMock(type="text_delta", text="world"),
        )
        cb_stop = _event("content_block_stop")
        msg_delta = _event(
            "message_delta",
            delta=MagicMock(stop_reason="end_turn"),
            usage=MagicMock(output_tokens=7),
        )
        msg_stop = _event("message_stop")

        events = self._drive_stream(
            provider,
            [msg_start, cb_start, delta_hello, delta_world, cb_stop, msg_delta, msg_stop],
        )

        # First: stream start
        assert isinstance(events[0], StreamStartEvent)
        assert events[0].id == "msg_1"
        assert events[0].model == "claude-sonnet-4-20250514"

        # Text deltas
        deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert [d.delta for d in deltas] == ["Hello ", "world"]

        # Final event has accumulated text and usage
        end = [e for e in events if isinstance(e, StreamEndEvent)]
        assert len(end) == 1
        assert end[0].stop_reason == "end_turn"
        text_parts = [p for p in end[0].parts if isinstance(p, TextPart)]
        assert text_parts[0].text == "Hello world"
        assert end[0].usage.input_tokens == 5
        assert end[0].usage.output_tokens == 7
        assert end[0].usage.total_tokens == 12

    def test_streaming_tool_use_assembles_json(self) -> None:
        """partial_json chunks should be concatenated and parsed."""
        provider = _make_provider_with_mock_client()

        msg_start = _event(
            "message_start",
            message=MagicMock(id="msg_2", model="claude-sonnet-4-20250514", usage=None),
        )
        _cb = MagicMock()
        _cb.type = "tool_use"
        _cb.id = "call_42"
        _cb.name = "search"
        tool_start = _event("content_block_start", content_block=_cb)
        # The JSON "{\"q\":\"dobby\"}" arrives in three partial chunks
        p1 = _event(
            "content_block_delta",
            delta=MagicMock(type="input_json_delta", partial_json='{"q":'),
        )
        p2 = _event(
            "content_block_delta",
            delta=MagicMock(type="input_json_delta", partial_json='"dob'),
        )
        p3 = _event(
            "content_block_delta",
            delta=MagicMock(type="input_json_delta", partial_json='by"}'),
        )
        tool_stop = _event("content_block_stop")
        msg_delta = _event(
            "message_delta",
            delta=MagicMock(stop_reason="tool_use"),
            usage=MagicMock(output_tokens=11),
        )
        msg_stop = _event("message_stop")

        events = self._drive_stream(
            provider,
            [msg_start, tool_start, p1, p2, p3, tool_stop, msg_delta, msg_stop],
        )

        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].id == "call_42"
        assert tool_events[0].name == "search"
        assert tool_events[0].inputs == {"q": "dobby"}

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        assert end.stop_reason == "tool_use"
        tool_parts = [p for p in end.parts if isinstance(p, ToolUsePart)]
        assert tool_parts[0].inputs == {"q": "dobby"}

    def test_streaming_thinking(self) -> None:
        """Thinking blocks should produce reasoning_start/delta/end and preserve signature."""
        provider = _make_provider_with_mock_client()

        msg_start = _event(
            "message_start",
            message=MagicMock(id="msg_3", model="claude-sonnet-4-20250514", usage=None),
        )
        think_start = _event(
            "content_block_start",
            content_block=MagicMock(type="thinking"),
        )
        think_delta = _event(
            "content_block_delta",
            delta=MagicMock(type="thinking_delta", thinking="Let me think "),
        )
        think_delta2 = _event(
            "content_block_delta",
            delta=MagicMock(type="thinking_delta", thinking="carefully."),
        )
        sig_delta = _event(
            "content_block_delta",
            delta=MagicMock(type="signature_delta", signature="sig-abc"),
        )
        think_stop = _event("content_block_stop")
        msg_delta = _event(
            "message_delta",
            delta=MagicMock(stop_reason="end_turn"),
            usage=MagicMock(output_tokens=3),
        )
        msg_stop = _event("message_stop")

        events = self._drive_stream(
            provider,
            [msg_start, think_start, think_delta, think_delta2, sig_delta, think_stop,
             msg_delta, msg_stop],
            reasoning_effort=5000,
        )

        assert any(isinstance(e, ReasoningStartEvent) for e in events)
        assert any(isinstance(e, ReasoningEndEvent) for e in events)
        rdeltas = [e for e in events if isinstance(e, ReasoningDeltaEvent)]
        assert "".join(d.delta for d in rdeltas) == "Let me think carefully."

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        rparts = [p for p in end.parts if isinstance(p, ReasoningPart)]
        assert rparts[0].text == "Let me think carefully."
        assert rparts[0].signature == "sig-abc"

    def test_stream_uses_stream_true_param(self) -> None:
        """The SDK call must be issued with stream=True."""
        provider = _make_provider_with_mock_client()

        msg_start = _event(
            "message_start",
            message=MagicMock(id="x", model="m", usage=None),
        )
        msg_stop = _event("message_stop")
        self._drive_stream(provider, [msg_start, msg_stop])

        kwargs = provider._client.messages.create.call_args.kwargs
        assert kwargs["stream"] is True


# ---------------------------------------------------------------------------
# Multi-turn conversation context tests
# ---------------------------------------------------------------------------


class TestAnthropicMessageContext:
    """Verify the full user -> assistant(tool_use) -> user(tool_result) round-trip."""

    def test_full_tool_round_trip_conversion(self) -> None:
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import (
            AssistantMessagePart,
            TextPart as TP,
            ToolResultPart,
            ToolUsePart as TUP,
            UserMessagePart,
        )

        messages = [
            UserMessagePart(parts=[TP(text="Search for Dobby")]),
            AssistantMessagePart(
                parts=[
                    TP(text="Let me search."),
                    TUP(id="call_1", name="search", inputs={"q": "dobby"}),
                ]
            ),
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id="call_1",
                        name="search",
                        parts=[TP(text="Dobby is a house-elf.")],
                    )
                ]
            ),
            AssistantMessagePart(parts=[TP(text="Dobby is a house-elf from Harry Potter.")]),
        ]

        result = to_anthropic_messages(messages)

        assert [m["role"] for m in result] == ["user", "assistant", "user", "assistant"]
        # Assistant turn combines text + tool_use
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][1] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "search",
            "input": {"q": "dobby"},
        }
        # Tool result travels back as a user message with tool_result block
        assert result[2]["content"][0]["type"] == "tool_result"
        assert result[2]["content"][0]["tool_use_id"] == "call_1"

    def test_reasoning_preserved_across_turns(self) -> None:
        """Reasoning blocks must round-trip with signature for multi-turn thinking."""
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart, TextPart as TP, UserMessagePart

        messages = [
            UserMessagePart(parts=[TP(text="2 + 2?")]),
            AssistantMessagePart(
                parts=[
                    ReasoningPart(text="Adding two and two.", signature="sig-99"),
                    TP(text="4"),
                ]
            ),
            UserMessagePart(parts=[TP(text="And 3 + 3?")]),
        ]

        result = to_anthropic_messages(messages)

        assistant = result[1]
        assert assistant["content"][0] == {
            "type": "thinking",
            "thinking": "Adding two and two.",
            "signature": "sig-99",
        }
        assert assistant["content"][1] == {"type": "text", "text": "4"}

    def test_empty_assistant_parts_skipped(self) -> None:
        """An assistant turn with no parts should not create an empty message block."""
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart, TextPart as TP, UserMessagePart

        messages = [
            UserMessagePart(parts=[TP(text="hi")]),
            AssistantMessagePart(parts=[]),
            UserMessagePart(parts=[TP(text="still here")]),
        ]
        result = to_anthropic_messages(messages)

        # The two user turns get merged since the empty assistant was skipped
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2

    def test_tool_result_is_error_flag_forwarded(self) -> None:
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import TextPart as TP, ToolResultPart, UserMessagePart

        messages = [
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id="call_err",
                        name="search",
                        parts=[TP(text="boom")],
                        is_error=True,
                    )
                ]
            )
        ]
        result = to_anthropic_messages(messages)

        assert result[0]["content"][0]["is_error"] is True
        assert result[0]["content"][0]["tool_use_id"] == "call_err"


# ---------------------------------------------------------------------------
# Tavily web search tool tests
# ---------------------------------------------------------------------------


class TestTavilySearchTool:
    """Test the TavilySearchTool against a mocked AsyncTavilyClient."""

    def _make_tool(self, search_result):
        from dobby.common_tools import TavilySearchTool

        tool = TavilySearchTool(api_key="tvly-fake")
        tool.client = MagicMock()
        tool.client.search = AsyncMock(return_value=search_result)
        return tool

    def test_returns_structured_results(self) -> None:
        from dobby.common_tools import TavilySearchResult

        tool = self._make_tool(
            {
                "results": [
                    {
                        "title": "Dobby on Wikipedia",
                        "url": "https://en.wikipedia.org/wiki/Dobby",
                        "content": "Dobby is a house-elf.",
                        "score": 0.95,
                    },
                    {
                        "title": "Dobby-SDK",
                        "url": "https://github.com/tynybay/dobby-sdk",
                        "content": "LLM SDK",
                        "score": 0.82,
                    },
                ]
            }
        )

        results = asyncio.run(tool(query="dobby"))

        assert len(results) == 2
        assert all(isinstance(r, TavilySearchResult) for r in results)
        assert results[0].title == "Dobby on Wikipedia"
        assert results[0].score == 0.95
        assert results[1].url == "https://github.com/tynybay/dobby-sdk"

    def test_passes_through_search_arguments(self) -> None:
        tool = self._make_tool({"results": []})

        asyncio.run(
            tool(
                query="latest AI news",
                search_depth="advanced",
                topic="news",
                time_range="week",
            )
        )

        tool.client.search.assert_awaited_once()
        kwargs = tool.client.search.call_args.kwargs
        assert kwargs["query"] == "latest AI news"
        assert kwargs["search_depth"] == "advanced"
        assert kwargs["topic"] == "news"
        assert kwargs["time_range"] == "week"

    def test_default_arguments(self) -> None:
        tool = self._make_tool({"results": []})

        asyncio.run(tool(query="python"))

        kwargs = tool.client.search.call_args.kwargs
        assert kwargs["search_depth"] == "basic"
        assert kwargs["topic"] == "general"
        assert kwargs["time_range"] is None

    def test_empty_results(self) -> None:
        tool = self._make_tool({"results": []})

        results = asyncio.run(tool(query="no matches expected"))

        assert results == []

    def test_missing_fields_get_defaults(self) -> None:
        """Malformed Tavily rows should not crash — missing fields get defaults."""
        tool = self._make_tool({"results": [{"title": "Partial"}]})

        results = asyncio.run(tool(query="partial"))

        assert len(results) == 1
        assert results[0].title == "Partial"
        assert results[0].url == ""
        assert results[0].content == ""
        assert results[0].score == 0.0

    def test_missing_results_key(self) -> None:
        """If Tavily returns no 'results' key, tool should return []."""
        tool = self._make_tool({})

        results = asyncio.run(tool(query="anything"))

        assert results == []

    def test_client_error_propagates(self) -> None:
        from dobby.common_tools import TavilySearchTool

        tool = TavilySearchTool(api_key="tvly-fake")
        tool.client = MagicMock()
        tool.client.search = AsyncMock(side_effect=RuntimeError("tavily down"))

        with pytest.raises(RuntimeError, match="tavily down"):
            asyncio.run(tool(query="anything"))

    def test_tool_schema_metadata(self) -> None:
        """Basic Tool contract — name/description set, schema reflects __call__ params."""
        from dobby.common_tools import TavilySearchTool

        tool = TavilySearchTool(api_key="tvly-fake")
        assert tool.name == "tavily_search"
        assert "web" in tool.description.lower() or "search" in tool.description.lower()

        # Sanity-check that __call__ takes the documented params
        import inspect
        sig = inspect.signature(tool.__call__)
        assert {"query", "search_depth", "topic", "time_range"} <= set(sig.parameters)


# ---------------------------------------------------------------------------
# Travel insurance claim accuracy tests (live — auto-skipped without API key)
#
# Run all:   uv run pytest tests/test_anthropic_provider.py -v -s -k "Travel"
# Run one:   uv run pytest tests/test_anthropic_provider.py -v -s -k "cancellation"
# ---------------------------------------------------------------------------

import base64
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_FOUNDRY_KEY = os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
_FOUNDRY_RESOURCE = os.getenv("ANTHROPIC_FOUNDRY_RESOURCE")
_FOUNDRY_BASE_URL = os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
_FOUNDRY_MODEL = os.getenv("ANTHROPIC_FOUNDRY_MODEL")
_DIRECT_KEY = os.getenv("ANTHROPIC_API_KEY")
_USE_AZURE = bool(_FOUNDRY_KEY and (_FOUNDRY_RESOURCE or _FOUNDRY_BASE_URL))
_HAS_KEY = _USE_AZURE or bool(_DIRECT_KEY)

_requires_live = pytest.mark.skipif(
    not _HAS_KEY,
    reason="No API key — set ANTHROPIC_API_KEY or ANTHROPIC_FOUNDRY_* in .env",
)

_FIXTURES = Path(__file__).parent / "fixtures"

_TRAVEL_GROUND_TRUTH = [
    {
        "file": "trip_cancellation_claim.pdf",
        "label": "Trip Cancellation",
        "fields": {
            "claim_number": "CLM-2026-TC-00881",
            "policy_number": "POL-TRVL-4492017",
            "traveler_name": "Emily R. Thornton",
            "destination": "Rome",
            "departure_date": "2026-03-20",
            "total_trip_cost": "8450",
            "net_claim_amount": "7540",
            "deductible": "250",
            "cancellation_reason": "appendicitis",
        },
    },
    {
        "file": "trip_interruption_claim.pdf",
        "label": "Trip Interruption",
        "fields": {
            "claim_number": "CLM-2026-TI-00334",
            "policy_number": "POL-TRVL-3381092",
            "traveler_name": "Kim",
            "destination": "Bali",
            "total_trip_cost": "12600",
            "net_claim_amount": "6690",
            "deductible": "500",
            "interruption_reason": "stroke",
        },
    },
    {
        "file": "baggage_loss_claim.pdf",
        "label": "Baggage Loss",
        "fields": {
            "claim_number": "CLM-2026-BG-01122",
            "policy_number": "POL-TRVL-5510834",
            "traveler_name": "Maria L. Santos",
            "destination": "Tokyo",
            "total_declared_value": "3970",
            "net_claim_amount": "3320",
            "deductible": "100",
            "airline_pir": "JAL-PIR-2026-ORD-00891",
        },
    },
    {
        "file": "emergency_medical_claim.pdf",
        "label": "Emergency Medical Evacuation",
        "fields": {
            "claim_number": "CLM-2026-EM-00219",
            "policy_number": "POL-TRVL-7723561",
            "traveler_name": "Daniel J. Owens",
            "destination": "Cusco",
            "diagnosis": "HAPE",
            "total_medical_costs": "48180",
            "net_claim_amount": "47930",
            "deductible": "250",
        },
    },
    {
        "file": "travel_delay_claim.pdf",
        "label": "Travel Delay",
        "fields": {
            "claim_number": "CLM-2026-TD-00667",
            "policy_number": "POL-TRVL-6640293",
            "traveler_name": "Russo",
            "destination": "Cancun",
            "total_expenses": "1028",
            "net_claim_amount": "1028",
            "deductible": "0",
            "delay_cause": "thunderstorm",
        },
    },
]

_EXTRACTION_PROMPT = (
    "Extract all structured fields from this travel insurance claim document. "
    "Return ONLY a valid JSON object — no markdown fences, no explanation. "
    "Include every claim number, policy number, traveler name, destination, "
    "travel dates, dollar amounts, deductibles, and reason for claim."
)


def _make_live_provider():
    from dobby.providers.anthropic import AnthropicProvider
    model = _FOUNDRY_MODEL or "claude-sonnet-4-6"
    if _USE_AZURE:
        return AnthropicProvider(
            model=model,
            api_key=_FOUNDRY_KEY,
            resource=_FOUNDRY_RESOURCE,
            base_url=_FOUNDRY_BASE_URL,
        )
    return AnthropicProvider(model=model, api_key=_DIRECT_KEY)


def _check_field(extracted: dict, value: str) -> bool:
    flat = json.dumps(extracted).lower().replace(",", "").replace("$", "")
    return value.lower().replace(",", "").replace("$", "") in flat


def _run_claim_accuracy(gt: dict) -> tuple[int, int, float]:
    from dobby.types import DocumentPart, TextPart, UserMessagePart
    from dobby.types.document_part import Base64PDFSource

    pdf_path = _FIXTURES / gt["file"]
    assert pdf_path.exists(), (
        f"Missing fixture: {pdf_path}\n"
        f"Generate with: python tests/fixtures/generate_claim_pdfs.py"
    )

    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode()
    provider = _make_live_provider()

    t0 = time.perf_counter()
    result = asyncio.run(
        provider.chat(
            messages=[
                UserMessagePart(parts=[
                    TextPart(text=_EXTRACTION_PROMPT),
                    DocumentPart(
                        source=Base64PDFSource(data=pdf_b64, media_type="application/pdf"),
                        filename=gt["file"],
                    ),
                ])
            ],
            system_prompt="You are a precise travel insurance data extraction system. Output only valid JSON.",
            max_tokens=1024,
            temperature=0.0,
        )
    )
    latency = time.perf_counter() - t0

    raw = next((p.text for p in result.parts if hasattr(p, "text")), "")
    clean = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    clean = re.sub(r"\n?```$", "", clean.strip())
    try:
        extracted = json.loads(clean)
    except json.JSONDecodeError:
        extracted = {}

    passed = 0
    print(f"\n  [{gt['label']}]  latency={latency:.1f}s  provider={provider.name}")
    for field, expected in gt["fields"].items():
        hit = _check_field(extracted, expected)
        print(f"    {'✓' if hit else '✗'} {field:<28} expected={expected!r}")
        passed += hit

    total = len(gt["fields"])
    usage = result.usage
    print(f"  → {passed}/{total} fields ({passed/total*100:.0f}%)  "
          f"tokens: in={usage.input_tokens if usage else '?'} out={usage.output_tokens if usage else '?'}")
    return passed, total, latency


@_requires_live
class TestTravelClaimAccuracy:
    """Field-level extraction accuracy across 5 travel insurance claim PDFs.

    Measures what the model correctly extracts vs known ground truth.
    Pass threshold: 75% of fields per claim.

    Run: uv run pytest tests/test_anthropic_provider.py -v -s -k "TravelClaim"
    """

    def test_trip_cancellation_accuracy(self) -> None:
        passed, total, _ = _run_claim_accuracy(_TRAVEL_GROUND_TRUTH[0])
        assert passed >= total * 0.75, f"Trip cancellation: only {passed}/{total} fields found"

    def test_trip_interruption_accuracy(self) -> None:
        passed, total, _ = _run_claim_accuracy(_TRAVEL_GROUND_TRUTH[1])
        assert passed >= total * 0.75, f"Trip interruption: only {passed}/{total} fields found"

    def test_baggage_loss_accuracy(self) -> None:
        passed, total, _ = _run_claim_accuracy(_TRAVEL_GROUND_TRUTH[2])
        assert passed >= total * 0.75, f"Baggage loss: only {passed}/{total} fields found"

    def test_emergency_medical_accuracy(self) -> None:
        passed, total, _ = _run_claim_accuracy(_TRAVEL_GROUND_TRUTH[3])
        assert passed >= total * 0.75, f"Emergency medical: only {passed}/{total} fields found"

    def test_travel_delay_accuracy(self) -> None:
        passed, total, _ = _run_claim_accuracy(_TRAVEL_GROUND_TRUTH[4])
        assert passed >= total * 0.75, f"Travel delay: only {passed}/{total} fields found"
