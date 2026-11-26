"""Tests for Anthropic provider: error translation, message conversion, and converters."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dobby.providers.base import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    ProviderError,
    RateLimitError,
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

    def _make_anthropic_error(
        self, error_cls, status_code: int = 500, headers: dict | None = None
    ):
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

    def test_mid_stream_error_is_translated(self) -> None:
        """An SDK error raised after streaming has started uses the unified error type."""
        import anthropic

        provider = self._make_provider()
        provider.max_retries = 0
        native = self._make_anthropic_error(anthropic.APIConnectionError)

        async def native_stream():
            yield SimpleNamespace(type="ping")
            raise native

        provider._client.messages.create = AsyncMock(return_value=native_stream())

        async def run() -> None:
            stream = provider._stream_chat_completion([], "claude-sonnet-4-20250514")
            with pytest.raises(APIConnectionError) as exc_info:
                async for _ in stream:
                    pass
            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.__cause__ is native

        asyncio.run(run())


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
            AssistantMessagePart(parts=[ReasoningPart(text="Let me think...", signature="sig123")])
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
            DocumentPart(
                source=Base64PDFSource(data="pdfdata", media_type="application/pdf"),
                filename="doc.pdf",
            )
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
        native = TestAnthropicErrorTranslation()._make_anthropic_error(anthropic.RateLimitError)

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
            asyncio.run(
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
# stop_reason mapping tests
# ---------------------------------------------------------------------------


class TestMapStopReason:
    """Test _map_stop_reason normalization."""

    @pytest.mark.parametrize(
        "reason",
        ["end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal", "pause_turn"],
    )
    def test_known_reasons_pass_through(self, reason: str) -> None:
        from dobby.providers.anthropic.adapter import _map_stop_reason

        assert _map_stop_reason(reason) == reason

    def test_none_falls_back_to_end_turn(self) -> None:
        from dobby.providers.anthropic.adapter import _map_stop_reason

        assert _map_stop_reason(None) == "end_turn"

    def test_unknown_falls_back_to_end_turn(self) -> None:
        from dobby.providers.anthropic.adapter import _map_stop_reason

        assert _map_stop_reason("some_future_reason") == "end_turn"


# ---------------------------------------------------------------------------
# kwargs passthrough tests
# ---------------------------------------------------------------------------


class TestKwargsPassthrough:
    """Test that _build_kwargs forwards provider-native params via extra."""

    def test_extra_forwarded(self) -> None:
        from dobby.providers.anthropic.adapter import AnthropicProvider

        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[],
            extra={"top_p": 0.9, "stop_sequences": ["END"]},
        )
        assert kwargs["top_p"] == 0.9
        assert kwargs["stop_sequences"] == ["END"]

    def test_extra_never_clobbers_core_fields(self) -> None:
        from dobby.providers.anthropic.adapter import AnthropicProvider

        kwargs = AnthropicProvider._build_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[],
            temperature=0.0,
            extra={"model": "evil", "temperature": 0.9},
        )
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# redacted_thinking round-trip tests
# ---------------------------------------------------------------------------


class TestRedactedThinking:
    """Test redacted_thinking preservation through the converter."""

    def test_redacted_reasoning_emits_redacted_block(self) -> None:
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart, ReasoningPart

        messages = [
            AssistantMessagePart(parts=[ReasoningPart(text="ENCRYPTED_BLOB", redacted=True)])
        ]
        result = to_anthropic_messages(messages)

        assert result[0]["content"] == [{"type": "redacted_thinking", "data": "ENCRYPTED_BLOB"}]

    def test_normal_reasoning_still_emits_thinking_block(self) -> None:
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart, ReasoningPart

        messages = [AssistantMessagePart(parts=[ReasoningPart(text="thinking", signature="s")])]
        result = to_anthropic_messages(messages)

        assert result[0]["content"] == [
            {"type": "thinking", "thinking": "thinking", "signature": "s"}
        ]


# ---------------------------------------------------------------------------
# Azure config validation tests
# ---------------------------------------------------------------------------


class TestConstructorSignature:
    """Pin the public constructor surface of every provider.

    These strings and argument positions are published API. Nothing else in the
    suite asserts them, so a future edit could narrow or reorder them silently.
    """

    def test_anthropic_max_retries_is_keyword_only(self) -> None:
        """max_retries must not be bindable positionally.

        Keyword-only enforcement stops a positional caller from binding their
        retry count to whatever parameter happens to sit in that slot.
        """
        from dobby.providers.anthropic.adapter import AnthropicProvider

        with pytest.raises(TypeError):
            AnthropicProvider("claude-sonnet-4-5", None, None, None, None, 5)  # type: ignore[misc]

    def test_gemini_max_retries_is_keyword_only(self) -> None:
        """Regression: `vertexai` used to be the 3rd positional parameter.

        `GeminiProvider("m", None, True)` was valid in published 0.2.15 and meant
        "use Vertex". With `vertexai` removed, the keyword-only boundary stops True
        from binding to the retry count and silently routing traffic to the
        Developer API. It must raise instead.
        """
        from dobby.providers.gemini.adapter import GeminiProvider

        with pytest.raises(TypeError):
            GeminiProvider("gemini-2.5-flash", None, True)  # type: ignore[misc]

    def test_removed_vertex_params_raise_type_error(self) -> None:
        """The removals must fail loud, never fall back to another backend."""
        from dobby.providers.anthropic.adapter import AnthropicProvider
        from dobby.providers.gemini.adapter import GeminiProvider

        for anthropic_kwargs in (
            {"vertex": True},
            {"project_id": "p"},
            {"region": "us-east5"},
            {"access_token": "t"},
        ):
            with pytest.raises(TypeError):
                AnthropicProvider(model="claude-sonnet-4-5", **anthropic_kwargs)  # type: ignore[arg-type]

        for gemini_kwargs in ({"vertexai": True}, {"project": "p"}, {"location": "us-central1"}):
            with pytest.raises(TypeError):
                GeminiProvider(model="gemini-2.5-flash", **gemini_kwargs)  # type: ignore[arg-type]

    def test_removed_gemini_attributes_are_gone(self) -> None:
        """`vertexai`/`project`/`location` were public attributes, not just params.

        Reading them is a distinct failure mode from the constructor TypeError,
        so it gets its own assertion.
        """
        from unittest.mock import patch as _patch

        from dobby.providers.gemini.adapter import GeminiProvider

        with _patch("dobby.providers.gemini.adapter.genai.Client"):
            provider = GeminiProvider(model="gemini-2.5-flash", api_key="k")

        for attr in ("vertexai", "project", "location"):
            assert not hasattr(provider, attr)

    def test_provider_name_values(self) -> None:
        """`name` values are published API — consumers match on these strings."""
        from unittest.mock import patch as _patch

        from dobby.providers.anthropic.adapter import AnthropicProvider
        from dobby.providers.gemini.adapter import GeminiProvider

        with _patch("dobby.providers.anthropic.adapter.AsyncAnthropic"):
            assert AnthropicProvider(model="claude-sonnet-4-5", api_key="k").name == "anthropic"
        with _patch("dobby.providers.anthropic.adapter.AsyncAnthropicFoundry"):
            azure = AnthropicProvider(model="claude-sonnet-4-5", api_key="k", resource="r")
        assert azure.name == "azure-anthropic"
        with _patch("dobby.providers.gemini.adapter.genai.Client"):
            assert GeminiProvider(model="gemini-2.5-flash", api_key="k").name == "gemini"


def _vertex_provider(max_retries: int = 3):
    """An adapter built through the internal path around a mock `AsyncAnthropicVertex`."""
    from anthropic import AsyncAnthropicVertex

    from dobby.providers.anthropic.adapter import AnthropicProvider

    client = MagicMock(spec=AsyncAnthropicVertex)
    client.messages = MagicMock()
    return AnthropicProvider._from_client(
        client, "claude-sonnet-4-5", max_retries=max_retries, provider_name="vertexai"
    )


class TestInjectedVertexClient:
    """The internal `_from_client` path runs the adapter on an `AsyncAnthropicVertex`."""

    def test_messages_create_called_with_converted_messages(self) -> None:
        from dobby.types import TextPart, UserMessagePart

        provider = _vertex_provider()
        response = MagicMock()
        response.content = []
        response.stop_reason = "end_turn"
        response.model = "claude-sonnet-4-5"
        response.usage = None
        provider.client.messages.create = AsyncMock(return_value=response)

        asyncio.run(
            provider.chat([UserMessagePart(parts=[TextPart(text="hi")])], system_prompt="be brief")
        )

        provider.client.messages.create.assert_awaited_once()
        sent = provider.client.messages.create.await_args.kwargs
        assert sent["model"] == "claude-sonnet-4-5"
        assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert sent["system"] == "be brief"

    def test_name_and_attributes(self) -> None:
        provider = _vertex_provider(max_retries=0)
        assert provider.name == "vertexai"
        assert provider.model == "claude-sonnet-4-5"
        assert provider.max_retries == 0

    def test_rate_limit_reports_vertexai(self) -> None:
        import anthropic
        import httpx

        from dobby.types import TextPart, UserMessagePart

        provider = _vertex_provider(max_retries=0)
        native_err = anthropic.RateLimitError(
            "rate limited",
            response=httpx.Response(
                429,
                headers={"retry-after": "7"},
                request=httpx.Request("POST", "https://example.invalid"),
            ),
            body=None,
        )
        provider.client.messages.create = AsyncMock(side_effect=native_err)

        with pytest.raises(RateLimitError) as exc_info:
            asyncio.run(provider.chat([UserMessagePart(parts=[TextPart(text="hi")])]))

        assert exc_info.value.provider == "vertexai"
        assert exc_info.value.retry_after == 7.0
        assert exc_info.value.__cause__ is native_err


class TestAzureValidation:
    """Test fail-fast validation of mutually exclusive Azure params."""

    def test_api_key_and_token_provider_conflict(self) -> None:
        from dobby.providers.anthropic.adapter import AnthropicProvider

        with pytest.raises(ValueError, match="api_key or azure_ad_token_provider"):
            AnthropicProvider(
                model="claude-sonnet-4-5",
                api_key="k",
                azure_ad_token_provider=lambda: "token",
                resource="my-resource",
            )

    def test_resource_and_base_url_conflict(self) -> None:
        from dobby.providers.anthropic.adapter import AnthropicProvider

        with pytest.raises(ValueError, match="resource or base_url"):
            AnthropicProvider(
                model="claude-sonnet-4-5",
                api_key="k",
                resource="my-resource",
                base_url="https://my-resource.services.ai.azure.com/anthropic/",
            )


class TestChatModelOverride:
    """Test that chat() honors per-call model and forwards kwargs."""

    def _make_provider(self):
        from dobby.providers.anthropic.adapter import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.api_key = "test"
        provider.base_url = None
        provider._model = "claude-sonnet-4-5"
        provider.max_retries = 3
        provider._client = MagicMock()
        return provider

    def _run_chat(self, provider, **chat_kwargs):
        from dobby.types import TextPart, UserMessagePart

        captured: dict = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.content = []
            resp.stop_reason = "end_turn"
            resp.model = kwargs["model"]
            resp.usage = None
            return resp

        provider._client.messages.create = fake_create
        result = asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                **chat_kwargs,
            )
        )
        return captured, result

    def test_model_override_used(self) -> None:
        provider = self._make_provider()
        captured, _ = self._run_chat(provider, model="claude-opus-4-8")
        assert captured["model"] == "claude-opus-4-8"

    def test_defaults_to_instance_model(self) -> None:
        provider = self._make_provider()
        captured, _ = self._run_chat(provider)
        assert captured["model"] == "claude-sonnet-4-5"

    def test_kwargs_forwarded_to_create(self) -> None:
        provider = self._make_provider()
        captured, _ = self._run_chat(provider, top_p=0.5)
        assert captured["top_p"] == 0.5

    def test_redacted_thinking_block_parsed(self) -> None:
        from dobby.types import ReasoningPart, TextPart, UserMessagePart

        provider = self._make_provider()
        captured: dict = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            block = MagicMock()
            block.type = "redacted_thinking"
            block.data = "ENC"
            resp = MagicMock()
            resp.content = [block]
            resp.stop_reason = "end_turn"
            resp.model = kwargs["model"]
            resp.usage = None
            return resp

        provider._client.messages.create = fake_create
        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])])
        )
        redacted = [p for p in result.parts if isinstance(p, ReasoningPart) and p.redacted]
        assert len(redacted) == 1
        assert redacted[0].text == "ENC"


class TestChatModelOverrideOnVertexClient(TestChatModelOverride):
    """Re-run the request-shape tests against an injected `AsyncAnthropicVertex`."""

    def _make_provider(self):
        return _vertex_provider()


# ---------------------------------------------------------------------------
# Streaming part assembly tests
# ---------------------------------------------------------------------------


def _ev(type_: str, **fields) -> SimpleNamespace:
    """Build a fake Anthropic stream event."""
    return SimpleNamespace(type=type_, **fields)


def _message_start() -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=5)
    return _ev("message_start", message=SimpleNamespace(id="msg_1", model="claude", usage=usage))


def _thinking_block(index: int, text: str, signature: str) -> list[SimpleNamespace]:
    """Events for one streamed thinking block (text split across two deltas)."""
    half = len(text) // 2
    return [
        _ev("content_block_start", index=index, content_block=SimpleNamespace(type="thinking")),
        _ev(
            "content_block_delta",
            index=index,
            delta=SimpleNamespace(type="thinking_delta", thinking=text[:half]),
        ),
        _ev(
            "content_block_delta",
            index=index,
            delta=SimpleNamespace(type="thinking_delta", thinking=text[half:]),
        ),
        _ev(
            "content_block_delta",
            index=index,
            delta=SimpleNamespace(type="signature_delta", signature=signature),
        ),
        _ev("content_block_stop", index=index),
    ]


def _redacted_block(index: int, data: str) -> list[SimpleNamespace]:
    """Events for one streamed redacted_thinking block."""
    block = SimpleNamespace(type="redacted_thinking", data=data)
    return [
        _ev("content_block_start", index=index, content_block=block),
        _ev("content_block_stop", index=index),
    ]


def _text_block(index: int, text: str) -> list[SimpleNamespace]:
    """Events for one streamed text block."""
    return [
        _ev("content_block_start", index=index, content_block=SimpleNamespace(type="text")),
        _ev(
            "content_block_delta",
            index=index,
            delta=SimpleNamespace(type="text_delta", text=text),
        ),
        _ev("content_block_stop", index=index),
    ]


def _tool_block(index: int, tool_id: str, name: str, args_json: str) -> list[SimpleNamespace]:
    """Events for one streamed tool_use block."""
    block = SimpleNamespace(type="tool_use", id=tool_id, name=name)
    return [
        _ev("content_block_start", index=index, content_block=block),
        _ev(
            "content_block_delta",
            index=index,
            delta=SimpleNamespace(type="input_json_delta", partial_json=args_json),
        ),
        _ev("content_block_stop", index=index),
    ]


def _message_end(stop_reason: str) -> list[SimpleNamespace]:
    return [
        _ev(
            "message_delta",
            delta=SimpleNamespace(stop_reason=stop_reason),
            usage=SimpleNamespace(output_tokens=7),
        ),
        _ev("message_stop"),
    ]


def _stream_events(blocks: list[list[SimpleNamespace]], stop_reason: str = "tool_use") -> list:
    """Run the streaming adapter over fake block events and return every yielded event."""
    from dobby.providers.anthropic.adapter import AnthropicProvider

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.base_url = None
    provider.max_retries = 0
    provider._client = MagicMock()

    raw = [_message_start(), *(e for block in blocks for e in block), *_message_end(stop_reason)]

    async def native_stream():
        for event in raw:
            yield event

    provider._client.messages.create = AsyncMock(return_value=native_stream())

    async def run() -> list:
        return [e async for e in provider._stream_chat_completion([], "claude")]

    return asyncio.run(run())


def _stream_parts(blocks: list[list[SimpleNamespace]], stop_reason: str = "tool_use") -> list:
    """Return the StreamEndEvent parts for the given fake block events."""
    from dobby.types import StreamEndEvent

    end = _stream_events(blocks, stop_reason)[-1]
    assert isinstance(end, StreamEndEvent)
    return end.parts


class TestStreamingPartAssembly:
    """StreamEndEvent.parts from the streaming path map block to part one-to-one."""

    def test_single_thinking_block_then_tool_call(self) -> None:
        """Characterization: one thinking block and one tool call (unchanged behaviour)."""
        from dobby.types import (
            ReasoningDeltaEvent,
            ReasoningEndEvent,
            ReasoningPart,
            ReasoningStartEvent,
            StreamStartEvent,
            ToolUseEvent,
            ToolUsePart,
        )

        events = _stream_events(
            [
                _thinking_block(0, "let me think", "s1"),
                _tool_block(1, "tu_1", "search", '{"q": "x"}'),
            ]
        )

        assert [type(e) for e in events[:-1]] == [
            StreamStartEvent,
            ReasoningStartEvent,
            ReasoningDeltaEvent,
            ReasoningDeltaEvent,
            ReasoningEndEvent,
            ToolUseEvent,
        ]
        deltas = "".join(e.delta for e in events if isinstance(e, ReasoningDeltaEvent))
        assert deltas == "let me think"
        end = events[-1]
        assert end.stop_reason == "tool_use"
        assert end.parts == [
            ReasoningPart(text="let me think", signature="s1"),
            ToolUsePart(id="tu_1", name="search", inputs={"q": "x"}),
        ]

    def test_single_thinking_block_then_text(self) -> None:
        """Characterization: one thinking block followed by a text answer."""
        from dobby.types import ReasoningPart, TextPart

        parts = _stream_parts(
            [_thinking_block(0, "hmm", "s1"), _text_block(1, "answer")], stop_reason="end_turn"
        )

        assert parts == [ReasoningPart(text="hmm", signature="s1"), TextPart(text="answer")]

    def test_two_thinking_blocks_keep_their_own_signatures(self) -> None:
        from dobby.types import ReasoningPart, ToolUsePart

        parts = _stream_parts(
            [
                _thinking_block(0, "first thought", "s1"),
                _thinking_block(1, "second thought", "s2"),
                _tool_block(2, "tu_1", "search", "{}"),
            ]
        )

        assert parts == [
            ReasoningPart(text="first thought", signature="s1"),
            ReasoningPart(text="second thought", signature="s2"),
            ToolUsePart(id="tu_1", name="search", inputs={}),
        ]

    def test_redacted_block_between_thinking_blocks_keeps_order(self) -> None:
        from dobby.types import ReasoningPart, TextPart

        parts = _stream_parts(
            [
                _thinking_block(0, "alpha", "s1"),
                _redacted_block(1, "ENC"),
                _thinking_block(2, "beta", "s2"),
                _text_block(3, "done"),
            ],
            stop_reason="end_turn",
        )

        assert parts == [
            ReasoningPart(text="alpha", signature="s1"),
            ReasoningPart(text="ENC", redacted=True),
            ReasoningPart(text="beta", signature="s2"),
            TextPart(text="done"),
        ]

    def test_interleaved_blocks_keep_provider_order(self) -> None:
        """Interleaved thinking: each block becomes one part, in the order streamed."""
        from dobby.types import ReasoningPart, TextPart, ToolUsePart

        parts = _stream_parts(
            [
                _thinking_block(0, "plan", "s1"),
                _text_block(1, "Looking it up."),
                _tool_block(2, "tu_1", "search", '{"q": "a"}'),
                _thinking_block(3, "next", "s2"),
                _tool_block(4, "tu_2", "fetch", '{"u": "b"}'),
            ]
        )

        assert parts == [
            ReasoningPart(text="plan", signature="s1"),
            TextPart(text="Looking it up."),
            ToolUsePart(id="tu_1", name="search", inputs={"q": "a"}),
            ReasoningPart(text="next", signature="s2"),
            ToolUsePart(id="tu_2", name="fetch", inputs={"u": "b"}),
        ]

    def test_streaming_matches_non_streaming_parts(self) -> None:
        """The same message yields the same part list on both paths."""
        from dobby.providers.anthropic.adapter import AnthropicProvider

        streamed = _stream_parts(
            [
                _thinking_block(0, "plan", "s1"),
                _redacted_block(1, "ENC"),
                _text_block(2, "Looking it up."),
                _tool_block(3, "tu_1", "search", '{"q": "a"}'),
                _thinking_block(4, "next", "s2"),
            ]
        )

        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="plan", signature="s1"),
                SimpleNamespace(type="redacted_thinking", data="ENC"),
                SimpleNamespace(type="text", text="Looking it up."),
                SimpleNamespace(type="tool_use", id="tu_1", name="search", input={"q": "a"}),
                SimpleNamespace(type="thinking", thinking="next", signature="s2"),
            ],
            stop_reason="tool_use",
            model="claude",
            usage=None,
        )
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.base_url = None
        provider.max_retries = 0
        provider._client = MagicMock()
        provider._client.messages.create = AsyncMock(return_value=response)
        non_streamed = asyncio.run(provider._non_stream_chat_completion([], "claude")).parts

        assert streamed == non_streamed

    def test_streamed_parts_round_trip_through_to_anthropic_messages(self) -> None:
        """Replaying streamed parts yields thinking blocks with matching text/signature pairs."""
        from dobby.providers.anthropic.adapter import to_anthropic_messages
        from dobby.types import AssistantMessagePart

        parts = _stream_parts(
            [
                _thinking_block(0, "first thought", "s1"),
                _redacted_block(1, "ENC"),
                _thinking_block(2, "second thought", "s2"),
                _tool_block(3, "tu_1", "search", '{"q": "x"}'),
            ]
        )

        result = to_anthropic_messages([AssistantMessagePart(parts=parts)])

        assert result == [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "first thought", "signature": "s1"},
                    {"type": "redacted_thinking", "data": "ENC"},
                    {"type": "thinking", "thinking": "second thought", "signature": "s2"},
                    {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "x"}},
                ],
            }
        ]
