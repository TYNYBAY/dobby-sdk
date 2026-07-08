"""Tests for Gemini provider: per-call model override and kwargs passthrough."""

import asyncio
from unittest.mock import MagicMock

from dobby.types import TextPart, UserMessagePart


class TestGeminiChatModelOverride:
    """Test chat() honors per-call model and forwards kwargs into the config."""

    def _make_provider(self):
        from dobby.providers.gemini.adapter import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model = "gemini-2.5-flash"
        provider.max_retries = 3
        provider._client = MagicMock()
        return provider

    def _run_chat(self, provider, **chat_kwargs):
        captured: dict = {}

        async def fake_generate(*, model, contents, config):
            captured["model"] = model
            captured["config"] = config
            resp = MagicMock()
            resp.candidates = None
            resp.usage_metadata = None
            return resp

        provider._client.aio.models.generate_content = fake_generate
        asyncio.get_event_loop().run_until_complete(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                **chat_kwargs,
            )
        )
        return captured

    def test_model_override_used(self) -> None:
        provider = self._make_provider()
        captured = self._run_chat(provider, model="gemini-2.5-pro")
        assert captured["model"] == "gemini-2.5-pro"

    def test_defaults_to_instance_model(self) -> None:
        provider = self._make_provider()
        captured = self._run_chat(provider)
        assert captured["model"] == "gemini-2.5-flash"

    def test_kwargs_forwarded_to_config(self) -> None:
        provider = self._make_provider()
        captured = self._run_chat(provider, top_p=0.4)
        assert captured["config"].top_p == 0.4
