"""Tests for OpenAI provider reasoning_effort validation."""

import asyncio
from unittest.mock import MagicMock

import pytest

from dobby.providers.openai.adapter import _validate_reasoning_effort


class TestOpenAIReasoningEffort:
    """Test reasoning_effort validation for OpenAI provider."""

    @pytest.mark.parametrize("value", ["none", "minimal", "low", "medium", "high", "xhigh"])
    def test_valid_effort_strings(self, value: str) -> None:
        assert _validate_reasoning_effort(value) == value

    def test_strips_and_lowercases(self) -> None:
        assert _validate_reasoning_effort("  High  ") == "high"
        assert _validate_reasoning_effort("LOW") == "low"

    def test_int_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="requires reasoning_effort as str"):
            _validate_reasoning_effort(5000)  # type: ignore[arg-type]

    def test_invalid_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid reasoning_effort"):
            _validate_reasoning_effort("turbo")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid reasoning_effort"):
            _validate_reasoning_effort("")


class TestOpenAIBuildKwargs:
    """Test _build_kwargs forwards provider-native params via extra."""

    def test_extra_forwarded(self) -> None:
        from dobby.providers.openai.adapter import OpenAIProvider

        kwargs = OpenAIProvider._build_kwargs(
            model="gpt-4o",
            input=[],
            extra={"top_p": 0.3, "metadata": {"k": "v"}},
        )
        assert kwargs["top_p"] == 0.3
        assert kwargs["metadata"] == {"k": "v"}

    def test_extra_never_clobbers_core_fields(self) -> None:
        from dobby.providers.openai.adapter import OpenAIProvider

        kwargs = OpenAIProvider._build_kwargs(
            model="gpt-4o",
            input=[],
            extra={"model": "evil"},
        )
        assert kwargs["model"] == "gpt-4o"


class TestOpenAIChatModelOverride:
    """Test chat() honors per-call model and forwards kwargs end-to-end."""

    def _make_provider(self):
        from dobby.providers.openai.adapter import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4o"
        provider.azure_deployment_id = None
        provider.max_retries = 3
        provider._client = MagicMock()
        return provider

    def _run_chat(self, provider, **chat_kwargs):
        from dobby.types import TextPart, UserMessagePart

        captured: dict = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.output = []
            resp.usage = None
            resp.model = kwargs["model"]
            return resp

        provider._client.responses.create = fake_create
        asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                **chat_kwargs,
            )
        )
        return captured

    def test_model_override_used(self) -> None:
        provider = self._make_provider()
        captured = self._run_chat(provider, model="gpt-5.1")
        assert captured["model"] == "gpt-5.1"

    def test_defaults_to_instance_model(self) -> None:
        provider = self._make_provider()
        captured = self._run_chat(provider)
        assert captured["model"] == "gpt-4o"

    def test_kwargs_forwarded_to_create(self) -> None:
        provider = self._make_provider()
        captured = self._run_chat(provider, top_p=0.7)
        assert captured["top_p"] == 0.7
