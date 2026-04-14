"""Tests for OpenAI provider reasoning_effort validation."""

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
