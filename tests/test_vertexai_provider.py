"""Tests for VertexAIProvider: constructor, auth resolution, and token refresh."""

import asyncio
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import MagicMock, patch

from pydantic import BaseModel
import pytest

from dobby.providers.vertexai.adapter import VertexAIProvider
from dobby.providers.vertexai.converters import to_vertexai_messages, to_vertexai_tool
from dobby.tools import Tool
from dobby.types import (
    AssistantMessagePart,
    Base64ImageSource,
    ImagePart,
    ReasoningPart,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    UserMessagePart,
)


def _mock_credentials(valid: bool = True, token: str = "test-token") -> MagicMock:
    """Build a MagicMock standing in for a google.auth.credentials.Credentials."""
    creds = MagicMock()
    creds.valid = valid
    creds.token = token
    return creds


# ---------------------------------------------------------------------------
# Constructor / auth resolution tests
# ---------------------------------------------------------------------------


class TestVertexAIConstructor:
    """Test VertexAIProvider.__init__ auth resolution and instance attributes."""

    def test_explicit_credentials_skips_default(self) -> None:
        creds = _mock_credentials()
        with patch("dobby.providers.vertexai.adapter.google.auth.default") as mock_default:
            provider = VertexAIProvider(
                model="meta/llama-3.1-405b-instruct-maas",
                project="my-project",
                credentials=creds,
            )

        mock_default.assert_not_called()
        assert provider._credentials is creds

    def test_no_credentials_calls_default_with_scopes(self) -> None:
        creds = _mock_credentials()
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        with patch(
            "dobby.providers.vertexai.adapter.google.auth.default",
            return_value=(creds, "my-project"),
        ) as mock_default:
            provider = VertexAIProvider(
                model="meta/llama-3.1-405b-instruct-maas",
                project="my-project",
                scopes=scopes,
            )

        mock_default.assert_called_once_with(scopes=scopes)
        assert provider._credentials is creds

    def test_no_credentials_default_scopes_is_none_by_default(self) -> None:
        creds = _mock_credentials()
        with patch(
            "dobby.providers.vertexai.adapter.google.auth.default",
            return_value=(creds, "my-project"),
        ) as mock_default:
            VertexAIProvider(model="meta/llama-3.1-405b-instruct-maas", project="my-project")

        mock_default.assert_called_once_with(scopes=None)

    def test_name_returns_vertexai(self) -> None:
        provider = VertexAIProvider(
            model="meta/llama-3.1-405b-instruct-maas",
            project="my-project",
            credentials=_mock_credentials(),
        )
        assert provider.name == "vertexai"

    def test_model_returns_constructor_model(self) -> None:
        provider = VertexAIProvider(
            model="meta/llama-3.1-405b-instruct-maas",
            project="my-project",
            credentials=_mock_credentials(),
        )
        assert provider.model == "meta/llama-3.1-405b-instruct-maas"

    def test_max_retries_stored_on_instance(self) -> None:
        provider = VertexAIProvider(
            model="meta/llama-3.1-405b-instruct-maas",
            project="my-project",
            credentials=_mock_credentials(),
            max_retries=7,
        )
        assert provider.max_retries == 7

    def test_client_constructed_with_bound_bearer_token_callable(self) -> None:
        with patch("dobby.providers.vertexai.adapter.AsyncOpenAI") as mock_openai_cls:
            provider = VertexAIProvider(
                model="meta/llama-3.1-405b-instruct-maas",
                project="my-project",
                credentials=_mock_credentials(),
            )

        _, kwargs = mock_openai_cls.call_args
        api_key_callable = kwargs["api_key"]

        # The callable must be the provider's own bound _bearer_token method, not a
        # static string or an unbound function.
        assert api_key_callable == provider._bearer_token
        assert api_key_callable.__self__ is provider
        assert api_key_callable.__func__ is VertexAIProvider._bearer_token
        assert "default_headers" not in kwargs

    def test_client_base_url_targets_vertex_maas_endpoint(self) -> None:
        with patch("dobby.providers.vertexai.adapter.AsyncOpenAI") as mock_openai_cls:
            VertexAIProvider(
                model="meta/llama-3.1-405b-instruct-maas",
                project="my-project",
                location="europe-west4",
                credentials=_mock_credentials(),
            )

        _, kwargs = mock_openai_cls.call_args
        assert kwargs["base_url"] == (
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/my-project"
            "/locations/europe-west4/endpoints/openapi"
        )


# ---------------------------------------------------------------------------
# _bearer_token tests
# ---------------------------------------------------------------------------


class TestBearerToken:
    """Test VertexAIProvider._bearer_token refresh/lock behavior."""

    def _make_provider(self, creds: MagicMock) -> VertexAIProvider:
        return VertexAIProvider(
            model="meta/llama-3.1-405b-instruct-maas",
            project="my-project",
            credentials=creds,
        )

    def test_returns_existing_token_without_refresh_when_valid(self) -> None:
        creds = _mock_credentials(valid=True, token="fresh-token")
        provider = self._make_provider(creds)

        token = asyncio.run(provider._bearer_token())

        assert token == "fresh-token"
        creds.refresh.assert_not_called()

    def test_refreshes_when_not_valid(self) -> None:
        creds = _mock_credentials(valid=False, token="stale-token")

        def _refresh(request: object) -> None:
            creds.valid = True
            creds.token = "refreshed-token"

        creds.refresh = MagicMock(side_effect=_refresh)
        provider = self._make_provider(creds)

        token = asyncio.run(provider._bearer_token())

        creds.refresh.assert_called_once()
        assert token == "refreshed-token"

    def test_refresh_failure_propagates(self) -> None:
        creds = _mock_credentials(valid=False)
        creds.refresh = MagicMock(side_effect=RuntimeError("refresh boom"))
        provider = self._make_provider(creds)

        with pytest.raises(RuntimeError, match="refresh boom"):
            asyncio.run(provider._bearer_token())

    def test_concurrent_calls_refresh_only_once(self) -> None:
        creds = _mock_credentials(valid=False, token="stale-token")
        call_count = 0

        def _refresh(request: object) -> None:
            nonlocal call_count
            call_count += 1
            creds.valid = True
            creds.token = "refreshed-token"

        creds.refresh = MagicMock(side_effect=_refresh)
        provider = self._make_provider(creds)

        async def _run() -> list[str]:
            return list(await asyncio.gather(provider._bearer_token(), provider._bearer_token()))

        tokens = asyncio.run(_run())

        assert call_count == 1
        assert tokens == ["refreshed-token", "refreshed-token"]


# ---------------------------------------------------------------------------
# to_vertexai_messages tests
# ---------------------------------------------------------------------------


class TestToVertexAIMessages:
    """Test to_vertexai_messages() Chat Completions message conversion."""

    def test_single_user_text_message(self) -> None:
        messages = [UserMessagePart(parts=[TextPart(text="Hello there")])]

        result = to_vertexai_messages(messages)

        assert result == [{"role": "user", "content": "Hello there"}]

    def test_user_message_with_image_becomes_content_list(self) -> None:
        messages = [
            UserMessagePart(
                parts=[
                    TextPart(text="What is this?"),
                    ImagePart(source=Base64ImageSource(data="abcd", media_type="image/png")),
                ]
            )
        ]

        result = to_vertexai_messages(messages)

        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abcd"}},
                ],
            }
        ]

    def test_assistant_tool_call_populates_tool_calls_no_content_leakage(self) -> None:
        messages = [
            AssistantMessagePart(
                parts=[ToolUsePart(id="call_1", name="get_weather", inputs={"city": "SF"})]
            )
        ]

        result = to_vertexai_messages(messages)

        assert len(result) == 1
        assistant_message = result[0]
        assert assistant_message["role"] == "assistant"
        assert assistant_message["content"] is None
        assert assistant_message["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
            }
        ]
        # No bare tool-call leakage into content.
        assert "get_weather" not in (assistant_message["content"] or "")

    def test_assistant_text_and_tool_call_together(self) -> None:
        messages = [
            AssistantMessagePart(
                parts=[
                    TextPart(text="Let me check that."),
                    ToolUsePart(id="call_1", name="get_weather", inputs={"city": "SF"}),
                ]
            )
        ]

        result = to_vertexai_messages(messages)

        assert result[0]["content"] == "Let me check that."
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_result_part_converts_to_tool_message(self) -> None:
        messages = [
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id="call_1",
                        name="get_weather",
                        parts=[TextPart(text="Sunny, 72F")],
                    )
                ]
            )
        ]

        result = to_vertexai_messages(messages)

        assert result == [{"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 72F"}]

    def test_tool_result_part_is_error_prefixes_content(self) -> None:
        messages = [
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id="call_1",
                        name="get_weather",
                        parts=[TextPart(text="city not found")],
                        is_error=True,
                    )
                ]
            )
        ]

        result = to_vertexai_messages(messages)

        assert result == [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Failed to execute tool: city not found",
            }
        ]

    def test_tool_result_part_is_error_with_no_content(self) -> None:
        messages = [
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id="call_1", name="get_weather", parts=[], is_error=True
                    )
                ]
            )
        ]

        result = to_vertexai_messages(messages)

        assert result[0]["content"] == "Failed to execute tool:"

    def test_reasoning_part_is_dropped(self) -> None:
        messages = [
            AssistantMessagePart(
                parts=[
                    ReasoningPart(text="thinking about the weather..."),
                    TextPart(text="It's sunny."),
                ]
            )
        ]

        result = to_vertexai_messages(messages)

        assert len(result) == 1
        assert result[0] == {"role": "assistant", "content": "It's sunny."}
        # No crash, and no key/text referencing the reasoning content anywhere.
        assert "thinking" not in str(result)

    def test_assistant_message_with_only_reasoning_emits_nothing(self) -> None:
        """A dropped-only ReasoningPart must not emit an empty assistant message.

        content=None with no tool_calls is invalid on most Chat Completions
        servers.
        """
        messages = [
            AssistantMessagePart(parts=[ReasoningPart(text="thinking...")]),
        ]

        result = to_vertexai_messages(messages)

        assert result == []

    def test_multi_turn_conversation_order_and_shape(self) -> None:
        messages = [
            UserMessagePart(parts=[TextPart(text="What's the weather in SF?")]),
            AssistantMessagePart(
                parts=[ToolUsePart(id="call_1", name="get_weather", inputs={"city": "SF"})]
            ),
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id="call_1",
                        name="get_weather",
                        parts=[TextPart(text="Sunny, 72F")],
                    )
                ]
            ),
            AssistantMessagePart(parts=[TextPart(text="It's sunny and 72F in SF.")]),
        ]

        result = to_vertexai_messages(messages)

        assert [m["role"] for m in result] == ["user", "assistant", "tool", "assistant"]
        assert result[0] == {"role": "user", "content": "What's the weather in SF?"}
        assert result[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result[2] == {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 72F"}
        assert result[3] == {"role": "assistant", "content": "It's sunny and 72F in SF."}

    def test_system_prompt_prepended_at_caller_layer(self) -> None:
        """to_vertexai_messages() takes no system_prompt param.

        Mirrors OpenAIProvider.chat(), which inserts the system message into the
        message list before calling its converter (dobby/providers/openai/
        adapter.py). VertexAIProvider.chat() (U3) is expected to prepend the
        system message the same way; this test locks in that prepending pattern
        at the caller layer this converter is designed for.
        """
        messages = to_vertexai_messages([UserMessagePart(parts=[TextPart(text="hi")])])
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        assert messages[0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        assert messages[1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# to_vertexai_tool tests
# ---------------------------------------------------------------------------


class _WeatherModel(BaseModel):
    """Get the weather for a city."""

    city: str
    units: str = "celsius"


class TestToVertexAITool:
    """Test to_vertexai_tool() Chat Completions tool-schema re-nesting."""

    def test_pydantic_model_tool_renests_openai_format(self) -> None:
        tool = Tool.from_model(_WeatherModel, name="get_weather", description="Get the weather")

        openai_format = tool.to_openai_format()
        result = to_vertexai_tool(tool)

        assert result == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": openai_format["parameters"],
            },
        }

    def test_parameter_list_tool_renests_openai_format(self) -> None:
        @dataclass
        class GetWeatherTool(Tool):
            description: ClassVar[str] = "Get the weather for a city"

            def __call__(self, city: str, units: str = "celsius") -> dict[str, str]:
                return {"city": city}

        tool = GetWeatherTool()

        openai_format = tool.to_openai_format()
        result = to_vertexai_tool(tool)

        assert result == {
            "type": "function",
            "function": {
                "name": "GetWeatherTool",
                "description": "Get the weather for a city",
                "parameters": openai_format["parameters"],
            },
        }
        assert set(result["function"]["parameters"]["properties"]) == {"city", "units"}
