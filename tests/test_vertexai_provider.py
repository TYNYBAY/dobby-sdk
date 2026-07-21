"""Tests for VertexAIProvider: constructor, auth resolution, and token refresh."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import logging
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import openai
from pydantic import BaseModel
import pytest

from dobby.providers.base import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    ProviderError,
    RateLimitError,
)
from dobby.providers.vertexai import adapter as vertexai_adapter_module
from dobby.providers.vertexai.adapter import VertexAIProvider
from dobby.providers.vertexai.converters import to_vertexai_messages, to_vertexai_tool
from dobby.tools import Tool
from dobby.types import (
    AssistantMessagePart,
    Base64ImageSource,
    ImagePart,
    ReasoningPart,
    StreamEndEvent,
    StreamErrorEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolResultPart,
    ToolUseEvent,
    ToolUsePart,
    Usage,
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

    @pytest.mark.parametrize("bad_model", ["", "   ", None])
    def test_empty_or_none_model_id_rejected_at_construction(self, bad_model) -> None:
        """Plain input validation, kept separate from the removed family guard.

        Without it, `model=None` constructs fine and reaches the wire as a null
        model field, turning a config typo into an opaque server-side error.
        """
        with pytest.raises(ValueError, match="non-empty model id"):
            VertexAIProvider(
                model=bad_model,
                project="my-project",
                credentials=_mock_credentials(),
            )

    def test_whitespace_only_per_call_override_rejected(self) -> None:
        """A whitespace-only override is a typo, not a fallback signal."""
        provider = _make_chat_provider()

        async def _run() -> None:
            await provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                stream=False,
                model="   ",
            )

        with pytest.raises(ValueError, match="non-empty model id"):
            asyncio.run(_run())
        provider._client.chat.completions.create.assert_not_called()

    @pytest.mark.parametrize(
        "model_id",
        [
            "meta/llama-3.1-405b-instruct-maas",
            "openai/gpt-oss-120b",
            "google/gemini-2.5-flash",
            "gemini-2.5-flash",
            "claude-sonnet-4-5",
            "anthropic/claude-sonnet-4-5",
            "publishers/anthropic/models/claude-sonnet-4-5",
        ],
    )
    def test_model_ids_forwarded_verbatim_without_family_validation(self, model_id: str) -> None:
        """No model id is rejected on family grounds -- the endpoint decides.

        The previous native-Gemini/native-Claude guard was removed: it rejected
        legitimately-named self-deployed containers (e.g. `gemini-finetune-v2`),
        matched only unqualified prefixes so fully-qualified ids slipped through
        anyway, and diverged from how comparable SDKs treat model names.
        """
        provider = VertexAIProvider(
            model=model_id,
            project="my-project",
            credentials=_mock_credentials(),
        )
        assert provider.model == model_id

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


def _base_url(**kwargs) -> str:
    """Construct a provider with a patched client and return the base_url it built."""
    with patch("dobby.providers.vertexai.adapter.AsyncOpenAI") as mock_openai_cls:
        VertexAIProvider(project="my-project", credentials=_mock_credentials(), **kwargs)
    return mock_openai_cls.call_args.kwargs["base_url"]


class TestEndpointRouting:
    """Model Garden vs self-deployed endpoints: URL shape, body, and validation.

    Reference: Google's OpenAI-compat docs (migrate/openai/auth-and-credentials,
    migrate/openai/examples, maas/call-open-model-apis) and the Endpoint
    `dedicatedEndpointDns` schema.
    """

    # --- URL construction ---------------------------------------------------

    def test_model_garden_url_unchanged(self) -> None:
        """Backwards compatibility: the default path is byte-identical to before."""
        assert _base_url(model="meta/llama-3.1-405b-instruct-maas") == (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project"
            "/locations/us-central1/endpoints/openapi"
        )

    def test_global_location_drops_the_region_prefix(self) -> None:
        """`global` uses the bare host, not `global-aiplatform.googleapis.com`."""
        url = _base_url(model="meta/llama-3.1-405b-instruct-maas", location="global")
        assert url.startswith("https://aiplatform.googleapis.com/")
        assert "global-aiplatform" not in url
        assert "/locations/global/" in url

    def test_deployed_endpoint_url_uses_endpoint_id_segment(self) -> None:
        assert _base_url(endpoint_id="5464397967697903616") == (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project"
            "/locations/us-central1/endpoints/5464397967697903616"
        )

    def test_dedicated_endpoint_host_replaces_the_regional_host(self) -> None:
        """A dedicated endpoint stops being served by the shared regional DNS."""
        url = _base_url(
            endpoint_id="546",
            endpoint_host="546.us-central1-987.prediction.vertexai.goog",
            api_version="v1beta1",
        )
        assert url == (
            "https://546.us-central1-987.prediction.vertexai.goog/v1beta1"
            "/projects/my-project/locations/us-central1/endpoints/546"
        )

    @pytest.mark.parametrize(
        "supplied_host",
        [
            "546.us-central1-987.prediction.vertexai.goog",
            "https://546.us-central1-987.prediction.vertexai.goog",
            "https://546.us-central1-987.prediction.vertexai.goog/",
        ],
    )
    def test_endpoint_host_scheme_is_stripped_if_present(self, supplied_host: str) -> None:
        """The API returns `dedicatedEndpointDns` bare; its schema documents a scheme."""
        url = _base_url(endpoint_id="546", endpoint_host=supplied_host)
        assert url.startswith("https://546.us-central1-987.prediction.vertexai.goog/v1/")
        assert "https://https://" not in url

    def test_api_version_is_configurable(self) -> None:
        url = _base_url(model="meta/llama", api_version="v1beta1")
        assert "/v1beta1/projects/" in url

    # --- request body -------------------------------------------------------

    def test_model_garden_sends_the_model_id_in_the_body(self) -> None:
        provider = _make_chat_provider()
        provider._client.chat.completions.create = AsyncMock(return_value=_make_response("ok"))

        asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        _, kwargs = provider._client.chat.completions.create.call_args
        assert kwargs["model"] == "meta/llama-3.1-405b-instruct-maas"

    def test_deployed_endpoint_sends_empty_model_in_the_body(self) -> None:
        """The endpoint selects the model; Vertex ignores this field.

        Google's REST samples omit it and their OpenAI-SDK samples pass `""`.
        We send `""` because the OpenAI SDK requires the argument.
        """
        provider = VertexAIProvider(
            endpoint_id="5464397967697903616",
            project="my-project",
            credentials=_mock_credentials(),
        )
        provider._client = MagicMock()
        provider._client.chat.completions.create = AsyncMock(return_value=_make_response("ok"))

        asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        _, kwargs = provider._client.chat.completions.create.call_args
        assert kwargs["model"] == ""

    def test_deployed_endpoint_per_call_override_still_sends_empty_model(self) -> None:
        """An override relabels the response; it cannot change what the endpoint serves."""
        provider = VertexAIProvider(
            endpoint_id="546", project="my-project", credentials=_mock_credentials()
        )
        provider._client = MagicMock()
        provider._client.chat.completions.create = AsyncMock(return_value=_make_response("ok"))

        asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                stream=False,
                model="whatever",
            )
        )

        _, kwargs = provider._client.chat.completions.create.call_args
        assert kwargs["model"] == ""

    # --- configuration validation ------------------------------------------

    def test_model_defaults_to_a_label_for_deployed_endpoints(self) -> None:
        provider = VertexAIProvider(
            endpoint_id="546", project="my-project", credentials=_mock_credentials()
        )
        assert provider.model == "endpoint-546"
        assert provider.endpoint_id == "546"

    def test_explicit_model_is_kept_as_a_label_for_deployed_endpoints(self) -> None:
        provider = VertexAIProvider(
            model="gemma-2-9b-it",
            endpoint_id="546",
            project="my-project",
            credentials=_mock_credentials(),
        )
        assert provider.model == "gemma-2-9b-it"

    def test_model_still_required_for_model_garden(self) -> None:
        """A type checker rejects this via the overloads; the runtime check backs it up."""
        with pytest.raises(ValueError, match="non-empty model id"):
            VertexAIProvider(  # type: ignore[call-overload]
                project="my-project", credentials=_mock_credentials()
            )

    def test_endpoint_host_without_endpoint_id_rejected(self) -> None:
        """Same: no overload accepts endpoint_host without endpoint_id."""
        with pytest.raises(ValueError, match="endpoint_host is only valid together"):
            VertexAIProvider(  # type: ignore[call-overload]
                model="meta/llama",
                endpoint_host="546.us-central1-987.prediction.vertexai.goog",
                project="my-project",
                credentials=_mock_credentials(),
            )

    @pytest.mark.parametrize("bad_endpoint_id", ["", "   "])
    def test_empty_endpoint_id_rejected(self, bad_endpoint_id: str) -> None:
        with pytest.raises(ValueError, match="non-empty endpoint_id"):
            VertexAIProvider(
                endpoint_id=bad_endpoint_id,
                project="my-project",
                credentials=_mock_credentials(),
            )

    def test_empty_api_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty api_version"):
            VertexAIProvider(
                model="meta/llama",
                api_version="  ",
                project="my-project",
                credentials=_mock_credentials(),
            )

    def test_new_params_are_keyword_only(self) -> None:
        """Positional callers must not be able to bind into the new params."""
        with pytest.raises(TypeError):
            VertexAIProvider(
                "meta/llama", "my-project", "us-central1", _mock_credentials(), None, 3, "546"
            )  # type: ignore[misc]

    def test_existing_positional_call_still_works(self) -> None:
        """Backwards compatibility for the pre-existing positional signature."""
        with patch("dobby.providers.vertexai.adapter.AsyncOpenAI"):
            provider = VertexAIProvider(
                "meta/llama-3.1-405b-instruct-maas",
                "my-project",
                "us-central1",
                _mock_credentials(),
                None,
                7,
            )
        assert provider.model == "meta/llama-3.1-405b-instruct-maas"
        assert provider.project == "my-project"
        assert provider.max_retries == 7
        assert provider.endpoint_id is None


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


# ---------------------------------------------------------------------------
# chat(stream=False) tests
# ---------------------------------------------------------------------------


def _make_chat_provider() -> VertexAIProvider:
    """Provider with a mocked underlying AsyncOpenAI client for chat() tests."""
    provider = VertexAIProvider(
        model="meta/llama-3.1-405b-instruct-maas",
        project="my-project",
        credentials=_mock_credentials(),
    )
    provider._client = MagicMock()
    return provider


def _make_message(
    content: str | None = None, tool_calls: list[SimpleNamespace] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _make_tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_usage(
    prompt_tokens: int = 10, completion_tokens: int = 5, total_tokens: int = 15
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _make_response(
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = "stop",
    usage: SimpleNamespace | None = None,
    model: str = "meta/llama-3.1-405b-instruct-maas",
) -> SimpleNamespace:
    choice = SimpleNamespace(
        message=_make_message(content=content, tool_calls=tool_calls),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


class TestNonStreamChatCompletion:
    """Test VertexAIProvider.chat(stream=False)."""

    def test_plain_text_response(self) -> None:
        provider = _make_chat_provider()
        response = _make_response(content="Hello!", finish_reason="stop")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        assert result.parts == [TextPart(text="Hello!")]
        assert result.stop_reason == "end_turn"

    def test_tool_calls_response(self) -> None:
        provider = _make_chat_provider()
        tool_calls = [
            _make_tool_call("call_1", "get_weather", '{"city": "SF"}'),
            _make_tool_call("call_2", "get_time", '{"tz": "UTC"}'),
        ]
        response = _make_response(content=None, tool_calls=tool_calls, finish_reason="tool_calls")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        assert result.stop_reason == "tool_use"
        assert result.parts == [
            ToolUsePart(id="call_1", name="get_weather", inputs={"city": "SF"}),
            ToolUsePart(id="call_2", name="get_time", inputs={"tz": "UTC"}),
        ]

    def test_tool_calls_present_wins_over_mismatched_finish_reason(self) -> None:
        """Nonconformant finish_reason="stop" alongside tool_calls still yields "tool_use".

        Guards against a server reporting finish_reason="stop" while still
        including tool_calls on the message.
        """
        provider = _make_chat_provider()
        tool_calls = [_make_tool_call("call_1", "get_weather", '{"city": "SF"}')]
        response = _make_response(content=None, tool_calls=tool_calls, finish_reason="stop")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        assert result.stop_reason == "tool_use"

    def test_finish_reason_length_maps_to_max_tokens(self) -> None:
        provider = _make_chat_provider()
        response = _make_response(content="cut off", finish_reason="length")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        assert result.stop_reason == "max_tokens"

    def test_finish_reason_content_filter_maps_through(self) -> None:
        provider = _make_chat_provider()
        response = _make_response(content="", finish_reason="content_filter")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        assert result.stop_reason == "content_filter"

    def test_unknown_finish_reason_falls_back_and_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _make_chat_provider()
        response = _make_response(content="hmm", finish_reason="some_future_value")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        with caplog.at_level(logging.DEBUG, logger="dobby"):
            result = asyncio.run(
                provider.chat(
                    messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False
                )
            )

        assert result.stop_reason == "end_turn"
        assert any("some_future_value" in record.message for record in caplog.records)

    def test_usage_populates_dobby_usage(self) -> None:
        provider = _make_chat_provider()
        usage = _make_usage(prompt_tokens=100, completion_tokens=42, total_tokens=142)
        response = _make_response(content="hi", finish_reason="stop", usage=usage)
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        assert result.usage == Usage(input_tokens=100, output_tokens=42, total_tokens=142)

    def test_no_extra_headers_or_manual_auth_kwargs(self) -> None:
        provider = _make_chat_provider()
        response = _make_response(content="hi", finish_reason="stop")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        _, kwargs = provider._client.chat.completions.create.call_args
        assert "extra_headers" not in kwargs
        assert "api_key" not in kwargs
        assert "headers" not in kwargs

    def test_instance_model_used_by_default(self) -> None:
        provider = _make_chat_provider()
        response = _make_response(content="hi", finish_reason="stop")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        _, kwargs = provider._client.chat.completions.create.call_args
        assert kwargs["model"] == "meta/llama-3.1-405b-instruct-maas"

    def test_per_call_model_overrides_instance_model(self) -> None:
        provider = _make_chat_provider()
        response = _make_response(content="hi", finish_reason="stop")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                stream=False,
                model="meta/llama-3.1-70b-instruct-maas",
            )
        )

        _, kwargs = provider._client.chat.completions.create.call_args
        assert kwargs["model"] == "meta/llama-3.1-70b-instruct-maas"

    @pytest.mark.parametrize("model_id", ["google/gemini-2.5-flash", "claude-sonnet-4-5"])
    def test_per_call_override_to_previously_rejected_id_is_forwarded(self, model_id: str) -> None:
        """The removed guard fired at two sites: construction AND per-call override.

        A constructor-only test would still pass if a per-call-only guard were
        reintroduced, so the permissive contract is pinned at both sites.
        """
        provider = _make_chat_provider()
        provider._client.chat.completions.create = AsyncMock(return_value=_make_response("ok"))

        async def _run() -> None:
            await provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                stream=False,
                model=model_id,
            )

        asyncio.run(_run())

        _, kwargs = provider._client.chat.completions.create.call_args
        assert kwargs["model"] == model_id

    def test_per_call_model_override_empty_string_falls_back_to_instance_model(self) -> None:
        """An empty-string override is treated as no override.

        This matches `model or self._model` semantics rather than being evaluated
        by the guard itself -- the instance model was already validated at
        construction.
        """
        provider = _make_chat_provider()
        response = _make_response(content="hi", finish_reason="stop")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                stream=False,
                model="",
            )
        )

        _, kwargs = provider._client.chat.completions.create.call_args
        assert kwargs["model"] == "meta/llama-3.1-405b-instruct-maas"

    def test_system_prompt_prepended_as_system_message(self) -> None:
        provider = _make_chat_provider()
        response = _make_response(content="hi", finish_reason="stop")
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                stream=False,
                system_prompt="You are a helpful assistant.",
            )
        )

        _, kwargs = provider._client.chat.completions.create.call_args
        assert kwargs["messages"][0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        assert kwargs["messages"][1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# chat() error translation tests
# ---------------------------------------------------------------------------


def _make_openai_error(
    error_cls: type[Exception], status_code: int = 500, headers: dict | None = None
) -> Exception:
    """Create a mock OpenAI SDK error (mirrors tests/test_provider_errors.py)."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}

    if error_cls is openai.RateLimitError:
        err = openai.RateLimitError.__new__(openai.RateLimitError)
        err.response = response
        err.status_code = 429
        err.message = "Rate limited"
        err.body = None
        return err
    if error_cls is openai.APITimeoutError:
        err = openai.APITimeoutError.__new__(openai.APITimeoutError)
        err.message = "Timed out"
        err.request = MagicMock()
        return err
    if error_cls is openai.APIConnectionError:
        err = openai.APIConnectionError.__new__(openai.APIConnectionError)
        err.message = "Connection failed"
        err.request = MagicMock()
        return err
    if error_cls is openai.InternalServerError:
        err = openai.InternalServerError.__new__(openai.InternalServerError)
        err.response = response
        err.status_code = status_code
        err.message = "Server error"
        err.body = None
        return err
    if error_cls is openai.APIStatusError:
        err = openai.APIStatusError.__new__(openai.APIStatusError)
        err.response = response
        err.status_code = status_code
        err.message = "API error"
        err.body = None
        return err
    return error_cls()


class TestIncompleteUsagePayload:
    """A usage object whose token counts are all None must not kill the stream.

    Observed live from Vertex's `openai/gpt-oss-20b-maas`: the SSE terminal chunk
    carries a usage object with prompt_tokens/completion_tokens/total_tokens all
    None. Guarding on `chunk.usage is not None` let those through into `Usage(...)`,
    raising a pydantic ValidationError mid-stream and aborting the response.
    Every mock in this suite supplied real integers, so no unit test caught it.
    """

    def _null_usage(self) -> SimpleNamespace:
        return SimpleNamespace(prompt_tokens=None, completion_tokens=None, total_tokens=None)

    def test_streaming_survives_null_usage_counts(self) -> None:
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(
                choices=[_make_stream_choice(_make_delta(content="hi"), finish_reason="stop")]
            ),
            _make_stream_chunk(choices=[], usage=self._null_usage()),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        end = events[-1]
        assert isinstance(end, StreamEndEvent)
        assert end.usage is None
        assert end.parts == [TextPart(text="hi")]

    def test_non_streaming_survives_null_usage_counts(self) -> None:
        provider = _make_chat_provider()
        response = _make_response("ok", usage=self._null_usage())
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        assert result.usage is None
        assert result.parts == [TextPart(text="ok")]

    def test_partially_null_usage_is_dropped_not_zero_filled(self) -> None:
        """Zero-filling would report a real token spend as free."""
        provider = _make_chat_provider()
        partial = SimpleNamespace(prompt_tokens=10, completion_tokens=None, total_tokens=None)
        provider._client.chat.completions.create = AsyncMock(
            return_value=_make_response("ok", usage=partial)
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        assert result.usage is None

    def test_complete_usage_still_reported(self) -> None:
        provider = _make_chat_provider()
        provider._client.chat.completions.create = AsyncMock(
            return_value=_make_response("ok", usage=_make_usage(7, 3, 10))
        )

        result = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False)
        )

        assert result.usage == Usage(input_tokens=7, output_tokens=3, total_tokens=10)


class TestKwargsPassthrough:
    """`chat(**kwargs)` must reach the wire, as it does on every sibling provider.

    Previously accepted and silently discarded: a caller migrating from
    GeminiProvider or OpenAIProvider lost max_tokens with no error and no warning.
    """

    def _run(self, provider: VertexAIProvider, **chat_kwargs) -> dict:
        provider._client.chat.completions.create = AsyncMock(return_value=_make_response("ok"))
        asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                stream=False,
                **chat_kwargs,
            )
        )
        _, kwargs = provider._client.chat.completions.create.call_args
        return kwargs

    def test_extra_params_forwarded_verbatim(self) -> None:
        kwargs = self._run(
            _make_chat_provider(), max_tokens=64, top_p=0.5, stop=["\n"], extra_body={"google": {}}
        )
        assert kwargs["max_tokens"] == 64
        assert kwargs["top_p"] == 0.5
        assert kwargs["stop"] == ["\n"]
        assert kwargs["extra_body"] == {"google": {}}

    def test_no_extra_params_leaves_payload_unchanged(self) -> None:
        """Backwards compatibility: the payload without kwargs is exactly as before."""
        kwargs = self._run(_make_chat_provider())
        assert set(kwargs) == {"model", "messages", "temperature"}

    def test_owned_fields_cannot_be_clobbered(self) -> None:
        """`extra` is applied first, so the provider's own fields win.

        Note `chat()` itself is already safe: `model`, `messages`, `temperature`
        and `tools` are explicit parameters, so passing them again raises
        TypeError before `**kwargs` is even formed. This asserts the layer below,
        where an `extra` dict is the only way those keys could arrive.
        """
        provider = _make_chat_provider()

        built = provider._build_kwargs(
            model="meta/llama-3.1-405b-instruct-maas",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            extra={"model": "sneaky", "messages": [], "temperature": 99, "max_tokens": 8},
        )

        assert built["model"] == "meta/llama-3.1-405b-instruct-maas"
        assert built["messages"] == [{"role": "user", "content": "hi"}]
        assert built["temperature"] == 0.0
        assert built["max_tokens"] == 8

    def test_chat_rejects_duplicate_owned_params(self) -> None:
        """Belt and braces: the explicit-parameter collision is a loud TypeError."""
        provider = _make_chat_provider()

        with pytest.raises(TypeError):
            asyncio.run(
                provider.chat(
                    messages=[UserMessagePart(parts=[TextPart(text="hi")])],
                    stream=False,
                    **{"messages": []},  # type: ignore[misc]
                )
            )

    def test_deployed_endpoint_model_stays_empty_despite_extra(self) -> None:
        provider = VertexAIProvider(
            endpoint_id="546", project="my-project", credentials=_mock_credentials()
        )
        provider._client = MagicMock()
        kwargs = self._run(provider, max_tokens=32, model="anything")
        assert kwargs["model"] == ""
        assert kwargs["max_tokens"] == 32

    def test_extra_params_forwarded_on_the_streaming_path(self) -> None:
        provider = _make_chat_provider()
        provider._client.chat.completions.create = AsyncMock(
            return_value=_chunk_stream(
                [
                    _make_stream_chunk(
                        choices=[_make_stream_choice(_make_delta(content="hi"), "stop")]
                    )
                ]
            )
        )

        asyncio.run(_collect_stream_events(provider, max_tokens=64))

        _, kwargs = provider._client.chat.completions.create.call_args
        assert kwargs["max_tokens"] == 64
        assert kwargs["stream"] is True


class TestDedicatedEndpointHint:
    """A dedicated endpoint stops being served by the shared regional DNS.

    That surfaces as a bare 404 or connection failure. Nothing client-side can
    detect `dedicatedEndpointEnabled` without an admin API call this provider
    deliberately does not make, so the error names the setting that fixes it --
    but only when the failure shape actually matches.
    """

    def _provider(self, **kwargs) -> VertexAIProvider:
        return VertexAIProvider(project="my-project", credentials=_mock_credentials(), **kwargs)

    def test_404_on_deployed_endpoint_suggests_endpoint_host(self) -> None:
        provider = self._provider(endpoint_id="546")

        with pytest.raises(ProviderError, match="dedicatedEndpointDns") as exc:
            provider._translate_error(_make_openai_error(openai.APIStatusError, 404))

        assert exc.value.status_code == 404
        assert "'546'" in str(exc.value)

    def test_connection_error_on_deployed_endpoint_suggests_endpoint_host(self) -> None:
        provider = self._provider(endpoint_id="546")

        with pytest.raises(APIConnectionError, match="dedicatedEndpointDns"):
            provider._translate_error(_make_openai_error(openai.APIConnectionError))

    def test_no_hint_when_endpoint_host_already_supplied(self) -> None:
        """The user already did the thing the hint would suggest."""
        provider = self._provider(
            endpoint_id="546", endpoint_host="546.us-central1-987.prediction.vertexai.goog"
        )

        with pytest.raises(ProviderError) as exc:
            provider._translate_error(_make_openai_error(openai.APIStatusError, 404))

        assert "dedicatedEndpointDns" not in str(exc.value)

    def test_no_hint_for_model_garden(self) -> None:
        """Model Garden has no dedicated DNS, so the hint would be noise."""
        provider = self._provider(model="meta/llama-3.1-405b-instruct-maas")

        with pytest.raises(ProviderError) as exc:
            provider._translate_error(_make_openai_error(openai.APIStatusError, 404))

        assert "dedicatedEndpointDns" not in str(exc.value)

    def test_non_404_status_errors_are_unaffected(self) -> None:
        """A 403 is an IAM problem, not a routing one -- don't misdirect."""
        provider = self._provider(endpoint_id="546")

        with pytest.raises(ProviderError) as exc:
            provider._translate_error(_make_openai_error(openai.APIStatusError, 403))

        assert exc.value.status_code == 403
        assert "dedicatedEndpointDns" not in str(exc.value)

    def test_500_still_maps_to_internal_server_error(self) -> None:
        """The new 404 branch must not shadow the 5xx branch below it."""
        provider = self._provider(endpoint_id="546")

        with pytest.raises(InternalServerError) as exc:
            provider._translate_error(_make_openai_error(openai.InternalServerError, 503))

        assert exc.value.status_code == 503


class TestVertexAIErrorTranslation:
    """Test VertexAIProvider chat() error translation via _translate_error."""

    @pytest.mark.parametrize(
        ("openai_error_cls", "dobby_error_cls", "status_code"),
        [
            (openai.RateLimitError, RateLimitError, 429),
            (openai.APITimeoutError, APITimeoutError, 500),
            (openai.APIConnectionError, APIConnectionError, 500),
            (openai.InternalServerError, InternalServerError, 502),
            (openai.APIStatusError, ProviderError, 403),
        ],
    )
    def test_chat_translates_openai_errors(
        self,
        openai_error_cls: type[Exception],
        dobby_error_cls: type[ProviderError],
        status_code: int,
    ) -> None:
        provider = _make_chat_provider()
        native_err = _make_openai_error(openai_error_cls, status_code)
        provider._client.chat.completions.create = AsyncMock(side_effect=native_err)

        async def _run() -> None:
            await provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=False
            )

        with pytest.raises(dobby_error_cls) as exc_info:
            asyncio.run(_run())

        assert exc_info.value.provider == "vertexai"
        assert exc_info.value.__cause__ is native_err


# ---------------------------------------------------------------------------
# chat(stream=True) tests
# ---------------------------------------------------------------------------


def _make_tool_call_delta(
    index: int,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_delta(
    content: str | None = None, tool_calls: list[SimpleNamespace] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _make_stream_choice(
    delta: SimpleNamespace, finish_reason: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(delta=delta, finish_reason=finish_reason)


def _make_stream_chunk(
    choices: list[SimpleNamespace] | None = None,
    usage: SimpleNamespace | None = None,
    chunk_id: str = "chatcmpl-test",
    model: str = "meta/llama-3.1-405b-instruct-maas",
) -> SimpleNamespace:
    return SimpleNamespace(choices=choices or [], usage=usage, id=chunk_id, model=model)


async def _chunk_stream(chunks: list[SimpleNamespace]) -> AsyncIterator[SimpleNamespace]:
    for chunk in chunks:
        yield chunk


class _FailingChunkStream:
    """Async iterator that yields a prefix of chunks then raises mid-stream."""

    def __init__(self, chunks: list[SimpleNamespace], fail_after: int, exc: Exception) -> None:
        self._chunks = chunks
        self._fail_after = fail_after
        self._exc = exc
        self._index = 0

    def __aiter__(self) -> "_FailingChunkStream":
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._index == self._fail_after:
            raise self._exc
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


async def _collect_stream_events(provider: VertexAIProvider, **kwargs: Any) -> list[Any]:
    stream = await provider.chat(
        messages=[UserMessagePart(parts=[TextPart(text="hi")])], stream=True, **kwargs
    )
    return [event async for event in stream]


class TestStreamChatCompletion:
    """Test VertexAIProvider.chat(stream=True)."""

    def test_text_stream_accumulates_into_single_text_part(self) -> None:
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(choices=[_make_stream_choice(_make_delta(content="Hel"))]),
            _make_stream_chunk(choices=[_make_stream_choice(_make_delta(content="lo"))]),
            _make_stream_chunk(
                choices=[_make_stream_choice(_make_delta(content="!"), finish_reason="stop")]
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert [e.delta for e in text_deltas] == ["Hel", "lo", "!"]

        end_event = events[-1]
        assert isinstance(end_event, StreamEndEvent)
        assert end_event.parts == [TextPart(text="Hello!")]
        assert end_event.stop_reason == "end_turn"

    def test_stream_start_event_yielded_exactly_once_on_first_chunk(self) -> None:
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(choices=[_make_stream_choice(_make_delta(content="a"))]),
            _make_stream_chunk(
                choices=[_make_stream_choice(_make_delta(content="b"), finish_reason="stop")]
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        start_events = [e for e in events if isinstance(e, StreamStartEvent)]
        assert len(start_events) == 1
        assert events[0] is start_events[0]

    def test_fragmented_tool_call_accumulates_by_index(self) -> None:
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(
                choices=[
                    _make_stream_choice(
                        _make_delta(
                            tool_calls=[
                                _make_tool_call_delta(
                                    0, call_id="call_1", name="get_weather", arguments='{"ci'
                                )
                            ]
                        )
                    )
                ]
            ),
            _make_stream_chunk(
                choices=[
                    _make_stream_choice(
                        _make_delta(tool_calls=[_make_tool_call_delta(0, arguments='ty": "SF"}')])
                    )
                ]
            ),
            _make_stream_chunk(
                choices=[_make_stream_choice(_make_delta(), finish_reason="tool_calls")]
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        end_event = events[-1]
        assert isinstance(end_event, StreamEndEvent)
        assert end_event.parts == [
            ToolUsePart(id="call_1", name="get_weather", inputs={"city": "SF"})
        ]
        assert end_event.stop_reason == "tool_use"

        tool_use_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert len(tool_use_events) == 1
        assert tool_use_events[0].id == "call_1"
        assert tool_use_events[0].inputs == {"city": "SF"}

    def test_two_parallel_tool_calls_accumulate_independently(self) -> None:
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(
                choices=[
                    _make_stream_choice(
                        _make_delta(
                            tool_calls=[
                                _make_tool_call_delta(
                                    0, call_id="call_1", name="get_weather", arguments='{"city":'
                                ),
                                _make_tool_call_delta(
                                    1, call_id="call_2", name="get_time", arguments='{"tz":'
                                ),
                            ]
                        )
                    )
                ]
            ),
            _make_stream_chunk(
                choices=[
                    _make_stream_choice(
                        _make_delta(
                            tool_calls=[
                                _make_tool_call_delta(0, arguments=' "SF"}'),
                                _make_tool_call_delta(1, arguments=' "UTC"}'),
                            ]
                        )
                    )
                ]
            ),
            _make_stream_chunk(
                choices=[_make_stream_choice(_make_delta(), finish_reason="tool_calls")]
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        end_event = events[-1]
        assert end_event.parts == [
            ToolUsePart(id="call_1", name="get_weather", inputs={"city": "SF"}),
            ToolUsePart(id="call_2", name="get_time", inputs={"tz": "UTC"}),
        ]

    def test_finish_reason_tool_calls_maps_to_stop_reason_tool_use(self) -> None:
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(
                choices=[
                    _make_stream_choice(
                        _make_delta(
                            tool_calls=[
                                _make_tool_call_delta(
                                    0, call_id="call_1", name="noop", arguments="{}"
                                )
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        assert events[-1].stop_reason == "tool_use"

    def test_usage_in_dedicated_final_empty_choices_chunk(self) -> None:
        provider = _make_chat_provider()
        usage = _make_usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [
            _make_stream_chunk(
                choices=[_make_stream_choice(_make_delta(content="hi"), finish_reason="stop")]
            ),
            _make_stream_chunk(choices=[], usage=usage),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        assert events[-1].usage == Usage(input_tokens=10, output_tokens=5, total_tokens=15)

    def test_usage_on_chunk_with_non_empty_choices_still_captured(self) -> None:
        """Regression test for the idempotent `usage is not None` check.

        Third-party (vLLM-backed) servers may attach usage to a chunk that
        also carries non-empty choices, deviating from native OpenAI's
        dedicated-empty-choices-chunk convention.
        """
        provider = _make_chat_provider()
        usage = _make_usage(prompt_tokens=20, completion_tokens=8, total_tokens=28)
        chunks = [
            _make_stream_chunk(
                choices=[_make_stream_choice(_make_delta(content="hi"), finish_reason="stop")],
                usage=usage,
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        assert events[-1].usage == Usage(input_tokens=20, output_tokens=8, total_tokens=28)

    def test_tool_call_id_and_name_on_later_chunk_still_accumulates(self) -> None:
        """Regression test for the idempotent set-if-present merge.

        Third-party servers may send id/name on a later fragment for an
        index instead of the first one; the merge must be idempotent
        (set-if-present), not "only check the first chunk".
        """
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(
                choices=[
                    _make_stream_choice(
                        _make_delta(tool_calls=[_make_tool_call_delta(0, arguments='{"city"')])
                    )
                ]
            ),
            _make_stream_chunk(
                choices=[
                    _make_stream_choice(
                        _make_delta(
                            tool_calls=[
                                _make_tool_call_delta(
                                    0, call_id="call_1", name="get_weather", arguments=': "SF"}'
                                )
                            ]
                        )
                    )
                ]
            ),
            _make_stream_chunk(
                choices=[_make_stream_choice(_make_delta(), finish_reason="tool_calls")]
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        end_event = events[-1]
        assert end_event.parts == [
            ToolUsePart(id="call_1", name="get_weather", inputs={"city": "SF"})
        ]

    def test_exceeding_distinct_index_bound_yields_stream_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vertexai_adapter_module, "_MAX_TOOL_CALL_INDICES", 2)
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(
                choices=[
                    _make_stream_choice(
                        _make_delta(
                            tool_calls=[
                                _make_tool_call_delta(0, call_id="c0", name="t0", arguments="{}"),
                                _make_tool_call_delta(1, call_id="c1", name="t1", arguments="{}"),
                                _make_tool_call_delta(2, call_id="c2", name="t2", arguments="{}"),
                            ]
                        )
                    )
                ]
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        error_events = [e for e in events if isinstance(e, StreamErrorEvent)]
        assert len(error_events) == 1
        # Stops accumulating instead of growing unboundedly: no StreamEndEvent
        # is ever produced once the bound is exceeded.
        assert not any(isinstance(e, StreamEndEvent) for e in events)

    def test_exceeding_arguments_length_bound_yields_stream_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            vertexai_adapter_module, "_MAX_ACCUMULATED_TOOL_CALL_ARGUMENTS_LENGTH", 5
        )
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(
                choices=[
                    _make_stream_choice(
                        _make_delta(
                            tool_calls=[
                                _make_tool_call_delta(
                                    0,
                                    call_id="call_1",
                                    name="get_weather",
                                    arguments="0123456789",
                                )
                            ]
                        )
                    )
                ]
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        events = asyncio.run(_collect_stream_events(provider))

        error_events = [e for e in events if isinstance(e, StreamErrorEvent)]
        assert len(error_events) == 1
        assert not any(isinstance(e, StreamEndEvent) for e in events)

    def test_mid_stream_exception_translates_through_translate_error(self) -> None:
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(choices=[_make_stream_choice(_make_delta(content="hi"))]),
        ]
        native_err = _make_openai_error(openai.APIConnectionError)
        provider._client.chat.completions.create = AsyncMock(
            return_value=_FailingChunkStream(chunks, fail_after=1, exc=native_err)
        )

        with pytest.raises(APIConnectionError) as exc_info:
            asyncio.run(_collect_stream_events(provider))

        assert exc_info.value.provider == "vertexai"
        assert exc_info.value.__cause__ is native_err

    def test_stream_create_kwargs_include_stream_options_no_manual_auth(self) -> None:
        provider = _make_chat_provider()
        chunks = [
            _make_stream_chunk(
                choices=[_make_stream_choice(_make_delta(content="hi"), finish_reason="stop")]
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))

        asyncio.run(_collect_stream_events(provider))

        _, kwargs = provider._client.chat.completions.create.call_args
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}
        assert "extra_headers" not in kwargs
        assert "api_key" not in kwargs
        assert "headers" not in kwargs
