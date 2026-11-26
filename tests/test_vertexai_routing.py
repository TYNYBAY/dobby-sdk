"""Tests for VertexAIProvider routing: native Gemini, native Anthropic, OpenAI-compatible."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
from google.genai import types as genai_types
import httpx
import pytest

from dobby import AgentExecutor
from dobby.providers.anthropic.adapter import AnthropicProvider
from dobby.providers.gemini.adapter import GeminiProvider
from dobby.providers.vertexai.adapter import VertexAIProvider
from dobby.providers.vertexai.converters import to_vertexai_tool
from dobby.tools import Tool
from dobby.types import (
    StreamEndEvent,
    TextPart,
    UserMessagePart,
)

GEMINI = "google/gemini-2.5-flash"
CLAUDE = "anthropic/claude-sonnet-4-5"
LLAMA = "meta/llama-3.3-70b-instruct-maas"
HI = [UserMessagePart(parts=[TextPart(text="hi")])]

ADAPTER = "dobby.providers.vertexai.adapter"


def _mock_credentials(valid: bool = True, token: str | None = "test-token") -> MagicMock:
    """A MagicMock standing in for google.auth credentials."""
    creds = MagicMock()
    creds.valid = valid
    creds.expired = not valid
    creds.token = token
    creds.quota_project_id = None
    return creds


@contextmanager
def _patched_sdks() -> Iterator[SimpleNamespace]:
    """Patch both native SDK clients at their import site in the Vertex adapter."""
    with (
        patch(f"{ADAPTER}.genai") as mock_genai,
        patch(f"{ADAPTER}.AsyncAnthropicVertex") as mock_anthropic_cls,
    ):
        yield SimpleNamespace(genai=mock_genai, anthropic_cls=mock_anthropic_cls)


def _provider(model: str | None = GEMINI, **kwargs: Any) -> VertexAIProvider:
    """Build a provider with mocked credentials and a mocked AsyncOpenAI client."""
    kwargs.setdefault("credentials", _mock_credentials())
    provider = VertexAIProvider(model=model, project="my-project", **kwargs)
    provider._client = MagicMock()
    provider._client.chat.completions.create = AsyncMock()
    return provider


def _gemini_response(
    parts: list[genai_types.Part], finish_reason: str = "STOP"
) -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=parts),
                finish_reason=finish_reason,
            )
        ]
    )


def _claude_message(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> Any:
    return anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    )


def _chat_completion_response(text: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="m")


@dataclass
class EchoTool(Tool):
    """Return the given value."""

    name = "echo"
    description = "Echo a value."

    async def __call__(self, value: str) -> str:
        return value


# ---------------------------------------------------------------------------
# Route resolution
# ---------------------------------------------------------------------------


class TestRouteResolution:
    """The route is chosen once, at construction, from the instance model."""

    @pytest.mark.parametrize(
        ("kwargs", "route"),
        [
            ({"model": GEMINI}, "gemini"),
            ({"model": "google/gemini-3-pro-preview"}, "gemini"),
            ({"model": CLAUDE}, "anthropic"),
            ({"model": LLAMA}, "openai"),
            ({"model": "google/gemma-3-27b-it-maas"}, "openai"),
            ({"model": "gemini-2.5-flash"}, "openai"),
            ({"model": "google/gemini-x", "endpoint_id": "123"}, "openai"),
            ({"model": GEMINI, "routing": "openai"}, "openai"),
            ({"model": CLAUDE, "routing": "openai"}, "openai"),
        ],
    )
    def test_route_per_publisher(self, kwargs: dict[str, Any], route: str) -> None:
        with _patched_sdks():
            provider = _provider(**kwargs)
        assert provider._route == route

    def test_native_routes_build_their_adapters(self) -> None:
        with _patched_sdks():
            gemini = _provider(GEMINI)
            claude = _provider(CLAUDE)
            llama = _provider(LLAMA)
        assert isinstance(gemini._native, GeminiProvider)
        assert gemini._native.model == "gemini-2.5-flash"
        assert isinstance(claude._native, AnthropicProvider)
        assert claude._native.model == "claude-sonnet-4-5"
        assert llama._native is None

    def test_openai_route_builds_no_native_client(self) -> None:
        with _patched_sdks() as sdks:
            _provider(GEMINI, routing="openai")
            _provider(CLAUDE, routing="openai")
            _provider(LLAMA)
        sdks.genai.Client.assert_not_called()
        sdks.anthropic_cls.assert_not_called()

    def test_invalid_routing_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="routing"):
            VertexAIProvider(
                model=GEMINI,
                project="my-project",
                credentials=_mock_credentials(),
                routing="fast",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("model", [GEMINI, CLAUDE, LLAMA])
    def test_name_and_model_unchanged_on_every_route(self, model: str) -> None:
        with _patched_sdks():
            provider = _provider(model)
        assert provider.name == "vertexai"
        assert provider.model == model
        if provider._native is not None:
            assert provider._native.name == "vertexai"

    def test_route_is_private(self) -> None:
        with _patched_sdks():
            provider = _provider(GEMINI)
        assert not hasattr(provider, "route")


# ---------------------------------------------------------------------------
# Native SDK client construction
# ---------------------------------------------------------------------------


class TestNativeClientConstruction:
    """Native SDK clients share the provider's resolved credentials, project and location."""

    def test_genai_client_built_for_vertex_with_provider_credentials(self) -> None:
        creds = _mock_credentials()
        with _patched_sdks() as sdks:
            provider = _provider(GEMINI, location="europe-west4", credentials=creds)
        sdks.genai.Client.assert_called_once_with(
            vertexai=True, credentials=creds, project="my-project", location="europe-west4"
        )
        assert provider._native.client is sdks.genai.Client.return_value

    def test_anthropic_vertex_client_built_with_retries_off(self) -> None:
        creds = _mock_credentials()
        with _patched_sdks() as sdks:
            provider = _provider(CLAUDE, location="global", credentials=creds, max_retries=5)
        sdks.anthropic_cls.assert_called_once_with(
            region="global", project_id="my-project", credentials=creds, max_retries=0
        )
        assert provider._native.client is sdks.anthropic_cls.return_value
        assert provider._native.max_retries == 5

    def test_gemini_adapter_inherits_max_retries(self) -> None:
        with _patched_sdks():
            provider = _provider(GEMINI, max_retries=2)
        assert provider._native.max_retries == 2

    def test_openai_client_still_built_on_native_routes(self) -> None:
        with _patched_sdks(), patch(f"{ADAPTER}.AsyncOpenAI") as mock_openai_cls:
            provider = VertexAIProvider(
                model=GEMINI, project="my-project", credentials=_mock_credentials()
            )
        assert provider.client is mock_openai_cls.return_value


# ---------------------------------------------------------------------------
# chat() delegation
# ---------------------------------------------------------------------------


def _gemini_provider(**kwargs: Any) -> tuple[VertexAIProvider, MagicMock]:
    with _patched_sdks() as sdks:
        provider = _provider(GEMINI, **kwargs)
    client = sdks.genai.Client.return_value
    client.aio.models.generate_content = AsyncMock(
        return_value=_gemini_response([genai_types.Part(text="hello")])
    )
    return provider, client


def _claude_provider(**kwargs: Any) -> tuple[VertexAIProvider, MagicMock]:
    with _patched_sdks() as sdks:
        provider = _provider(CLAUDE, **kwargs)
    client = sdks.anthropic_cls.return_value
    client.messages.create = AsyncMock(
        return_value=_claude_message([{"type": "text", "text": "hello"}])
    )
    return provider, client


class TestGeminiRouteChat:
    """chat() on the Gemini route goes to the genai client, never AsyncOpenAI."""

    def test_sends_model_without_publisher_prefix(self) -> None:
        provider, client = _gemini_provider()

        result = asyncio.run(provider.chat(HI))

        call = client.aio.models.generate_content.await_args.kwargs
        assert call["model"] == "gemini-2.5-flash"
        assert call["contents"][0].parts[0].text == "hi"
        assert result.parts == [TextPart(text="hello")]
        provider._client.chat.completions.create.assert_not_called()

    def test_per_call_override_strips_publisher_prefix(self) -> None:
        provider, client = _gemini_provider()

        asyncio.run(provider.chat(HI, model="google/gemini-2.5-pro"))

        assert client.aio.models.generate_content.await_args.kwargs["model"] == "gemini-2.5-pro"

    def test_reasoning_effort_and_max_tokens_reach_gemini_config(self) -> None:
        provider, client = _gemini_provider()

        asyncio.run(provider.chat(HI, reasoning_effort="high", max_tokens=64, top_p=0.5))

        config = client.aio.models.generate_content.await_args.kwargs["config"]
        assert config.thinking_config.thinking_level == "HIGH"
        assert config.max_output_tokens == 64
        assert config.top_p == 0.5

    def test_format_tools_reach_genai_as_gemini_tools(self) -> None:
        provider, client = _gemini_provider()
        tools = provider.format_tools([EchoTool()])

        asyncio.run(provider.chat(HI, tools=tools))

        sent = client.aio.models.generate_content.await_args.kwargs["config"].tools
        assert all(isinstance(tool, genai_types.Tool) for tool in sent)
        assert sent[0].function_declarations[0].name == "echo"

    @pytest.mark.parametrize(
        "kwarg",
        [
            "stop",
            "n",
            "logprobs",
            "top_logprobs",
            "response_format",
            "extra_body",
            "max_completion_tokens",
            "tool_choice",
            "parallel_tool_calls",
            "user",
            "stream_options",
        ],
    )
    def test_chat_completions_kwargs_rejected_before_network(self, kwarg: str) -> None:
        provider, client = _gemini_provider()

        with pytest.raises(TypeError) as exc_info:
            asyncio.run(provider.chat(HI, **{kwarg: ["x"]}))

        message = str(exc_info.value)
        assert repr(kwarg) in message
        assert "gemini" in message
        assert 'routing="openai"' in message
        client.aio.models.generate_content.assert_not_called()

    def test_native_gemini_fields_with_shared_names_pass_through(self) -> None:
        """`seed` and the penalties mean the same on Gemini, so they are not rejected."""
        provider, client = _gemini_provider()

        asyncio.run(provider.chat(HI, seed=7, frequency_penalty=0.5, presence_penalty=0.25))

        config = client.aio.models.generate_content.await_args.kwargs["config"]
        assert config.seed == 7
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.25


class TestAnthropicRouteChat:
    """chat() on the Anthropic route goes to AsyncAnthropicVertex.messages.create."""

    def test_sends_model_without_publisher_prefix(self) -> None:
        provider, client = _claude_provider()

        result = asyncio.run(provider.chat(HI, system_prompt="be brief"))

        sent = client.messages.create.await_args.kwargs
        assert sent["model"] == "claude-sonnet-4-5"
        assert sent["system"] == "be brief"
        assert result.parts == [TextPart(text="hello")]
        provider._client.chat.completions.create.assert_not_called()

    def test_int_reasoning_effort_becomes_thinking_budget(self) -> None:
        provider, client = _claude_provider()

        asyncio.run(provider.chat(HI, reasoning_effort=2048, max_tokens=4096))

        sent = client.messages.create.await_args.kwargs
        assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert sent["max_tokens"] == 4096

    def test_str_reasoning_effort_names_the_way_back(self) -> None:
        provider, client = _claude_provider()

        with pytest.raises(TypeError) as exc_info:
            asyncio.run(provider.chat(HI, reasoning_effort="high"))

        assert 'routing="openai"' in str(exc_info.value)
        assert "anthropic" in str(exc_info.value)
        client.messages.create.assert_not_called()

    @pytest.mark.parametrize(
        ("kwarg", "value"),
        [
            ("extra_body", {"x": 1}),
            ("tool_choice", {"type": "function", "function": {"name": "echo"}}),
            ("max_completion_tokens", 64),
            ("parallel_tool_calls", False),
            ("seed", 7),
            ("frequency_penalty", 0.5),
            ("presence_penalty", 0.5),
        ],
    )
    def test_chat_completions_kwargs_rejected_before_network(
        self, kwarg: str, value: object
    ) -> None:
        """Claude has no seed or penalty fields, so these stay rejected here."""
        provider, client = _claude_provider()

        with pytest.raises(TypeError) as exc_info:
            asyncio.run(provider.chat(HI, **{kwarg: value}))

        message = str(exc_info.value)
        assert repr(kwarg) in message
        assert "anthropic" in message
        assert 'routing="openai"' in message
        client.messages.create.assert_not_called()

    def test_format_tools_reach_messages_create_in_anthropic_format(self) -> None:
        provider, client = _claude_provider()
        tools = provider.format_tools([EchoTool()])

        asyncio.run(provider.chat(HI, tools=tools))

        sent = client.messages.create.await_args.kwargs["tools"]
        assert sent == [EchoTool().to_anthropic_format()]
        assert sent[0]["name"] == "echo"
        assert "input_schema" in sent[0]

    def test_other_kwargs_pass_through(self) -> None:
        provider, client = _claude_provider()

        asyncio.run(provider.chat(HI, top_p=0.9))

        assert client.messages.create.await_args.kwargs["top_p"] == 0.9


class TestOpenAIRouteChat:
    """Everything not routed natively keeps today's Chat Completions request."""

    @pytest.mark.parametrize(
        ("model", "kwargs"),
        [
            (LLAMA, {}),
            (GEMINI, {"routing": "openai"}),
            (CLAUDE, {"routing": "openai"}),
        ],
    )
    def test_request_matches_todays_shape(self, model: str, kwargs: dict[str, Any]) -> None:
        with _patched_sdks() as sdks:
            provider = _provider(model, **kwargs)
        provider._client.chat.completions.create = AsyncMock(
            return_value=_chat_completion_response()
        )

        asyncio.run(provider.chat(HI, stop=["x"], extra_body={"k": "v"}))

        assert provider._client.chat.completions.create.await_args.kwargs == {
            "stop": ["x"],
            "extra_body": {"k": "v"},
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
        }
        sdks.genai.Client.assert_not_called()
        sdks.anthropic_cls.assert_not_called()

    def test_endpoint_id_with_gemini_label_uses_chat_completions(self) -> None:
        provider = _provider("google/gemini-x", endpoint_id="123")
        provider._client.chat.completions.create = AsyncMock(
            return_value=_chat_completion_response()
        )

        asyncio.run(provider.chat(HI))

        assert provider._client.chat.completions.create.await_args.kwargs["model"] == ""

    def test_format_tools_returns_nested_chat_completions_format(self) -> None:
        provider = _provider(LLAMA)
        assert provider.format_tools([EchoTool()]) == [to_vertexai_tool(EchoTool())]


# ---------------------------------------------------------------------------
# AgentExecutor integration
# ---------------------------------------------------------------------------


async def _run(executor: AgentExecutor) -> list:
    return [event async for event in executor.run_stream(messages=list(HI))]


def test_executor_round_trip_on_gemini_route() -> None:
    """AgentExecutor sends Gemini-format tools and completes a tool round trip."""
    with _patched_sdks() as sdks:
        provider = _provider(GEMINI)
    client = sdks.genai.Client.return_value
    turns = [
        [
            _gemini_response(
                [
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(name="echo", args={"value": "A"})
                    )
                ]
            )
        ],
        [_gemini_response([genai_types.Part(text="done")])],
    ]
    requests: list[dict[str, Any]] = []

    async def fake_stream(*, model, contents, config):
        requests.append({"model": model, "contents": contents, "config": config})
        chunks = turns[len(requests) - 1]

        async def gen() -> AsyncIterator[genai_types.GenerateContentResponse]:
            for chunk in chunks:
                yield chunk

        return gen()

    client.aio.models.generate_content_stream = fake_stream
    executor = AgentExecutor("vertexai", provider, tools=[EchoTool()])

    events = asyncio.run(_run(executor))

    assert len(requests) == 2
    assert all(request["model"] == "gemini-2.5-flash" for request in requests)
    assert all(isinstance(tool, genai_types.Tool) for tool in requests[0]["config"].tools)
    history = requests[1]["contents"]
    assert [content.role for content in history] == ["user", "model", "user"]
    assert history[1].parts[0].function_call.name == "echo"
    assert history[2].parts[0].function_response.name == "echo"
    assert history[2].parts[0].function_response.response == {"text": "A"}
    end = [event for event in events if isinstance(event, StreamEndEvent)][-1]
    assert end.parts == [TextPart(text="done")]
    provider._client.chat.completions.create.assert_not_called()


def _ev(type_: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=type_, **fields)


def _claude_stream(blocks: list[list[SimpleNamespace]], stop_reason: str) -> list:
    """Raw Anthropic stream events for the given content blocks."""
    start = _ev(
        "message_start",
        message=SimpleNamespace(
            id="msg_1", model="claude-sonnet-4-5", usage=SimpleNamespace(input_tokens=5)
        ),
    )
    end = [
        _ev(
            "message_delta",
            delta=SimpleNamespace(stop_reason=stop_reason),
            usage=SimpleNamespace(output_tokens=7),
        ),
        _ev("message_stop"),
    ]
    return [start, *(event for block in blocks for event in block), *end]


def _claude_tool_block(index: int, tool_id: str, value: str) -> list[SimpleNamespace]:
    block = SimpleNamespace(type="tool_use", id=tool_id, name="echo")
    return [
        _ev("content_block_start", index=index, content_block=block),
        _ev(
            "content_block_delta",
            index=index,
            delta=SimpleNamespace(
                type="input_json_delta", partial_json=json.dumps({"value": value})
            ),
        ),
        _ev("content_block_stop", index=index),
    ]


def _claude_text_block(index: int, text: str) -> list[SimpleNamespace]:
    return [
        _ev("content_block_start", index=index, content_block=SimpleNamespace(type="text")),
        _ev(
            "content_block_delta", index=index, delta=SimpleNamespace(type="text_delta", text=text)
        ),
        _ev("content_block_stop", index=index),
    ]


def test_executor_round_trip_on_claude_route_batches_the_tool_turn() -> None:
    """A parallel tool turn reaches messages.create as one assistant + one user turn."""
    with _patched_sdks() as sdks:
        provider = _provider(CLAUDE)
    client = sdks.anthropic_cls.return_value
    turns = [
        _claude_stream(
            [
                _claude_text_block(0, "Checking."),
                _claude_tool_block(1, "toolu_a", "A"),
                _claude_tool_block(2, "toolu_b", "B"),
            ],
            "tool_use",
        ),
        _claude_stream([_claude_text_block(0, "done")], "end_turn"),
    ]
    requests: list[dict[str, Any]] = []

    async def fake_create(**kwargs):
        requests.append(kwargs)
        events = turns[len(requests) - 1]

        async def gen():
            for event in events:
                yield event

        return gen()

    client.messages.create = fake_create
    executor = AgentExecutor("vertexai", provider, tools=[EchoTool()])

    events = asyncio.run(_run(executor))

    assert len(requests) == 2
    assert requests[0]["model"] == "claude-sonnet-4-5"
    assert requests[0]["tools"] == [EchoTool().to_anthropic_format()]
    history = requests[1]["messages"]
    assert [message["role"] for message in history] == ["user", "assistant", "user"]
    assistant_blocks = history[1]["content"]
    assert [block["type"] for block in assistant_blocks] == ["text", "tool_use", "tool_use"]
    assert [block["id"] for block in assistant_blocks[1:]] == ["toolu_a", "toolu_b"]
    result_blocks = history[2]["content"]
    assert [block["type"] for block in result_blocks] == ["tool_result", "tool_result"]
    assert [block["tool_use_id"] for block in result_blocks] == ["toolu_a", "toolu_b"]
    end = [event for event in events if isinstance(event, StreamEndEvent)][-1]
    assert end.parts == [TextPart(text="done")]
    provider._client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Credential refresh on the native routes
# ---------------------------------------------------------------------------


def _expired_credentials() -> MagicMock:
    """Credentials whose refresh sets a token, as google-auth does."""
    creds = _mock_credentials(valid=False, token=None)

    def _refresh(request: object) -> None:
        creds.token = "refreshed-token"
        creds.expired = False
        creds.valid = True

    creds.refresh = MagicMock(side_effect=_refresh)
    return creds


class TestNativeCredentialRefresh:
    """Each native SDK refreshes the shared credentials and sends the new token.

    The real SDK clients are built (not patched); only their HTTP transport is
    replaced, so the SDKs' own refresh paths run.
    """

    def test_gemini_route_refreshes_expired_credentials_once(self) -> None:
        creds = _expired_credentials()
        provider = VertexAIProvider(model=GEMINI, project="my-project", credentials=creds)
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "hi"}]},
                            "finishReason": "STOP",
                        }
                    ]
                },
            )

        provider._native.client._api_client._async_httpx_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

        result = asyncio.run(provider.chat(HI))

        assert result.parts == [TextPart(text="hi")]
        creds.refresh.assert_called_once()
        assert len(seen) == 1
        assert seen[0].headers["authorization"] == "Bearer refreshed-token"
        assert "/projects/my-project/locations/us-central1/" in str(seen[0].url)
        assert str(seen[0].url).endswith("/models/gemini-2.5-flash:generateContent")

    def test_anthropic_route_refreshes_expired_credentials_once(self) -> None:
        creds = _expired_credentials()
        provider = VertexAIProvider(model=CLAUDE, project="my-project", credentials=creds)
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider._native.client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        result = asyncio.run(provider.chat(HI))

        assert result.parts == [TextPart(text="hi")]
        creds.refresh.assert_called_once()
        assert len(seen) == 1
        assert seen[0].headers["authorization"] == "Bearer refreshed-token"
        assert "/projects/my-project/locations/us-central1/" in str(seen[0].url)
        assert "claude-sonnet-4-5" in str(seen[0].url)
