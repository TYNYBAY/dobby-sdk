"""Tests for the Gemini provider adapter.

Covers request building (config, kwargs passthrough, model override), response
parsing for both the streaming and non-streaming paths, and tool-call ID minting.
"""

import asyncio
import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from google.genai import types as genai_types
import pytest

from dobby import AgentExecutor
from dobby.exceptions import ApprovalRequired
from dobby.providers.base import ToolCallTruncatedError
from dobby.tools import Tool
from dobby.types import (
    AssistantMessagePart,
    MessagePart,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolResultEvent,
    ToolResultPart,
    ToolUseErrorEvent,
    ToolUseEvent,
    ToolUsePart,
    UserMessagePart,
)

HI: list[MessagePart] = [UserMessagePart(parts=[TextPart(text="hi")])]


def _make_provider():
    from dobby.providers.gemini.adapter import GeminiProvider

    provider = GeminiProvider.__new__(GeminiProvider)
    provider._model = "gemini-2.5-flash"
    provider.max_retries = 1
    provider._client = MagicMock()
    return provider


def _call(
    name: str, args: dict[str, Any] | None = None, call_id: str | None = None
) -> genai_types.Part:
    return genai_types.Part(
        function_call=genai_types.FunctionCall(name=name, args=args or {}, id=call_id)
    )


def _response(
    parts: list[genai_types.Part], finish_reason: str | None = "STOP", usage: bool = True
) -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=parts),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=(
            genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=3, candidates_token_count=5, total_token_count=8
            )
            if usage
            else None
        ),
    )


def _capture_non_stream(provider, response: genai_types.GenerateContentResponse) -> dict:
    captured: dict = {}

    async def fake_generate(*, model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return response

    provider._client.aio.models.generate_content = fake_generate
    return captured


def _capture_stream(provider, chunks: list[genai_types.GenerateContentResponse]) -> dict:
    captured: dict = {}

    async def fake_stream(*, model, contents, config):
        captured["model"] = model
        captured["config"] = config

        async def gen() -> AsyncIterator[genai_types.GenerateContentResponse]:
            for chunk in chunks:
                yield chunk

        return gen()

    provider._client.aio.models.generate_content_stream = fake_stream
    return captured


async def _collect(provider, messages, **kwargs) -> list:
    return [event async for event in await provider.chat(messages, stream=True, **kwargs)]


class TestGeminiChatModelOverride:
    """Test chat() honors per-call model and forwards kwargs into the config."""

    def _run_chat(self, provider, **chat_kwargs):
        captured = _capture_non_stream(provider, _response([]))
        asyncio.run(provider.chat(messages=HI, **chat_kwargs))
        return captured

    def test_model_override_used(self) -> None:
        provider = _make_provider()
        captured = self._run_chat(provider, model="gemini-2.5-pro")
        assert captured["model"] == "gemini-2.5-pro"

    def test_defaults_to_instance_model(self) -> None:
        provider = _make_provider()
        captured = self._run_chat(provider)
        assert captured["model"] == "gemini-2.5-flash"

    def test_kwargs_forwarded_to_config(self) -> None:
        provider = _make_provider()
        captured = self._run_chat(provider, top_p=0.4)
        assert captured["config"].top_p == 0.4


class TestInjectedClient:
    """The internal `_from_client` path runs the adapter on a pre-built client."""

    def test_calls_generate_content_with_model_and_converted_contents(self) -> None:
        from google import genai

        from dobby.providers.gemini.adapter import GeminiProvider

        client = MagicMock(spec=genai.Client)
        client.aio.models.generate_content = AsyncMock(
            return_value=_response([genai_types.Part(text="hello")])
        )
        provider = GeminiProvider._from_client(
            client, "gemini-2.5-pro", max_retries=0, provider_name="vertexai"
        )

        result = asyncio.run(provider.chat(HI))

        client.aio.models.generate_content.assert_awaited_once()
        call = client.aio.models.generate_content.await_args.kwargs
        assert call["model"] == "gemini-2.5-pro"
        assert [c.role for c in call["contents"]] == ["user"]
        assert call["contents"][0].parts[0].text == "hi"
        assert result.parts == [TextPart(text="hello")]
        assert provider.client is client
        assert provider.model == "gemini-2.5-pro"
        assert provider.max_retries == 0
        assert provider.name == "vertexai"

    def test_name_defaults_to_gemini(self) -> None:
        from dobby.providers.gemini.adapter import GeminiProvider

        provider = GeminiProvider._from_client(MagicMock(), "gemini-2.5-flash")
        assert provider.name == "gemini"
        assert provider.max_retries == 3


class TestRequestBuilding:
    """Characterize how chat() builds the GenerateContentConfig."""

    def test_config_defaults(self) -> None:
        provider = _make_provider()
        captured = _capture_non_stream(provider, _response([]))
        asyncio.run(provider.chat(HI))
        config = captured["config"]
        assert config.temperature == 0.0
        assert config.automatic_function_calling.disable is True
        assert config.system_instruction is None
        assert config.tools is None
        assert config.max_output_tokens is None
        assert config.thinking_config is None

    def test_system_prompt_tools_and_temperature(self) -> None:
        provider = _make_provider()
        captured = _capture_non_stream(provider, _response([]))
        tool = genai_types.Tool(
            function_declarations=[genai_types.FunctionDeclaration(name="search")]
        )
        asyncio.run(provider.chat(HI, system_prompt="be brief", temperature=0.7, tools=[tool]))
        config = captured["config"]
        assert config.system_instruction == "be brief"
        assert config.temperature == 0.7
        assert config.tools == [tool]

    def test_messages_converted_to_contents(self) -> None:
        provider = _make_provider()
        captured = _capture_non_stream(provider, _response([]))
        asyncio.run(provider.chat(HI))
        contents = captured["contents"]
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "hi"

    def test_max_tokens_mapped_to_max_output_tokens(self) -> None:
        provider = _make_provider()
        captured = _capture_non_stream(provider, _response([]))
        asyncio.run(provider.chat(HI, max_tokens=256))
        config = captured["config"]
        assert config.max_output_tokens == 256
        assert "max_tokens" not in config.model_dump(exclude_none=True)

    def test_streaming_uses_same_config(self) -> None:
        provider = _make_provider()
        captured = _capture_stream(provider, [_response([genai_types.Part(text="x")])])
        asyncio.run(_collect(provider, HI, system_prompt="sys", max_tokens=64))
        assert captured["config"].system_instruction == "sys"
        assert captured["config"].max_output_tokens == 64


class TestNonStreamParsing:
    """Characterize _parse_response via the non-streaming chat path."""

    def _chat(self, response, messages=HI) -> StreamEndEvent:
        provider = _make_provider()
        _capture_non_stream(provider, response)
        return asyncio.run(provider.chat(messages))

    def test_text_response(self) -> None:
        end = self._chat(_response([genai_types.Part(text="hello")]))
        assert end.parts == [TextPart(text="hello")]
        assert end.stop_reason == "end_turn"
        assert end.model == "gemini-2.5-flash"
        assert end.usage is not None
        assert (end.usage.input_tokens, end.usage.output_tokens, end.usage.total_tokens) == (
            3,
            5,
            8,
        )

    def test_no_usage(self) -> None:
        end = self._chat(_response([genai_types.Part(text="x")], usage=False))
        assert end.usage is None

    def test_tool_call_with_default_signature(self) -> None:
        end = self._chat(_response([_call("search", {"q": "x"})]))
        assert end.stop_reason == "tool_use"
        (part,) = end.parts
        assert isinstance(part, ToolUsePart)
        assert part.id == "call_search"
        assert part.name == "search"
        assert part.inputs == {"q": "x"}
        expected = base64.b64encode(b"skip_thought_signature_validator").decode()
        assert part.metadata == {"signature": expected}

    def test_tool_call_keeps_thought_signature(self) -> None:
        part = _call("search")
        part.thought_signature = b"sig"
        end = self._chat(_response([part]))
        tool_part = end.parts[0]
        assert isinstance(tool_part, ToolUsePart)
        assert tool_part.metadata == {"signature": base64.b64encode(b"sig").decode()}

    def test_safety_finish_reason(self) -> None:
        end = self._chat(_response([genai_types.Part(text="x")], finish_reason="SAFETY"))
        assert end.stop_reason == "content_filter"

    def test_max_tokens_text_only(self) -> None:
        end = self._chat(_response([genai_types.Part(text="x")], finish_reason="MAX_TOKENS"))
        assert end.stop_reason == "max_tokens"

    def test_max_tokens_tool_call_raises_truncated(self) -> None:
        with pytest.raises(ToolCallTruncatedError) as exc_info:
            self._chat(_response([_call("search", {"q": "par"})], finish_reason="MAX_TOKENS"))
        assert exc_info.value.tool_name == "search"
        assert exc_info.value.tool_id == "call_search"
        assert exc_info.value.partial_inputs == {"q": "par"}


class TestStreamParsing:
    """Characterize the streaming event sequence."""

    def test_text_and_tool_stream(self) -> None:
        provider = _make_provider()
        chunks = [
            _response([genai_types.Part(text="hel")], finish_reason=None, usage=False),
            _response([genai_types.Part(text="lo"), _call("search", {"q": "x"})]),
        ]
        _capture_stream(provider, chunks)
        events = asyncio.run(_collect(provider, HI))

        assert isinstance(events[0], StreamStartEvent)
        assert [e.delta for e in events if isinstance(e, TextDeltaEvent)] == ["hel", "lo"]
        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert [e.id for e in tool_events] == ["call_search"]
        end = events[-1]
        assert isinstance(end, StreamEndEvent)
        assert end.stop_reason == "tool_use"
        assert end.parts[0] == TextPart(text="hello")
        assert isinstance(end.parts[1], ToolUsePart)
        assert end.parts[1].id == "call_search"
        assert end.usage is not None and end.usage.total_tokens == 8

    def test_truncated_stream_tool_call_emits_error_event(self) -> None:
        provider = _make_provider()
        _capture_stream(provider, [_response([_call("search")], finish_reason="MAX_TOKENS")])
        events = asyncio.run(_collect(provider, HI))
        errors = [e for e in events if isinstance(e, ToolUseErrorEvent)]
        assert [(e.id, e.name) for e in errors] == [("call_search", "search")]
        assert not any(isinstance(e, ToolUseEvent) for e in events)
        assert events[-1].stop_reason == "max_tokens"


class TestReasoningEffort:
    """U1: reasoning_effort maps onto Gemini's ThinkingConfig."""

    def _config(self, **chat_kwargs) -> genai_types.GenerateContentConfig:
        provider = _make_provider()
        captured = _capture_non_stream(provider, _response([]))
        asyncio.run(provider.chat(HI, **chat_kwargs))
        return captured["config"]

    def test_none_sets_no_thinking_config(self) -> None:
        assert self._config(reasoning_effort=None).thinking_config is None

    def test_int_sets_thinking_budget(self) -> None:
        thinking = self._config(reasoning_effort=1024).thinking_config
        assert thinking.thinking_budget == 1024
        assert thinking.thinking_level is None
        assert thinking.include_thoughts is None

    def test_str_sets_thinking_level(self) -> None:
        thinking = self._config(reasoning_effort="high").thinking_config
        assert thinking.thinking_level == genai_types.ThinkingLevel.HIGH
        assert thinking.thinking_budget is None

    def test_explicit_thinking_config_still_forwarded(self) -> None:
        thinking = genai_types.ThinkingConfig(thinking_budget=7)
        assert self._config(thinking_config=thinking).thinking_config == thinking

    def test_max_tokens_with_reasoning_effort(self) -> None:
        config = self._config(reasoning_effort=128, max_tokens=512)
        assert config.max_output_tokens == 512
        assert config.thinking_config.thinking_budget == 128

    @pytest.mark.parametrize("bad", [1.5, True, False, ["high"]])
    def test_invalid_type_raises_before_network(self, bad) -> None:
        provider = _make_provider()
        provider._client.aio.models.generate_content = MagicMock(
            side_effect=AssertionError("network called")
        )
        with pytest.raises(TypeError, match="int .*str .*None"):
            asyncio.run(provider.chat(HI, reasoning_effort=bad))
        provider._client.aio.models.generate_content.assert_not_called()

    def test_invalid_type_raises_before_stream(self) -> None:
        provider = _make_provider()
        provider._client.aio.models.generate_content_stream = MagicMock()
        with pytest.raises(TypeError):
            asyncio.run(provider.chat(HI, stream=True, reasoning_effort=1.5))
        provider._client.aio.models.generate_content_stream.assert_not_called()

    def test_reasoning_effort_with_thinking_config_is_ambiguous(self) -> None:
        provider = _make_provider()
        provider._client.aio.models.generate_content = MagicMock()
        with pytest.raises(ValueError, match="reasoning_effort or thinking_config"):
            asyncio.run(
                provider.chat(
                    HI,
                    reasoning_effort=64,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=7),
                )
            )
        provider._client.aio.models.generate_content.assert_not_called()

    def test_agent_executor_run_completes(self) -> None:
        """Reproduces the reasoning_effort ValidationError every executor run hit."""
        provider = _make_provider()
        captured = _capture_stream(provider, [_response([genai_types.Part(text="done")])])
        executor = AgentExecutor("gemini", provider)

        async def run() -> list:
            return [event async for event in executor.run_stream(HI, max_tokens=100)]

        events = asyncio.run(run())
        assert isinstance(events[-1], StreamEndEvent)
        assert events[-1].parts == [TextPart(text="done")]
        assert captured["config"].thinking_config is None
        assert captured["config"].max_output_tokens == 100


def _history_with_search(result_text: str, *, is_error: bool) -> list:
    """History where `call_search` was called and answered with one result."""
    return [
        UserMessagePart(parts=[TextPart(text="find x")]),
        AssistantMessagePart(parts=[ToolUsePart(id="call_search", name="search", inputs={})]),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id="call_search",
                    name="search",
                    parts=[TextPart(text=result_text)],
                    is_error=is_error,
                )
            ]
        ),
    ]


APPROVAL_PLACEHOLDER = str({"approval_required": True})
CANCEL_PLACEHOLDER = str({"cancelled": True})


class TestToolCallIds:
    """U2: Gemini tool calls without an ID get unique IDs."""

    def _ids(self, parts, messages=HI, *, stream: bool = False) -> list[str]:
        provider = _make_provider()
        if stream:
            _capture_stream(provider, [_response(parts)])
            events = asyncio.run(_collect(provider, messages))
            end = events[-1]
            event_ids = [e.id for e in events if isinstance(e, ToolUseEvent)]
            part_ids = [p.id for p in end.parts if isinstance(p, ToolUsePart)]
            assert event_ids == part_ids
            return part_ids
        _capture_non_stream(provider, _response(parts))
        end = asyncio.run(provider.chat(messages))
        return [p.id for p in end.parts if isinstance(p, ToolUsePart)]

    @pytest.mark.parametrize("stream", [False, True])
    def test_single_call_unchanged(self, stream) -> None:
        assert self._ids([_call("search")], stream=stream) == ["call_search"]

    @pytest.mark.parametrize("stream", [False, True])
    def test_parallel_same_tool(self, stream) -> None:
        parts = [_call("search"), _call("search"), _call("search")]
        assert self._ids(parts, stream=stream) == [
            "call_search",
            "call_search_2",
            "call_search_3",
        ]

    @pytest.mark.parametrize("stream", [False, True])
    def test_different_tools(self, stream) -> None:
        parts = [_call("search"), _call("fetch")]
        assert self._ids(parts, stream=stream) == ["call_search", "call_fetch"]

    @pytest.mark.parametrize("stream", [False, True])
    def test_provided_id_kept(self, stream) -> None:
        parts = [_call("search", call_id="call_search"), _call("search", call_id="abc")]
        assert self._ids(parts, stream=stream) == ["call_search", "abc"]

    def test_provided_id_kept_even_when_taken_in_history(self) -> None:
        history = _history_with_search("ok", is_error=False)
        parts = [_call("search", call_id="call_search")]
        assert self._ids(parts, history) == ["call_search"]

    def test_provided_id_reserves_it_for_missing_ids(self) -> None:
        parts = [_call("search", call_id="call_search"), _call("search")]
        assert self._ids(parts) == ["call_search", "call_search_2"]

    @pytest.mark.parametrize("stream", [False, True])
    def test_history_real_result_takes_id(self, stream) -> None:
        history = _history_with_search("ok", is_error=False)
        assert self._ids([_call("search")], history, stream=stream) == ["call_search_2"]

    def test_history_real_error_result_takes_id(self) -> None:
        history = _history_with_search("boom", is_error=True)
        assert self._ids([_call("search")], history) == ["call_search_2"]

    def test_history_suffixed_ids_skipped(self) -> None:
        history = _history_with_search("ok", is_error=False)
        history += [
            AssistantMessagePart(
                parts=[ToolUsePart(id="call_search_2", name="search", inputs={})]
            ),
            UserMessagePart(
                parts=[ToolResultPart(tool_use_id="call_search_2", name="search", parts=[])]
            ),
        ]
        assert self._ids([_call("search")], history) == ["call_search_3"]

    @pytest.mark.parametrize("placeholder", [APPROVAL_PLACEHOLDER, CANCEL_PLACEHOLDER])
    @pytest.mark.parametrize("stream", [False, True])
    def test_placeholder_only_result_keeps_id(self, placeholder, stream) -> None:
        history = _history_with_search(placeholder, is_error=True)
        assert self._ids([_call("search")], history, stream=stream) == ["call_search"]

    def test_placeholder_text_without_error_flag_is_real(self) -> None:
        history = _history_with_search(APPROVAL_PLACEHOLDER, is_error=False)
        assert self._ids([_call("search")], history) == ["call_search_2"]

    def test_call_without_result_keeps_id(self) -> None:
        history = _history_with_search("ok", is_error=False)[:2]
        assert self._ids([_call("search")], history) == ["call_search"]

    def test_result_before_call_does_not_take_id(self) -> None:
        history = _history_with_search("ok", is_error=False)
        history = [history[0], history[2], history[1]]
        assert self._ids([_call("search")], history) == ["call_search"]

    def test_history_iterable_consumed_once(self) -> None:
        history = _history_with_search("ok", is_error=False)
        provider = _make_provider()
        captured = _capture_non_stream(provider, _response([_call("search")]))
        end = asyncio.run(provider.chat(iter(history)))
        assert len(captured["contents"]) == 3
        tool_part = end.parts[0]
        assert isinstance(tool_part, ToolUsePart)
        assert tool_part.id == "call_search_2"

    def test_truncated_stream_call_matches_helper(self) -> None:
        history = _history_with_search("ok", is_error=False)
        parts = [_call("search"), _call("search")]
        provider = _make_provider()
        _capture_stream(provider, [_response(parts, finish_reason="MAX_TOKENS")])
        events = asyncio.run(_collect(provider, history))
        error_ids = [e.id for e in events if isinstance(e, ToolUseErrorEvent)]
        assert error_ids == self._ids(parts, history)

    def test_truncated_non_stream_call_uses_minted_id(self) -> None:
        history = _history_with_search("ok", is_error=False)
        provider = _make_provider()
        _capture_non_stream(provider, _response([_call("search")], finish_reason="MAX_TOKENS"))
        with pytest.raises(ToolCallTruncatedError) as exc_info:
            asyncio.run(provider.chat(history))
        assert exc_info.value.tool_id == "call_search_2"

    def test_stream_matches_non_stream_mixed(self) -> None:
        history = _history_with_search("ok", is_error=False)
        parts = [
            _call("search"),
            _call("fetch"),
            _call("search", call_id="g1"),
            _call("search"),
        ]
        expected = ["call_search_2", "call_fetch", "g1", "call_search_3"]
        assert self._ids(parts, history) == expected
        assert self._ids(parts, history, stream=True) == expected


@dataclass
class SearchTool(Tool):
    """Approval-gated tool used by the executor integration tests."""

    name = "search"
    description = "Search for something."
    requires_approval = True
    calls: int = 0

    async def __call__(self, q: str) -> str:
        self.calls += 1
        return f"results for {q}"


def _scripted_stream(provider, turns: list[list[genai_types.Part]]) -> None:
    """Make the provider stream one scripted response per chat() call."""
    responses = iter(turns)

    async def fake_stream(*, model, contents, config):
        response = _response(next(responses))

        async def gen() -> AsyncIterator[genai_types.GenerateContentResponse]:
            yield response

        return gen()

    provider._client.aio.models.generate_content_stream = fake_stream


class TestExecutorApprovalIds:
    """U2 integration: approval IDs survive resume and do not leak to new calls."""

    def test_resumed_call_keeps_id_and_approval(self) -> None:
        provider = _make_provider()
        _scripted_stream(
            provider,
            [[_call("search", {"q": "x"})], [genai_types.Part(text="done")]],
        )
        tool = SearchTool()
        executor = AgentExecutor("gemini", provider, tools=[tool])
        history = _history_with_search(APPROVAL_PLACEHOLDER, is_error=True)

        async def run() -> list:
            return [
                event
                async for event in executor.run_stream(
                    history, approved_tool_calls={"call_search"}
                )
            ]

        events = asyncio.run(run())
        assert tool.calls == 1
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert [(r.tool_use_id, r.is_error) for r in results] == [("call_search", False)]

    def test_later_call_to_same_tool_needs_new_approval(self) -> None:
        provider = _make_provider()
        _scripted_stream(
            provider,
            [
                [_call("search", {"q": "x"})],
                [_call("search", {"q": "y"})],
                [genai_types.Part(text="unreachable")],
            ],
        )
        tool = SearchTool()
        executor = AgentExecutor("gemini", provider, tools=[tool])

        async def run() -> None:
            async for _ in executor.run_stream(HI, approved_tool_calls={"call_search"}):
                pass

        with pytest.raises(ApprovalRequired) as exc_info:
            asyncio.run(run())
        assert tool.calls == 1
        assert exc_info.value.tool_call_id == "call_search_2"
