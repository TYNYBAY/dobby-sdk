"""Tests for defensive handling of max_tokens-truncated tool calls.

Covers both the OpenAI (Responses API) and Gemini adapters, streaming and
non-streaming. A truncated tool call must never crash the stream as a raw
``JSONDecodeError`` (streaming) or silently return partial arguments
(non-streaming):

- Streaming: yields a ``ToolUseErrorEvent``; the stream still completes with a
  ``StreamEndEvent`` and any valid tool call in the same stream is delivered.
- Non-streaming: raises a typed ``ToolCallTruncatedError`` instead of returning
  partial inputs.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from dobby.providers.openai.adapter import OpenAIProvider
from dobby.providers.gemini.adapter import GeminiProvider

from dobby.providers.base import (
    RETRYABLE_ERRORS,
    ProviderError,
    ToolCallTruncatedError,
)
from dobby.types import (
    StreamEndEvent,
    ToolUseErrorEvent,
    ToolUsePart,
)


async def _aiter(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# ToolCallTruncatedError / ToolUseErrorEvent contract
# ---------------------------------------------------------------------------


class TestTruncationContract:
    """The shared error/event types behave as the design requires."""

    def test_error_is_provider_error(self) -> None:
        err = ToolCallTruncatedError("truncated", provider="openai")
        assert isinstance(err, ProviderError)

    def test_error_carries_tool_metadata(self) -> None:
        err = ToolCallTruncatedError(
            "truncated",
            provider="openai",
            tool_name="get_weather",
            tool_id="call_1",
            partial_inputs='{"city": "San Fr',
        )
        assert err.tool_name == "get_weather"
        assert err.tool_id == "call_1"
        assert err.partial_inputs == '{"city": "San Fr'

    def test_error_not_retryable(self) -> None:
        # Truncation is deterministic; retrying with the same budget repeats it.
        assert ToolCallTruncatedError not in RETRYABLE_ERRORS

    def test_event_is_discriminated(self) -> None:
        event = ToolUseErrorEvent(
            id="call_1", name="get_weather", raw_arguments="{", error="boom"
        )
        assert event.type == "tool_use_error"


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def _make_openai_provider() -> "object":
    

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.api_key = "test"
    provider.base_url = None
    provider._model = "gpt-4"
    provider.azure_deployment_id = None
    provider.max_retries = 0  # bypass retry wrapping for deterministic tests
    provider._client = MagicMock()
    return provider


def _openai_stream_events(tool_items, terminal="completed"):
    """Build a minimal OpenAI Responses stream around the given function calls.

    tool_items: list of (call_id, name, arguments) tuples.
    terminal: "completed" (normal end) or "incomplete" (max_output_tokens cutoff).
        A real max_tokens-truncated stream ends with ``response.incomplete``, not
        ``response.completed`` — the adapter must still emit a StreamEndEvent.
    """
    events = [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1", model="gpt-4"),
        )
    ]
    for call_id, name, arguments in tool_items:
        events.append(
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call",
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                ),
            )
        )
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
    if terminal == "incomplete":
        events.append(
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(
                    usage=usage,
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                ),
            )
        )
    else:
        events.append(
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage=usage),
            )
        )
    return events

class TestOpenAIStreamingTruncation:
    """Streaming truncation yields ToolUseErrorEvent, stream still completes."""

    def test_truncated_single_tool_call(self) -> None:
        provider = _make_openai_provider()
        events = _openai_stream_events(
            [("call_1", "get_weather", '{"city": "San Fran')]  # truncated JSON
        )
        provider._client.responses.create = AsyncMock(return_value=_aiter(events))

        async def run():
            collected = []
            async for event in provider._stream_chat_completion(
                [], 0.0, "gpt-4", None, None, None
            ):
                collected.append(event)
            return collected

        collected = asyncio.run(run())

        # Exactly one ToolUseErrorEvent naming the broken tool, carrying raw payload
        errors = [e for e in collected if isinstance(e, ToolUseErrorEvent)]
        assert len(errors) == 1
        assert errors[0].name == "get_weather"
        assert errors[0].id == "call_1"
        assert errors[0].raw_arguments == '{"city": "San Fran'

        # Stream still completes; broken tool absent from final parts
        end = [e for e in collected if isinstance(e, StreamEndEvent)]
        assert len(end) == 1
        assert not any(isinstance(p, ToolUsePart) for p in end[0].parts)

    def test_incomplete_terminated_stream_still_ends(self) -> None:
        # Real max_tokens-truncated streams end with `response.incomplete`, not
        # `response.completed`. The stream must still emit a terminal
        # StreamEndEvent with stop_reason="max_tokens" (regression: R3).
        provider = _make_openai_provider()
        events = _openai_stream_events(
            [("call_1", "save_essay", '{"essay": "The history of')],  # truncated
            terminal="incomplete",
        )
        provider._client.responses.create = AsyncMock(return_value=_aiter(events))

        async def run():
            collected = []
            async for event in provider._stream_chat_completion(
                [], 0.0, "gpt-4", None, None, None
            ):
                collected.append(event)
            return collected

        collected = asyncio.run(run())

        errors = [e for e in collected if isinstance(e, ToolUseErrorEvent)]
        assert len(errors) == 1

        # Stream completes despite the incomplete terminal; cutoff is observable
        end = [e for e in collected if isinstance(e, StreamEndEvent)]
        assert len(end) == 1
        assert end[0].stop_reason == "max_tokens"
        assert not any(isinstance(p, ToolUsePart) for p in end[0].parts)

    def test_mixed_valid_and_invalid_tool_calls(self) -> None:
        provider = _make_openai_provider()
        events = _openai_stream_events(
            [
                ("call_ok", "get_weather", '{"city": "NYC"}'),  # valid
                ("call_bad", "get_time", '{"tz": "America/New_Y'),  # truncated
            ]
        )
        provider._client.responses.create = AsyncMock(return_value=_aiter(events))

        async def run():
            collected = []
            async for event in provider._stream_chat_completion(
                [], 0.0, "gpt-4", None, None, None
            ):
                collected.append(event)
            return collected

        collected = asyncio.run(run())

        errors = [e for e in collected if isinstance(e, ToolUseErrorEvent)]
        assert len(errors) == 1
        assert errors[0].name == "get_time"

        # The valid tool is delivered in the final StreamEndEvent
        end = next(e for e in collected if isinstance(e, StreamEndEvent))
        tool_parts = [p for p in end.parts if isinstance(p, ToolUsePart)]
        assert len(tool_parts) == 1
        assert tool_parts[0].name == "get_weather"
        assert tool_parts[0].inputs == {"city": "NYC"}


class TestOpenAINonStreamingTruncation:
    """Non-streaming truncation raises ToolCallTruncatedError."""

    def test_truncated_tool_call_raises(self) -> None:
        provider = _make_openai_provider()
        response = SimpleNamespace(
            model="gpt-4",
            output=[
                SimpleNamespace(
                    type="function_call",
                    id="call_1",
                    name="get_weather",
                    arguments='{"city": "San Fran',  # truncated JSON
                )
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        provider._client.responses.create = AsyncMock(return_value=response)

        async def run():
            return await provider._non_stream_chat_completion(
                [], "gpt-4", None, None, None
            )

        with pytest.raises(ToolCallTruncatedError) as exc_info:
            asyncio.run(run())

        assert exc_info.value.tool_name == "get_weather"
        assert exc_info.value.tool_id == "call_1"
        assert exc_info.value.provider == "openai"


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


def _make_gemini_provider() -> "object":

    provider = GeminiProvider.__new__(GeminiProvider)
    provider.api_key = "test"
    provider.vertexai = False
    provider.project = None
    provider.location = "us-central1"
    provider._model = "gemini-2.5-flash"
    provider.max_retries = 0
    provider._client = MagicMock()
    return provider


def _gemini_function_call_part(name="get_weather", args=None):
    return SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(id=None, name=name, args=args or {"city": "San"}),
        thought_signature=None,
    )


def _gemini_response(finish_reason, parts):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason=finish_reason,
                content=SimpleNamespace(parts=parts),
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15
        ),
    )


class TestGeminiNonStreamingTruncation:
    """MAX_TOKENS + a tool call raises ToolCallTruncatedError."""

    def test_truncated_tool_call_raises(self) -> None:
        provider = _make_gemini_provider()
        response = _gemini_response("MAX_TOKENS", [_gemini_function_call_part()])

        with pytest.raises(ToolCallTruncatedError) as exc_info:
            provider._parse_response(response)

        assert exc_info.value.tool_name == "get_weather"
        assert exc_info.value.provider == "gemini"

    def test_completed_tool_call_does_not_raise(self) -> None:
        # STOP finish reason: a normal tool call must NOT be treated as truncated.
        provider = _make_gemini_provider()
        response = _gemini_response("STOP", [_gemini_function_call_part()])

        result = provider._parse_response(response)
        assert result.stop_reason == "tool_use"
        assert any(isinstance(p, ToolUsePart) for p in result.parts)


class TestGeminiStreamingTruncation:
    """Streaming MAX_TOKENS + a tool call yields ToolUseErrorEvent."""

    def test_truncated_tool_call_yields_event(self) -> None:
        provider = _make_gemini_provider()
        chunk = SimpleNamespace(
            prompt_feedback=None,
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5, total_token_count=15
            ),
            candidates=[
                SimpleNamespace(
                    finish_reason="MAX_TOKENS",
                    content=SimpleNamespace(parts=[_gemini_function_call_part()]),
                )
            ],
        )
        provider._client.aio.models.generate_content_stream = AsyncMock(
            return_value=_aiter([chunk])
        )

        async def run():
            collected = []
            async for event in provider._stream_chat_completion(None, None):
                collected.append(event)
            return collected

        collected = asyncio.run(run())

        errors = [e for e in collected if isinstance(e, ToolUseErrorEvent)]
        assert len(errors) == 1
        assert errors[0].name == "get_weather"

        # Stream completes; the truncated tool is absent and stop_reason reflects it
        end = next(e for e in collected if isinstance(e, StreamEndEvent))
        assert not any(isinstance(p, ToolUsePart) for p in end.parts)
        assert end.stop_reason == "max_tokens"
