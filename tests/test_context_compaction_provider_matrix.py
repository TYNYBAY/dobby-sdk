# ruff: noqa: E402
"""Phase 15: context compaction across OpenAI, Gemini, Anthropic, and Vertex AI.

Uses real provider adapters with mocked SDK clients (no live API calls).
Recovered compaction ``AgentExecutor`` from the in-tree ``94b5a8f`` snapshot.
"""

# isort: off
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from dobby.providers.anthropic.adapter import AnthropicProvider
from dobby.providers.base import ProviderError
from dobby.providers.gemini.adapter import GeminiProvider
from dobby.providers.openai.adapter import OpenAIProvider
from dobby.providers.vertexai.adapter import VertexAIProvider
from dobby.providers.vertexai.converters import to_vertexai_tool
from dobby.types import TextPart, UserMessagePart

from tests.compaction_subject import load_recovered

load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context._tokens import estimate_input_tokens
from recovered_dobby.tools import Tool
from recovered_dobby import types as rec_types
from recovered_dobby.types import (
    AssistantMessagePart,
    StreamEndEvent,
    TextPart as RecoveredTextPart,
    ToolResultEvent,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart as RecoveredUserMessagePart,
)

# isort: on

_RECOVERED_STREAM_EVENT_CLASSES: tuple[type[Any], ...] = (
    rec_types.StreamStartEvent,
    rec_types.TextDeltaEvent,
    rec_types.ReasoningDeltaEvent,
    rec_types.ReasoningStartEvent,
    rec_types.ReasoningEndEvent,
    rec_types.ToolUseEvent,
    rec_types.StreamErrorEvent,
    rec_types.StreamEndEvent,
)
_RECOVERED_STREAM_EVENTS: dict[str, type[Any]] = {
    cls.model_fields["type"].default: cls for cls in _RECOVERED_STREAM_EVENT_CLASSES
}


def _to_recovered_stream_event(event: Any) -> Any:
    """Recovered executor isinstance-checks its own StreamEvent classes."""
    if not hasattr(event, "model_dump"):
        return event
    payload = event.model_dump()
    event_type = payload.get("type")
    recovered_cls = _RECOVERED_STREAM_EVENTS.get(event_type)
    if recovered_cls is None:
        return event
    return recovered_cls.model_validate(payload)


def _wrap_chat_with_recovered_events(provider: Any) -> None:
    """Outermost wrapper: production adapters emit ``dobby.types`` stream events."""
    inner = provider.chat

    async def bridged_chat(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("reasoning_effort") is None:
            kwargs.pop("reasoning_effort", None)
        result = await inner(*args, **kwargs)
        if kwargs.get("stream") is False:
            return _to_recovered_stream_event(result)

        async def bridged_stream() -> AsyncIterator[Any]:
            async for event in result:
                yield _to_recovered_stream_event(event)

        return bridged_stream()

    provider.chat = bridged_chat  # type: ignore[method-assign]

_PROVIDER_IDS = ("openai", "gemini", "anthropic", "vertexai")
_PLACEHOLDER = "[Tool result cleared to save context.]"
_FACT_RE = re.compile(r"\[DOBBY-FACT:[^\]]+\]")
_WINDOW = 128_000
_TRIGGER = int(0.8 * _WINDOW)


async def _aiter(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


class _NoopTool(Tool):
    name = "noop"
    description = "Ack."

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


class _EchoTool(Tool):
    name = "echo"
    description = "Echo label."

    def __call__(self, label: str = "") -> dict[str, str]:
        return {"label": label}


class _FailTool(Tool):
    name = "fail_tool"
    description = "Always fails."

    def __call__(self) -> dict[str, str]:
        raise RuntimeError("boom")


@dataclass(frozen=True)
class _Turn:
    """One agent model turn delivered through a provider adapter."""

    tool_calls: list[tuple[str, str, dict[str, Any]]]
    usage_input: int | None
    usage_output: int = 0
    anthropic_cache_read: int | None = None
    gemini_zero_usage: bool = False
    vertex_usage_incomplete: bool = False


@dataclass(frozen=True)
class ProviderCase:
    id: str
    executor_key: Literal["openai", "gemini", "anthropic"]
    build: Callable[[], Any]
    wire_tools: Callable[[AgentExecutor], None] | None = None


def _trim_policy(*, keep_last_n: int = 2, mode: str = "trim") -> ContextPolicy:
    return ContextPolicy(
        context_window=_WINDOW,
        trigger_pct=0.8,
        keep_last_n=keep_last_n,
        mode=mode,
    )


def _usage(input_tokens: int | None, **extra: int | None) -> Usage | None:
    if input_tokens is None:
        return None
    return Usage(
        input_tokens=input_tokens,
        output_tokens=extra.get("output_tokens") or 0,
        total_tokens=input_tokens + (extra.get("output_tokens") or 0),
        cache_creation_input_tokens=extra.get("cache_creation_input_tokens"),
        cache_read_input_tokens=extra.get("cache_read_input_tokens"),
    )


def _make_openai() -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.api_key = "test"
    provider.base_url = None
    provider._model = "gpt-4o"
    provider.azure_deployment_id = None
    provider.max_retries = 0
    provider._client = MagicMock()
    return provider


def _make_gemini() -> GeminiProvider:
    provider = GeminiProvider.__new__(GeminiProvider)
    provider._model = "gemini-2.5-flash"
    provider.max_retries = 0
    provider._client = MagicMock()
    return provider


def _make_anthropic() -> AnthropicProvider:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.api_key = "test"
    provider.base_url = None
    provider._model = "claude-sonnet-4-20250514"
    provider.max_retries = 0
    provider._client = MagicMock()
    return provider


def _make_vertex() -> VertexAIProvider:
    creds = MagicMock()
    creds.valid = True
    creds.token = "test-token"
    provider = VertexAIProvider(
        model="meta/llama-3.1-405b-instruct-maas",
        project="my-project",
        credentials=creds,
    )
    provider.max_retries = 0
    provider._client = MagicMock()
    return provider


def _wire_vertex_tools(executor: AgentExecutor) -> None:
    def _schemas() -> list[Any]:
        return [to_vertexai_tool(tool) for tool in executor.tools.values()]

    executor.get_tools_schema = _schemas  # type: ignore[method-assign]


PROVIDER_CASES: tuple[ProviderCase, ...] = (
    ProviderCase("openai", "openai", _make_openai),
    ProviderCase("gemini", "gemini", _make_gemini),
    ProviderCase("anthropic", "anthropic", _make_anthropic),
    ProviderCase("vertexai", "openai", _make_vertex, _wire_vertex_tools),
)


def _openai_stream_events(turn: _Turn) -> list[Any]:
    events: list[Any] = [
        SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_1", model="gpt-4o"),
        )
    ]
    for call_id, name, inputs in turn.tool_calls:
        events.append(
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call",
                    call_id=call_id,
                    name=name,
                    arguments=json.dumps(inputs),
                ),
            )
        )
    usage_ns = None
    if turn.usage_input is not None:
        usage_ns = SimpleNamespace(
            input_tokens=turn.usage_input,
            output_tokens=turn.usage_output,
            total_tokens=turn.usage_input + turn.usage_output,
        )
    events.append(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(usage=usage_ns),
        )
    )
    return events


def _gemini_chunks(turn: _Turn) -> list[Any]:
    chunks: list[Any] = []
    for call_id, name, inputs in turn.tool_calls:
        part = SimpleNamespace(
            text=None,
            function_call=SimpleNamespace(id=call_id, name=name, args=inputs),
            thought_signature=None,
        )
        chunks.append(
            SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason=None)],
                prompt_feedback=None,
                usage_metadata=None,
            )
        )
    usage_meta = None
    if turn.gemini_zero_usage:
        usage_meta = None
    elif turn.usage_input is not None:
        usage_meta = SimpleNamespace(
            prompt_token_count=turn.usage_input,
            candidates_token_count=turn.usage_output,
            total_token_count=turn.usage_input + turn.usage_output,
        )
    chunks.append(
        SimpleNamespace(
            candidates=[
                SimpleNamespace(content=None, finish_reason="STOP" if not turn.tool_calls else None)
            ],
            prompt_feedback=None,
            usage_metadata=usage_meta,
        )
    )
    return chunks


def _anthropic_events(turn: _Turn) -> list[Any]:
    events: list[Any] = []
    start_usage = None
    if turn.usage_input is not None:
        start_usage = SimpleNamespace(
            input_tokens=turn.usage_input,
            output_tokens=turn.usage_output,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=turn.anthropic_cache_read,
        )
    events.append(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(id="msg_1", model="claude", usage=start_usage),
        )
    )
    for call_id, name, inputs in turn.tool_calls:
        events.append(
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use", id=call_id, name=name),
            )
        )
        events.append(
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="input_json_delta", partial_json=json.dumps(inputs)),
            )
        )
        events.append(SimpleNamespace(type="content_block_stop"))
    delta_usage = (
        SimpleNamespace(output_tokens=turn.usage_output) if turn.usage_input is not None else None
    )
    events.append(
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use" if turn.tool_calls else "end_turn"),
            usage=delta_usage,
        )
    )
    events.append(SimpleNamespace(type="message_stop"))
    return events


def _vertex_chunks(turn: _Turn) -> list[Any]:
    if turn.tool_calls:

        def _delta(
            index: int, call_id: str, name: str, arguments: str
        ) -> SimpleNamespace:
            return SimpleNamespace(
                index=index,
                id=call_id,
                function=SimpleNamespace(name=name, arguments=arguments),
            )

        chunks: list[Any] = []
        for idx, (call_id, name, inputs) in enumerate(turn.tool_calls):
            arg_str = json.dumps(inputs)
            mid = max(1, len(arg_str) // 2)
            chunks.append(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[_delta(idx, call_id, name, arg_str[:mid])],
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                    id="chatcmpl-test",
                    model="meta/llama",
                )
            )
            chunks.append(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[_delta(idx, None, None, arg_str[mid:])],
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                    id="chatcmpl-test",
                    model="meta/llama",
                )
            )
        chunks.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
                id="chatcmpl-test",
                model="meta/llama",
            )
        )
    else:
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="done", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
                id="chatcmpl-test",
                model="meta/llama",
            )
        ]
    if turn.usage_input is not None and not turn.vertex_usage_incomplete:
        chunks.append(
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=turn.usage_input,
                    completion_tokens=turn.usage_output,
                    total_tokens=turn.usage_input + turn.usage_output,
                ),
                id="chatcmpl-test",
                model="meta/llama",
            )
        )
    elif turn.vertex_usage_incomplete:
        chunks.append(
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                ),
                id="chatcmpl-test",
                model="meta/llama",
            )
        )
    return chunks


def _configure_client_mock(provider: Any, case_id: str, turn: _Turn) -> None:
    if case_id == "openai":
        provider._client.responses.create = AsyncMock(
            return_value=_aiter(_openai_stream_events(turn))
        )
    elif case_id == "gemini":
        provider._client.aio.models.generate_content_stream = AsyncMock(
            return_value=_aiter(_gemini_chunks(turn))
        )
    elif case_id == "anthropic":
        provider._client.messages.create = AsyncMock(return_value=_aiter(_anthropic_events(turn)))
    elif case_id == "vertexai":
        provider._client.chat.completions.create = AsyncMock(
            return_value=_aiter(_vertex_chunks(turn))
        )
    else:
        raise AssertionError(f"unknown provider case {case_id}")


def _install_turn_sequence(provider: Any, case_id: str, turns: list[_Turn]) -> None:
    """Configure mocked SDK for a sequence of agent ``chat(stream=True)`` calls."""
    call_index = 0
    real_chat = type(provider).chat

    async def chat_sequence(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_index
        if kwargs.get("reasoning_effort") is None:
            kwargs.pop("reasoning_effort", None)
        if kwargs.get("stream") is False:
            return await real_chat(provider, *args, **kwargs)
        turn = turns[min(call_index, len(turns) - 1)]
        call_index += 1
        _configure_client_mock(provider, case_id, turn)
        return await real_chat(provider, *args, **kwargs)

    provider.chat = chat_sequence  # type: ignore[method-assign]
    _wrap_chat_with_recovered_events(provider)


def _make_executor(
    case: ProviderCase,
    policy: ContextPolicy,
    tools: list[Tool] | None = None,
) -> AgentExecutor:
    llm = case.build()
    executor = AgentExecutor(
        case.executor_key,
        llm,
        tools=tools or [_NoopTool()],
        context_policy=policy,
    )
    if case.wire_tools is not None:
        case.wire_tools(executor)
    return executor


async def _run_executor(
    executor: AgentExecutor,
    *,
    turns: list[_Turn],
    case_id: str,
    seed: list[Any] | None = None,
    max_iterations: int = 8,
) -> tuple[list[Any], list[Any]]:
    _install_turn_sequence(executor.llm, case_id, turns)
    messages: list[Any] = seed or [
        RecoveredUserMessagePart(parts=[RecoveredTextPart(text="start")])
    ]
    events: list[Any] = []
    async for event in executor.run_stream(messages, max_iterations=max_iterations):
        events.append(event)
    return events, messages


def _context_blob_from_events(events: list[Any]) -> str:
    return "\n".join(str(e.result) for e in events if isinstance(e, ToolResultEvent))


@pytest.fixture(params=PROVIDER_CASES, ids=[c.id for c in PROVIDER_CASES])
def provider_case(request: pytest.FixtureRequest) -> ProviderCase:
    return request.param


# --- matrix dimensions ---


@pytest.mark.parametrize("provider_case", PROVIDER_CASES, ids=_PROVIDER_IDS)
def test_token_usage_triggers_compaction(provider_case: ProviderCase) -> None:
    """Usage at the configured threshold triggers trim before the next model call."""
    policy = _trim_policy(keep_last_n=1)
    turns = [
        _Turn([("w1", "noop", {})], usage_input=_TRIGGER - 1),
        _Turn([("c1", "noop", {})], usage_input=_TRIGGER),
        _Turn([], usage_input=_TRIGGER),
    ]
    events, _ = asyncio.run(_run_executor(_make_executor(provider_case, policy), turns=turns, case_id=provider_case.id))
    edits = [e for e in events if isinstance(e, ContextEditEvent)]
    assert len(edits) >= 1


@pytest.mark.parametrize("provider_case", PROVIDER_CASES, ids=_PROVIDER_IDS)
def test_streaming_tool_call_round_trip(provider_case: ProviderCase) -> None:
    """Streaming adapter delivers tool calls; executor records one tool round-trip."""
    policy = _trim_policy(keep_last_n=2)
    turns = [
        _Turn([("tc1", "echo", {"label": "alpha"})], usage_input=_TRIGGER - 5000),
        _Turn([], usage_input=_TRIGGER - 5000),
    ]
    executor = _make_executor(provider_case, policy, tools=[_EchoTool()])
    events, _messages = asyncio.run(
        _run_executor(executor, turns=turns, case_id=provider_case.id, max_iterations=3)
    )
    assert not any(isinstance(e, ContextEditEvent) for e in events)
    assert "alpha" in _context_blob_from_events(events)


@pytest.mark.parametrize("provider_case", PROVIDER_CASES, ids=_PROVIDER_IDS)
def test_parallel_tool_calls(provider_case: ProviderCase) -> None:
    """Two tool calls in one streamed response become two paired round-trips."""
    policy = _trim_policy(keep_last_n=2)
    turns = [
        _Turn(
            [
                ("p0", "echo", {"label": "one"}),
                ("p1", "echo", {"label": "two"}),
            ],
            usage_input=_TRIGGER - 5000,
        ),
        _Turn([], usage_input=_TRIGGER - 5000),
    ]
    executor = _make_executor(provider_case, policy, tools=[_EchoTool()])
    events, _ = asyncio.run(
        _run_executor(executor, turns=turns, case_id=provider_case.id, max_iterations=3)
    )
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) >= 2


@pytest.mark.parametrize("provider_case", PROVIDER_CASES, ids=_PROVIDER_IDS)
def test_tool_error_handling_with_compaction(provider_case: ProviderCase) -> None:
    """Execution error rows stay paired and trim clears error text outside keep window."""
    policy = _trim_policy(keep_last_n=1)
    turns = [
        _Turn([("ok1", "noop", {})], usage_input=_TRIGGER - 1),
        _Turn([("bad", "fail_tool", {})], usage_input=_TRIGGER),
        _Turn([], usage_input=_TRIGGER),
    ]
    executor = _make_executor(provider_case, policy, tools=[_NoopTool(), _FailTool()])
    events, _ = asyncio.run(
        _run_executor(executor, turns=turns, case_id=provider_case.id, max_iterations=4)
    )
    assert any(isinstance(e, ContextEditEvent) for e in events)


@pytest.mark.parametrize("provider_case", PROVIDER_CASES, ids=_PROVIDER_IDS)
def test_context_limit_trigger_respects_policy_window(provider_case: ProviderCase) -> None:
    """Compaction fires at ``trigger_pct * context_window`` from reported usage."""
    policy = _trim_policy(keep_last_n=0)
    turns = [
        _Turn([("w1", "noop", {})], usage_input=_TRIGGER - 1),
        _Turn([("x1", "noop", {})], usage_input=_TRIGGER),
        _Turn([], usage_input=_TRIGGER),
    ]
    events, _ = asyncio.run(
        _run_executor(_make_executor(provider_case, policy), turns=turns, case_id=provider_case.id)
    )
    assert sum(isinstance(e, ContextEditEvent) for e in events) >= 1


@pytest.mark.parametrize("provider_case", PROVIDER_CASES, ids=_PROVIDER_IDS)
def test_summarization_preserves_fact_in_digest(provider_case: ProviderCase) -> None:
    """Summarize mode calls ``stream=False`` on the provider and writes a digest."""
    policy = ContextPolicy(
        context_window=_WINDOW,
        trigger_pct=0.8,
        keep_last_n=1,
        mode="summarize",
    )

    class _FactTool(Tool):
        name = "fact_tool"
        description = "Return a labeled fact."

        def __call__(self) -> dict[str, str]:
            return {"payload": "[DOBBY-FACT:id:seed=matrix]"}

    summarize_calls: list[bool] = []

    case = provider_case
    agent_turns = [
        _Turn([("f1", "fact_tool", {})], usage_input=_TRIGGER - 1),
        _Turn([("n1", "noop", {})], usage_input=_TRIGGER),
        _Turn([], usage_input=_TRIGGER),
    ]
    llm_agent = case.build()
    _install_turn_sequence(llm_agent, case.id, agent_turns)

    executor = AgentExecutor(
        case.executor_key,
        llm_agent,
        tools=[_FactTool(), _NoopTool()],
        context_policy=policy,
    )
    if case.wire_tools is not None:
        case.wire_tools(executor)

    agent_chat = llm_agent.chat

    async def chat_hybrid(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream") is False:
            summarize_calls.append(True)
            return StreamEndEvent(
                model="summarizer",
                parts=[RecoveredTextPart(text="<summary>[DOBBY-FACT:id:seed=matrix]</summary>")],
                stop_reason="end_turn",
                usage=_usage(0),
            )
        return await agent_chat(*args, **kwargs)

    llm_agent.chat = chat_hybrid  # type: ignore[method-assign]
    _wrap_chat_with_recovered_events(llm_agent)

    events: list[Any] = []
    messages = [RecoveredUserMessagePart(parts=[RecoveredTextPart(text="q")])]

    async def _drive() -> None:
        async for event in executor.run_stream(messages, max_iterations=5):
            events.append(event)

    asyncio.run(_drive())
    assert summarize_calls, "expected at least one summarize stream=False call"
    assert any(isinstance(e, ContextEditEvent) for e in events)
    assert _FACT_RE.search("\n".join(str(e) for e in events) + str(messages))


@pytest.mark.parametrize("provider_case", PROVIDER_CASES, ids=_PROVIDER_IDS)
def test_missing_usage_metadata_fallback(provider_case: ProviderCase) -> None:
    """Omitted usage still compacts when the outgoing transcript estimate crosses the trigger."""
    policy = _trim_policy(keep_last_n=1)
    huge = "H" * (_TRIGGER * 4 + 400)
    seed = [
        RecoveredUserMessagePart(parts=[RecoveredTextPart(text="seed")]),
        *[
            part
            for i in range(2)
            for part in (
                AssistantMessagePart(parts=[ToolUsePart(id=f"h{i}", name="noop", inputs={})]),
                RecoveredUserMessagePart(
                    parts=[
                        ToolResultPart(
                            tool_use_id=f"h{i}",
                            name="noop",
                            parts=[RecoveredTextPart(text=huge)],
                        )
                    ]
                ),
            )
        ],
    ]
    missing_flags: dict[str, Any] = {}
    if provider_case.id == "vertexai":
        missing_flags["vertex_usage_incomplete"] = True
    if provider_case.id == "gemini":
        missing_flags["gemini_zero_usage"] = True
    turns = [
        _Turn([("w1", "noop", {})], usage_input=None, **missing_flags),
        _Turn([], usage_input=None, **missing_flags),
    ]

    events, _ = asyncio.run(
        _run_executor(
            _make_executor(provider_case, policy),
            turns=turns,
            case_id=provider_case.id,
            seed=seed,
            max_iterations=3,
        )
    )
    assert estimate_input_tokens(seed) >= _TRIGGER
    # Anthropic materializes Usage(0) when the stream omits counts, so the executor
    # records reported input rather than estimating at StreamEnd. Compaction still
    # fires: max(reported, estimate(outgoing_messages)) crosses the trigger here.
    assert any(isinstance(e, ContextEditEvent) for e in events)


@pytest.mark.parametrize("provider_case", PROVIDER_CASES, ids=_PROVIDER_IDS)
def test_summarize_provider_error_does_not_mutate_history(provider_case: ProviderCase) -> None:
    """Summarizer ``ProviderError`` aborts without ``ContextEditEvent``."""
    policy = ContextPolicy(
        context_window=_WINDOW,
        trigger_pct=0.8,
        keep_last_n=1,
        mode="summarize",
    )
    llm = provider_case.build()
    turns = [
        _Turn([("a1", "noop", {})], usage_input=_TRIGGER - 1),
        _Turn([("a2", "noop", {})], usage_input=_TRIGGER),
        _Turn([], usage_input=_TRIGGER),
    ]
    _install_turn_sequence(llm, provider_case.id, turns)

    agent_chat = llm.chat

    async def chat_hybrid(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream") is False:
            raise ProviderError("summarizer failed", provider=provider_case.id)
        return await agent_chat(*args, **kwargs)

    llm.chat = chat_hybrid  # type: ignore[method-assign]
    _wrap_chat_with_recovered_events(llm)
    executor = AgentExecutor(
        provider_case.executor_key,
        llm,
        tools=[_NoopTool()],
        context_policy=policy,
    )
    if provider_case.wire_tools is not None:
        provider_case.wire_tools(executor)

    messages = [RecoveredUserMessagePart(parts=[RecoveredTextPart(text="q")])]
    before = json.dumps([str(m) for m in messages])

    collected: list[Any] = []

    async def _drive() -> None:
        async for event in executor.run_stream(messages, max_iterations=4):
            collected.append(event)

    with pytest.raises(ProviderError):
        asyncio.run(_drive())
    assert json.dumps([str(m) for m in messages]) == before
    assert not any(isinstance(e, ContextEditEvent) for e in collected)


def test_anthropic_cache_read_not_counted_in_compaction_trigger() -> None:
    """Witness: executor uses ``usage.input_tokens`` only, not cache-read totals."""
    case = next(c for c in PROVIDER_CASES if c.id == "anthropic")
    policy = _trim_policy(keep_last_n=1)
    turns = [
        _Turn([("w1", "noop", {})], usage_input=_TRIGGER - 5000, anthropic_cache_read=_TRIGGER + 50_000),
        _Turn([("c1", "noop", {})], usage_input=_TRIGGER - 5000, anthropic_cache_read=_TRIGGER + 50_000),
        _Turn([], usage_input=_TRIGGER - 5000, anthropic_cache_read=_TRIGGER + 50_000),
    ]
    events, _ = asyncio.run(
        _run_executor(_make_executor(case, policy), turns=turns, case_id=case.id, max_iterations=4)
    )
    assert not any(isinstance(e, ContextEditEvent) for e in events)


def test_recovered_executor_vertex_schema_gap_witness() -> None:
    """Recovered executor has no ``vertexai`` tool-schema branch (production does)."""
    import recovered_dobby.executor as rec_exec

    hints = getattr(rec_exec.AgentExecutor.__init__, "__annotations__", {})
    provider_hint = str(hints.get("provider", ""))
    assert "vertexai" not in provider_hint


def test_vertex_incomplete_usage_yields_none_in_stream_end() -> None:
    """Vertex adapter drops all-None usage payloads instead of coercing to zero."""
    from dobby.types import StreamEndEvent as MainStreamEndEvent

    provider = _make_vertex()
    turn = _Turn([], usage_input=100, vertex_usage_incomplete=True)
    _configure_client_mock(provider, "vertexai", turn)

    async def _collect() -> MainStreamEndEvent:
        end: MainStreamEndEvent | None = None
        stream = await provider.chat([UserMessagePart(parts=[TextPart(text="hi")])], stream=True)
        async for event in stream:
            if isinstance(event, MainStreamEndEvent):
                end = event
        assert end is not None
        return end

    end = asyncio.run(_collect())
    assert end.usage is None
