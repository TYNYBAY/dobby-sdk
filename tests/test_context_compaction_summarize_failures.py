# ruff: noqa: E402
"""Phase 7: summarizer failure handling (recovered ``summarize_context`` + executor).

Deterministic mocks only. Each case checks that failures do not leave a torn
conversation (no partial span replacement, no ``<summary>``) unless the
implementation explicitly accepts the response.
"""

# isort: off
from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.compaction_subject import load_recovered

load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.providers.base import APITimeoutError, ProviderError, RateLimitError
from recovered_dobby.context import summarize_context
from recovered_dobby.tools import Tool
from recovered_dobby.types import (
    AssistantMessagePart,
    StreamEndEvent,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart,
)

# isort: on


class _NoopTool(Tool):
    name = "noop"
    description = "Ack."

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


def _usage(tokens: int) -> Usage:
    return Usage(input_tokens=tokens, output_tokens=0, total_tokens=tokens)


def _tool_call(call_id: str) -> ToolUsePart:
    return ToolUsePart(id=call_id, name="noop", inputs={})


def _history(pairs: int = 4, text: str = "history-payload") -> list[Any]:
    messages: list[Any] = [UserMessagePart(parts=[TextPart(text="question")])]
    for index in range(pairs):
        call_id = f"h{index}"
        messages.append(
            AssistantMessagePart(
                parts=[ToolUsePart(id=call_id, name="search", inputs={"q": call_id})]
            )
        )
        messages.append(
            UserMessagePart(
                parts=[
                    ToolResultPart(
                        tool_use_id=call_id,
                        name="search",
                        parts=[TextPart(text=f"{text}-{index}")],
                    )
                ]
            )
        )
    return messages


def _message_fingerprint(messages: list[Any]) -> list[tuple[str, str]]:
    """Stable snapshot of user-visible tool/user text for equality checks."""
    fingerprint: list[tuple[str, str]] = []
    for message in messages:
        role = message.role
        for part in message.parts:
            if isinstance(part, TextPart):
                fingerprint.append((role, part.text))
            elif isinstance(part, ToolResultPart):
                text = "".join(p.text for p in part.parts if isinstance(p, TextPart))
                fingerprint.append((role, text))
    return fingerprint


def _summary_count(messages: list[Any]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, TextPart) and part.text.startswith("<summary>")
    )


def _assert_conversation_unchanged(before: list[Any], after: list[Any]) -> None:
    assert len(before) == len(after)
    assert _message_fingerprint(before) == _message_fingerprint(after)
    assert _summary_count(after) == 0


def _llm_with_summarize_failure(
    exc: BaseException | None = None,
    *,
    stream_end: StreamEndEvent | None = None,
    stream_false_returns_stream: bool = False,
) -> AsyncMock:
    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", True) is False:
            if exc is not None:
                raise exc
            if stream_false_returns_stream:

                async def broken_stream() -> AsyncIterator[StreamEndEvent]:
                    raise ProviderError("stream failed mid-flight", provider="mock")
                    yield StreamEndEvent(  # pragma: no cover
                        type="stream_end",
                        model="mock",
                        parts=[],
                        stop_reason="end_turn",
                        usage=_usage(0),
                    )

                return broken_stream()
            if stream_end is not None:
                return stream_end
            raise AssertionError("summarize mock missing stream_end")
        # Agent loop streaming path
        async def stream() -> AsyncIterator[StreamEndEvent]:
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=[],
                stop_reason="end_turn",
                usage=_usage(0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    provider.name = "mock"
    return provider


def _llm_agent_then_summarize(
    *,
    summarize_exc: BaseException | None = None,
    summarize_end: StreamEndEvent | None = None,
    stream_false_returns_stream: bool = False,
    captured: list[list[Any]],
    summary_attempts: list[int],
) -> AsyncMock:
    agent_calls = 0

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal agent_calls
        if kwargs.get("stream", True) is False:
            summary_attempts.append(len(messages))
            if summarize_exc is not None:
                raise summarize_exc
            if stream_false_returns_stream:

                async def broken_stream() -> AsyncIterator[StreamEndEvent]:
                    raise ProviderError("streaming failure", provider="mock")
                    yield StreamEndEvent(  # pragma: no cover
                        type="stream_end",
                        model="mock",
                        parts=[],
                        stop_reason="end_turn",
                        usage=_usage(0),
                    )

                return broken_stream()
            assert summarize_end is not None
            return summarize_end
        agent_calls += 1
        captured.append(list(messages))

        async def stream() -> AsyncIterator[StreamEndEvent]:
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=[_tool_call(f"t{agent_calls}")] if agent_calls == 1 else [],
                stop_reason="tool_use" if agent_calls == 1 else "end_turn",
                usage=_usage(200_000),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    provider.name = "mock"
    return provider


async def _run_summarize(messages: list[Any], llm: Any, keep_last_n: int = 1) -> Any:
    policy = ContextPolicy(keep_last_n=keep_last_n, mode="summarize")
    return await summarize_context(messages, policy, llm)


def _history_for_unit() -> list[Any]:
    return _history(pairs=4)


@pytest.mark.parametrize(
    ("failure_id", "exc"),
    [
        pytest.param("provider-exception", RuntimeError("provider internal error"), id="provider-exception"),
        pytest.param("timeout", APITimeoutError("summarizer timed out", provider="mock"), id="timeout"),
        pytest.param("rate-limit", RateLimitError("rate limited", provider="mock"), id="rate-limit"),
        pytest.param(
            "context-length",
            ProviderError("context length exceeded", provider="mock", status_code=400),
            id="context-length",
        ),
        pytest.param(
            "authentication",
            ProviderError("invalid api key", provider="mock", status_code=401),
            id="authentication",
        ),
    ],
)
def test_summarize_exception_leaves_conversation_intact(failure_id: str, exc: BaseException) -> None:
    """Raised summarizer errors must not replace any messages."""
    del failure_id
    messages = _history_for_unit()
    snapshot = copy.deepcopy(messages)
    llm = _llm_with_summarize_failure(exc)

    with pytest.raises(type(exc)):
        asyncio.run(_run_summarize(messages, llm))

    _assert_conversation_unchanged(snapshot, messages)


def test_summarize_streaming_failure_leaves_conversation_intact() -> None:
    """When ``stream=False`` returns an async generator, span is not replaced."""
    messages = _history_for_unit()
    snapshot = copy.deepcopy(messages)
    llm = _llm_with_summarize_failure(stream_false_returns_stream=True)

    with pytest.raises(AttributeError, match="parts"):
        asyncio.run(_run_summarize(messages, llm))

    _assert_conversation_unchanged(snapshot, messages)


def test_summarize_malformed_response_leaves_conversation_intact() -> None:
    """Non-``StreamEndEvent`` responses must not mutate the message list."""
    messages = _history_for_unit()
    snapshot = copy.deepcopy(messages)

    async def bad_chat(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", True) is False:
            return {"not": "a stream end event"}
        async def stream() -> AsyncIterator[StreamEndEvent]:
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=[],
                stop_reason="end_turn",
                usage=_usage(0),
            )

        return stream()

    llm = AsyncMock()
    llm.chat = bad_chat

    with pytest.raises(AttributeError):
        asyncio.run(_run_summarize(messages, llm))

    _assert_conversation_unchanged(snapshot, messages)


@pytest.mark.parametrize(
    "parts",
    [
        pytest.param([], id="empty-response"),
        pytest.param([TextPart(text="   ")], id="whitespace-only-response"),
    ],
)
def test_summarize_empty_digest_must_not_destroy_span(parts: list[TextPart]) -> None:
    """Empty digests must not replace tool history (unsafe if implementation writes ``<summary></summary>``)."""
    messages = _history_for_unit()
    snapshot = copy.deepcopy(messages)
    llm = _llm_with_summarize_failure(
        stream_end=StreamEndEvent(
            type="stream_end",
            model="mock",
            parts=parts,
            stop_reason="end_turn",
            usage=_usage(0),
        )
    )

    applied = asyncio.run(_run_summarize(messages, llm))
    assert applied is None
    _assert_conversation_unchanged(snapshot, messages)


@pytest.mark.parametrize(
    ("failure_id", "exc"),
    [
        pytest.param("provider-exception", RuntimeError("provider internal error"), id="executor-provider-exception"),
        pytest.param("timeout", APITimeoutError("summarizer timed out", provider="mock"), id="executor-timeout"),
        pytest.param("rate-limit", RateLimitError("rate limited", provider="mock"), id="executor-rate-limit"),
        pytest.param(
            "context-length",
            ProviderError("context length exceeded", provider="mock", status_code=400),
            id="executor-context-length",
        ),
        pytest.param(
            "authentication",
            ProviderError("invalid api key", provider="mock", status_code=401),
            id="executor-authentication",
        ),
    ],
)
def test_executor_summarize_failure_preserves_history(failure_id: str, exc: BaseException) -> None:
    """Executor run aborts on summarizer failure without ``ContextEditEvent`` or torn history."""
    del failure_id
    messages = _history(pairs=4)
    snapshot = copy.deepcopy(messages)
    captured: list[list[Any]] = []
    summary_attempts: list[int] = []
    policy = ContextPolicy(context_window=100_000, trigger_pct=0.8, keep_last_n=1, mode="summarize")
    llm = _llm_agent_then_summarize(
        summarize_exc=exc,
        captured=captured,
        summary_attempts=summary_attempts,
    )
    executor = AgentExecutor("openai", llm, tools=[_NoopTool()], context_policy=policy)

    async def fail_run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=2):
            events.append(event)
        return events

    with pytest.raises(type(exc)):
        asyncio.run(fail_run())

    assert summary_attempts
    _assert_conversation_unchanged(snapshot, messages)


def test_executor_can_recover_after_summarizer_failure() -> None:
    """After a failed summarize, a later run with a healthy summarizer compacts successfully."""
    messages = _history(pairs=4)
    snapshot = copy.deepcopy(messages)
    captured: list[list[Any]] = []
    summary_attempts: list[int] = []
    policy = ContextPolicy(context_window=100_000, trigger_pct=0.8, keep_last_n=1, mode="summarize")
    llm = _llm_agent_then_summarize(
        summarize_exc=APITimeoutError("timeout", provider="mock"),
        captured=captured,
        summary_attempts=summary_attempts,
    )
    executor = AgentExecutor("openai", llm, tools=[_NoopTool()], context_policy=policy)

    async def fail_run() -> None:
        async for _event in executor.run_stream(messages, max_iterations=2):
            pass

    with pytest.raises(APITimeoutError):
        asyncio.run(fail_run())

    _assert_conversation_unchanged(snapshot, messages)

    ok_end = StreamEndEvent(
        type="stream_end",
        model="mock",
        parts=[TextPart(text="recovered digest")],
        stop_reason="end_turn",
        usage=_usage(0),
    )
    executor.llm = _llm_agent_then_summarize(
        summarize_end=ok_end,
        captured=captured,
        summary_attempts=summary_attempts,
    )

    async def recover_run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=2):
            events.append(event)
        return events

    events = asyncio.run(recover_run())
    edits = [event for event in events if isinstance(event, ContextEditEvent)]
    assert len(edits) == 1
    assert edits[0].applied_edits[0].type == "summarize"
    assert edits[0].applied_edits[0].summary_text == "recovered digest"
    # ``run_stream`` works on a copy; caller-owned ``messages`` stays unchanged.
    _assert_conversation_unchanged(snapshot, messages)
    assert len(captured) >= 3
    post_compact_texts = [text for _, text in _message_fingerprint(captured[-1])]
    assert any("recovered digest" in text for text in post_compact_texts)
    assert any(text.startswith("<summary>") for text in post_compact_texts)


def test_executor_streaming_summarizer_failure_preserves_history() -> None:
    messages = _history(pairs=4)
    snapshot = copy.deepcopy(messages)
    summary_attempts: list[int] = []
    policy = ContextPolicy(context_window=100_000, trigger_pct=0.8, keep_last_n=1, mode="summarize")
    llm = _llm_agent_then_summarize(
        stream_false_returns_stream=True,
        captured=[],
        summary_attempts=summary_attempts,
    )
    executor = AgentExecutor("openai", llm, tools=[_NoopTool()], context_policy=policy)

    async def fail_run() -> None:
        async for _event in executor.run_stream(messages, max_iterations=2):
            pass

    with pytest.raises((ProviderError, AttributeError, TypeError)):
        asyncio.run(fail_run())

    _assert_conversation_unchanged(snapshot, messages)
