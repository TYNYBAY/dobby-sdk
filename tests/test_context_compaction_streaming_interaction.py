# ruff: noqa: E402
"""Phase 13: compaction interaction with streaming tool execution."""

# isort: off
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar
from unittest.mock import AsyncMock, patch

from dobby import AgentExecutor as CurrentAgentExecutor
from dobby.exceptions import ModelRetryExhaustedError
from dobby.tools import Tool as CurrentTool
from dobby.types import StreamEndEvent as CurrentStreamEndEvent
from dobby.types import TextPart as CurrentTextPart
from dobby.types import ToolResultEvent
from dobby.types import ToolResultPart as CurrentToolResultPart
from dobby.types import ToolStreamEvent
from dobby.types import ToolUsePart as CurrentToolUsePart
from dobby.types import Usage as CurrentUsage
from dobby.types import UserMessagePart as CurrentUserMessagePart

from tests.compaction_subject import load_recovered

load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context import edit_context
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

_PLACEHOLDER = "[Tool result cleared to save context.]"


def _usage(input_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _trim_policy(*, keep_last_n: int = 2) -> ContextPolicy:
    return ContextPolicy(
        context_window=128_000,
        trigger_pct=0.8,
        keep_last_n=keep_last_n,
        mode="trim",
    )


def _result_parts(messages: list[Any]) -> list[ToolResultPart]:
    return [
        part
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]


def _use_parts(messages: list[Any]) -> list[ToolUsePart]:
    return [
        part
        for message in messages
        if isinstance(message, AssistantMessagePart)
        for part in message.parts
        if isinstance(part, ToolUsePart)
    ]


def _result_texts(messages: list[Any]) -> list[str]:
    texts: list[str] = []
    for part in _result_parts(messages):
        texts.append("".join(piece.text for piece in part.parts if isinstance(piece, TextPart)))
    return texts


def _context_blob(messages: list[Any]) -> str:
    return "\n".join(_result_texts(messages))


def _assert_ids_paired(messages: list[Any]) -> None:
    uses = {part.id: part.name for part in _use_parts(messages)}
    for result in _result_parts(messages):
        assert result.tool_use_id in uses
        assert result.name == uses[result.tool_use_id]


def _project_current_messages(messages: list[Any]) -> list[Any]:
    projected: list[Any] = []
    for message in messages:
        if message.role == "assistant":
            parts: list[Any] = []
            for part in message.parts:
                if isinstance(part, CurrentToolUsePart):
                    parts.append(
                        ToolUsePart(
                            id=part.id,
                            name=part.name,
                            inputs=dict(part.inputs),
                        )
                    )
                elif isinstance(part, CurrentTextPart):
                    parts.append(TextPart(text=part.text))
            projected.append(AssistantMessagePart(parts=parts))
        else:
            parts = []
            for part in message.parts:
                if isinstance(part, CurrentToolResultPart):
                    inner = [
                        TextPart(text=piece.text)
                        for piece in part.parts
                        if isinstance(piece, CurrentTextPart)
                    ]
                    parts.append(
                        ToolResultPart(
                            tool_use_id=part.tool_use_id,
                            name=part.name,
                            parts=inner,
                            is_error=part.is_error,
                        )
                    )
                elif isinstance(part, CurrentTextPart):
                    parts.append(TextPart(text=part.text))
            projected.append(UserMessagePart(parts=parts))
    return projected


def _pair(call_id: str, text: str, *, name: str = "noop") -> tuple[AssistantMessagePart, UserMessagePart]:
    return (
        AssistantMessagePart(parts=[ToolUsePart(id=call_id, name=name, inputs={})]),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id=call_id,
                    name=name,
                    parts=[TextPart(text=text)],
                )
            ]
        ),
    )


@dataclass
class _StreamerTool(CurrentTool):
    name = "streamer"
    description = "Yield progress then a payload."
    stream_output: ClassVar[bool] = True
    edits_context: ClassVar[bool] = False

    async def __call__(self, label: Annotated[str, "Label"]):
        runs = getattr(self, "_runs", 0)
        self._runs = runs + 1  # type: ignore[attr-defined]
        yield ToolStreamEvent(type="progress", data=f"chunk-{label}")
        yield {"label": label, "run": runs}


@dataclass
class _FailAfterYieldTool(CurrentTool):
    name = "stream_fail"
    description = "Yield once then fail."
    stream_output: ClassVar[bool] = True
    edits_context: ClassVar[bool] = False

    async def __call__(self):
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        yield ToolStreamEvent(type="progress", data="started")
        raise TimeoutError("after yield")


@dataclass
class _RetryBeforeYieldTool(CurrentTool):
    name = "stream_retry"
    description = "Fail before yield once, then stream."
    stream_output: ClassVar[bool] = True
    edits_context: ClassVar[bool] = False
    retryable_exceptions = (TimeoutError,)

    async def __call__(self):
        attempts = getattr(self, "_attempts", 0)
        self._attempts = attempts + 1  # type: ignore[attr-defined]
        if attempts == 0:
            raise TimeoutError("before yield")
        yield ToolStreamEvent(type="progress", data="ok")
        yield "recovered"


@dataclass
class _StreamingTypedTool(CurrentTool):
    name = "stream_typed"
    description = "Stream an integer."
    stream_output: ClassVar[bool] = True
    edits_context: ClassVar[bool] = False

    async def __call__(self, count: int):
        yield count


@dataclass
class _EchoTool(CurrentTool):
    name = "echo"
    description = "Echo label."
    edits_context: ClassVar[bool] = False

    async def __call__(self, label: Annotated[str, "Label"]) -> dict[str, str]:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        return {"label": label}


@dataclass
class _NoopTool(CurrentTool):
    name = "noop"
    description = "Ack."
    edits_context: ClassVar[bool] = False

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


def _run_current_streaming(
    tool_calls: list[CurrentToolUsePart],
    tools: list[CurrentTool],
) -> tuple[list[Any], list[Any]]:
    call_count = 0

    async def mock_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1

        async def stream() -> Any:
            yield CurrentStreamEndEvent(
                type="stream_end",
                model="mock",
                parts=tool_calls if call_count == 1 else [],
                stop_reason="tool_use" if call_count == 1 else "end_turn",
                usage=CurrentUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    executor = CurrentAgentExecutor("openai", provider, tools=tools)
    holder: dict[str, list[Any]] = {}
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def collect() -> tuple[list[Any], list[Any]]:
        events: list[Any] = []
        async for event in executor.run_stream([]):
            events.append(event)
        return events, holder["messages"]

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events, messages = asyncio.run(collect())
    return events, messages


def _scripted_recovered(
    turns: list[tuple[list[Any], Usage | None]],
    captured: list[list[Any]],
) -> Any:
    call_count = 0

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        idx = min(call_count, len(turns) - 1)
        call_count += 1
        captured.append(list(messages))
        parts, usage = turns[idx]

        async def stream() -> Any:
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=usage,
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    return provider


def _drive_recovered(
    policy: ContextPolicy,
    turns: list[tuple[list[Any], Usage | None]],
    messages: list[Any],
    *,
    tools: list[CurrentTool],
) -> tuple[list[Any], list[list[Any]]]:
    captured: list[list[Any]] = []
    executor = AgentExecutor(
        "openai",
        _scripted_recovered(turns, captured),
        tools=tools,
        context_policy=policy,
    )

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=len(turns)):
            events.append(event)
        return events

    return asyncio.run(run()), captured


def test_streaming_tool_before_compaction_preserves_pairing() -> None:
    """Captured streaming round-trip is one pair; trim keeps the final payload."""
    streamer = _StreamerTool()
    events, live = _run_current_streaming(
        [CurrentToolUsePart(id="st-1", name="streamer", inputs={"label": "alpha"})],
        [streamer],
    )
    stream_events = [event for event in events if isinstance(event, ToolStreamEvent)]
    assert len(stream_events) == 1
    assert stream_events[0].data == "chunk-alpha"
    assert streamer._runs == 1  # type: ignore[attr-defined]

    projected = _project_current_messages(live)
    assert len(projected) == 2
    _assert_ids_paired(projected)
    payload = _result_texts(projected)[0]
    assert "alpha" in payload

    filler: list[Any] = []
    for index in range(4):
        filler.extend(_pair(f"f{index}", f"fill-{index}"))
    history = filler + projected
    edited, applied = edit_context(history, _trim_policy(keep_last_n=1))
    assert applied is not None
    assert _PLACEHOLDER in _result_texts(edited)[0]
    assert "alpha" in _result_texts(edited)[-1]
    assert getattr(streamer, "_runs", 0) == 1


def test_repeated_streaming_turns_with_compaction_between() -> None:
    """Two streaming turns on recovered executor; compaction fires before the second send."""
    streamer = _StreamerTool()
    policy = _trim_policy()
    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="warm", name="noop", inputs={})], _usage(trigger - 1)),
        ([ToolUsePart(id="s1", name="streamer", inputs={"label": "first"})], _usage(trigger)),
        ([ToolUsePart(id="s2", name="streamer", inputs={"label": "second"})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, captured = _drive_recovered(
        policy, turns, [], tools=[streamer, _NoopTool()]
    )
    assert streamer._runs == 2  # type: ignore[attr-defined]
    edits = [event for event in events if isinstance(event, ContextEditEvent)]
    assert len(edits) >= 1
    assert len(captured) >= 3
    assert "first" in _context_blob(captured[2])
    final_blob = _context_blob(captured[-1])
    assert "first" in final_blob
    assert "second" in final_blob


def test_streaming_results_followed_by_auto_compaction() -> None:
    """Filler + streaming turn, then compaction on the following noop turn."""
    streamer = _StreamerTool()
    policy = _trim_policy()
    trigger = policy.trigger_tokens
    filler: list[Any] = []
    for index in range(5):
        filler.extend(_pair(f"fill-{index}", f"payload-{index}"))
    turns = [
        ([ToolUsePart(id="stream-1", name="streamer", inputs={"label": "live"})], _usage(trigger)),
        ([ToolUsePart(id="noop-1", name="noop", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, captured = _drive_recovered(
        policy, turns, filler, tools=[streamer, _NoopTool()]
    )
    assert streamer._runs == 1  # type: ignore[attr-defined]
    assert any(isinstance(event, ContextEditEvent) for event in events)
    assert "live" in _context_blob(captured[-1])
    assert _PLACEHOLDER in _context_blob(captured[1])


def test_streaming_error_after_yield_paired_and_trimmed() -> None:
    """Error result stays paired; trim clears the error text from older window."""
    tool = _FailAfterYieldTool()
    events, messages = _run_current_streaming(
        [CurrentToolUsePart(id="err-1", name="stream_fail", inputs={})],
        [tool],
    )
    stream_events = [event for event in events if isinstance(event, ToolStreamEvent)]
    assert len(stream_events) == 1
    assert tool._calls == 1  # type: ignore[attr-defined]
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert results[0].is_error is True
    assert "[tool_execution_error]" in str(results[0].result)

    projected = _project_current_messages(messages)
    _assert_ids_paired(projected)
    edited, applied = edit_context(projected, _trim_policy(keep_last_n=0))
    assert applied is not None
    assert _result_texts(edited) == [_PLACEHOLDER]
    assert _result_parts(edited)[0].tool_use_id == "err-1"


def test_streaming_retry_before_first_yield_no_duplicate_chunks() -> None:
    """Host retry on streaming tool: one progress event, one final result after recovery."""
    tool = _RetryBeforeYieldTool()
    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events, messages = _run_current_streaming(
            [CurrentToolUsePart(id="retry-stream", name="stream_retry", inputs={})],
            [tool],
        )
    stream_events = [event for event in events if isinstance(event, ToolStreamEvent)]
    assert tool._attempts == 2  # type: ignore[attr-defined]
    assert [event.data for event in stream_events] == ["ok"]
    projected = _project_current_messages(messages)
    assert _result_texts(projected) == ["recovered"]
    _assert_ids_paired(projected)


def test_streaming_validation_correction_preserves_tool_use_id() -> None:
    """Invalid streaming args produce one correction row with stable id."""
    tool = _StreamingTypedTool()
    call_count = 0

    async def mock_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        parts = (
            [CurrentToolUsePart(id="st-typed", name="stream_typed", inputs={"count": "bad"})]
            if call_count == 1
            else []
        )

        async def stream() -> Any:
            yield CurrentStreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=CurrentUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    executor = CurrentAgentExecutor("openai", provider, tools=[tool])
    holder: dict[str, list[Any]] = {}
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def run() -> None:
        try:
            async for _event in executor.run_stream([], max_model_corrections=0):
                pass
        except ModelRetryExhaustedError:
            pass

    asyncio.run(run())
    parts = [
        part
        for message in holder["messages"]
        if isinstance(message, CurrentUserMessagePart)
        for part in message.parts
        if isinstance(part, CurrentToolResultPart)
    ]
    assert len(parts) == 1
    assert parts[0].tool_use_id == "st-typed"
    assert parts[0].parts[0].text.startswith("[tool_input_invalid]")
    assert getattr(tool, "_calls", 0) == 0  # body never ran


def test_parallel_echo_plus_streaming_same_turn() -> None:
    """Parallel echo batch completes, then streaming tool runs; trim keeps both ids."""
    echo = _EchoTool()
    streamer = _StreamerTool()
    calls = [
        CurrentToolUsePart(id="p1", name="echo", inputs={"label": "parallel"}),
        CurrentToolUsePart(id="s-par", name="streamer", inputs={"label": "stream"}),
    ]
    call_count = 0

    async def mock_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1

        async def stream() -> Any:
            yield CurrentStreamEndEvent(
                type="stream_end",
                model="mock",
                parts=calls if call_count == 1 else [],
                stop_reason="tool_use" if call_count == 1 else "end_turn",
                usage=CurrentUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    executor = CurrentAgentExecutor("openai", provider, tools=[echo, streamer])
    holder: dict[str, list[Any]] = {}
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def collect() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream([]):
            events.append(event)
        return events

    events = asyncio.run(collect())
    stream_events = [event for event in events if isinstance(event, ToolStreamEvent)]
    assert len(stream_events) == 1
    assert echo._calls == 1  # type: ignore[attr-defined]
    assert streamer._runs == 1  # type: ignore[attr-defined]

    projected = _project_current_messages(holder["messages"])
    assert len(_result_parts(projected)) == 2
    ids = [part.tool_use_id for part in _result_parts(projected)]
    assert ids == ["p1", "s-par"]
    edited, applied = edit_context(projected, _trim_policy(keep_last_n=1))
    assert applied is not None
    assert [part.tool_use_id for part in _result_parts(edited)] == ["p1", "s-par"]
    _assert_ids_paired(edited)


def test_usable_context_after_streaming_compaction() -> None:
    """Post-compaction send still exposes the kept streaming payload for a later read."""
    streamer = _StreamerTool()
    policy = _trim_policy(keep_last_n=2)
    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="warm", name="noop", inputs={})], _usage(trigger - 1)),
        ([ToolUsePart(id="s-keep", name="streamer", inputs={"label": "KEEP-ME"})], _usage(trigger)),
        ([ToolUsePart(id="read", name="noop", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, captured = _drive_recovered(policy, turns, [], tools=[streamer, _NoopTool()])
    assert any(isinstance(event, ContextEditEvent) for event in events)
    assert "KEEP-ME" in _context_blob(captured[-1])
