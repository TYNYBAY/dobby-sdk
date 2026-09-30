# ruff: noqa: E402
"""Phase 12: compaction interaction with parallel tool batches."""

# isort: off
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest

from dobby import AgentExecutor as CurrentAgentExecutor
from dobby.tools import Tool as CurrentTool
from dobby.types import StreamEndEvent as CurrentStreamEndEvent
from dobby.types import TextPart as CurrentTextPart
from dobby.types import ToolResultEvent
from dobby.types import ToolResultPart as CurrentToolResultPart
from dobby.types import ToolUsePart as CurrentToolUsePart
from dobby.types import Usage as CurrentUsage

from tests.compaction_subject import load_recovered

load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context import edit_context, summarize_context
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

_BATCH_SIZES = (2, 3, 5, 10, 50)
_PLACEHOLDER = "[Tool result cleared to save context.]"
_FACT_RE = re.compile(r"\[DOBBY-FACT:[^\]]+\]")


def _usage(input_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _trim_policy(*, keep_last_n: int = 2, mode: str = "trim") -> ContextPolicy:
    return ContextPolicy(
        context_window=128_000,
        trigger_pct=0.8,
        keep_last_n=keep_last_n,
        mode=mode,
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


def _assert_ids_paired(messages: list[Any]) -> None:
    uses = {part.id: part.name for part in _use_parts(messages)}
    for result in _result_parts(messages):
        assert result.tool_use_id in uses
        assert result.name == uses[result.tool_use_id]


def _assert_one_pair_per_tool(messages: list[Any], expected_ids: list[str]) -> None:
    """Parallel batches become alternating assistant/user rows, one tool call each."""
    assert len(messages) == len(expected_ids) * 2
    for index, call_id in enumerate(expected_ids):
        use_msg = messages[index * 2]
        result_msg = messages[index * 2 + 1]
        assert isinstance(use_msg, AssistantMessagePart)
        assert isinstance(result_msg, UserMessagePart)
        assert len(use_msg.parts) == 1
        assert len(result_msg.parts) == 1
        assert use_msg.parts[0].id == call_id
        assert result_msg.parts[0].tool_use_id == call_id


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


def _pair(
    call_id: str,
    text: str,
    *,
    name: str = "noop",
    inputs: dict[str, Any] | None = None,
) -> tuple[AssistantMessagePart, UserMessagePart]:
    return (
        AssistantMessagePart(
            parts=[ToolUsePart(id=call_id, name=name, inputs={} if inputs is None else inputs)]
        ),
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
class _EchoTool(CurrentTool):
    name = "echo"
    description = "Return the label."
    edits_context: ClassVar[bool] = False

    async def __call__(self, label: Annotated[str, "Label"]) -> dict[str, str]:
        calls: list[str] = getattr(self, "_calls", [])
        calls.append(label)
        self._calls = calls  # type: ignore[attr-defined]
        return {"label": label}


@dataclass
class _NoopTool(CurrentTool):
    name = "noop"
    description = "Ack."
    edits_context: ClassVar[bool] = False

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


@dataclass
class _FailingTool(CurrentTool):
    name = "failing"
    description = "Always fail."
    edits_context: ClassVar[bool] = False

    async def __call__(self) -> None:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        raise ValueError("fail")


@dataclass
class _RetryParallelTool(CurrentTool):
    name = "retrying_tool"
    description = "Fail once then succeed."
    edits_context: ClassVar[bool] = False
    retryable_exceptions = (TimeoutError,)

    async def __call__(self) -> dict[str, str]:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        if calls == 0:
            raise TimeoutError("transient")
        return {"status": "ok"}


@dataclass
class _SiblingTool(CurrentTool):
    name = "sibling_tool"
    description = "Parallel sibling."
    edits_context: ClassVar[bool] = False

    async def __call__(self) -> dict[str, str]:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        return {"status": "sibling"}


@dataclass
class _TypedTool(CurrentTool):
    name = "typed"
    description = "Accept int."
    edits_context: ClassVar[bool] = False

    async def __call__(self, count: int) -> int:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        return count


def _capture_parallel_batch(
    batch_size: int,
    *,
    tools: list[CurrentTool] | None = None,
    tool_calls: list[CurrentToolUsePart] | None = None,
) -> list[Any]:
    calls = tool_calls or [
        CurrentToolUsePart(id=f"id-{index}", name="echo", inputs={"label": str(index)})
        for index in range(batch_size)
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
    executor = CurrentAgentExecutor("openai", provider, tools=tools or [_EchoTool()])
    holder: dict[str, list[Any]] = {}
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def run() -> None:
        async for _event in executor.run_stream([]):
            pass

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(run())
    return holder["messages"]


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
    provider.name = "scripted"
    return provider


def _bind_host_retry(recovered: AgentExecutor) -> None:
    bridge = CurrentAgentExecutor("openai", AsyncMock(), tools=list(recovered.tools.values()))

    async def invoke_with_retry(tool: Any, inputs: dict[str, Any], context: Any) -> Any:
        return await bridge._invoke_tool(tool, inputs, context)

    recovered._invoke_tool = invoke_with_retry  # type: ignore[method-assign, assignment]


def _drive_recovered(
    policy: ContextPolicy,
    turns: list[tuple[list[Any], Usage | None]],
    messages: list[Any],
    *,
    tools: list[CurrentTool] | None = None,
    bind_retry: bool = False,
) -> tuple[list[Any], list[list[Any]]]:
    captured: list[list[Any]] = []
    executor = AgentExecutor(
        "openai",
        _scripted_recovered(turns, captured),
        tools=tools or [_NoopTool()],
        context_policy=policy,
    )
    if bind_retry:
        _bind_host_retry(executor)

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=len(turns)):
            events.append(event)
        return events

    return asyncio.run(run()), captured


def _fact_summarizer() -> AsyncMock:
    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", True) is False:
            span = " ".join(
                part.text
                for message in messages
                if isinstance(message, UserMessagePart)
                for part in message.parts
                if isinstance(part, TextPart)
            )
            markers = " ".join(m.group(0) for m in _FACT_RE.finditer(span))
            return StreamEndEvent(
                type="stream_end",
                model="mock-summarizer",
                parts=[TextPart(text=markers or "empty")],
                stop_reason="end_turn",
                usage=_usage(0),
            )
        raise AssertionError("unexpected streaming summarizer call")

    provider = AsyncMock()
    provider.chat = mock_chat
    return provider


@pytest.mark.parametrize("batch_size", _BATCH_SIZES, ids=lambda n: f"parallel-before-{n}")
def test_parallel_batch_before_compaction_keep_last_n(batch_size: int) -> None:
    """Batch captured on current executor; filler cleared, last two batch pairs kept."""
    live = _capture_parallel_batch(batch_size)
    batch = _project_current_messages(live)
    expected_ids = [f"id-{index}" for index in range(batch_size)]
    _assert_one_pair_per_tool(batch, expected_ids)

    filler: list[Any] = []
    for index in range(4):
        filler.extend(_pair(f"f{index}", f"filler-{index}", name="noop"))

    history = filler + batch
    edited, applied = edit_context(history, _trim_policy(keep_last_n=2))
    if batch_size <= 2:
        assert applied is None or applied.cleared_tool_uses == 4
    else:
        assert applied is not None
        assert applied.cleared_tool_uses == len(filler) // 2 + batch_size - 2
    texts = _result_texts(edited)
    assert texts[-2:] == [
        str({"label": str(batch_size - 2)}),
        str({"label": str(batch_size - 1)}),
    ]
    if batch_size > 2:
        assert texts[-3] == _PLACEHOLDER or batch_size == 3
    _assert_ids_paired(edited)


@pytest.mark.parametrize("batch_size", _BATCH_SIZES, ids=lambda n: f"parallel-after-{n}")
def test_parallel_batch_after_compaction_runs_once_each(batch_size: int) -> None:
    """Auto-trim on filler, then parallel batch executes exactly once per tool."""
    echo = _EchoTool()
    policy = _trim_policy(keep_last_n=2)
    trigger = policy.trigger_tokens
    filler: list[Any] = []
    for index in range(6):
        filler.extend(_pair(f"fill-{index}", f"fill-{index}", name="noop"))

    batch_parts = [
        ToolUsePart(id=f"p-{index}", name="echo", inputs={"label": str(index)})
        for index in range(batch_size)
    ]
    turns = [
        ([ToolUsePart(id="warm", name="noop", inputs={})], _usage(trigger)),
        (batch_parts, _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, captured = _drive_recovered(policy, turns, filler, tools=[echo, _NoopTool()])
    assert len(getattr(echo, "_calls", [])) == batch_size  # type: ignore[attr-defined]
    assert len(set(getattr(echo, "_calls", []))) == batch_size  # type: ignore[attr-defined]
    edits = [event for event in events if isinstance(event, ContextEditEvent)]
    assert len(edits) >= 1
    assert _PLACEHOLDER in " ".join(_result_texts(captured[1]))
    batch_ids = [f"p-{index}" for index in range(batch_size)]
    send_during_batch = captured[1]
    result_ids = [part.tool_use_id for part in _result_parts(send_during_batch)]
    assert not any(batch_id in result_ids for batch_id in batch_ids)
    final_send = captured[-1]
    result_ids_final = [part.tool_use_id for part in _result_parts(final_send)]
    assert result_ids_final[-batch_size:] == batch_ids
    _assert_ids_paired(final_send)


def test_partial_parallel_batches_trim_keep_last_n() -> None:
    """Two batches (5 + 3 pairs): keep_last_n=4 retains last pair of first batch + all of second."""
    batch_a = _project_current_messages(_capture_parallel_batch(5))
    batch_b_calls = [
        CurrentToolUsePart(id=f"b-{index}", name="echo", inputs={"label": f"b{index}"})
        for index in range(3)
    ]
    batch_b = _project_current_messages(
        _capture_parallel_batch(0, tool_calls=batch_b_calls)
    )
    history = batch_a + batch_b
    edited, applied = edit_context(history, _trim_policy(keep_last_n=4))
    assert applied is not None
    assert applied.cleared_tool_uses == 4
    texts = _result_texts(edited)
    assert texts[:4] == [_PLACEHOLDER] * 4
    assert texts[4] == str({"label": "4"})
    assert texts[-3:] == [str({"label": f"b{index}"}) for index in range(3)]
    _assert_ids_paired(edited)


def test_partial_parallel_batches_summarize() -> None:
    marker = "[DOBBY-FACT:id:batch-a=first-five]"
    batch_a = _project_current_messages(_capture_parallel_batch(5))
    batch_a[1] = UserMessagePart(
        parts=[
            ToolResultPart(
                tool_use_id="id-0",
                name="echo",
                parts=[TextPart(text=f"{str({'label': '0'})}\n{marker}")],
            )
        ]
    )
    batch_b = _project_current_messages(_capture_parallel_batch(3))
    history = batch_a + batch_b
    policy = _trim_policy(keep_last_n=3, mode="summarize")
    applied = asyncio.run(summarize_context(history, policy, _fact_summarizer()))
    assert applied is not None
    assert marker in applied.summary_text
    assert str({"label": "2"}) in _result_texts(history)


def test_parallel_batch_with_failed_tool_in_batch() -> None:
    """One success and one execution error; ids stay paired after trim."""
    calls = [
        CurrentToolUsePart(id="ok", name="echo", inputs={"label": "good"}),
        CurrentToolUsePart(id="bad", name="failing", inputs={}),
    ]
    failing = _FailingTool()
    live = _capture_parallel_batch(0, tools=[_EchoTool(), failing], tool_calls=calls)
    projected = _project_current_messages(live)
    assert failing._calls == 1  # type: ignore[attr-defined]
    texts = _result_texts(projected)
    assert str({"label": "good"}) in texts
    assert any("[tool_execution_error]" in text for text in texts)

    edited, applied = edit_context(projected, _trim_policy(keep_last_n=1))
    assert applied is not None
    ids = [part.tool_use_id for part in _result_parts(edited)]
    assert ids == ["ok", "bad"]
    _assert_ids_paired(edited)


def test_parallel_batch_with_host_retry_and_sibling() -> None:
    """Retrying tool + sibling: two results, two invocations on retry side, one on sibling."""
    calls = [
        CurrentToolUsePart(id="tc1", name="retrying_tool", inputs={}),
        CurrentToolUsePart(id="tc2", name="sibling_tool", inputs={}),
    ]
    retry_tool = _RetryParallelTool()
    sibling = _SiblingTool()
    messages = _capture_parallel_batch(0, tools=[retry_tool, sibling], tool_calls=calls)
    assert retry_tool._calls == 2  # type: ignore[attr-defined]
    assert sibling._calls == 1  # type: ignore[attr-defined]

    projected = _project_current_messages(messages)
    edited, applied = edit_context(projected, _trim_policy(keep_last_n=1))
    assert applied is not None
    assert [part.tool_use_id for part in _result_parts(edited)] == ["tc1", "tc2"]
    assert retry_tool._calls == 2  # type: ignore[attr-defined]
    assert sibling._calls == 1  # type: ignore[attr-defined]


def test_parallel_batch_with_model_correction_and_sibling() -> None:
    """Unknown parallel tool + typed sibling; trim preserves three paired ids."""
    typed = _TypedTool()
    calls = [
        CurrentToolUsePart(id="unknown", name="missing", inputs={}),
        CurrentToolUsePart(id="sibling", name="typed", inputs={"count": 1}),
    ]
    provider_turns = [[calls[0], calls[1]], []]
    call_count = 0

    async def mock_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        parts = provider_turns[call_count - 1] if call_count <= len(provider_turns) else []

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
    executor = CurrentAgentExecutor("openai", provider, tools=[typed])
    holder: dict[str, list[Any]] = {}
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def run() -> None:
        async for _event in executor.run_stream([]):
            pass

    asyncio.run(run())
    projected = _project_current_messages(holder["messages"])
    assert typed._calls == 1  # type: ignore[attr-defined]
    assert len(_result_parts(projected)) == 2

    edited, applied = edit_context(projected, _trim_policy(keep_last_n=2))
    assert applied is None
    _assert_ids_paired(edited)


@pytest.mark.parametrize("batch_size", (5, 50), ids=lambda n: f"no-dup-{n}")
def test_parallel_batch_executor_invocation_counts(batch_size: int) -> None:
    """Each echo label in a parallel batch is recorded exactly once."""
    echo = _EchoTool()
    policy = _trim_policy()
    trigger = policy.trigger_tokens
    parts = [
        ToolUsePart(id=f"e-{index}", name="echo", inputs={"label": str(index)})
        for index in range(batch_size)
    ]
    turns = [(parts, _usage(trigger)), ([], _usage(trigger))]
    _drive_recovered(policy, turns, [], tools=[echo], bind_retry=False)
    assert getattr(echo, "_calls", []) == [str(index) for index in range(batch_size)]
