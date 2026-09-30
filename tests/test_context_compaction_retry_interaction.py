# ruff: noqa: E402
"""Phase 10: compaction interaction with host-side tool retry and model correction.

Host-side retry uses the current ``dobby`` executor; compaction uses the recovered
``dobby-compaction-94b5a8f`` package. Integration runs patch the recovered executor
to delegate ``_invoke_tool`` to production retry logic (test-only bridge).
"""

# isort: off
from __future__ import annotations

import asyncio
import copy
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
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


def _recovered_root() -> Path:
    repo = Path(__file__).resolve().parents[1]
    pointer = repo / ".git" / "worktrees" / "dobby-compaction-94b5a8f" / "gitdir"
    return Path(pointer.read_text(encoding="utf-8").strip()).parent


def _load_recovered() -> Any:
    name = "recovered_dobby"
    if name in sys.modules:
        return sys.modules[name]
    root = _recovered_root()
    init = root / "dobby" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name,
        init,
        submodule_search_locations=[str(root / "dobby")],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load recovered compaction package from {init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context import edit_context, summarize_context
from recovered_dobby.tools import Tool
from recovered_dobby.types import (
    AssistantMessagePart,
    StreamEndEvent,
    TextPart,
    ToolResultEvent as RecoveredToolResultEvent,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart,
)

# isort: on

_PLACEHOLDER = "[Tool result cleared to save context.]"
_FACT_RE = re.compile(r"\[DOBBY-FACT:(?P<category>[a-z]+):(?P<key>[^=\]]+)=(?P<value>[^\]]+)\]")


def _usage(input_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _trim_policy(*, keep_last_n: int = 2, mode: str = "trim") -> ContextPolicy:
    return ContextPolicy(
        context_window=128_000,
        trigger_pct=0.8,
        keep_last_n=keep_last_n,
        mode=mode,
    )


def _pair(
    call_id: str,
    text: str,
    *,
    name: str = "echo",
    inputs: dict[str, Any] | None = None,
    is_error: bool = False,
) -> tuple[AssistantMessagePart, UserMessagePart]:
    return (
        AssistantMessagePart(
            parts=[
                ToolUsePart(
                    id=call_id,
                    name=name,
                    inputs={} if inputs is None else inputs,
                )
            ]
        ),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id=call_id,
                    name=name,
                    parts=[TextPart(text=text)],
                    is_error=is_error,
                )
            ]
        ),
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


def _scripted_provider(
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


def _bind_host_retry_from_current(recovered: AgentExecutor) -> CurrentAgentExecutor:
    """Test bridge: recovered executor tool bodies use production host-side retry."""
    bridge = CurrentAgentExecutor(
        "openai",
        AsyncMock(),
        tools=list(recovered.tools.values()),
    )

    async def invoke_with_retry(
        tool: Tool,
        inputs: dict[str, Any],
        context: Any,
    ) -> Any:
        return await bridge._invoke_tool(tool, inputs, context)

    recovered._invoke_tool = invoke_with_retry  # type: ignore[method-assign, assignment]
    return bridge


class _NoopTool(CurrentTool):
    name = "noop"
    description = "Ack."
    edits_context: ClassVar[bool] = False

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


def _drive_recovered(
    policy: ContextPolicy,
    turns: list[tuple[list[Any], Usage | None]],
    messages: list[Any],
    *,
    tools: list[Tool] | None = None,
    bind_retry: bool = False,
) -> tuple[list[Any], list[list[Any]], AgentExecutor]:
    captured: list[list[Any]] = []
    llm = _scripted_provider(turns, captured)
    executor = AgentExecutor(
        "openai",
        llm,
        tools=tools or [_NoopTool()],
        context_policy=policy,
    )
    if bind_retry:
        _bind_host_retry_from_current(executor)

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=len(turns)):
            events.append(event)
        return events

    events = asyncio.run(run())
    return events, captured, executor


def _capture_current_run(
    tool_calls: list[CurrentToolUsePart],
    tools: list[CurrentTool],
) -> tuple[list[Any], list[ToolResultEvent]]:
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

    async def run() -> list[ToolResultEvent]:
        results: list[ToolResultEvent] = []
        async for event in executor.run_stream([]):
            if isinstance(event, ToolResultEvent):
                results.append(event)
        return results

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        results = asyncio.run(run())
    return holder["messages"], results


def _model_correction_history() -> list[Any]:
    """Two model-correction rounds then a successful call (distinct tool call ids)."""
    messages: list[Any] = []
    messages.extend(
        _pair(
            "call-v1",
            "[tool_retry] use count as integer",
            name="typed",
            inputs={"count": "bad"},
            is_error=True,
        )
    )
    messages.extend(
        _pair(
            "call-v2",
            "[tool_retry] still wrong type",
            name="typed",
            inputs={"count": "also-bad"},
            is_error=True,
        )
    )
    messages.extend(
        _pair(
            "call-v3",
            "42",
            name="typed",
            inputs={"count": 42},
            is_error=False,
        )
    )
    return messages


def _fact_preserving_summarizer(
    summary_calls: list[dict[str, Any]] | None = None,
) -> AsyncMock:
    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", True) is False:
            record = {"messages": copy.deepcopy(messages)}
            if summary_calls is not None:
                summary_calls.append(record)
            span_text = " ".join(
                part.text
                for message in messages
                if isinstance(message, UserMessagePart)
                for part in message.parts
                if isinstance(part, TextPart)
            )
            markers = " ".join(m.group(0) for m in _FACT_RE.finditer(span_text))
            digest = markers or "empty-span"
            return StreamEndEvent(
                type="stream_end",
                model="mock-summarizer",
                parts=[TextPart(text=digest)],
                stop_reason="end_turn",
                usage=_usage(0),
            )
        raise AssertionError("unexpected streaming chat in summarize test")

    provider = AsyncMock()
    provider.chat = mock_chat
    provider.name = "deterministic-summarizer"
    return provider


@dataclass
class _FlakyOnceTool(CurrentTool):
    name = "flaky"
    description = "Fail once with TimeoutError, then succeed."
    edits_context: ClassVar[bool] = False
    retryable_exceptions = (TimeoutError,)

    async def __call__(self) -> dict[str, str]:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        if calls == 0:
            raise TimeoutError("transient")
        return {"status": "ok", "attempts": str(self._calls)}  # type: ignore[attr-defined]


@dataclass
class _MultiRetryTool(CurrentTool):
    name = "multi_flaky"
    description = "Fail three times then succeed."
    edits_context: ClassVar[bool] = False
    max_retries = 3
    retryable_exceptions = (TimeoutError,)

    async def __call__(self) -> dict[str, str]:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        if calls < 3:
            raise TimeoutError(f"fail-{calls}")
        return {"status": "ok", "attempts": str(self._calls)}  # type: ignore[attr-defined]


@dataclass
class _RetryingParallelTool(CurrentTool):
    name = "retrying_tool"
    description = "Parallel tool that retries once."
    edits_context: ClassVar[bool] = False
    retryable_exceptions = (TimeoutError,)

    async def __call__(self) -> dict[str, str]:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        if calls == 0:
            raise TimeoutError("transient")
        return {"label": "retrying"}


@dataclass
class _SiblingTool(CurrentTool):
    name = "sibling_tool"
    description = "Parallel sibling."
    edits_context: ClassVar[bool] = False

    async def __call__(self) -> dict[str, str]:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        return {"label": "sibling"}


def test_host_retry_before_compaction_preserves_single_success_pair() -> None:
    """Host retry completes before history is seeded; compaction keeps the final pair."""
    flaky = _FlakyOnceTool()
    live_messages, results = _capture_current_run(
        [CurrentToolUsePart(id="call-flaky", name="flaky", inputs={})],
        [flaky],
    )
    assert flaky._calls == 2  # type: ignore[attr-defined]
    assert len(results) == 1
    assert results[0].tool_use_id == "call-flaky"
    assert results[0].result == {"status": "ok", "attempts": "2"}

    history = _project_current_messages(live_messages)
    assert len(_result_parts(history)) == 1
    assert _result_texts(history) == [str({"status": "ok", "attempts": "2"})]

    policy = _trim_policy()
    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="t-warm", name="noop", inputs={})], _usage(trigger)),
        ([ToolUsePart(id="t-next", name="noop", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, captured, _executor = _drive_recovered(policy, turns, history)
    edits = [event for event in events if isinstance(event, ContextEditEvent)]
    assert len(edits) == 1
    assert len(captured) >= 2
    send_after_compact = captured[1]
    assert len(_result_parts(send_after_compact)) == 2
    assert _result_texts(send_after_compact)[-1] == str({"ok": "yes"})
    assert _result_texts(send_after_compact)[0] == str({"status": "ok", "attempts": "2"})
    assert flaky._calls == 2  # type: ignore[attr-defined]


def test_host_retry_after_compaction_with_retry_bridge() -> None:
    """Flaky tool runs after auto-trim; host retry still succeeds without extra pairs."""
    policy = _trim_policy()
    trigger = policy.trigger_tokens
    filler: list[Any] = []
    for index in range(4):
        filler.extend(_pair(f"h{index}", f"filler-{index}", name="noop"))

    flaky = _FlakyOnceTool()
    turns = [
        ([ToolUsePart(id="warm", name="noop", inputs={})], _usage(trigger - 1)),
        ([ToolUsePart(id="call-flaky", name="flaky", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events, captured, _executor = _drive_recovered(
            policy,
            turns,
            filler,
            tools=[flaky, _NoopTool()],
            bind_retry=True,
        )
    edits = [event for event in events if isinstance(event, ContextEditEvent)]
    assert len(edits) == 1
    assert flaky._calls == 2  # type: ignore[attr-defined]
    assert len(_result_parts(captured[-1])) >= 1
    last_texts = _result_texts(captured[-1])
    assert str({"status": "ok", "attempts": "2"}) in last_texts


def test_model_correction_retry_chain_trim_counts_each_round_trip() -> None:
    """Each model-correction round-trip is one pair; trim can clear early [tool_retry] rows."""
    history = _model_correction_history()
    edited, applied = edit_context(history, _trim_policy(keep_last_n=1))
    assert applied is not None
    assert applied.cleared_tool_uses == 2
    texts = _result_texts(edited)
    assert texts == [_PLACEHOLDER, _PLACEHOLDER, "42"]
    ids = [part.tool_use_id for part in _result_parts(edited)]
    assert ids == ["call-v1", "call-v2", "call-v3"]
    inputs = [part.inputs for part in _use_parts(edited)]
    assert inputs == [{"count": "bad"}, {"count": "also-bad"}, {"count": 42}]


def test_model_correction_chain_summarize_retains_markers_in_digest() -> None:
    marker = "[DOBBY-FACT:constraint:retry_hint=keep_for_correction]"
    history = _model_correction_history()
    history[1] = UserMessagePart(
        parts=[
            ToolResultPart(
                tool_use_id="call-v1",
                name="typed",
                parts=[TextPart(text=f"[tool_retry] bad\n{marker}")],
                is_error=True,
            )
        ]
    )
    llm = _fact_preserving_summarizer()
    policy = _trim_policy(keep_last_n=1, mode="summarize")

    applied = asyncio.run(summarize_context(history, policy, llm))
    assert applied is not None
    assert marker in applied.summary_text
    user_text = "\n".join(
        part.text
        for message in history
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, TextPart)
    )
    assert "<summary>" in user_text
    assert marker in user_text
    assert "42" in _result_texts(history)


def test_failed_tool_result_then_model_retry_loses_error_text_after_trim() -> None:
    """Oracle: cleared [tool_execution_error] text is not visible after trim."""
    history: list[Any] = []
    history.extend(
        _pair(
            "call-fail",
            "[tool_execution_error] The tool failed unexpectedly.",
            name="broken",
            is_error=True,
        )
    )
    history.extend(
        _pair(
            "call-retry",
            "recovered-value",
            name="broken",
            inputs={"fix": True},
        )
    )
    edited, applied = edit_context(history, _trim_policy(keep_last_n=1))
    assert applied is not None
    blob = "\n".join(_result_texts(edited))
    assert _PLACEHOLDER in blob
    assert "recovered-value" in blob
    assert "[tool_execution_error]" not in blob


def test_multiple_host_retries_then_compaction() -> None:
    tool = _MultiRetryTool()
    live, results = _capture_current_run(
        [CurrentToolUsePart(id="call-multi", name="multi_flaky", inputs={})],
        [tool],
    )
    assert tool._calls == 4  # type: ignore[attr-defined]
    assert results[0].result == {"status": "ok", "attempts": "4"}

    projected = _project_current_messages(live)
    edited, applied = edit_context(projected, _trim_policy(keep_last_n=0))
    assert applied is not None
    assert _result_texts(edited) == [_PLACEHOLDER]
    assert _result_parts(edited)[0].tool_use_id == "call-multi"
    assert tool._calls == 4  # type: ignore[attr-defined]


def test_parallel_retry_batch_trim_preserves_ids_and_single_sibling_run() -> None:
    retry_tool = _RetryingParallelTool()
    sibling = _SiblingTool()
    calls = [
        CurrentToolUsePart(id="tc1", name="retrying_tool", inputs={}),
        CurrentToolUsePart(id="tc2", name="sibling_tool", inputs={}),
    ]
    live, results = _capture_current_run(calls, [retry_tool, sibling])
    assert retry_tool._calls == 2  # type: ignore[attr-defined]
    assert sibling._calls == 1  # type: ignore[attr-defined]
    assert [event.tool_use_id for event in results] == ["tc1", "tc2"]

    projected = _project_current_messages(live)
    edited, applied = edit_context(projected, _trim_policy(keep_last_n=1))
    assert applied is not None
    assert [part.tool_use_id for part in _result_parts(edited)] == ["tc1", "tc2"]
    assert _result_texts(edited)[-1] == str({"label": "sibling"})
    assert retry_tool._calls == 2  # type: ignore[attr-defined]
    assert sibling._calls == 1  # type: ignore[attr-defined]


def test_compaction_does_not_reinvoke_tools_on_later_turns() -> None:
    """Tool invocation counts stay flat across an auto-trim turn."""
    policy = _trim_policy()
    trigger = policy.trigger_tokens
    counter_tool = _FlakyOnceTool()
    turns = [
        ([ToolUsePart(id="c1", name="flaky", inputs={})], _usage(trigger - 1)),
        ([ToolUsePart(id="c2", name="noop", inputs={})], _usage(trigger)),
        ([ToolUsePart(id="c3", name="noop", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events, _captured, _executor = _drive_recovered(
            policy,
            turns,
            [],
            tools=[counter_tool, _NoopTool()],
            bind_retry=True,
        )
    edits = [event for event in events if isinstance(event, ContextEditEvent)]
    assert len(edits) == 1
    assert counter_tool._calls == 2  # type: ignore[attr-defined]


def test_recovered_executor_without_retry_bridge_fails_flaky_once() -> None:
    """Documented gap: recovered executor alone does not host-retry before emitting errors."""
    policy = _trim_policy()
    trigger = policy.trigger_tokens
    flaky = _FlakyOnceTool()
    turns = [
        ([ToolUsePart(id="only", name="flaky", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events, _captured, _executor = _drive_recovered(
            policy,
            turns,
            [],
            tools=[flaky],
            bind_retry=False,
        )
    results = [event for event in events if isinstance(event, RecoveredToolResultEvent)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert flaky._calls == 1  # type: ignore[attr-defined]
