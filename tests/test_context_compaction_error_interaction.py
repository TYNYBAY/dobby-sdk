# ruff: noqa: E402
"""Phase 14: compaction interaction with tool and execution errors."""

# isort: off
from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar
from unittest.mock import AsyncMock

from dobby import AgentExecutor as CurrentAgentExecutor
from dobby.exceptions import ModelRetry
from dobby.tools import Tool as CurrentTool
from dobby.types import StreamEndEvent as CurrentStreamEndEvent
from dobby.types import TextPart as CurrentTextPart
from dobby.types import ToolResultEvent
from dobby.types import ToolResultPart as CurrentToolResultPart
from dobby.types import ToolStreamEvent
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


def _pair(
    call_id: str,
    text: str,
    *,
    name: str = "tool",
    is_error: bool = False,
) -> tuple[AssistantMessagePart, UserMessagePart]:
    return (
        AssistantMessagePart(parts=[ToolUsePart(id=call_id, name=name, inputs={})]),
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


def _result_texts(messages: list[Any]) -> list[str]:
    return [
        "".join(p.text for p in part.parts if isinstance(p, TextPart))
        for part in _result_parts(messages)
    ]


def _assert_ids_paired(messages: list[Any]) -> None:
    uses = {part.id: part.name for part in _use_parts(messages)}
    for result in _result_parts(messages):
        assert result.tool_use_id in uses
        assert result.name == uses[result.tool_use_id]


def _use_parts(messages: list[Any]) -> list[ToolUsePart]:
    return [
        part
        for message in messages
        if isinstance(message, AssistantMessagePart)
        for part in message.parts
        if isinstance(part, ToolUsePart)
    ]


def _context_blob(messages: list[Any]) -> str:
    return "\n".join(_result_texts(messages))


def _project_current_messages(messages: list[Any]) -> list[Any]:
    projected: list[Any] = []
    for message in messages:
        if message.role == "assistant":
            parts: list[Any] = []
            for part in message.parts:
                if isinstance(part, CurrentToolUsePart):
                    parts.append(
                        ToolUsePart(id=part.id, name=part.name, inputs=dict(part.inputs))
                    )
            projected.append(AssistantMessagePart(parts=parts))
        else:
            parts = []
            for part in message.parts:
                if isinstance(part, CurrentToolResultPart):
                    parts.append(
                        ToolResultPart(
                            tool_use_id=part.tool_use_id,
                            name=part.name,
                            parts=[
                                TextPart(text=p.text)
                                for p in part.parts
                                if isinstance(p, CurrentTextPart)
                            ],
                            is_error=part.is_error,
                        )
                    )
            projected.append(UserMessagePart(parts=parts))
    return projected


@dataclass
class _OkTool(CurrentTool):
    name = "ok_tool"
    description = "Succeed."
    edits_context: ClassVar[bool] = False

    async def __call__(self) -> dict[str, str]:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        return {"status": "ok"}


@dataclass
class _FailTool(CurrentTool):
    name = "fail_tool"
    description = "Raise ValueError."
    edits_context: ClassVar[bool] = False

    async def __call__(self) -> None:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        raise ValueError("boom")


@dataclass
class _BodyRetryTool(CurrentTool):
    name = "body_retry"
    description = "ModelRetry from body."
    edits_context: ClassVar[bool] = False

    async def __call__(self) -> None:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        raise ModelRetry("fix inputs")


@dataclass
class _TypedTool(CurrentTool):
    name = "typed"
    description = "Int arg."
    edits_context: ClassVar[bool] = False

    async def __call__(self, count: int) -> int:
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        return count


@dataclass
class _StreamFailTool(CurrentTool):
    name = "stream_fail"
    description = "Stream then fail."
    stream_output: ClassVar[bool] = True
    edits_context: ClassVar[bool] = False

    async def __call__(self):
        calls = getattr(self, "_calls", 0)
        self._calls = calls + 1  # type: ignore[attr-defined]
        yield ToolStreamEvent(type="progress", data="tick")
        raise RuntimeError("stream boom")


@dataclass
class _NoopTool(CurrentTool):
    name = "noop"
    description = "Ack."
    edits_context: ClassVar[bool] = False

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


def _run_batch(
    tool_calls: list[CurrentToolUsePart],
    tools: list[CurrentTool],
) -> tuple[list[ToolResultEvent], list[Any]]:
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

    async def collect() -> list[ToolResultEvent]:
        results: list[ToolResultEvent] = []
        async for event in executor.run_stream([]):
            if isinstance(event, ToolResultEvent):
                results.append(event)
        return results

    results = asyncio.run(collect())
    return results, holder["messages"]


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


def test_success_then_execution_error_trim_preserves_is_error_flag() -> None:
    """Success pair cleared to placeholder; error pair keeps ``is_error=True``."""
    ok = _OkTool()
    fail = _FailTool()
    results, live = _run_batch(
        [
            CurrentToolUsePart(id="ok-1", name="ok_tool", inputs={}),
            CurrentToolUsePart(id="bad-1", name="fail_tool", inputs={}),
        ],
        [ok, fail],
    )
    assert ok._calls == 1  # type: ignore[attr-defined]
    assert fail._calls == 1  # type: ignore[attr-defined]
    assert results[0].is_error is False
    assert results[1].is_error is True
    assert results[1].tool_use_id == "bad-1"

    history = _project_current_messages(live)
    edited, applied = edit_context(history, _trim_policy(keep_last_n=1))
    assert applied is not None
    parts = _result_parts(edited)
    assert parts[0].is_error is False
    assert parts[0].parts[0].text == _PLACEHOLDER
    assert parts[1].is_error is True
    assert "[tool_execution_error]" in parts[1].parts[0].text
    _assert_ids_paired(edited)


def test_failed_error_text_cleared_when_outside_keep_window() -> None:
    """Oracle: detailed error text is lost when the pair is trimmed."""
    history: list[Any] = []
    history.extend(
        _pair(
            "err-old",
            "[tool_execution_error] The tool failed unexpectedly.",
            name="fail_tool",
            is_error=True,
        )
    )
    history.extend(_pair("ok-new", str({"status": "ok"}), name="ok_tool", is_error=False))
    edited, applied = edit_context(history, _trim_policy(keep_last_n=1))
    assert applied is not None
    blob = _context_blob(edited)
    assert _PLACEHOLDER in blob
    assert "[tool_execution_error]" not in blob
    assert _result_parts(edited)[0].is_error is True


def test_error_fact_summarized_when_span_cleared() -> None:
    marker = "[DOBBY-FACT:error:upstream=HTTP 503 timeout]"
    history: list[Any] = []
    history.extend(
        _pair(
            "e1",
            f"[tool_execution_error] failed\n{marker}",
            name="fail_tool",
            is_error=True,
        )
    )
    history.extend(_pair("e2", str({"status": "ok"}), name="ok_tool"))
    applied = asyncio.run(
        summarize_context(history, _trim_policy(keep_last_n=1, mode="summarize"), _fact_summarizer())
    )
    assert applied is not None
    assert marker in applied.summary_text


def test_execution_error_then_model_correction() -> None:
    """Failure does not block a later correction attempt with a new id."""
    fail = _FailTool()
    body = _BodyRetryTool()
    typed = _TypedTool()
    call_count = 0
    turns = [
        [CurrentToolUsePart(id="f1", name="fail_tool", inputs={})],
        [CurrentToolUsePart(id="b1", name="body_retry", inputs={})],
        [CurrentToolUsePart(id="t1", name="typed", inputs={"count": 2})],
        [],
    ]

    async def mock_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        parts = turns[call_count - 1] if call_count <= len(turns) else []

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
    executor = CurrentAgentExecutor(
        "openai", provider, tools=[fail, body, typed]
    )
    async def run() -> list[ToolResultEvent]:
        out: list[ToolResultEvent] = []
        async for event in executor.run_stream([], max_model_corrections=3):
            if isinstance(event, ToolResultEvent):
                out.append(event)
        return out

    events = asyncio.run(run())
    assert fail._calls == 1  # type: ignore[attr-defined]
    assert body._calls == 1  # type: ignore[attr-defined]
    assert typed._calls == 1  # type: ignore[attr-defined]
    assert [event.tool_use_id for event in events] == ["f1", "b1", "t1"]
    assert events[0].is_error is True
    assert events[1].is_error is True
    assert str(events[1].result).startswith("[tool_retry]")
    assert events[2].is_error is False


def test_multiple_errors_trim_keep_last_n() -> None:
    history: list[Any] = []
    for index in range(3):
        history.extend(
            _pair(
                f"err-{index}",
                f"[tool_execution_error] fail-{index}",
                name="fail_tool",
                is_error=True,
            )
        )
    history.extend(_pair("win", "success", name="ok_tool", is_error=False))
    edited, applied = edit_context(history, _trim_policy(keep_last_n=2))
    assert applied is not None
    parts = _result_parts(edited)
    assert [part.is_error for part in parts] == [True, True, True, False]
    assert parts[0].parts[0].text == _PLACEHOLDER
    assert parts[1].parts[0].text == _PLACEHOLDER
    assert "fail-2" in parts[2].parts[0].text
    assert parts[3].parts[0].text == "success"


def test_parallel_mixed_success_and_failure() -> None:
    ok = _OkTool()
    fail = _FailTool()
    results, live = _run_batch(
        [
            CurrentToolUsePart(id="p-ok", name="ok_tool", inputs={}),
            CurrentToolUsePart(id="p-bad", name="fail_tool", inputs={}),
        ],
        [ok, fail],
    )
    assert [event.is_error for event in results] == [False, True]
    projected = _project_current_messages(live)
    edited, applied = edit_context(projected, _trim_policy(keep_last_n=2))
    assert applied is None
    assert [part.is_error for part in _result_parts(edited)] == [False, True]
    assert ok._calls == 1  # type: ignore[attr-defined]
    assert fail._calls == 1  # type: ignore[attr-defined]
    _assert_ids_paired(edited)


def test_streaming_error_paired_and_trimmed() -> None:
    tool = _StreamFailTool()
    results, live = _run_batch(
        [CurrentToolUsePart(id="sf-1", name="stream_fail", inputs={})],
        [tool],
    )
    assert tool._calls == 1  # type: ignore[attr-defined]
    assert results[0].is_error is True
    assert results[0].tool_use_id == "sf-1"
    projected = _project_current_messages(live)
    assert _result_parts(projected)[0].is_error is True
    edited, applied = edit_context(projected, _trim_policy(keep_last_n=0))
    assert applied is not None
    part = _result_parts(edited)[0]
    assert part.tool_use_id == "sf-1"
    assert part.is_error is True
    assert part.parts[0].text == _PLACEHOLDER


def test_unknown_tool_correction_error_then_trim() -> None:
    """Missing tool emits correction error; trim keeps id and ``is_error=True``."""
    typed = _TypedTool()
    results, live = _run_batch(
        [
            CurrentToolUsePart(id="missing", name="not_registered", inputs={}),
            CurrentToolUsePart(id="typed-1", name="typed", inputs={"count": 1}),
        ],
        [typed],
    )
    assert results[0].is_error is True
    assert str(results[0].result).startswith("[tool_not_found]")
    assert results[1].is_error is False
    edited, applied = edit_context(_project_current_messages(live), _trim_policy(keep_last_n=1))
    assert applied is not None
    parts = _result_parts(edited)
    assert parts[0].is_error is True
    assert parts[0].parts[0].text == _PLACEHOLDER
    assert parts[1].is_error is False


def test_compaction_does_not_reexecute_failed_tools() -> None:
    """Failure counted once; later compaction noop does not call fail tool again."""
    fail = _FailTool()
    _results, live = _run_batch(
        [CurrentToolUsePart(id="f1", name="fail_tool", inputs={})],
        [fail],
    )
    assert _results[0].is_error is True
    history = _project_current_messages(live)
    policy = _trim_policy()
    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="n1", name="noop", inputs={})], _usage(trigger)),
        ([ToolUsePart(id="n2", name="noop", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, _captured = _drive_recovered(
        policy, turns, history, tools=[_NoopTool()]
    )
    assert fail._calls == 1  # type: ignore[attr-defined]
    assert any(isinstance(event, ContextEditEvent) for event in events)


def test_usable_context_after_error_and_compaction() -> None:
    """Kept success payload remains readable in the post-compaction send list."""
    ok = _OkTool()
    fail = _FailTool()
    results, live = _run_batch(
        [
            CurrentToolUsePart(id="bad", name="fail_tool", inputs={}),
            CurrentToolUsePart(id="good", name="ok_tool", inputs={}),
        ],
        [fail, ok],
    )
    assert results[1].is_error is False
    history = _project_current_messages(live)
    policy = _trim_policy(keep_last_n=2)
    trigger = policy.trigger_tokens
    turns = [
        ([ToolUsePart(id="n1", name="noop", inputs={})], _usage(trigger)),
        ([], _usage(trigger)),
    ]
    events, captured = _drive_recovered(
        policy, turns, history, tools=[_NoopTool()]
    )
    assert any(isinstance(event, ContextEditEvent) for event in events)
    assert str({"status": "ok"}) in _context_blob(captured[-1])
    assert fail._calls == 1  # type: ignore[attr-defined]
    assert ok._calls == 1  # type: ignore[attr-defined]
