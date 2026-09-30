# ruff: noqa: E402
"""Trigger and threshold tests for the recovered context-compaction implementation.

The subject is the historical compaction package recovered in the
``dobby-compaction-94b5a8f`` worktree. It is loaded under ``recovered_dobby``
so the current tree's executor is left untouched. Responses are scripted.
"""

# isort: off
from __future__ import annotations

import asyncio
import importlib.util
import sys
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError


def _recovered_root() -> Path:
    """Return the worktree that holds the recovered compaction sources."""
    repo = Path(__file__).resolve().parents[1]
    pointer = repo / ".git" / "worktrees" / "dobby-compaction-94b5a8f" / "gitdir"
    git_path = Path(pointer.read_text(encoding="utf-8").strip())
    return git_path.parent


def _load_recovered() -> Any:
    """Import the recovered package without replacing the installed ``dobby``."""
    name = "recovered_dobby"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
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


_RECOVERED = _load_recovered()

from recovered_dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from recovered_dobby.context._tokens import estimate_input_tokens
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

# Configured percentages. The oracle is decimal ``pct * context_window``,
# not ``int(pct * context_window)``.
_PCTS: tuple[tuple[str, str], ...] = (
    ("0%", "0"),
    ("1%", "0.01"),
    ("50%", "0.50"),
    ("79.99%", "0.7999"),
    ("80%", "0.80"),
    ("80.01%", "0.8001"),
    ("99%", "0.99"),
    ("100%", "1"),
)
_WINDOWS: tuple[tuple[str, int], ...] = (
    ("small", 1),
    ("normal", 128_000),
    ("large", 10**18),
)
_PLACEHOLDER = "[Tool result cleared to save context.]"


class _NoopTool(Tool):
    name = "noop"
    description = "Return a tiny acknowledgement."

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


def _huge_tool(chars: int) -> Tool:
    """Build a tool whose string result is large enough to cross ``chars``."""

    class _HugeTool(Tool):
        name = "huge"
        description = "Return a large blob."

        def __call__(self) -> dict[str, str]:
            return {"blob": "H" * chars}

    return _HugeTool()


def _usage(input_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _tool_call(call_id: str, name: str = "noop") -> ToolUsePart:
    return ToolUsePart(id=call_id, name=name, inputs={})


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


def _configured_line(pct_text: str, window: int) -> Decimal:
    return Decimal(pct_text) * window


def _at_line(pct_text: str, window: int) -> int:
    """Smallest integer token count on or above the configured percentage."""
    return int(_configured_line(pct_text, window).to_integral_value(rounding=ROUND_CEILING))


def _boundary_cases() -> list[Any]:
    cases: list[Any] = []
    for window_name, window in _WINDOWS:
        for pct_label, pct_text in _PCTS:
            exactly = _at_line(pct_text, window)
            points: list[tuple[str, int, bool]] = [("exactly", exactly, True)]
            if exactly > 0:
                points.append(("just-below", exactly - 1, False))
            if exactly > 1:
                points.append(("below", 0, False))
            points.append(("just-above", exactly + 1, True))
            points.append(("far-above", exactly + window + 1, True))
            for case_name, tokens, expect in points:
                cases.append(
                    pytest.param(
                        window_name,
                        window,
                        pct_label,
                        pct_text,
                        case_name,
                        tokens,
                        expect,
                        id=f"{window_name}-{pct_label}-{case_name}",
                    )
                )
    return cases


def _first_iteration_cases() -> list[Any]:
    cases: list[Any] = []
    for window_name, window in _WINDOWS:
        for pct_label, pct_text in _PCTS:
            cases.append(
                pytest.param(
                    window_name,
                    window,
                    pct_label,
                    pct_text,
                    id=f"{window_name}-{pct_label}",
                )
            )
    return cases


def _require_policy(window: int, pct_text: str, **overrides: Any) -> ContextPolicy:
    pct = float(pct_text)
    try:
        return ContextPolicy(context_window=window, trigger_pct=pct, **overrides)
    except ValidationError as exc:
        pytest.fail(
            f"ContextPolicy rejected trigger_pct={pct_text} ({pct}) context_window={window}: {exc}"
        )


def _scripted(turns: list[tuple[list[Any], Usage | None]], captured: list[list[Any]]) -> Any:
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


def _summarizing(
    turns: list[tuple[list[Any], Usage | None]],
    captured: list[list[Any]],
    summary_calls: list[list[Any]],
    *,
    summary_error: BaseException | None = None,
) -> Any:
    call_count = 0

    async def mock_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        if kwargs.get("stream", True) is False:
            summary_calls.append(list(messages))
            if summary_error is not None:
                raise summary_error
            return StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=[TextPart(text="DIGEST")],
                stop_reason="end_turn",
                usage=_usage(0),
            )
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


def _drive(
    policy: ContextPolicy | None,
    turns: list[tuple[list[Any], Usage | None]],
    messages: list[Any],
    *,
    tools: list[Tool] | None = None,
    provider: Any | None = None,
    captured: list[list[Any]] | None = None,
) -> tuple[list[Any], list[list[Any]], AgentExecutor]:
    recorded: list[list[Any]] = [] if captured is None else captured
    llm = provider if provider is not None else _scripted(turns, recorded)
    kwargs: dict[str, Any] = {}
    if policy is not None:
        kwargs["context_policy"] = policy
    executor = AgentExecutor("openai", llm, tools=tools or [_NoopTool()], **kwargs)

    async def run() -> list[Any]:
        events: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=len(turns)):
            events.append(event)
        return events

    return asyncio.run(run()), recorded, executor


def _edits(events: list[Any]) -> list[ContextEditEvent]:
    return [event for event in events if isinstance(event, ContextEditEvent)]


def _compacted(messages: list[Any], placeholder: str) -> bool:
    for message in messages:
        if not isinstance(message, UserMessagePart):
            continue
        for part in message.parts:
            if not isinstance(part, ToolResultPart):
                continue
            for inner in part.parts:
                if isinstance(inner, TextPart) and inner.text == placeholder:
                    return True
    return False


def _assert_boundary(
    *,
    window_name: str,
    window: int,
    pct_label: str,
    pct_text: str,
    case_name: str,
    tokens: int,
    expect: bool,
    policy: ContextPolicy,
    events: list[Any],
    captured: list[list[Any]],
) -> None:
    assert len(captured) == 2
    edits = _edits(events)
    first = _compacted(captured[0], policy.placeholder)
    second = _compacted(captured[1], policy.placeholder)
    assert not first, f"{window_name} {pct_label} compacted on the first model turn"
    assert second is expect and (len(edits) == 1) is expect, (
        f"{window_name} context_window={window} trigger_pct={pct_label} "
        f"case={case_name} input_tokens={tokens} "
        f"configured_threshold={_configured_line(pct_text, window)} "
        f"production_trigger_tokens={policy.trigger_tokens} "
        f"compacted={second} edits={len(edits)} expected_compaction={expect}"
    )


@pytest.mark.parametrize(
    ("window_name", "window", "pct_label", "pct_text", "case_name", "tokens", "expect"),
    _boundary_cases(),
)
def test_threshold_boundary(
    window_name: str,
    window: int,
    pct_label: str,
    pct_text: str,
    case_name: str,
    tokens: int,
    expect: bool,
) -> None:
    """Compaction follows the configured percentage of the context window."""
    policy = _require_policy(window, pct_text, keep_last_n=1)
    turns = [
        ([_tool_call("t1")], _usage(tokens)),
        ([], _usage(tokens)),
    ]
    events, captured, _executor = _drive(policy, turns, _history())
    _assert_boundary(
        window_name=window_name,
        window=window,
        pct_label=pct_label,
        pct_text=pct_text,
        case_name=case_name,
        tokens=tokens,
        expect=expect,
        policy=policy,
        events=events,
        captured=captured,
    )


@pytest.mark.parametrize(
    ("window_name", "window", "pct_label", "pct_text"),
    _first_iteration_cases(),
)
def test_first_iteration_does_not_compact(
    window_name: str,
    window: int,
    pct_label: str,
    pct_text: str,
) -> None:
    """The first model turn has no previous usage and must not compact."""
    policy = _require_policy(window, pct_text, keep_last_n=1)
    far_above = _at_line(pct_text, window) + window + 1
    events, captured, _executor = _drive(policy, [([], _usage(far_above))], _history())
    assert len(captured) == 1
    assert _edits(events) == [], (
        f"{window_name} context_window={window} trigger_pct={pct_label} "
        "compacted on the first iteration"
    )
    assert not _compacted(captured[0], policy.placeholder)


@pytest.mark.parametrize(
    ("window_name", "window"),
    [pytest.param(name, window, id=name) for name, window in _WINDOWS],
)
def test_disabled_policy_does_not_compact(window_name: str, window: int) -> None:
    """Omitting the policy leaves every model request uncompacted."""
    turns = [
        ([_tool_call("t1")], _usage(window)),
        ([], _usage(window)),
    ]
    events, captured, _executor = _drive(None, turns, _history())
    assert _edits(events) == [], f"{window_name} compacted with no policy"
    assert all(not _compacted(call, _PLACEHOLDER) for call in captured)


def test_trim_repeats_while_usage_stays_at_threshold() -> None:
    """Trim fires again on every later turn that remains at the threshold."""
    policy = _require_policy(128_000, "0.80", keep_last_n=1)
    tokens = policy.trigger_tokens
    turns = [
        ([_tool_call("t1")], _usage(tokens)),
        ([_tool_call("t2")], _usage(tokens)),
        ([_tool_call("t3")], _usage(tokens)),
        ([], _usage(tokens)),
    ]
    events, captured, _executor = _drive(policy, turns, _history())
    assert len(captured) == 4
    assert not _compacted(captured[0], policy.placeholder)
    assert all(_compacted(captured[index], policy.placeholder) for index in (1, 2, 3))
    assert len(_edits(events)) == 3


def test_summarize_repeats_only_after_the_token_count_increases() -> None:
    """Summarize runs once per token-count, then again after the count grows."""
    policy = _require_policy(100_000, "0.80", keep_last_n=1, mode="summarize")
    captured: list[list[Any]] = []
    summary_calls: list[list[Any]] = []
    turns = [
        ([_tool_call("t1")], _usage(200_000)),
        ([_tool_call("t2")], _usage(200_000)),
        ([_tool_call("t3")], _usage(250_000)),
        ([], _usage(250_000)),
    ]
    provider = _summarizing(turns, captured, summary_calls)
    events, _recorded, _executor = _drive(
        policy,
        turns,
        _history(),
        provider=provider,
        captured=captured,
    )
    assert len(summary_calls) == 2
    assert len(_edits(events)) == 2


def test_failed_summarize_does_not_mark_context_compacted() -> None:
    """A summarizer exception must not stick on the executor."""
    policy = _require_policy(100_000, "0.80", keep_last_n=1, mode="summarize")
    messages = _history()
    snapshot = list(messages)
    captured: list[list[Any]] = []
    summary_calls: list[list[Any]] = []
    failing = _summarizing(
        [
            ([_tool_call("t1")], _usage(200_000)),
            ([], _usage(200_000)),
        ],
        captured,
        summary_calls,
        summary_error=RuntimeError("summarizer unavailable"),
    )
    executor_box: dict[str, AgentExecutor] = {}

    async def fail_run() -> None:
        kwargs = {"context_policy": policy}
        executor = AgentExecutor("openai", failing, tools=[_NoopTool()], **kwargs)
        executor_box["executor"] = executor
        async for _event in executor.run_stream(messages, max_iterations=2):
            pass

    with pytest.raises(RuntimeError, match="summarizer unavailable"):
        asyncio.run(fail_run())

    assert summary_calls
    assert messages == snapshot
    summary_calls_ok: list[list[Any]] = []
    recovered_capture: list[list[Any]] = []
    executor_box["executor"].llm = _summarizing(
        [
            ([_tool_call("t1")], _usage(200_000)),
            ([], _usage(200_000)),
        ],
        recovered_capture,
        summary_calls_ok,
    )

    async def recover_run() -> list[Any]:
        events: list[Any] = []
        async for event in executor_box["executor"].run_stream(messages, max_iterations=2):
            events.append(event)
        return events

    events = asyncio.run(recover_run())
    assert summary_calls_ok
    assert len(_edits(events)) == 1


def test_noop_summarize_does_not_mark_context_compacted() -> None:
    """A summarize that clears nothing must not suppress a later real one."""
    policy = _require_policy(100_000, "0.80", keep_last_n=5, mode="summarize")
    captured: list[list[Any]] = []
    summary_calls: list[list[Any]] = []
    tokens = 200_000
    turns = [
        ([_tool_call("t1")], _usage(tokens)),
        ([_tool_call("t2")], _usage(tokens)),
        ([], _usage(tokens)),
    ]
    provider = _summarizing(turns, captured, summary_calls)
    events, _recorded, _executor = _drive(
        policy,
        turns,
        _history(pairs=4),
        provider=provider,
        captured=captured,
    )
    assert len(summary_calls) == 1
    assert len(_edits(events)) == 1


@pytest.mark.parametrize(
    ("window", "pct_text", "previous"),
    [
        pytest.param(1, "1", 0, id="small-window-1-pct-100"),
        pytest.param(128_000, "0.80", 102_399, id="normal-window-128000-pct-80"),
        pytest.param(1_000_000, "0.80", 799_999, id="large-window-1000000-pct-80"),
    ],
)
def test_next_request_cannot_exceed_window_before_compaction(
    window: int,
    pct_text: str,
    previous: int,
) -> None:
    """An outgoing request must not already exceed the window before the trigger."""
    policy = _require_policy(window, pct_text, keep_last_n=0)
    assert previous < policy.trigger_tokens
    tool = _huge_tool((window + 10) * 4)
    messages = [UserMessagePart(parts=[TextPart(text="hi")])]
    turns = [
        ([_tool_call("c1", name="huge")], _usage(previous)),
        ([], _usage(previous)),
    ]
    events, captured, _executor = _drive(policy, turns, messages, tools=[tool])
    assert len(captured) == 2
    first_estimate = estimate_input_tokens(captured[0])
    second_estimate = estimate_input_tokens(captured[1])
    assert first_estimate <= window
    compacted = _compacted(captured[1], policy.placeholder) or bool(_edits(events))
    assert second_estimate <= window or compacted, (
        f"next request estimate={second_estimate} exceeds context_window={window} "
        f"before compaction; previous input_tokens={previous} "
        f"production_trigger_tokens={policy.trigger_tokens} compacted={compacted}"
    )
