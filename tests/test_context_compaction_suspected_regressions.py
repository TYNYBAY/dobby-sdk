"""Focused regressions for four suspected context-compaction bugs.

These tests call the shipped executor, ``_find_tool_pairs`` / trim / summarize,
and the provider converters. They do not change production behavior.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

from dobby import AgentExecutor
from dobby.context import ContextPolicy, edit_context, summarize_context
from dobby.context.edit import _find_tool_pairs
from dobby.providers import (
    to_anthropic_messages,
    to_gemini_messages,
    to_openai_messages,
    to_vertexai_messages,
)
from dobby.tools import CompactContextTool, Tool
from dobby.types import (
    AssistantMessagePart,
    StreamEndEvent,
    TextPart,
    ToolResultEvent,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart,
)

_SUMMARY = "digest-kept"


def _usage() -> Usage:
    return Usage(input_tokens=0, output_tokens=0, total_tokens=0)


def _policy(*, keep_last_n: int) -> ContextPolicy:
    return ContextPolicy(
        context_window=128_000,
        trigger_pct=0.8,
        keep_last_n=keep_last_n,
        mode="summarize",
    )


def _pair(call_id: str, name: str, result: str) -> tuple[AssistantMessagePart, UserMessagePart]:
    return (
        AssistantMessagePart(parts=[ToolUsePart(id=call_id, name=name, inputs={"q": call_id})]),
        UserMessagePart(
            parts=[
                ToolResultPart(
                    tool_use_id=call_id,
                    name=name,
                    parts=[TextPart(text=result)],
                )
            ]
        ),
    )


def _uses(messages: list[Any]) -> list[ToolUsePart]:
    found: list[ToolUsePart] = []
    for message in messages:
        if isinstance(message, AssistantMessagePart):
            found.extend(part for part in message.parts if isinstance(part, ToolUsePart))
    return found


def _results_parts(messages: list[Any]) -> list[ToolResultPart]:
    found: list[ToolResultPart] = []
    for message in messages:
        if isinstance(message, UserMessagePart):
            found.extend(part for part in message.parts if isinstance(part, ToolResultPart))
    return found


def _result_text(part: ToolResultPart) -> str:
    return "".join(inner.text for inner in part.parts if isinstance(inner, TextPart))


class _Scripted:
    """One streaming agent turn per scripted response; ``stream=False`` summarizes."""

    name = "scripted"

    def __init__(self, turns: list[list[Any]], *, summary_text: str = _SUMMARY) -> None:
        self.turns = turns
        self.summary_text = summary_text
        self.agent_calls: list[list[Any]] = []
        self._index = 0

    async def chat(self, messages: list[Any], **kwargs: Any) -> Any:
        if kwargs.get("stream") is False:
            return StreamEndEvent(
                model="summarizer",
                parts=[TextPart(text=self.summary_text)],
                stop_reason="end_turn",
                usage=_usage(),
            )
        self.agent_calls.append(list(messages))
        parts = self.turns[min(self._index, len(self.turns) - 1)]
        self._index += 1

        async def stream() -> Any:
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=_usage(),
            )

        return stream()


def _drive(
    turns: list[list[Any]],
    messages: list[Any],
    tools: list[Tool],
    *,
    keep_last_n: int,
) -> tuple[list[Any], _Scripted]:
    provider = _Scripted(turns)
    executor = AgentExecutor(
        "openai",
        provider,  # type: ignore[arg-type]
        tools=tools,
        context_policy=_policy(keep_last_n=keep_last_n),
    )

    async def run() -> list[Any]:
        collected: list[Any] = []
        async for event in executor.run_stream(messages, max_iterations=len(turns)):
            collected.append(event)
        return collected

    return asyncio.run(run()), provider


def _compact_call(call_id: str, keep_last_n: int) -> ToolUsePart:
    return ToolUsePart(
        id=call_id,
        name="compact_context",
        inputs={"instructions": "keep ids", "keep_last_n": keep_last_n},
    )


@dataclass
class _Echo(Tool):
    name = "echo"
    description = "Return the given value."

    def __call__(self, value: str) -> str:
        return value


def test_keep_last_n_zero_keeps_compact_pair_and_patches_host_result() -> None:
    """``compact_context(keep_last_n=0)`` must leave its own round-trip visible and patched."""
    history = [UserMessagePart(parts=[TextPart(text="question")])]
    for index in range(2):
        use, result = _pair(f"h{index}", "search", f"payload-{index}")
        history.extend((use, result))
    events, provider = _drive(
        [[_compact_call("compact-1", 0)], []],
        history,
        [CompactContextTool()],
        keep_last_n=3,
    )
    yielded = next(
        event
        for event in events
        if isinstance(event, ToolResultEvent) and event.name == "compact_context"
    )
    visible = provider.agent_calls[1]
    uses = [part for part in _uses(visible) if part.name == "compact_context"]
    results = [part for part in _results_parts(visible) if part.name == "compact_context"]
    stored = _result_text(results[0]) if results else None

    assert yielded.result["status"] == "context_compacted"
    assert uses and results, (
        "compact_context tool-use/tool-result pair is missing from the next model-visible "
        f"history; yielded host result was {yielded.result!r} and stored result was {stored!r}"
    )
    assert uses[0].id == results[0].tool_use_id == yielded.tool_use_id
    assert stored == str(yielded.result)
    assert "payload-0" not in str(visible)
    assert "payload-1" not in str(visible)
    assert any("<summary>digest-kept</summary>" in str(message) for message in visible)


def test_same_turn_sibling_survives_compact_keep_last_n_one() -> None:
    """A normal tool call emitted with ``compact_context`` must still be visible afterwards."""
    events, provider = _drive(
        [
            [
                ToolUsePart(id="echo-1", name="echo", inputs={"value": "sibling-payload"}),
                _compact_call("compact-1", 1),
            ],
            [],
        ],
        [UserMessagePart(parts=[TextPart(text="question")])],
        [_Echo(), CompactContextTool()],
        keep_last_n=1,
    )
    visible = provider.agent_calls[1]
    echo_uses = [part for part in _uses(visible) if part.id == "echo-1"]
    echo_results = [part for part in _results_parts(visible) if part.tool_use_id == "echo-1"]
    compact_uses = [part for part in _uses(visible) if part.id == "compact-1"]
    compact_results_parts = [
        part for part in _results_parts(visible) if part.tool_use_id == "compact-1"
    ]
    compact_events = [
        event
        for event in events
        if isinstance(event, ToolResultEvent) and event.name == "compact_context"
    ]

    assert compact_events
    assert compact_events[0].result == {
        "status": "context_unchanged",
        "reason": "nothing_to_compact",
    }
    assert compact_uses and compact_results_parts
    assert (
        compact_uses[0].id == compact_results_parts[0].tool_use_id == compact_events[0].tool_use_id
    )
    assert _result_text(compact_results_parts[0]) == str(compact_events[0].result)
    assert echo_uses, "sibling tool use echo-1 was removed from the next model-visible history"
    assert echo_results, (
        "sibling tool result echo-1 was removed from the next model-visible history"
    )
    assert _result_text(echo_results[0]) == "sibling-payload"


def test_same_turn_sibling_does_not_change_keep_last_n_for_older_history() -> None:
    """``keep_last_n=1`` still summarizes older pairs while keeping this turn's tools."""
    history = [UserMessagePart(parts=[TextPart(text="question")])]
    for label, payload in (("old-a", "stale-payload"), ("old-b", "recent-old-payload")):
        use, result = _pair(label, "search", payload)
        history.extend((use, result))
    events, provider = _drive(
        [
            [
                ToolUsePart(id="echo-1", name="echo", inputs={"value": "sibling-payload"}),
                _compact_call("compact-1", 1),
            ],
            [],
        ],
        history,
        [_Echo(), CompactContextTool()],
        keep_last_n=1,
    )
    visible = provider.agent_calls[1]
    compact_event = next(
        event
        for event in events
        if isinstance(event, ToolResultEvent) and event.name == "compact_context"
    )
    assert compact_event.result["status"] == "context_compacted"
    assert any("<summary>digest-kept</summary>" in str(message) for message in visible)
    assert [part.id for part in _uses(visible)] == ["old-b", "echo-1", "compact-1"]
    assert [part.tool_use_id for part in _results_parts(visible)] == [
        "old-b",
        "echo-1",
        "compact-1",
    ]
    assert _result_text(_results_parts(visible)[1]) == "sibling-payload"
    assert "stale-payload" not in str(visible)
    assert "recent-old-payload" in str(visible)


def test_find_tool_pairs_matches_ids_across_trim_and_summarize() -> None:
    """ToolUse A, ToolUse B, ToolResult A, ToolResult B must pair by id on both edit paths."""
    use_a, result_a = _pair("A", "alpha", "result-A")
    use_b, result_b = _pair("B", "beta", "result-B")
    messages: list[Any] = [use_a, use_b, result_a, result_b]
    problems: list[str] = []

    actual: list[tuple[str, str]] = []
    for use_index, result_index in _find_tool_pairs(messages):
        use = next(part for part in messages[use_index].parts if isinstance(part, ToolUsePart))
        result = next(
            part for part in messages[result_index].parts if isinstance(part, ToolResultPart)
        )
        actual.append((use.id, result.tool_use_id))
    if actual != [("A", "A"), ("B", "B")]:
        problems.append(f"_find_tool_pairs returned {actual}, expected [('A', 'A'), ('B', 'B')]")

    adjacent = [_pair("C", "gamma", "result-C")[0], _pair("C", "gamma", "result-C")[1]]
    if _find_tool_pairs(adjacent) != [(0, 1)]:
        problems.append(f"adjacent executor pair was not preserved: {_find_tool_pairs(adjacent)}")

    use_only, _ = _pair("A", "alpha", "result-A")
    _, result_only = _pair("B", "beta", "result-B")
    if _find_tool_pairs([use_only, result_only]) != []:
        problems.append("unrelated ToolUse A / ToolResult B were paired")

    edited, applied = edit_context(messages, _policy(keep_last_n=0))
    edited_by_id = {part.tool_use_id: _result_text(part) for part in _results_parts(edited)}
    cleared = None if applied is None else applied.cleared_tool_uses
    if cleared != 2 or edited_by_id != {
        "A": "[Tool result cleared to save context.]",
        "B": "[Tool result cleared to save context.]",
    }:
        problems.append(
            f"trim keep_last_n=0 cleared {cleared} result(s) with texts {edited_by_id}"
        )

    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value=StreamEndEvent(
            model="summarizer",
            parts=[TextPart(text=_SUMMARY)],
            stop_reason="end_turn",
            usage=None,
        )
    )
    summarized = [use_a, use_b, result_a, result_b]
    asyncio.run(summarize_context(summarized, _policy(keep_last_n=0), llm))
    span = llm.chat.await_args.args[0][0].parts[0].text
    alpha_call = span.find("[tool call alpha")
    alpha_result = span.find("[tool result alpha: result-A]")
    beta_call = span.find("[tool call beta")
    beta_result = span.find("[tool result beta: result-B]")
    matched_span = (
        alpha_call != -1
        and alpha_result != -1
        and beta_call != -1
        and beta_result != -1
        and alpha_call < alpha_result
        and beta_call < beta_result
    )
    if not matched_span:
        problems.append(f"summarize span did not pair A with A and B with B: {span!r}")
    orphan_uses = [part.id for part in _uses(summarized)]
    orphan_results = [part.tool_use_id for part in _results_parts(summarized)]
    if orphan_uses or orphan_results:
        problems.append(
            "summarize left tool messages "
            f"uses={orphan_uses} results={orphan_results} instead of matching both pairs"
        )

    assert problems == []


def _compacted_history_with_retained_turns() -> list[Any]:
    """History ``summarize_context`` writes when text between old pairs is retained."""
    use_a, result_a = _pair("A", "search", "result-A")
    use_b, result_b = _pair("B", "search", "result-B")
    use_c, result_c = _pair("id-C", "search", "result-C")
    messages: list[Any] = [
        UserMessagePart(parts=[TextPart(text="user question")]),
        use_a,
        result_a,
        UserMessagePart(parts=[TextPart(text="NEVER-DELETE")]),
        AssistantMessagePart(parts=[TextPart(text="assistant note")]),
        use_b,
        result_b,
        use_c,
        result_c,
    ]
    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value=StreamEndEvent(
            model="summarizer",
            parts=[TextPart(text=_SUMMARY)],
            stop_reason="end_turn",
            usage=None,
        )
    )
    applied = asyncio.run(summarize_context(messages, _policy(keep_last_n=1), llm))
    assert applied is not None
    assert applied.type == "summarize"
    return messages


def _openai_problems(history: list[Any]) -> list[str]:
    try:
        items = to_openai_messages(history)
    except Exception as exc:
        return [f"openai converter raised {type(exc).__name__}: {exc}"]
    calls = [item for item in items if item.get("type") == "function_call"]
    outputs = [item for item in items if item.get("type") == "function_call_output"]
    call_ids = {item["call_id"] for item in calls}
    output_ids = {item["call_id"] for item in outputs}
    problems: list[str] = []
    blob = str(items)
    if call_ids != {"id-C"} or output_ids != {"id-C"}:
        problems.append(f"openai tool ids calls={call_ids} outputs={output_ids}")
    else:
        body = str(outputs[0]["output"])
        if "result-C" not in body:
            problems.append(f"openai dropped result-C: {body!r}")
    if _SUMMARY not in blob:
        problems.append("openai dropped the compaction summary")
    if "NEVER-DELETE" not in blob:
        problems.append("openai dropped NEVER-DELETE")
    return problems


def _anthropic_problems(history: list[Any]) -> list[str]:
    try:
        messages = to_anthropic_messages(history)
    except Exception as exc:
        return [f"anthropic converter raised {type(exc).__name__}: {exc}"]
    roles = [message["role"] for message in messages]
    problems: list[str] = []
    if any(roles[index] == roles[index + 1] for index in range(len(roles) - 1)):
        problems.append(f"anthropic roles do not alternate: {roles}")
    uses = [
        block
        for message in messages
        for block in message["content"]
        if block.get("type") == "tool_use"
    ]
    results = [
        block
        for message in messages
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    blob = str(messages)
    if [block["id"] for block in uses] != ["id-C"] or [
        block["tool_use_id"] for block in results
    ] != ["id-C"]:
        problems.append(f"anthropic tool pairing uses={uses} results={results}")
    elif "result-C" not in str(results[0]["content"]):
        problems.append(f"anthropic dropped result-C: {results[0]!r}")
    if _SUMMARY not in blob:
        problems.append("anthropic dropped the compaction summary")
    if "NEVER-DELETE" not in blob:
        problems.append("anthropic dropped NEVER-DELETE")
    return problems


def _gemini_problems(history: list[Any]) -> list[str]:
    try:
        contents = to_gemini_messages(history)
    except Exception as exc:
        return [f"gemini converter raised {type(exc).__name__}: {exc}"]
    roles = [content.role for content in contents]
    problems: list[str] = []
    if any(roles[index] == roles[index + 1] for index in range(len(roles) - 1)):
        problems.append(f"gemini roles do not alternate: {roles}")
    paired = False
    texts: list[str] = []
    for index, content in enumerate(contents):
        for part in content.parts or []:
            if getattr(part, "text", None):
                texts.append(part.text)
            call = getattr(part, "function_call", None)
            response = getattr(part, "function_response", None)
            if call is not None and call.name == "search" and index + 1 < len(contents):
                next_response = next(
                    (
                        inner.function_response
                        for inner in contents[index + 1].parts or []
                        if inner.function_response
                    ),
                    None,
                )
                if next_response is not None and "result-C" in str(next_response.response):
                    paired = True
            if response is not None:
                texts.append(str(response.response))
    blob = "\n".join(texts)
    if not paired:
        problems.append("gemini did not keep search call id-C adjacent to result-C")
    if _SUMMARY not in blob:
        problems.append("gemini dropped the compaction summary")
    if "NEVER-DELETE" not in blob:
        problems.append("gemini dropped NEVER-DELETE")
    return problems


def _vertex_problems(history: list[Any]) -> list[str]:
    try:
        messages = to_vertexai_messages(history)
    except Exception as exc:
        return [f"vertex converter raised {type(exc).__name__}: {exc}"]
    problems: list[str] = []
    call_ids: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            call_ids.append(call["id"])
    result_ids = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
    blob = str(messages)
    if call_ids != ["id-C"] or result_ids != ["id-C"]:
        problems.append(f"vertex tool ids calls={call_ids} results={result_ids}")
    else:
        body = next(message["content"] for message in messages if message["role"] == "tool")
        if body != "result-C":
            problems.append(f"vertex dropped result-C: {body!r}")
    if _SUMMARY not in blob:
        problems.append("vertex dropped the compaction summary")
    if "NEVER-DELETE" not in blob:
        problems.append("vertex dropped NEVER-DELETE")
    return problems


def test_compacted_history_converts_on_openai_anthropic_gemini_and_vertex() -> None:
    """A summarize write-back must convert on all four providers with tool ids intact."""
    history = _compacted_history_with_retained_turns()
    problems = [
        *_openai_problems(history),
        *_anthropic_problems(history),
        *_gemini_problems(history),
        *_vertex_problems(history),
    ]
    assert problems == []
