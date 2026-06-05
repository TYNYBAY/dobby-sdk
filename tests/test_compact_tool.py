"""Tests for CompactContextTool and the Tool.edits_context flag (U5)."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from dobby import AgentExecutor
from dobby.context import ContextPolicy
from dobby.tools import CompactContextTool, Tool
from dobby.types import (
    AssistantMessagePart,
    ContextEditEvent,
    StreamEndEvent,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart,
)

# --- Helpers ---


def _usage(input_tokens: int) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _tool_pair(call_id: str, text: str = "payload" * 20):
    use = AssistantMessagePart(parts=[ToolUsePart(id=call_id, name="search", inputs={})])
    result = UserMessagePart(
        parts=[ToolResultPart(tool_use_id=call_id, name="search", parts=[TextPart(text=text)])]
    )
    return use, result


@dataclass
class _NoopTool(Tool):
    name = "noop"
    description = "A no-op tool."

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


def _make_provider(turns, captured, summarizer_calls, summary_text="DIGEST"):
    """Mock provider: streams agent turns; answers stream=False summarizer calls."""
    call_count = 0

    async def mock_chat(messages, *args, stream=True, **kwargs):
        nonlocal call_count
        if not stream:
            summarizer_calls.append({"system_prompt": kwargs.get("system_prompt")})
            return StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=[TextPart(text=summary_text)],
                stop_reason="end_turn",
                usage=_usage(0),
            )

        idx = min(call_count, len(turns) - 1)
        call_count += 1
        captured.append(list(messages))
        parts, usage = turns[idx]

        async def gen():
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=usage,
            )

        return gen()

    provider = AsyncMock()
    provider.chat = mock_chat
    return provider


async def _collect(executor, messages):
    events = []
    async for event in executor.run_stream(messages=messages, system_prompt=None):
        events.append(event)
    return events


def _history(n: int = 3) -> list:
    msgs = [UserMessagePart(parts=[TextPart(text="question")])]
    for k in range(n):
        msgs.extend(_tool_pair(f"h{k}"))
    return msgs


# --- edits_context flag (pure dataclass, no LLM) ---


class TestEditsContextFlag:
    def test_compact_tool_sets_flag(self) -> None:
        assert CompactContextTool.edits_context is True
        assert CompactContextTool().edits_context is True

    def test_normal_tool_defaults_false(self) -> None:
        assert _NoopTool.edits_context is False

    def test_non_bool_edits_context_raises(self) -> None:
        with pytest.raises(TypeError):

            @dataclass
            class BadTool(Tool):
                name = "bad"
                description = "Bad edits_context"
                edits_context = "yes"  # type: ignore[assignment]

                def __call__(self) -> dict:
                    return {}


# --- CompactContextTool routed through the executor ---


class TestCompactToolExecution:
    def test_invoking_compact_triggers_summarize(self) -> None:
        captured: list = []
        summarizer_calls: list = []
        turns = [
            (
                [
                    ToolUsePart(
                        id="c1", name="compact_context", inputs={"instructions": "keep IDs"}
                    )
                ],
                _usage(10),
            ),
            ([], _usage(10)),
        ]
        provider = _make_provider(turns, captured, summarizer_calls)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            tools=[CompactContextTool()],
            context_policy=ContextPolicy(context_window=100_000, keep_last_n=1, mode="summarize"),
        )
        events = asyncio.run(_collect(executor, _history()))

        edits = [e for e in events if isinstance(e, ContextEditEvent)]
        assert len(edits) == 1
        assert edits[0].applied_edits[0].type == "summarize"
        assert len(summarizer_calls) == 1

    def test_instructions_reach_summarizer_prompt(self) -> None:
        captured: list = []
        summarizer_calls: list = []
        turns = [
            (
                [
                    ToolUsePart(
                        id="c1",
                        name="compact_context",
                        inputs={"instructions": "preserve the API keys"},
                    )
                ],
                _usage(10),
            ),
            ([], _usage(10)),
        ]
        provider = _make_provider(turns, captured, summarizer_calls)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            tools=[CompactContextTool()],
            context_policy=ContextPolicy(context_window=100_000, keep_last_n=1, mode="summarize"),
        )
        asyncio.run(_collect(executor, _history()))

        assert "preserve the API keys" in summarizer_calls[0]["system_prompt"]

    def test_guard_prevents_double_compaction(self) -> None:
        captured: list = []
        summarizer_calls: list = []
        # Turn 1 grows context (high usage). Turn 2: auto-trigger fires AND the
        # model also calls compact_context — the guard must yield exactly one.
        turns = [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(150_000)),
            (
                [ToolUsePart(id="c1", name="compact_context", inputs={"instructions": "x"})],
                _usage(150_000),
            ),
            ([], _usage(150_000)),
        ]
        provider = _make_provider(turns, captured, summarizer_calls)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            tools=[_NoopTool(), CompactContextTool()],
            context_policy=ContextPolicy(
                context_window=100_000, trigger_pct=0.8, keep_last_n=1, mode="summarize"
            ),
        )
        events = asyncio.run(_collect(executor, _history()))

        assert len([e for e in events if isinstance(e, ContextEditEvent)]) == 1
        assert len(summarizer_calls) == 1

    def test_noop_auto_trigger_does_not_suppress_explicit_compact(self) -> None:
        captured: list = []
        summarizer_calls: list = []
        # Turn 2 the auto-trigger fires but is a NO-OP (keep_last_n=10 > pairs), so it
        # must NOT block the explicit compact_context call (which overrides keep_last_n=1).
        turns = [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(150_000)),
            (
                [
                    ToolUsePart(
                        id="c1",
                        name="compact_context",
                        inputs={"instructions": "x", "keep_last_n": 1},
                    )
                ],
                _usage(150_000),
            ),
            ([], _usage(150_000)),
        ]
        provider = _make_provider(turns, captured, summarizer_calls)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            tools=[_NoopTool(), CompactContextTool()],
            context_policy=ContextPolicy(
                context_window=100_000, trigger_pct=0.8, keep_last_n=10, mode="summarize"
            ),
        )
        events = asyncio.run(_collect(executor, _history()))

        # The no-op auto path latched nothing, so the explicit request still runs.
        assert len([e for e in events if isinstance(e, ContextEditEvent)]) == 1
        assert len(summarizer_calls) == 1
