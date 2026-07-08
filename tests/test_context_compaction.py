"""Tests for context compaction: ContextPolicy, ContextEditEvent, trim, summarize."""

import asyncio
from unittest.mock import AsyncMock

from pydantic import ValidationError
import pytest

from dobby import AgentExecutor
from dobby.context import ContextPolicy, edit_context, summarize_context
from dobby.tools import Tool
from dobby.types import (
    AppliedEdit,
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


def _tool_use_msg(call_id: str, name: str = "search", **metadata) -> AssistantMessagePart:
    """An assistant message carrying a single tool call."""
    return AssistantMessagePart(
        parts=[
            ToolUsePart(
                id=call_id,
                name=name,
                inputs={"q": call_id},
                metadata=metadata or None,
            )
        ]
    )


def _tool_result_msg(call_id: str, text: str, name: str = "search") -> UserMessagePart:
    """A user message carrying the matching tool result."""
    return UserMessagePart(
        parts=[ToolResultPart(tool_use_id=call_id, name=name, parts=[TextPart(text=text)])]
    )


def _tool_pair(call_id: str, text: str = "a large tool payload" * 20):
    """An (assistant tool-use, user tool-result) message pair."""
    return _tool_use_msg(call_id), _tool_result_msg(call_id, text)


def _usage(input_tokens: int) -> Usage:
    """A Usage with the given input_tokens."""
    return Usage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)


def _make_streaming_provider(turns: list[tuple[list, Usage | None]], captured: list):
    """A mock provider for streaming chat that replays a fixed sequence of turns.

    Each turn is ``(parts, usage)``: ``parts`` becomes a single StreamEndEvent's
    parts (non-empty parts → stop_reason 'tool_use' so the loop continues).
    ``captured`` records the message list passed to each chat call, in order.
    """
    call_count = 0

    async def mock_chat(messages, *args, **kwargs):
        nonlocal call_count
        idx = min(call_count, len(turns) - 1)
        call_count += 1
        captured.append(list(messages))
        parts, usage = turns[idx]

        async def stream():
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


async def _collect(executor, messages, **kwargs):
    """Run run_stream and collect all events."""
    events = []
    async for event in executor.run_stream(messages=messages, system_prompt=None, **kwargs):
        events.append(event)
    return events


def _summarizer_llm(summary_text: str = "DIGEST", calls: list | None = None):
    """A mock provider whose ``chat`` always returns a non-streaming summary value."""

    async def mock_chat(messages, *args, stream=True, **kwargs):
        if calls is not None:
            calls.append({"stream": stream, "system_prompt": kwargs.get("system_prompt")})
        return StreamEndEvent(
            type="stream_end",
            model="mock",
            parts=[TextPart(text=summary_text)],
            stop_reason="end_turn",
            usage=_usage(0),
        )

    provider = AsyncMock()
    provider.chat = mock_chat
    return provider


def _make_summarizing_provider(
    turns: list[tuple[list, Usage | None]],
    captured: list,
    summarizer_calls: list,
    summary_text: str = "DIGEST",
):
    """Mock provider that streams agent turns and answers ``stream=False`` summaries.

    Streaming calls replay ``turns`` and record their messages in ``captured``;
    non-streaming (summarizer) calls record into ``summarizer_calls`` and return a
    single StreamEndEvent value carrying ``summary_text``.
    """
    call_count = 0

    async def mock_chat(messages, *args, stream=True, **kwargs):
        nonlocal call_count
        if not stream:
            summarizer_calls.append(
                {"messages": list(messages), "system_prompt": kwargs.get("system_prompt")}
            )
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


class _NoopTool(Tool):
    name = "noop"
    description = "A no-op tool that returns a small result."

    def __call__(self) -> dict[str, str]:
        return {"ok": "yes"}


def _summary_message(messages) -> UserMessagePart | None:
    """The first ``<summary>`` user message in a list, if any."""
    for m in messages:
        if (
            isinstance(m, UserMessagePart)
            and m.parts
            and isinstance(m.parts[0], TextPart)
            and m.parts[0].text.startswith("<summary>")
        ):
            return m
    return None


# --- U1: ContextPolicy + ContextEditEvent types ---


class TestContextPolicy:
    """ContextPolicy defaults, the trigger_tokens knob, and validation."""

    def test_defaults(self) -> None:
        policy = ContextPolicy()
        assert policy.context_window == 128_000
        assert policy.trigger_pct == 0.8
        assert policy.keep_last_n == 3
        assert policy.mode == "trim"
        assert policy.trigger_tokens == 102_400  # 0.8 * 128_000

    def test_trigger_tokens_testing_knob(self) -> None:
        policy = ContextPolicy(context_window=2000, trigger_pct=0.5)
        assert policy.trigger_tokens == 1000

    def test_invalid_trigger_pct_zero(self) -> None:
        with pytest.raises(ValidationError):
            ContextPolicy(trigger_pct=0)

    def test_invalid_trigger_pct_above_one(self) -> None:
        with pytest.raises(ValidationError):
            ContextPolicy(trigger_pct=1.5)

    def test_invalid_context_window_zero(self) -> None:
        with pytest.raises(ValidationError):
            ContextPolicy(context_window=0)

    def test_invalid_keep_last_n_negative(self) -> None:
        with pytest.raises(ValidationError):
            ContextPolicy(keep_last_n=-1)


class TestContextEditEvent:
    """ContextEditEvent / AppliedEdit construction and public import surface."""

    def test_constructs_with_applied_edit(self) -> None:
        event = ContextEditEvent(
            applied_edits=[AppliedEdit(type="clear_tool_uses", cleared_tool_uses=2)]
        )
        assert event.type == "context_edit"
        assert event.applied_edits[0].cleared_tool_uses == 2

    def test_importable_from_public_types(self) -> None:
        from dobby.types import ContextEditEvent as PublicEvent

        assert PublicEvent is ContextEditEvent


class TestPublicImportSurface:
    """U6: the compaction surface is importable from the top-level package."""

    def test_top_level_imports(self) -> None:
        from dobby import CompactContextTool, ContextEditEvent, ContextPolicy

        assert ContextPolicy is not None
        assert ContextEditEvent is not None
        assert CompactContextTool is not None


# --- U2: edit_context() trim algorithm ---


class TestEditContextTrim:
    """The deterministic, turn-aware trimmer."""

    def test_keeps_last_n_clears_older(self) -> None:
        # 5 round-trips, keep_last_n=2 → 3 oldest results cleared.
        messages = []
        for k in range(5):
            messages.extend(_tool_pair(f"tc{k}", text=f"payload-{k}-xxxxxxxxxx"))
        policy = ContextPolicy(keep_last_n=2)

        new_messages, applied = edit_context(messages, policy)

        assert applied is not None
        assert applied.type == "clear_tool_uses"
        assert applied.cleared_tool_uses == 3

        # Result messages are at odd indices (1,3,5,7,9). The last two pairs
        # (indices 6/7 and 8/9) are kept verbatim; older results are placeholder.
        for result_idx in (1, 3, 5):
            part = new_messages[result_idx].parts[0]
            assert part.parts[0].text == policy.placeholder
        for result_idx in (7, 9):
            part = new_messages[result_idx].parts[0]
            assert part.parts[0].text.startswith("payload-")

    def test_non_destructive_object_identity(self) -> None:
        messages = []
        for k in range(4):
            messages.extend(_tool_pair(f"tc{k}"))
        original_objects = list(messages)
        policy = ContextPolicy(keep_last_n=1)

        new_messages, applied = edit_context(messages, policy)

        # Input list and its objects are untouched.
        assert messages == original_objects
        for orig, still in zip(messages, original_objects, strict=True):
            assert orig is still
        # Returned list is a new list object.
        assert new_messages is not messages
        # Kept entries reuse the same objects; cleared entries are new instances.
        assert new_messages[-1] is messages[-1]  # last result kept
        assert new_messages[1] is not messages[1]  # first result cleared
        # The cleared original is unchanged in the input.
        assert messages[1].parts[0].parts[0].text != policy.placeholder

    def test_in_flight_pair_never_trimmed(self) -> None:
        # Two complete pairs, then a trailing unanswered tool-use (in-flight).
        messages = [*_tool_pair("tc0"), *_tool_pair("tc1"), _tool_use_msg("tc2")]
        policy = ContextPolicy(keep_last_n=1)

        new_messages, applied = edit_context(messages, policy)

        # Only the oldest complete result (index 1) clears; the in-flight tool-use
        # at the tail is reused by identity and never touched.
        assert applied is not None
        assert applied.cleared_tool_uses == 1
        assert new_messages[-1] is messages[-1]
        assert isinstance(new_messages[-1], AssistantMessagePart)

    def test_parallel_batch_atomicity(self) -> None:
        # A parallel batch is one assistant message with N tool uses + one user
        # message with the N matching results — kept/cleared as a single unit.
        batch_use = AssistantMessagePart(
            parts=[ToolUsePart(id=f"p{i}", name="search", inputs={}) for i in range(3)]
        )
        batch_result = UserMessagePart(
            parts=[
                ToolResultPart(tool_use_id=f"p{i}", name="search", parts=[TextPart(text="x" * 50)])
                for i in range(3)
            ]
        )
        # Put the batch first (old), then keep_last_n=1 recent pair after it.
        messages = [batch_use, batch_result, *_tool_pair("recent")]
        policy = ContextPolicy(keep_last_n=1)

        new_messages, applied = edit_context(messages, policy)

        assert applied is not None
        # All three results in the batch cleared together (one unit).
        assert applied.cleared_tool_uses == 3
        cleared = new_messages[1]
        assert all(p.parts[0].text == policy.placeholder for p in cleared.parts)

    def test_openai_pairing_preserved(self) -> None:
        messages = []
        for k in range(4):
            messages.extend(_tool_pair(f"tc{k}"))
        new_messages, _ = edit_context(messages, ContextPolicy(keep_last_n=1))

        # Every tool-result still has its tool_use_id, and a preceding tool-use
        # message with the matching id exists.
        use_ids = {
            p.id
            for m in new_messages
            if isinstance(m, AssistantMessagePart)
            for p in m.parts
            if isinstance(p, ToolUsePart)
        }
        for m in new_messages:
            if isinstance(m, UserMessagePart):
                for p in m.parts:
                    if isinstance(p, ToolResultPart):
                        assert p.tool_use_id in use_ids

    def test_gemini_signature_preserved_and_skeleton_kept(self) -> None:
        # Kept pair carries a thought-signature in tool-use metadata.
        kept_use = _tool_use_msg("kept", thought_signature="sig-123")
        messages = [
            *_tool_pair("old0"),
            *_tool_pair("old1"),
            kept_use,
            _tool_result_msg("kept", "fresh"),
        ]
        new_messages, applied = edit_context(messages, ContextPolicy(keep_last_n=1))

        # Kept tool-use is object-identical → signature intact.
        assert new_messages[4] is kept_use
        assert new_messages[4].parts[0].metadata["thought_signature"] == "sig-123"
        # Cleared prior pairs keep their tool-use skeleton (call still present).
        assert isinstance(new_messages[0], AssistantMessagePart)
        assert new_messages[0].parts[0].id == "old0"

    def test_returns_none_when_nothing_old(self) -> None:
        messages = [*_tool_pair("tc0"), *_tool_pair("tc1")]
        result, applied = edit_context(messages, ContextPolicy(keep_last_n=3))

        assert applied is None
        assert result is messages


# --- U3: executor wiring — automatic trim ---


def _result_text(message) -> str | None:
    """First tool-result text in a message, or None if not a tool-result message."""
    if isinstance(message, UserMessagePart):
        for part in message.parts:
            if isinstance(part, ToolResultPart):
                return part.parts[0].text
    return None


class TestExecutorAutoTrim:
    """The between-turns trim hook in run_stream."""

    def _history(self) -> list:
        # Four pre-existing tool round-trips the executor can trim.
        msgs = []
        for k in range(4):
            msgs.extend(_tool_pair(f"h{k}", text=f"history-payload-{k}"))
        return msgs

    def test_trim_fires_above_threshold(self) -> None:
        captured: list = []
        # Turn 1 returns a tool call (loop continues) with a huge input usage;
        # turn 2 ends. Trigger = 0.8 * 100_000 = 80_000.
        turns = [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(150_000)),
            ([], _usage(150_000)),
        ]
        provider = _make_streaming_provider(turns, captured)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            context_policy=ContextPolicy(context_window=100_000, trigger_pct=0.8, keep_last_n=1),
        )
        events = asyncio.run(_collect(executor, self._history()))

        edits = [e for e in events if isinstance(e, ContextEditEvent)]
        assert len(edits) == 1
        assert edits[0].applied_edits[0].type == "clear_tool_uses"
        # The second chat call received a trimmed list (older results placeholdered).
        second_call = captured[1]
        placeholder = ContextPolicy().placeholder
        cleared = [m for m in second_call if _result_text(m) == placeholder]
        assert len(cleared) >= 1

    def test_below_threshold_no_trim(self) -> None:
        captured: list = []
        turns = [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(10_000)),
            ([], _usage(10_000)),
        ]
        provider = _make_streaming_provider(turns, captured)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            context_policy=ContextPolicy(context_window=100_000, trigger_pct=0.8),
        )
        events = asyncio.run(_collect(executor, self._history()))

        assert not [e for e in events if isinstance(e, ContextEditEvent)]
        # The second chat call received the full, untrimmed history.
        placeholder = ContextPolicy().placeholder
        assert not [m for m in captured[1] if _result_text(m) == placeholder]

    def test_no_policy_is_regression_safe(self) -> None:
        captured: list = []
        turns = [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(150_000)),
            ([], _usage(150_000)),
        ]
        provider = _make_streaming_provider(turns, captured)
        executor = AgentExecutor(provider="openai", llm=provider)  # no policy
        events = asyncio.run(_collect(executor, self._history()))

        assert not [e for e in events if isinstance(e, ContextEditEvent)]

    def test_usage_none_fallback_engages_trigger(self) -> None:
        captured: list = []
        # usage=None on every turn; a tiny window makes the chars/4 estimate cross.
        turns = [
            ([ToolUsePart(id="t1", name="noop", inputs={})], None),
            ([], None),
        ]
        provider = _make_streaming_provider(turns, captured)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            context_policy=ContextPolicy(context_window=20, trigger_pct=0.5, keep_last_n=1),
        )
        events = asyncio.run(_collect(executor, self._history()))

        assert [e for e in events if isinstance(e, ContextEditEvent)]

    def test_first_iteration_never_trims(self) -> None:
        captured: list = []
        # Loop ends on turn 1; the trigger has no prior usage to act on.
        turns = [([], _usage(150_000))]
        provider = _make_streaming_provider(turns, captured)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            context_policy=ContextPolicy(context_window=100_000, trigger_pct=0.8, keep_last_n=1),
        )
        events = asyncio.run(_collect(executor, self._history()))

        assert not [e for e in events if isinstance(e, ContextEditEvent)]
        # The only chat call received the full, untrimmed history.
        placeholder = ContextPolicy().placeholder
        assert not [m for m in captured[0] if _result_text(m) == placeholder]


# --- U4: summarize mode — write-back + Summarizer ---


class TestSummarizeContextUnit:
    """The summarize_context() pure-ish unit (with a mock summarizer llm)."""

    def test_writes_back_summary_and_stashes_originals(self) -> None:
        messages = [
            UserMessagePart(parts=[TextPart(text="question")]),
            *_tool_pair("h0", text="history-payload-0"),
            *_tool_pair("h1", text="history-payload-1"),
            *_tool_pair("h2", text="history-payload-2"),
        ]
        original_objs = list(messages)
        llm = _summarizer_llm("DIGEST")

        applied = asyncio.run(summarize_context(messages, ContextPolicy(keep_last_n=1), llm))

        assert applied is not None
        assert applied.type == "summarize"
        assert applied.summary_text == "DIGEST"
        # Span (h0, h1) summarized; h2 kept verbatim.
        summary = _summary_message(messages)
        assert summary is not None
        assert "DIGEST" in summary.parts[0].text
        # The replaced originals are stashed for audit/replay.
        assert applied.replaced_originals is not None
        assert all(obj in original_objs for obj in applied.replaced_originals)
        # The most recent pair (h2) survives verbatim.
        assert any(isinstance(m, AssistantMessagePart) and m.parts[0].id == "h2" for m in messages)

    def test_excludes_in_flight_pair(self) -> None:
        messages = [
            *_tool_pair("h0"),
            *_tool_pair("h1"),
            *_tool_pair("h2"),
            _tool_use_msg("inflight"),
        ]
        llm = _summarizer_llm("DIGEST")

        applied = asyncio.run(summarize_context(messages, ContextPolicy(keep_last_n=1), llm))

        assert applied is not None
        # The in-flight (unanswered) tool-use stays at the tail, never summarized.
        assert isinstance(messages[-1], AssistantMessagePart)
        assert messages[-1].parts[0].id == "inflight"

    def test_uses_stream_false(self) -> None:
        calls: list = []
        messages = [*_tool_pair("h0"), *_tool_pair("h1"), *_tool_pair("h2")]
        llm = _summarizer_llm("DIGEST", calls=calls)

        asyncio.run(summarize_context(messages, ContextPolicy(keep_last_n=1), llm))

        assert calls and calls[0]["stream"] is False

    def test_extra_instructions_augment_prompt(self) -> None:
        calls: list = []
        messages = [*_tool_pair("h0"), *_tool_pair("h1")]
        llm = _summarizer_llm("DIGEST", calls=calls)

        asyncio.run(
            summarize_context(
                messages,
                ContextPolicy(keep_last_n=0),
                llm,
                extra_instructions="focus on the API keys",
            )
        )

        assert "focus on the API keys" in calls[0]["system_prompt"]

    def test_returns_none_when_nothing_old(self) -> None:
        messages = [*_tool_pair("h0")]
        llm = _summarizer_llm("DIGEST")
        applied = asyncio.run(summarize_context(messages, ContextPolicy(keep_last_n=3), llm))
        assert applied is None


class TestExecutorAutoSummarize:
    """The summarize branch of the between-turns hook (write-back + watermark)."""

    def _history(self) -> list:
        msgs = [UserMessagePart(parts=[TextPart(text="question")])]
        for k in range(4):
            msgs.extend(_tool_pair(f"h{k}", text=f"history-payload-{k}"))
        return msgs

    def test_summarize_writes_back_and_calls_stream_false(self) -> None:
        captured: list = []
        summarizer_calls: list = []
        turns = [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(150_000)),
            ([], _usage(150_000)),
        ]
        provider = _make_summarizing_provider(turns, captured, summarizer_calls)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            tools=[_NoopTool()],
            context_policy=ContextPolicy(
                context_window=100_000, trigger_pct=0.8, keep_last_n=1, mode="summarize"
            ),
        )
        events = asyncio.run(_collect(executor, self._history()))

        edits = [e for e in events if isinstance(e, ContextEditEvent)]
        assert len(edits) == 1
        assert edits[0].applied_edits[0].type == "summarize"
        # Exactly one summarizer round-trip.
        assert len(summarizer_calls) == 1
        # The second streaming call saw the written-back <summary> turn, and the
        # old history payloads are gone.
        second_call = captured[1]
        assert _summary_message(second_call) is not None
        assert not any(_result_text(m) == "history-payload-0" for m in second_call)

    def test_computed_once_watermark(self) -> None:
        captured: list = []
        summarizer_calls: list = []
        # Three streaming turns all reporting the same high input usage. With the
        # watermark, only the first crossing summarizes.
        turns = [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(150_000)),
            ([ToolUsePart(id="t2", name="noop", inputs={})], _usage(150_000)),
            ([], _usage(150_000)),
        ]
        provider = _make_summarizing_provider(turns, captured, summarizer_calls)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            tools=[_NoopTool()],
            context_policy=ContextPolicy(
                context_window=100_000, trigger_pct=0.8, keep_last_n=1, mode="summarize"
            ),
        )
        events = asyncio.run(_collect(executor, self._history()))

        # Watermark suppresses re-summarize at the same token count.
        assert len(summarizer_calls) == 1
        assert len([e for e in events if isinstance(e, ContextEditEvent)]) == 1

    def test_non_destructive_to_caller(self) -> None:
        captured: list = []
        summarizer_calls: list = []
        turns = [
            ([ToolUsePart(id="t1", name="noop", inputs={})], _usage(150_000)),
            ([], _usage(150_000)),
        ]
        provider = _make_summarizing_provider(turns, captured, summarizer_calls)
        caller_messages = self._history()
        snapshot = list(caller_messages)
        executor = AgentExecutor(
            provider="openai",
            llm=provider,
            tools=[_NoopTool()],
            context_policy=ContextPolicy(
                context_window=100_000, trigger_pct=0.8, keep_last_n=1, mode="summarize"
            ),
        )
        asyncio.run(_collect(executor, caller_messages))

        # The caller's list and its objects are unchanged by write-back.
        assert caller_messages == snapshot
        for orig, still in zip(caller_messages, snapshot, strict=True):
            assert orig is still
