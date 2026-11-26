"""Tests that each tool turn is written to the executor's history as one whole turn."""

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

from pydantic import BaseModel
import pytest

from dobby import AgentExecutor
from dobby.exceptions import ApprovalRequired
from dobby.providers.anthropic.adapter import to_anthropic_messages
from dobby.providers.gemini.converters import to_gemini_messages
from dobby.providers.openai.adapter import to_openai_messages
from dobby.providers.vertexai.converters import to_vertexai_messages
from dobby.tools import Tool
from dobby.types import (
    AssistantMessagePart,
    MessagePart,
    ReasoningPart,
    ResponsePart,
    StreamEndEvent,
    TextPart,
    ToolResultEvent,
    ToolResultPart,
    ToolStreamEvent,
    ToolUseEndEvent,
    ToolUsePart,
    Usage,
    UserMessagePart,
)

USER_PROMPT = UserMessagePart(parts=[TextPart(text="Check both.")])


class ScriptedProvider:
    """Return one scripted model response per chat call and record each request."""

    def __init__(self, turns: list[list[ResponsePart]]) -> None:
        self.turns = turns
        self.calls: list[list[MessagePart]] = []

    def format_tools(self, tools):
        del tools
        return None

    async def chat(self, messages, **kwargs):
        """Record history and stream the next scripted response."""
        del kwargs
        self.calls.append(list(messages))
        call_index = len(self.calls) - 1
        parts = self.turns[call_index] if call_index < len(self.turns) else []

        async def stream():
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()


@dataclass
class EchoTool(Tool):
    """Return the given value after an optional delay."""

    name = "echo"
    description = "Echo a value."

    async def __call__(self, value: str, delay: float = 0.0) -> str:
        await asyncio.sleep(delay)
        return value


@dataclass
class TypedTool(Tool):
    """Accept a validated integer."""

    name = "typed"
    description = "Accept an integer."

    async def __call__(self, count: int) -> int:
        return count


@dataclass
class StreamTool(Tool):
    """Stream one progress event and return a value."""

    name = "stream"
    description = "Stream a value."
    stream_output = True

    async def __call__(self, value: str):
        yield ToolStreamEvent(type="progress", data=value)
        yield value


@dataclass
class FinishTool(Tool):
    """End the run."""

    name = "finish"
    description = "Finish the run."
    terminal = True

    async def __call__(self) -> str:
        return "finished"


@dataclass
class ApprovalTool(Tool):
    """Require host approval."""

    name = "approve_me"
    description = "Requires approval."
    requires_approval = True

    async def __call__(self) -> str:
        return "approved"


class Answer(BaseModel):
    """Structured output for final_result tests."""

    answer: str


def _call(call_id: str, name: str = "echo", **inputs) -> ToolUsePart:
    """Build a tool call."""
    return ToolUsePart(id=call_id, name=name, inputs=inputs)


def _result_texts(message: MessagePart) -> list[tuple[str, str, bool]]:
    """Return (tool_use_id, text, is_error) for each tool result in a user message."""
    assert isinstance(message, UserMessagePart)
    return [
        (part.tool_use_id, part.parts[0].text, part.is_error)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]


def _event_shape(events: list) -> list[tuple]:
    """Summarise tool result and end events for order and payload comparison."""
    shape = []
    for event in events:
        if isinstance(event, ToolResultEvent):
            shape.append(("result", event.tool_use_id, event.result, event.is_error))
        elif isinstance(event, ToolUseEndEvent):
            shape.append(("end", event.tool_use_id))
        elif isinstance(event, ToolStreamEvent):
            shape.append(("stream", event.data))
    return shape


async def _collect_events(executor: AgentExecutor, messages=None) -> list:
    """Collect all events from a completed run."""
    return [event async for event in executor.run_stream(messages=messages or [USER_PROMPT])]


async def _collect_with_flushes(executor: AgentExecutor, exception_type=None):
    """Collect events, the flush calls and any raised host exception."""
    events = []
    caught = None
    with patch.object(executor, "_flush_turn", wraps=executor._flush_turn) as flush:
        try:
            async for event in executor.run_stream(messages=[USER_PROMPT]):
                events.append(event)
        except BaseException as exception:
            if exception_type is None or not isinstance(exception, exception_type):
                raise
            caught = exception
    return events, flush, caught


def test_reasoning_text_and_two_calls_form_one_turn() -> None:
    turn = [
        ReasoningPart(text="plan", signature="sig"),
        TextPart(text="Checking."),
        _call("a", value="A"),
        _call("b", value="B"),
    ]
    provider = ScriptedProvider([turn, []])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[EchoTool()])

    events = asyncio.run(_collect_events(executor))

    assert len(provider.calls) == 2
    history = provider.calls[1]
    assert len(history) == 3
    assert history[0] is USER_PROMPT
    assert history[1] == AssistantMessagePart(parts=turn)
    assert _result_texts(history[2]) == [("a", "A", False), ("b", "B", False)]
    assert _event_shape(events) == [
        ("result", "a", "A", False),
        ("end", "a"),
        ("result", "b", "B", False),
        ("end", "b"),
    ]


def test_single_call_turn_keeps_text() -> None:
    turn = [TextPart(text="One moment."), _call("a", value="A")]
    provider = ScriptedProvider([turn, []])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[EchoTool()])

    asyncio.run(_collect_events(executor))

    history = provider.calls[1]
    assert history[1:] == [
        AssistantMessagePart(parts=turn),
        UserMessagePart(
            parts=[ToolResultPart(tool_use_id="a", name="echo", parts=[TextPart(text="A")])]
        ),
    ]


def test_results_follow_call_order_when_calls_finish_in_reverse() -> None:
    turn = [_call("a", value="A", delay=0.05), _call("b", value="B")]
    provider = ScriptedProvider([turn, []])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[EchoTool()])

    asyncio.run(_collect_events(executor))

    history = provider.calls[1]
    assert len(history) == 3
    assert _result_texts(history[2]) == [("a", "A", False), ("b", "B", False)]


def test_mixed_batch_results_follow_call_order_not_category_order() -> None:
    turn = [
        _call("s", "stream", value="S"),
        _call("p", value="P"),
        _call("t", "finish"),
    ]
    provider = ScriptedProvider([turn])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[EchoTool(), StreamTool(), FinishTool()],
    )

    events, flush, _ = asyncio.run(_collect_with_flushes(executor))

    assert len(provider.calls) == 1
    assert flush.call_count == 1
    history = flush.call_args.args[0]
    assert history[1] == AssistantMessagePart(parts=turn)
    assert _result_texts(history[2]) == [
        ("s", "S", False),
        ("p", "P", False),
        ("t", "finished", False),
    ]
    # Events keep category order (parallel, streaming, terminal) and the
    # successful terminal call still has no ToolUseEndEvent.
    assert _event_shape(events) == [
        ("result", "p", "P", False),
        ("end", "p"),
        ("stream", "S"),
        ("result", "s", "S", False),
        ("end", "s"),
        ("result", "t", "finished", False),
    ]


def test_model_correction_flushes_one_result_message() -> None:
    turn = [_call("bad", "typed", count="x"), _call("s", "stream", value="S")]
    provider = ScriptedProvider([turn, []])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[TypedTool(), StreamTool()],
    )

    events = asyncio.run(_collect_events(executor))

    history = provider.calls[1]
    assert len(history) == 3
    assert history[1] == AssistantMessagePart(parts=turn)
    results = _result_texts(history[2])
    assert [result[0] for result in results] == ["bad", "s"]
    assert results[0][1].startswith("[tool_input_invalid]")
    assert results[0][2] is True
    assert results[1] == ("s", str({"skipped": True, "reason": "model_correction"}), True)
    assert [shape[:2] for shape in _event_shape(events)] == [
        ("result", "bad"),
        ("end", "bad"),
        ("result", "s"),
        ("end", "s"),
    ]


def test_invalid_final_result_with_siblings_flushes_one_result_message() -> None:
    turn = [_call("a", value="A"), _call("f", "final_result", wrong="field")]
    provider = ScriptedProvider([turn, []])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[EchoTool()],
        output_type=Answer,
    )

    asyncio.run(_collect_events(executor))

    history = provider.calls[1]
    assert len(history) == 3
    assert history[1] == AssistantMessagePart(parts=turn)
    results = _result_texts(history[2])
    assert results[0] == ("a", str({"skipped": True, "reason": "final_result_invalid"}), True)
    assert results[1][0] == "f"
    assert results[1][1].startswith("[final_result_invalid]")
    assert results[1][2] is True


def test_valid_final_result_does_not_flush() -> None:
    provider = ScriptedProvider([[_call("f", "final_result", answer="yes")]])
    executor = AgentExecutor(provider="openai", llm=provider, output_type=Answer)

    _, flush, _ = asyncio.run(_collect_with_flushes(executor))

    assert flush.call_count == 0
    assert executor.last_output == Answer(answer="yes")


def test_approval_required_flushes_once_before_raising() -> None:
    turn = [_call("a", "approve_me"), _call("b", "stream", value="B")]
    provider = ScriptedProvider([turn])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[ApprovalTool(), StreamTool()],
    )

    events, flush, caught = asyncio.run(_collect_with_flushes(executor, ApprovalRequired))

    assert isinstance(caught, ApprovalRequired)
    assert caught.tool_call_id == "a"
    assert flush.call_count == 1
    history = flush.call_args.args[0]
    assert len(history) == 3
    assert history[1] == AssistantMessagePart(parts=turn)
    placeholder = str({"approval_required": True})
    assert _result_texts(history[2]) == [("a", placeholder, True), ("b", placeholder, True)]
    assert _event_shape(events) == [
        ("result", "a", {"approval_required": True}, True),
        ("end", "a"),
        ("result", "b", {"approval_required": True}, True),
        ("end", "b"),
    ]


class CancelTool(Tool):
    """Cancel the task running the tool, as host cancellation would."""

    name = "cancel_me"
    description = "Gets cancelled mid-call."

    async def __call__(self) -> str:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)
        return "unreachable"


def test_cancellation_flushes_once_before_raising() -> None:
    """A cancelled parallel batch still writes one whole turn, then re-raises."""
    turn = [_call("a", "cancel_me"), _call("b", value="B")]
    provider = ScriptedProvider([turn])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[CancelTool(), EchoTool()])

    events, flush, caught = asyncio.run(_collect_with_flushes(executor, asyncio.CancelledError))

    assert isinstance(caught, asyncio.CancelledError)
    assert flush.call_count == 1
    history = flush.call_args.args[0]
    assert len(history) == 3
    assert history[1] == AssistantMessagePart(parts=turn)
    assert _result_texts(history[2]) == [
        ("a", str({"cancelled": True}), True),
        ("b", "B", False),
    ]
    # Host cancellation yields no placeholder event for the cancelled call.
    assert ("result", "a", {"cancelled": True}, True) not in _event_shape(events)


@pytest.mark.parametrize("call_id", ["same", ""])
def test_calls_sharing_an_id_each_keep_their_result(call_id: str) -> None:
    turn = [_call(call_id, value="first"), _call(call_id, value="second")]
    provider = ScriptedProvider([turn, []])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[EchoTool()])

    asyncio.run(_collect_events(executor))

    history = provider.calls[1]
    assert history[1] == AssistantMessagePart(parts=turn)
    assert _result_texts(history[2]) == [(call_id, "first", False), (call_id, "second", False)]


def _batched_history() -> list[MessagePart]:
    """Return a batched tool turn as the executor now writes it."""
    return [
        USER_PROMPT,
        AssistantMessagePart(
            parts=[TextPart(text="Checking."), _call("a", value="A"), _call("b", value="B")]
        ),
        UserMessagePart(
            parts=[
                ToolResultPart(tool_use_id="a", name="echo", parts=[TextPart(text="A")]),
                ToolResultPart(tool_use_id="b", name="echo", parts=[TextPart(text="B")]),
            ]
        ),
    ]


def _split_history() -> list[MessagePart]:
    """Return the same tool turn in the older one-pair-per-call shape."""
    return [
        USER_PROMPT,
        AssistantMessagePart(parts=[_call("a", value="A")]),
        UserMessagePart(
            parts=[ToolResultPart(tool_use_id="a", name="echo", parts=[TextPart(text="A")])]
        ),
        AssistantMessagePart(parts=[_call("b", value="B")]),
        UserMessagePart(
            parts=[ToolResultPart(tool_use_id="b", name="echo", parts=[TextPart(text="B")])]
        ),
    ]


def test_batched_history_is_one_assistant_turn_for_anthropic() -> None:
    converted = to_anthropic_messages(_batched_history())

    assert [message["role"] for message in converted] == ["user", "assistant", "user"]
    tool_uses = [block["id"] for block in converted[1]["content"] if block["type"] == "tool_use"]
    assert tool_uses == ["a", "b"]
    assert [block["tool_use_id"] for block in converted[2]["content"]] == ["a", "b"]


def test_batched_history_is_one_assistant_turn_for_gemini() -> None:
    converted = to_gemini_messages(_batched_history())

    assert [content.role for content in converted] == ["user", "model", "user"]
    calls = [part.function_call.name for part in converted[1].parts if part.function_call]
    assert len(calls) == 2
    responses = [part.function_response for part in converted[2].parts]
    assert len(responses) == 2


def test_batched_history_is_one_assistant_turn_for_openai_responses() -> None:
    converted = to_openai_messages(_batched_history())

    kinds = [item.get("type") for item in converted]
    assert kinds == [
        "message",
        "message",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert [item["call_id"] for item in converted[2:4]] == ["a", "b"]


def test_batched_history_is_one_assistant_turn_for_vertexai() -> None:
    converted = to_vertexai_messages(_batched_history())

    assert [message["role"] for message in converted] == ["user", "assistant", "tool", "tool"]
    assert [call["id"] for call in converted[1]["tool_calls"]] == ["a", "b"]
    assert [message["tool_call_id"] for message in converted[2:]] == ["a", "b"]


@pytest.mark.parametrize(
    "convert",
    [to_anthropic_messages, to_gemini_messages, to_openai_messages, to_vertexai_messages],
)
def test_split_client_history_is_still_accepted(convert) -> None:
    assert convert(_split_history())


def test_split_client_history_is_sent_unchanged() -> None:
    provider = ScriptedProvider([[]])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[EchoTool()])
    history = _split_history()

    asyncio.run(_collect_events(executor, messages=history))

    assert provider.calls == [history]
