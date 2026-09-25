"""Tests for bounded model correction in AgentExecutor."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch
import warnings

from pydantic import BaseModel
import pytest

from dobby import AgentExecutor
from dobby.exceptions import ApprovalRequired, ModelRetry, ModelRetryExhaustedError
from dobby.executor import _control_flow_result, _resolve_max_model_corrections
from dobby.tools import Tool
from dobby.types import (
    StreamEndEvent,
    ToolResultEvent,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart,
)


class ScriptedProvider:
    """Return one scripted model response per chat call."""

    def __init__(self, turns: list[list[ToolUsePart]]) -> None:
        self.turns = turns
        self.calls: list[list] = []

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
class TypedTool(Tool):
    """Accept a validated integer."""

    name = "typed"
    description = "Accept an integer."

    async def __call__(self, count: int) -> int:
        return count


def _tool_results(messages) -> list[ToolResultPart]:
    """Return tool-result history parts in order."""
    return [
        part
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]


async def _collect_events(executor: AgentExecutor, **kwargs) -> list:
    """Collect all events from a completed executor run."""
    return [event async for event in executor.run_stream(messages=[], **kwargs)]


async def _collect_until_exception(executor, exception_type, **kwargs):
    """Collect events and assembled history until a host exception is raised."""
    events = []
    caught = None
    with patch.object(
        executor, "_emit_tool_result", wraps=executor._emit_tool_result
    ) as emit_result:
        try:
            async for event in executor.run_stream(messages=[], **kwargs):
                events.append(event)
        except exception_type as exception:
            caught = exception

    assert caught is not None
    messages = emit_result.call_args.args[3] if emit_result.call_args else []
    return events, messages, caught


def test_invalid_args_are_sent_to_the_second_model_turn() -> None:
    provider = ScriptedProvider(
        [[ToolUsePart(id="call-invalid", name="typed", inputs={"count": "bad"})], []]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[TypedTool()])

    asyncio.run(_collect_events(executor))

    assert len(provider.calls) == 2
    results = _tool_results(provider.calls[1])
    assert [result.tool_use_id for result in results] == ["call-invalid"]
    assert results[0].is_error is True
    assert results[0].parts[0].text.startswith("[tool_input_invalid]")


def test_unknown_tool_with_successful_sibling_counts_as_one_correction_batch() -> None:
    provider = ScriptedProvider(
        [
            [
                ToolUsePart(id="call-unknown", name="missing", inputs={}),
                ToolUsePart(id="call-success", name="typed", inputs={"count": 1}),
            ],
            [ToolUsePart(id="call-second-correction", name="still_missing", inputs={})],
            [],
        ]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[TypedTool()])

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_model_corrections=1,
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(provider.calls) == 2
    assert error.attempts == 2
    assert [result.tool_use_id for result in results] == [
        "call-unknown",
        "call-success",
        "call-second-correction",
    ]
    assert [result.is_error for result in results] == [True, False, True]
    assert [part.tool_use_id for part in _tool_results(messages)] == [
        "call-unknown",
        "call-success",
        "call-second-correction",
    ]


@pytest.mark.parametrize("middle_fails", [False, True], ids=["success", "execution-error"])
def test_run_wide_limit_survives_non_correction_progress(middle_fails: bool) -> None:
    @dataclass
    class MiddleTool(Tool):
        name = "middle"
        description = "Succeed or fail without requesting model correction."

        async def __call__(self) -> str:
            if middle_fails:
                raise ValueError("middle execution failure")
            return "ok"

    provider = ScriptedProvider(
        [
            [ToolUsePart(id="call-first", name="typed", inputs={"count": "bad"})],
            [ToolUsePart(id="call-middle", name="middle", inputs={})],
            [ToolUsePart(id="call-last", name="typed", inputs={"last": 1})],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[TypedTool(), MiddleTool()],
    )

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_model_corrections=1,
        )
    )

    assert len(provider.calls) == 3
    assert error.attempts == 2
    assert error.last_error is not None
    assert "last" in error.last_error.message
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-first",
        "call-middle",
        "call-last",
    ]
    assert [part.tool_use_id for part in _tool_results(messages)] == [
        "call-first",
        "call-middle",
        "call-last",
    ]


def test_execution_errors_do_not_consume_correction_budget() -> None:
    @dataclass
    class FailingTool(Tool):
        name = "failing"
        description = "Fail during execution."

        async def __call__(self) -> None:
            raise ValueError("failure")

    provider = ScriptedProvider([[ToolUsePart(id="call-failing", name="failing", inputs={})], []])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[FailingTool()])

    events = asyncio.run(_collect_events(executor, max_model_corrections=0))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(provider.calls) == 2
    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].error_details is not None
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."


def test_body_model_retry_consumes_correction_budget() -> None:
    @dataclass
    class RetryTool(Tool):
        name = "retrying"
        description = "Request model correction from the tool body."

        async def __call__(self) -> None:
            raise ModelRetry("retry body")

    provider = ScriptedProvider(
        [[ToolUsePart(id="call-retrying", name="retrying", inputs={})], []]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[RetryTool()])

    events, messages, error = asyncio.run(
        _collect_until_exception(executor, ModelRetryExhaustedError, max_model_corrections=0)
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert isinstance(error, ModelRetryExhaustedError)
    assert error.attempts == 1
    assert len(provider.calls) == 1
    assert len(results) == 1
    assert results[0].is_error is True
    assert str(results[0].result) == "[tool_retry] retry body"
    assert [part.tool_use_id for part in _tool_results(messages)] == ["call-retrying"]


def test_retry_exhausted_execution_error_does_not_consume_correction_budget() -> None:
    calls = 0

    @dataclass
    class TimeoutTool(Tool):
        name = "timeout"
        description = "Always time out."
        retryable_exceptions = (TimeoutError,)

        async def __call__(self) -> None:
            nonlocal calls
            calls += 1
            raise TimeoutError("transient")

    provider = ScriptedProvider([[ToolUsePart(id="call-timeout", name="timeout", inputs={})], []])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[TimeoutTool()])

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events = asyncio.run(_collect_events(executor, max_model_corrections=0))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert calls == 2
    assert len(provider.calls) == 2
    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].error_details is not None
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."


def test_streaming_validation_correction_exhausts_after_result() -> None:
    @dataclass
    class StreamingTool(Tool):
        name = "streaming"
        description = "Stream after validation."
        stream_output = True

        async def __call__(self, count: int):
            yield count

    provider = ScriptedProvider(
        [[ToolUsePart(id="call-streaming", name="streaming", inputs={"count": "bad"})], []]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[StreamingTool()])

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_model_corrections=0,
        )
    )

    assert len(provider.calls) == 1
    assert error.attempts == 1
    assert error.last_error is not None
    assert error.last_error.exception_type == "ModelRetry"
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-streaming"
    ]
    assert [part.tool_use_id for part in _tool_results(messages)] == ["call-streaming"]


def test_terminal_validation_correction_exhausts_after_result() -> None:
    @dataclass
    class TerminalTool(Tool):
        name = "terminal"
        description = "Terminate after validation."
        terminal = True

        async def __call__(self, count: int) -> int:
            return count

    provider = ScriptedProvider(
        [[ToolUsePart(id="call-terminal", name="terminal", inputs={"count": "bad"})], []]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[TerminalTool()])

    events, messages, error = asyncio.run(
        _collect_until_exception(executor, ModelRetryExhaustedError, max_model_corrections=0)
    )

    assert len(provider.calls) == 1
    assert error.attempts == 1
    assert error.last_error is not None
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-terminal"
    ]
    assert [part.tool_use_id for part in _tool_results(messages)] == ["call-terminal"]


@pytest.mark.parametrize("correction_kind", ["regular", "streaming"])
def test_successful_terminal_waits_for_mixed_batch_correction_gate(
    correction_kind: str,
) -> None:
    terminal_calls = 0

    @dataclass
    class StreamingTool(Tool):
        name = "streaming"
        description = "Stream after validation."
        stream_output = True

        async def __call__(self, count: int):
            yield count

    @dataclass
    class TerminalTool(Tool):
        name = "terminal"
        description = "Complete the run."
        terminal = True

        async def __call__(self) -> str:
            nonlocal terminal_calls
            terminal_calls += 1
            return "done"

    correction_name = "typed" if correction_kind == "regular" else "streaming"
    provider = ScriptedProvider(
        [
            [
                ToolUsePart(
                    id="call-correction",
                    name=correction_name,
                    inputs={"count": "bad"},
                ),
                ToolUsePart(id="call-terminal", name="terminal", inputs={}),
            ],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[TypedTool(), StreamingTool(), TerminalTool()],
    )

    events = asyncio.run(_collect_events(executor, max_model_corrections=1))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(provider.calls) == 2
    assert terminal_calls == 0
    assert [result.tool_use_id for result in results] == [
        "call-correction",
        "call-terminal",
    ]
    assert results[0].is_error is True
    assert results[1].result == {"skipped": True, "reason": "model_correction"}
    assert results[1].is_error is True
    assert results[1].is_terminal is False
    assert [part.tool_use_id for part in _tool_results(provider.calls[1])] == [
        "call-correction",
        "call-terminal",
    ]


def test_approval_does_not_become_model_retry_exhaustion() -> None:
    @dataclass
    class ControlFlowTool(Tool):
        name = "control"
        description = "Raise host control flow."
        requires_approval = True

        async def __call__(self) -> None:
            return None

    provider = ScriptedProvider([[ToolUsePart(id="call-control", name="control", inputs={})]])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[ControlFlowTool()])

    control_results = []

    def capture_control_flow_result(tool_call, exception):
        result = _control_flow_result(tool_call, exception)
        control_results.append(result)
        return result

    with patch(
        "dobby.executor._control_flow_result",
        side_effect=capture_control_flow_result,
    ):
        events, messages, error = asyncio.run(
            _collect_until_exception(executor, ApprovalRequired, max_model_corrections=0)
        )

    assert isinstance(error, ApprovalRequired)
    assert len(provider.calls) == 1
    assert len(control_results) == 1
    assert control_results[0].retry_model is False
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-control"
    ]
    assert [part.tool_use_id for part in _tool_results(messages)] == ["call-control"]


def test_parallel_corrections_count_once_per_batch() -> None:
    provider = ScriptedProvider(
        [
            [
                ToolUsePart(id="call-one", name="missing_one", inputs={}),
                ToolUsePart(id="call-two", name="missing_two", inputs={}),
            ],
            [],
        ]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[])

    events = asyncio.run(
        _collect_events(
            executor,
            max_model_corrections=1,
        )
    )

    assert len(provider.calls) == 2
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-one",
        "call-two",
    ]
    assert [part.tool_use_id for part in _tool_results(provider.calls[1])] == [
        "call-one",
        "call-two",
    ]


def test_exhaustion_waits_for_complete_multi_result_batch() -> None:
    provider = ScriptedProvider(
        [
            [
                ToolUsePart(id="call-correction", name="missing", inputs={}),
                ToolUsePart(id="call-later-success", name="typed", inputs={"count": 1}),
            ],
            [],
        ]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[TypedTool()])

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_model_corrections=0,
        )
    )

    assert len(provider.calls) == 1
    assert error.attempts == 1
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-correction",
        "call-later-success",
    ]
    assert [part.tool_use_id for part in _tool_results(messages)] == [
        "call-correction",
        "call-later-success",
    ]


def test_exhausted_multi_correction_batch_keeps_last_error_in_batch_order() -> None:
    provider = ScriptedProvider(
        [
            [
                ToolUsePart(id="call-one", name="missing_one", inputs={}),
                ToolUsePart(id="call-two", name="missing_two", inputs={}),
            ],
            [],
        ]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[])

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_model_corrections=0,
        )
    )

    assert len(provider.calls) == 1
    assert error.attempts == 1
    assert error.last_error is not None
    assert "missing_two" in error.last_error.message
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-one",
        "call-two",
    ]
    assert [part.tool_use_id for part in _tool_results(messages)] == [
        "call-one",
        "call-two",
    ]


def test_max_iterations_still_silently_limits_model_calls() -> None:
    provider = ScriptedProvider(
        [
            [ToolUsePart(id="call-one", name="missing_one", inputs={})],
            [ToolUsePart(id="call-two", name="missing_two", inputs={})],
            [ToolUsePart(id="call-three", name="missing_three", inputs={})],
        ]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[])

    events = asyncio.run(
        _collect_events(
            executor,
            max_iterations=2,
            max_model_corrections=10,
        )
    )

    assert len(provider.calls) == 2
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-one",
        "call-two",
    ]


def test_resolve_default_is_canonical_budget() -> None:
    assert _resolve_max_model_corrections(None) == 3
    assert _resolve_max_model_corrections(4) == 4


def test_canonical_max_model_corrections_does_not_warn() -> None:
    provider = ScriptedProvider(
        [[ToolUsePart(id="call-invalid", name="typed", inputs={"count": "bad"})], []]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[TypedTool()])

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        events, _, error = asyncio.run(
            _collect_until_exception(
                executor,
                ModelRetryExhaustedError,
                max_model_corrections=0,
            )
        )

    assert len(provider.calls) == 1
    assert error.attempts == 1
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-invalid"
    ]


def test_sequential_model_correction_skips_later_sequential_and_streaming() -> None:
    later_sequential_calls = 0
    later_streaming_calls = 0

    @dataclass
    class FirstSequentialTool(Tool):
        name = "first"
        description = "Request correction."
        sequential = True

        async def __call__(self, count: int) -> int:
            return count

    @dataclass
    class LaterSequentialTool(Tool):
        name = "later_sequential"
        description = "Must not run after correction."
        sequential = True

        async def __call__(self) -> str:
            nonlocal later_sequential_calls
            later_sequential_calls += 1
            return "later"

    @dataclass
    class LaterStreamingTool(Tool):
        name = "later_streaming"
        description = "Must not stream after correction."
        stream_output = True

        async def __call__(self):
            nonlocal later_streaming_calls
            later_streaming_calls += 1
            yield "streamed"

    provider = ScriptedProvider(
        [
            [
                ToolUsePart(id="call-first", name="first", inputs={"count": "bad"}),
                ToolUsePart(id="call-later-seq", name="later_sequential", inputs={}),
                ToolUsePart(id="call-later-stream", name="later_streaming", inputs={}),
            ],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[FirstSequentialTool(), LaterSequentialTool(), LaterStreamingTool()],
    )

    events = asyncio.run(_collect_events(executor, max_model_corrections=1))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert later_sequential_calls == 0
    assert later_streaming_calls == 0
    assert [result.tool_use_id for result in results] == [
        "call-first",
        "call-later-seq",
        "call-later-stream",
    ]
    assert results[0].is_error is True
    assert results[1].result == {"skipped": True, "reason": "model_correction"}
    assert results[2].result == {"skipped": True, "reason": "model_correction"}
    assert [part.tool_use_id for part in _tool_results(provider.calls[1])] == [
        "call-first",
        "call-later-seq",
        "call-later-stream",
    ]


def test_streaming_model_correction_skips_later_streaming_and_terminal() -> None:
    later_streaming_calls = 0
    terminal_calls = 0

    @dataclass
    class FirstStreamingTool(Tool):
        name = "first_streaming"
        description = "Stream then request correction."
        stream_output = True

        async def __call__(self, count: int):
            yield count

    @dataclass
    class LaterStreamingTool(Tool):
        name = "later_streaming"
        description = "Must not run."
        stream_output = True

        async def __call__(self):
            nonlocal later_streaming_calls
            later_streaming_calls += 1
            yield "later"

    @dataclass
    class TerminalTool(Tool):
        name = "terminal"
        description = "Must not run."
        terminal = True

        async def __call__(self) -> str:
            nonlocal terminal_calls
            terminal_calls += 1
            return "done"

    provider = ScriptedProvider(
        [
            [
                ToolUsePart(
                    id="call-first",
                    name="first_streaming",
                    inputs={"count": "bad"},
                ),
                ToolUsePart(id="call-later", name="later_streaming", inputs={}),
                ToolUsePart(id="call-terminal", name="terminal", inputs={}),
            ],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[FirstStreamingTool(), LaterStreamingTool(), TerminalTool()],
    )

    events = asyncio.run(_collect_events(executor, max_model_corrections=1))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert later_streaming_calls == 0
    assert terminal_calls == 0
    assert [result.tool_use_id for result in results] == [
        "call-first",
        "call-later",
        "call-terminal",
    ]
    assert results[1].result == {"skipped": True, "reason": "model_correction"}
    assert results[2].result == {"skipped": True, "reason": "model_correction"}


def test_multiple_terminals_are_all_represented_after_first_completes() -> None:
    second_calls = 0

    @dataclass
    class FirstTerminal(Tool):
        name = "first_terminal"
        description = "Ends the run."
        terminal = True

        async def __call__(self) -> str:
            return "done"

    @dataclass
    class SecondTerminal(Tool):
        name = "second_terminal"
        description = "Must be represented as skipped."
        terminal = True

        async def __call__(self) -> str:
            nonlocal second_calls
            second_calls += 1
            return "also done"

    provider = ScriptedProvider(
        [
            [
                ToolUsePart(id="call-first", name="first_terminal", inputs={}),
                ToolUsePart(id="call-second", name="second_terminal", inputs={}),
            ],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[FirstTerminal(), SecondTerminal()],
    )

    events = []
    messages = []
    with patch.object(
        executor, "_emit_tool_result", wraps=executor._emit_tool_result
    ) as emit_result:
        events = asyncio.run(_collect_events(executor))
        messages = emit_result.call_args.args[3] if emit_result.call_args else []

    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert second_calls == 0
    assert [result.tool_use_id for result in results] == ["call-first", "call-second"]
    assert results[0].result == "done"
    assert results[0].is_terminal is True
    assert results[1].result == {"skipped": True, "reason": "terminal"}
    assert results[1].is_error is True
    assert [part.tool_use_id for part in _tool_results(messages)] == [
        "call-first",
        "call-second",
    ]


def test_multiple_terminals_skipped_as_model_correction_when_first_retries() -> None:
    second_calls = 0

    @dataclass
    class FirstTerminal(Tool):
        name = "first_terminal"
        description = "Request correction then stop exit."
        terminal = True

        async def __call__(self, count: int) -> int:
            return count

    @dataclass
    class SecondTerminal(Tool):
        name = "second_terminal"
        description = "Must be skipped for correction."
        terminal = True

        async def __call__(self) -> str:
            nonlocal second_calls
            second_calls += 1
            return "done"

    provider = ScriptedProvider(
        [
            [
                ToolUsePart(
                    id="call-first",
                    name="first_terminal",
                    inputs={"count": "bad"},
                ),
                ToolUsePart(id="call-second", name="second_terminal", inputs={}),
            ],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[FirstTerminal(), SecondTerminal()],
    )

    events = asyncio.run(_collect_events(executor, max_model_corrections=1))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert second_calls == 0
    assert [result.tool_use_id for result in results] == ["call-first", "call-second"]
    assert results[0].is_error is True
    assert results[0].is_terminal is False
    assert results[1].result == {"skipped": True, "reason": "model_correction"}


def test_last_output_is_reset_at_start_of_each_run() -> None:
    class Output(BaseModel):
        value: int

    valid = {"value": 1}
    provider = ScriptedProvider(
        [
            [ToolUsePart(id="call-valid", name="final_result", inputs=valid)],
            [ToolUsePart(id="call-invalid", name="final_result", inputs={})],
        ]
    )
    executor = AgentExecutor(provider="openai", llm=provider, output_type=Output)

    asyncio.run(_collect_events(executor))
    assert executor.last_output == Output(value=1)

    with pytest.raises(ModelRetryExhaustedError):
        asyncio.run(_collect_events(executor, max_model_corrections=0))

    assert executor.last_output is None
