"""Tests for bounded model correction in AgentExecutor."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from dobby import AgentExecutor
from dobby.exceptions import ApprovalRequired, ModelRetry, ModelRetryExhaustedError
from dobby.executor import _control_flow_result
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
            max_model_retries=1,
            max_consecutive_model_retries=5,
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


def test_consecutive_limit_exhausts_after_results_without_extra_chat() -> None:
    provider = ScriptedProvider(
        [
            [ToolUsePart(id="call-first", name="typed", inputs={"first": 1})],
            [ToolUsePart(id="call-second", name="typed", inputs={"second": 1})],
            [],
        ]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[TypedTool()])

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_model_retries=5,
            max_consecutive_model_retries=1,
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(provider.calls) == 2
    assert [result.tool_use_id for result in results] == ["call-first", "call-second"]
    assert [part.tool_use_id for part in _tool_results(messages)] == [
        "call-first",
        "call-second",
    ]
    assert error.attempts == 2
    assert error.last_error is not None
    assert error.last_error.exception_type == "ModelRetry"
    assert "second" in error.last_error.message
    assert results[-1].result.startswith("[tool_input_invalid]")


def test_consecutive_limit_resets_after_non_correction_progress() -> None:
    provider = ScriptedProvider(
        [
            [ToolUsePart(id="call-first", name="typed", inputs={"first": 1})],
            [ToolUsePart(id="call-success", name="typed", inputs={"count": 1})],
            [ToolUsePart(id="call-third", name="typed", inputs={"third": 1})],
            [ToolUsePart(id="call-fourth", name="typed", inputs={"fourth": 1})],
            [],
        ]
    )
    executor = AgentExecutor(provider="openai", llm=provider, tools=[TypedTool()])

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_model_retries=10,
            max_consecutive_model_retries=1,
        )
    )

    assert len(provider.calls) == 4
    assert error.attempts == 2
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-first",
        "call-success",
        "call-third",
        "call-fourth",
    ]
    assert [part.tool_use_id for part in _tool_results(messages)] == [
        "call-first",
        "call-success",
        "call-third",
        "call-fourth",
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
            max_model_retries=1,
            max_consecutive_model_retries=10,
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

    events = asyncio.run(_collect_events(executor, max_model_retries=0))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(provider.calls) == 2
    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].error_details is not None
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."
    assert "failure" not in str(results[0].result)


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
        _collect_until_exception(executor, ModelRetryExhaustedError, max_model_retries=0)
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
        events = asyncio.run(_collect_events(executor, max_model_retries=0))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert calls == 2
    assert len(provider.calls) == 2
    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].error_details is not None
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."
    assert "transient" not in str(results[0].result)


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
            max_model_retries=5,
            max_consecutive_model_retries=0,
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
        _collect_until_exception(executor, ModelRetryExhaustedError, max_model_retries=0)
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

    events = asyncio.run(_collect_events(executor, max_model_retries=1))
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


@pytest.mark.parametrize("control_flow", ["approval", "cancellation"])
def test_host_control_flow_does_not_become_model_retry_exhaustion(control_flow: str) -> None:
    @dataclass
    class ControlFlowTool(Tool):
        name = "control"
        description = "Raise host control flow."
        requires_approval = control_flow == "approval"

        async def __call__(self) -> None:
            if control_flow == "cancellation":
                raise asyncio.CancelledError

    provider = ScriptedProvider([[ToolUsePart(id="call-control", name="control", inputs={})]])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[ControlFlowTool()])

    expected = ApprovalRequired if control_flow == "approval" else asyncio.CancelledError
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
            _collect_until_exception(executor, expected, max_model_retries=0)
        )

    assert isinstance(error, expected)
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
            max_model_retries=1,
            max_consecutive_model_retries=1,
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
            max_model_retries=0,
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
            max_model_retries=0,
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
            max_model_retries=10,
            max_consecutive_model_retries=10,
        )
    )

    assert len(provider.calls) == 2
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-one",
        "call-two",
    ]
