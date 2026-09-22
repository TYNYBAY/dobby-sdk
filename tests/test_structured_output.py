"""Tests for bounded structured-output correction in AgentExecutor."""

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, field_validator
import pytest

from dobby import AgentExecutor
from dobby.exceptions import ModelRetryExhaustedError
from dobby.tools import Tool
from dobby.types import (
    AssistantMessagePart,
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
        index = len(self.calls) - 1
        parts = self.turns[index] if index < len(self.turns) else []

        async def stream():
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()


class Details(BaseModel):
    """Nested structured-output details."""

    score: int


class StructuredResult(BaseModel):
    """Structured output used by correction tests."""

    model_config = ConfigDict(extra="forbid")

    name: str
    count: int
    details: Details


@dataclass
class TypedTool(Tool):
    """Accept a validated integer."""

    name = "typed"
    description = "Accept an integer."

    async def __call__(self, count: int) -> int:
        return count


def _valid_result() -> dict:
    """Return valid structured-output inputs."""
    return {"name": "result", "count": 1, "details": {"score": 2}}


def _final_call(call_id: str, inputs: dict) -> ToolUsePart:
    """Build a final_result call."""
    return ToolUsePart(id=call_id, name="final_result", inputs=inputs)


def _tool_results(messages) -> list[ToolResultPart]:
    """Return tool-result history parts in order."""
    return [
        part
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]


def _tool_use_ids(messages) -> list[str]:
    """Return assembled assistant tool-call IDs in conversation order."""
    return [
        part.id
        for message in messages
        if isinstance(message, AssistantMessagePart)
        for part in message.parts
        if isinstance(part, ToolUsePart)
    ]


def _assert_paired_final_result(messages, call_id: str) -> ToolResultPart:
    """Require a matching assistant tool-use and user tool-result for one call."""
    use_ids = _tool_use_ids(messages)
    results = _tool_results(messages)
    result_ids = [part.tool_use_id for part in results]
    assert call_id in use_ids
    assert call_id in result_ids
    assert use_ids.index(call_id) == result_ids.index(call_id)
    return results[result_ids.index(call_id)]


async def _collect_events(executor, **kwargs) -> list:
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


def test_invalid_final_result_is_sent_to_next_model_turn() -> None:
    invalid = _valid_result()
    invalid.pop("name")
    provider = ScriptedProvider([[_final_call("call-invalid", invalid)], []])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=StructuredResult,
    )

    events = asyncio.run(_collect_events(executor))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(provider.calls) == 2
    assert executor.last_output is None
    assert [result.tool_use_id for result in results] == ["call-invalid"]
    part = _assert_paired_final_result(provider.calls[1], "call-invalid")
    assert part.parts[0].text.startswith("[final_result_invalid]")


@pytest.mark.parametrize(
    ("mutate", "expected", "secret"),
    [
        (lambda data: data.pop("name"), "name: Field required", None),
        (
            lambda data: data.update(count="SECRET_WRONG_TYPE"),
            "count: Input should be a valid integer",
            "SECRET_WRONG_TYPE",
        ),
        (
            lambda data: data.update(extra="SECRET_EXTRA"),
            "extra: Extra inputs are not permitted",
            "SECRET_EXTRA",
        ),
        (
            lambda data: data["details"].update(score="SECRET_NESTED"),
            "details.score: Input should be a valid integer",
            "SECRET_NESTED",
        ),
    ],
    ids=["missing", "wrong-type", "extra", "nested"],
)
def test_final_result_feedback_is_safe_and_field_level(mutate, expected, secret) -> None:
    invalid = _valid_result()
    mutate(invalid)
    provider = ScriptedProvider([[_final_call("call-invalid", invalid)]])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=StructuredResult,
    )

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_final_result_retries=0,
        )
    )
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    part = _assert_paired_final_result(messages, "call-invalid")
    model_text = part.parts[0].text

    assert len(provider.calls) == 1
    assert error.attempts == 1
    assert result.result == model_text
    assert expected in model_text
    assert "Traceback" not in model_text
    assert "validation error" not in model_text.lower()
    assert "pydantic_core" not in model_text
    if secret is not None:
        assert secret not in model_text


def test_corrected_final_result_sets_output_without_third_chat() -> None:
    invalid = _valid_result()
    invalid.pop("name")
    valid = _valid_result()
    provider = ScriptedProvider(
        [
            [_final_call("call-invalid", invalid)],
            [_final_call("call-valid", valid)],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=StructuredResult,
    )

    events = asyncio.run(_collect_events(executor))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(provider.calls) == 2
    assert executor.last_output == StructuredResult.model_validate(valid)
    assert [result.tool_use_id for result in results] == ["call-invalid", "call-valid"]
    assert [result.is_error for result in results] == [True, False]


def test_zero_final_result_limit_exhausts_after_emission() -> None:
    provider = ScriptedProvider([[_final_call("call-invalid", {})], []])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=StructuredResult,
    )

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_final_result_retries=0,
        )
    )

    assert len(provider.calls) == 1
    assert error.attempts == 1
    assert error.last_error is not None
    assert error.last_error.exception_type == "ValidationError"
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-invalid"
    ]
    part = _assert_paired_final_result(messages, "call-invalid")
    assert part.parts[0].text.startswith("[final_result_invalid]")


def test_consecutive_final_result_limit_zero_exhausts_first_invalid() -> None:
    provider = ScriptedProvider([[_final_call("call-invalid", {})], []])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=StructuredResult,
    )

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_final_result_retries=10,
            max_consecutive_final_result_retries=0,
        )
    )

    assert len(provider.calls) == 1
    assert error.attempts == 1
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-invalid"
    ]
    _assert_paired_final_result(messages, "call-invalid")


def test_multiple_field_errors_count_as_one_structured_output_correction() -> None:
    provider = ScriptedProvider(
        [
            [_final_call("call-many-fields", {})],
            [_final_call("call-second", {"name": "x"})],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=StructuredResult,
    )

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_final_result_retries=1,
            max_consecutive_final_result_retries=10,
        )
    )

    assert len(provider.calls) == 2
    assert error.attempts == 2
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-many-fields",
        "call-second",
    ]
    first = _assert_paired_final_result(messages, "call-many-fields")
    second = _assert_paired_final_result(messages, "call-second")
    assert first.parts[0].text.startswith("[final_result_invalid]")
    assert "name: Field required" in first.parts[0].text
    assert "count: Field required" in first.parts[0].text
    assert "details: Field required" in first.parts[0].text
    assert second.parts[0].text.startswith("[final_result_invalid]")


def test_consecutive_final_result_limit() -> None:
    provider = ScriptedProvider(
        [
            [_final_call("call-one", {"count": 1, "details": {"score": 2}})],
            [_final_call("call-two", {"name": "x", "details": {"score": 2}})],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=StructuredResult,
    )

    events, messages, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_final_result_retries=10,
            max_consecutive_final_result_retries=1,
        )
    )

    assert len(provider.calls) == 2
    assert error.attempts == 2
    assert error.last_error is not None
    assert "count" in error.last_error.message
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-one",
        "call-two",
    ]
    assert [part.tool_use_id for part in _tool_results(messages)] == [
        "call-one",
        "call-two",
    ]


def test_run_wide_final_result_limit_does_not_reset() -> None:
    provider = ScriptedProvider(
        [
            [_final_call("call-one", {})],
            [ToolUsePart(id="call-progress", name="typed", inputs={"count": 1})],
            [_final_call("call-two", {})],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[TypedTool()],
        output_type=StructuredResult,
    )

    _, _, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_final_result_retries=1,
            max_consecutive_final_result_retries=10,
        )
    )

    assert len(provider.calls) == 3
    assert error.attempts == 2


@pytest.mark.parametrize(
    "middle_kind",
    ["success", "tool-correction", "execution-error"],
)
def test_non_final_result_iteration_resets_consecutive_counter(middle_kind: str) -> None:
    @dataclass
    class FailingTool(Tool):
        name = "failing"
        description = "Fail during execution."

        async def __call__(self) -> None:
            raise ValueError("execution failure")

    if middle_kind == "success":
        middle = ToolUsePart(id="call-middle", name="typed", inputs={"count": 1})
    elif middle_kind == "tool-correction":
        middle = ToolUsePart(id="call-middle", name="typed", inputs={"count": "bad"})
    else:
        middle = ToolUsePart(id="call-middle", name="failing", inputs={})

    provider = ScriptedProvider(
        [
            [_final_call("call-one", {})],
            [middle],
            [_final_call("call-three", {})],
            [_final_call("call-four", {})],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[TypedTool(), FailingTool()],
        output_type=StructuredResult,
    )

    _, _, error = asyncio.run(
        _collect_until_exception(
            executor,
            ModelRetryExhaustedError,
            max_final_result_retries=10,
            max_consecutive_final_result_retries=1,
            max_model_retries=10,
        )
    )

    assert len(provider.calls) == 4
    assert error.attempts == 2


@pytest.mark.parametrize(
    "tool_call",
    [
        ToolUsePart(id="call-tool", name="typed", inputs={"count": "bad"}),
        ToolUsePart(id="call-missing", name="missing", inputs={}),
    ],
    ids=["tool-input-invalid", "tool-not-found"],
)
def test_tool_correction_does_not_consume_final_result_budget(
    tool_call: ToolUsePart,
) -> None:
    provider = ScriptedProvider([[tool_call], []])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[TypedTool()],
        output_type=StructuredResult,
    )

    events = asyncio.run(
        _collect_events(
            executor,
            max_final_result_retries=0,
            max_consecutive_final_result_retries=0,
        )
    )

    assert len(provider.calls) == 2
    assert len([event for event in events if isinstance(event, ToolResultEvent)]) == 1


def test_final_result_correction_does_not_consume_model_budget() -> None:
    invalid = _valid_result()
    invalid.pop("name")
    valid = _valid_result()
    provider = ScriptedProvider(
        [
            [_final_call("call-invalid", invalid)],
            [_final_call("call-valid", valid)],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=StructuredResult,
    )

    asyncio.run(
        _collect_events(
            executor,
            max_model_retries=0,
            max_consecutive_model_retries=0,
        )
    )

    assert len(provider.calls) == 2
    assert executor.last_output == StructuredResult.model_validate(valid)


def test_invalid_final_result_skips_sibling_tools() -> None:
    sibling_called = False

    @dataclass
    class SiblingTool(Tool):
        name = "sibling"
        description = "Must not execute beside final_result."

        async def __call__(self) -> None:
            nonlocal sibling_called
            sibling_called = True

    provider = ScriptedProvider(
        [
            [
                _final_call("call-invalid", {}),
                ToolUsePart(id="call-sibling", name="sibling", inputs={}),
            ],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[SiblingTool()],
        output_type=StructuredResult,
    )

    events = asyncio.run(
        _collect_events(
            executor,
            max_final_result_retries=1,
        )
    )

    assert sibling_called is False
    assert len(provider.calls) == 2
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-invalid",
        "call-sibling",
    ]
    second_turn_use_ids = _tool_use_ids(provider.calls[1])
    second_turn_results = _tool_results(provider.calls[1])
    assert second_turn_use_ids == ["call-invalid", "call-sibling"]
    assert [part.tool_use_id for part in second_turn_results] == second_turn_use_ids
    assert second_turn_results[1].is_error is True
    assert second_turn_results[1].parts[0].text == (
        "{'skipped': True, 'reason': 'final_result_invalid'}"
    )


def test_structured_output_turn_resets_phase4_consecutive_budget() -> None:
    invalid = _valid_result()
    invalid.pop("name")
    provider = ScriptedProvider(
        [
            [ToolUsePart(id="call-tool-one", name="typed", inputs={"count": "bad"})],
            [_final_call("call-final-invalid", invalid)],
            [ToolUsePart(id="call-tool-two", name="typed", inputs={"count": "also-bad"})],
            [],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[TypedTool()],
        output_type=StructuredResult,
    )

    events = asyncio.run(
        _collect_events(
            executor,
            max_consecutive_model_retries=1,
            max_model_retries=10,
            max_final_result_retries=10,
            max_consecutive_final_result_retries=10,
        )
    )

    assert len(provider.calls) == 4
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-tool-one",
        "call-final-invalid",
        "call-tool-two",
    ]
    _assert_paired_final_result(provider.calls[2], "call-final-invalid")


def test_structured_output_respects_silent_max_iterations_cap() -> None:
    provider = ScriptedProvider(
        [
            [_final_call("call-one", {})],
            [_final_call("call-two", {})],
            [_final_call("call-three", {})],
        ]
    )
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=StructuredResult,
    )

    events = asyncio.run(
        _collect_events(
            executor,
            max_iterations=2,
            max_final_result_retries=10,
            max_consecutive_final_result_retries=10,
        )
    )

    assert len(provider.calls) == 2
    assert executor.last_output is None
    assert [event.tool_use_id for event in events if isinstance(event, ToolResultEvent)] == [
        "call-one",
        "call-two",
    ]


def test_valid_final_result_terminates_normally() -> None:
    valid = _valid_result()
    provider = ScriptedProvider([[_final_call("call-valid", valid)], []])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=StructuredResult,
    )

    with patch.object(executor, "_emit_tool_result") as emit_result:
        events = asyncio.run(_collect_events(executor))
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert len(provider.calls) == 1
    assert emit_result.call_count == 0
    assert executor.last_output == StructuredResult.model_validate(valid)
    assert [result.tool_use_id for result in results] == ["call-valid"]
    assert results[0].is_error is False


def test_final_result_without_output_type_remains_unknown_tool() -> None:
    provider = ScriptedProvider([[_final_call("call-final", {"value": "x"})], []])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[])

    events = asyncio.run(_collect_events(executor))
    result = next(event for event in events if isinstance(event, ToolResultEvent))

    assert result.tool_use_id == "call-final"
    assert result.result.startswith("[tool_not_found]")


def test_unexpected_output_validation_exception_remains_host_raised() -> None:
    class ExplosiveResult(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def explode(cls, value: str) -> str:
            raise RuntimeError("internal validator secret")

    provider = ScriptedProvider([[_final_call("call-final", {"value": "trigger"})]])
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        output_type=ExplosiveResult,
    )

    with patch.object(executor, "_emit_tool_result") as emit_result:
        with pytest.raises(RuntimeError, match="internal validator secret"):
            asyncio.run(_collect_events(executor))

    assert len(provider.calls) == 1
    assert emit_result.call_count == 0
