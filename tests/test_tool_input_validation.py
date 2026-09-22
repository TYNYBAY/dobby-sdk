"""Tests for tool input validation in AgentExecutor."""

import asyncio
from dataclasses import dataclass
from typing import Annotated
from unittest.mock import AsyncMock

from pydantic import BaseModel
import pytest

from dobby import AgentExecutor
from dobby.exceptions import ErrorCode, ModelRetry
from dobby.tools import Injected, Tool
from dobby.types import StreamEndEvent, ToolResultEvent, ToolStreamEvent, ToolUsePart, Usage


def _make_mock_provider(tool_calls: list[ToolUsePart]) -> AsyncMock:
    call_count = 0

    async def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        async def stream():
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=tool_calls if call_count == 1 else [],
                stop_reason="tool_use" if call_count == 1 else "end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    return provider


async def _collect_results(
    tools: list[Tool],
    tool_calls: list[ToolUsePart],
    **run_kwargs,
) -> list[ToolResultEvent]:
    events = await _collect_events(tools, tool_calls, **run_kwargs)
    return [event for event in events if isinstance(event, ToolResultEvent)]


async def _collect_events(
    tools: list[Tool],
    tool_calls: list[ToolUsePart],
    **run_kwargs,
) -> list:
    executor = AgentExecutor(
        provider="openai",
        llm=_make_mock_provider(tool_calls),
        tools=tools,
    )
    return [event async for event in executor.run_stream(messages=[], **run_kwargs)]


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"count": 1}, "label: Field required"),
        ({"count": "not-an-integer", "label": "x"}, "count: Input should be a valid integer"),
        ({"count": 1, "label": "x", "extra": True}, "extra: Extra inputs are not permitted"),
    ],
    ids=["missing-required", "invalid-type", "extra-key"],
)
def test_invalid_function_inputs_are_rejected_before_execution(
    inputs: dict,
    message: str,
) -> None:
    calls = 0

    @dataclass
    class TypedTool(Tool):
        name = "typed"
        description = "Accept typed inputs."

        async def __call__(self, count: int, label: str) -> dict:
            nonlocal calls
            calls += 1
            return {"count": count, "label": label}

    results = asyncio.run(
        _collect_results(
            [TypedTool()],
            [ToolUsePart(id="call-invalid", name="typed", inputs=inputs)],
        )
    )

    assert calls == 0
    assert len(results) == 1
    assert results[0].tool_use_id == "call-invalid"
    assert results[0].is_error is True
    assert str(results[0].result).startswith("[tool_input_invalid] Invalid tool arguments:")
    assert message in str(results[0].result)
    assert "Traceback" not in str(results[0].result)


def test_injected_context_is_preserved_and_cannot_be_supplied_by_model() -> None:
    @dataclass
    class Context:
        prefix: str

    received: list[tuple[Context, int]] = []

    @dataclass
    class ContextTool(Tool):
        name = "context_tool"
        description = "Use runtime context."

        async def __call__(
            self,
            context: Injected[Context],
            value: Annotated[int, "Value"],
            **kwargs,
        ) -> str:
            received.append((context, value))
            return f"{context.prefix}-{value}"

    context = Context(prefix="ctx")
    results = asyncio.run(
        _collect_results(
            [ContextTool()],
            [
                ToolUsePart(id="call-valid", name="context_tool", inputs={"value": "3"}),
                ToolUsePart(
                    id="call-injected",
                    name="context_tool",
                    inputs={"context": {"prefix": "model"}, "value": 4},
                ),
            ],
            context=context,
        )
    )

    assert received == [(context, 3)]
    assert [result.tool_use_id for result in results] == ["call-valid", "call-injected"]
    assert results[0].result == "ctx-3"
    assert results[1].is_error is True
    assert str(results[1].result).startswith("[tool_input_invalid]")
    assert "Runtime-injected parameters cannot be supplied" in str(results[1].result)


def test_unknown_tool_returns_correctable_result_without_disrupting_siblings() -> None:
    calls: list[str] = []

    @dataclass
    class EchoTool(Tool):
        name = "echo"
        description = "Echo a value."

        async def __call__(self, value: str) -> str:
            calls.append(value)
            return value

    results = asyncio.run(
        _collect_results(
            [EchoTool()],
            [
                ToolUsePart(id="call-1", name="echo", inputs={"value": "first"}),
                ToolUsePart(id="call-2", name="missing", inputs={}),
                ToolUsePart(id="call-3", name="echo", inputs={"value": "third"}),
            ],
        )
    )

    assert calls == ["first", "third"]
    assert [result.tool_use_id for result in results] == ["call-1", "call-2", "call-3"]
    assert [result.name for result in results] == ["echo", "missing", "echo"]
    assert results[1].is_error is True
    assert str(results[1].result).startswith("[tool_not_found]")


def test_final_result_is_unknown_without_configured_output_type() -> None:
    results = asyncio.run(
        _collect_results(
            [],
            [ToolUsePart(id="call-final", name="final_result", inputs={"value": "x"})],
        )
    )

    assert len(results) == 1
    assert results[0].tool_use_id == "call-final"
    assert results[0].name == "final_result"
    assert results[0].is_error is True
    assert str(results[0].result).startswith("[tool_not_found]")


def test_from_model_rejects_extra_keys_like_function_tools() -> None:
    class ModelInput(BaseModel):
        value: int

    tool = Tool.from_model(ModelInput, name="model_tool", description="Validate a model.")
    results = asyncio.run(
        _collect_results(
            [tool],
            [
                ToolUsePart(
                    id="call-model",
                    name="model_tool",
                    inputs={"value": 1, "extra": True},
                )
            ],
        )
    )

    assert len(results) == 1
    assert results[0].is_error is True
    assert "extra: Extra inputs are not permitted" in str(results[0].result)


def test_validation_happens_before_approval() -> None:
    called = False

    @dataclass
    class ApprovalTool(Tool):
        name = "approval"
        description = "Require approval."
        requires_approval = True

        async def __call__(self, count: int) -> int:
            nonlocal called
            called = True
            return count

    results = asyncio.run(
        _collect_results(
            [ApprovalTool()],
            [ToolUsePart(id="call-approval", name="approval", inputs={"count": "invalid"})],
        )
    )

    assert called is False
    assert len(results) == 1
    assert results[0].is_error is True
    assert str(results[0].result).startswith("[tool_input_invalid]")


def test_streaming_tool_inputs_are_validated_before_execution() -> None:
    called = False

    @dataclass
    class StreamingTool(Tool):
        name = "streaming"
        description = "Stream progress."
        stream_output = True

        async def __call__(self, count: int):
            nonlocal called
            called = True
            yield ToolStreamEvent(type="progress", data=count)

    results = asyncio.run(
        _collect_results(
            [StreamingTool()],
            [ToolUsePart(id="call-streaming", name="streaming", inputs={"count": "invalid"})],
        )
    )

    assert called is False
    assert len(results) == 1
    assert results[0].tool_use_id == "call-streaming"
    assert results[0].is_error is True
    assert str(results[0].result).startswith("[tool_input_invalid]")


def test_terminal_tool_inputs_are_validated_before_execution() -> None:
    called = False

    @dataclass
    class TerminalTool(Tool):
        name = "terminal"
        description = "End the run."
        terminal = True

        async def __call__(self, count: int) -> int:
            nonlocal called
            called = True
            return count

    events = asyncio.run(
        _collect_events(
            [TerminalTool()],
            [ToolUsePart(id="call-terminal", name="terminal", inputs={"count": "invalid"})],
        )
    )
    results = [event for event in events if isinstance(event, ToolResultEvent)]

    assert called is False
    assert len(results) == 1
    assert results[0].tool_use_id == "call-terminal"
    assert results[0].is_error is True
    assert results[0].is_terminal is False
    assert str(results[0].result).startswith("[tool_input_invalid]")
    assert len([event for event in events if isinstance(event, StreamEndEvent)]) == 2


def test_streaming_model_retry_skips_terminal_body() -> None:
    terminal_calls = 0

    @dataclass
    class StreamingTool(Tool):
        name = "body_streaming"
        description = "Raise from a streaming body."
        stream_output = True

        async def __call__(self, value: int):
            yield ToolStreamEvent(type="progress", data=value)
            raise ModelRetry("streaming body failure", code=ErrorCode.TOOL_INPUT_INVALID)

    @dataclass
    class TerminalTool(Tool):
        name = "body_terminal"
        description = "Raise from a terminal body."
        terminal = True

        async def __call__(self, value: int) -> int:
            nonlocal terminal_calls
            terminal_calls += 1
            raise ModelRetry("terminal body failure", code=ErrorCode.TOOL_INPUT_INVALID)

    results = asyncio.run(
        _collect_results(
            [StreamingTool(), TerminalTool()],
            [
                ToolUsePart(id="call-streaming-body", name="body_streaming", inputs={"value": 1}),
                ToolUsePart(id="call-terminal-body", name="body_terminal", inputs={"value": 2}),
            ],
        )
    )

    assert [result.tool_use_id for result in results] == [
        "call-streaming-body",
        "call-terminal-body",
    ]
    assert all(result.is_error for result in results)
    assert "streaming body failure" in str(results[0].result)
    assert results[1].result == {"skipped": True, "reason": "model_correction"}
    assert results[1].is_terminal is False
    assert terminal_calls == 0


def test_exception_inside_executed_tool_is_not_input_validation_error() -> None:
    @dataclass
    class BrokenTool(Tool):
        name = "broken"
        description = "Fail after valid input."

        async def __call__(self, count: int) -> int:
            raise TypeError(f"failure inside tool: {count}")

    results = asyncio.run(
        _collect_results(
            [BrokenTool()],
            [ToolUsePart(id="call-broken", name="broken", inputs={"count": 1})],
        )
    )

    assert len(results) == 1
    assert results[0].is_error is True
    assert "[tool_input_invalid]" not in str(results[0].result)
    assert str(results[0].result) == "[tool_execution_error] The tool failed unexpectedly."
    assert "failure inside tool: 1" not in str(results[0].result)
    assert results[0].error_details is not None
    assert results[0].error_details.message == "failure inside tool: 1"
