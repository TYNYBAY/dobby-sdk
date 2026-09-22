"""Tests for Phase 8 error correlation and host-only diagnostics."""

import asyncio
from dataclasses import dataclass
import logging
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel
import pytest
from tenacity import retry_if_exception_type, stop_after_attempt, wait_none

from dobby import AgentExecutor
from dobby.exceptions import ModelRetry, ToolFailure
from dobby.providers import ProviderError, RateLimitError
from dobby.providers._retry import with_retries
from dobby.tools import Tool
from dobby.types import (
    StreamEndEvent,
    ToolResultEvent,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart,
)


class _RecordingProvider:
    name = "recording"
    model = "mock"
    max_retries = 0

    def __init__(self, tool_calls: list[ToolUsePart]) -> None:
        self.tool_calls = tool_calls
        self.calls = 0
        self.messages: list[list[object]] = []

    async def chat(self, messages, **kwargs):
        del kwargs
        self.calls += 1
        self.messages.append(list(messages))

        async def stream():
            yield StreamEndEvent(
                model="mock",
                parts=self.tool_calls if self.calls == 1 else [],
                stop_reason="tool_use" if self.calls == 1 else "end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()


async def _run(tools, tool_calls, **kwargs):
    provider = _RecordingProvider(tool_calls)
    executor = AgentExecutor(provider="openai", llm=provider, tools=tools)
    events = [event async for event in executor.run_stream(messages=[], **kwargs)]
    return events, provider


def _results(events) -> list[ToolResultEvent]:
    return [event for event in events if isinstance(event, ToolResultEvent)]


def _history_results(provider: _RecordingProvider) -> list[ToolResultPart]:
    return [
        part
        for message in provider.messages[-1]
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]


def test_unexpected_error_has_correlated_host_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @dataclass
    class BrokenTool(Tool):
        name = "broken"
        description = "Fail unexpectedly."

        async def __call__(self) -> None:
            raise ValueError("secret diagnostic")

    with caplog.at_level(logging.ERROR, logger="dobby"):
        events, provider = asyncio.run(
            _run(
                [BrokenTool()],
                [ToolUsePart(id="call-broken", name="broken", inputs={})],
            )
        )

    result = _results(events)[0]
    details = result.error_details
    assert details is not None
    assert details.error_code == "tool_execution_error"
    assert details.run_id
    assert details.tool_name == "broken"
    assert details.tool_call_id == "call-broken"
    assert details.attempt == 1
    assert details.max_attempts == 1

    error_records = [record for record in caplog.records if record.exc_info]
    assert len(error_records) == 1
    assert error_records[0].run_id == details.run_id
    assert error_records[0].tool_call_id == "call-broken"

    history_text = _history_results(provider)[0].parts[0].text
    assert history_text == "[tool_execution_error] secret diagnostic"
    assert details.run_id not in history_text
    assert "attempt" not in history_text
    assert "Traceback" not in history_text


def test_parallel_same_name_calls_share_run_id_and_keep_attempts_independent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: dict[str, int] = {"left": 0, "right": 0}

    @dataclass
    class FlakyTool(Tool):
        name = "flaky"
        description = "Fail once per key."
        retryable_exceptions = (TimeoutError,)

        async def __call__(self, key: str) -> str:
            calls[key] += 1
            if calls[key] == 1:
                raise TimeoutError(key)
            return key

    tool_calls = [
        ToolUsePart(id="call-left", name="flaky", inputs={"key": "left"}),
        ToolUsePart(id="call-right", name="flaky", inputs={"key": "right"}),
    ]
    with (
        caplog.at_level(logging.WARNING, logger="dobby"),
        patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock),
    ):
        events, provider = asyncio.run(_run([FlakyTool()], tool_calls))

    retry_records = [
        record for record in caplog.records if getattr(record, "layer", None) == "tool"
    ]
    assert {record.tool_call_id for record in retry_records} == {"call-left", "call-right"}
    assert len({record.run_id for record in retry_records}) == 1
    assert {record.attempt for record in retry_records} == {1}
    assert all(record.max_attempts == 2 for record in retry_records)
    assert [result.result for result in _results(events)] == ["left", "right"]
    assert provider.calls == 2


@pytest.mark.parametrize(
    ("exception", "expected_level"),
    [
        (ToolFailure("expected failure"), logging.ERROR),
        (ModelRetry("correct this"), logging.WARNING),
    ],
)
def test_semantic_tool_errors_do_not_use_unexpected_exception_logging(
    exception: Exception,
    expected_level: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    @dataclass
    class SemanticTool(Tool):
        name = "semantic"
        description = "Raise a semantic error."

        async def __call__(self) -> None:
            raise exception

    with caplog.at_level(logging.WARNING, logger="dobby"):
        asyncio.run(
            _run(
                [SemanticTool()],
                [ToolUsePart(id="call-semantic", name="semantic", inputs={})],
            )
        )

    semantic_records = [
        record for record in caplog.records if getattr(record, "tool_call_id", None)
    ]
    assert semantic_records
    assert semantic_records[0].levelno == expected_level
    assert all(record.exc_info is None for record in semantic_records)


def test_tool_retry_exhaustion_reports_final_attempt_metadata() -> None:
    calls = 0

    @dataclass
    class TimeoutTool(Tool):
        name = "timeout"
        description = "Always time out."
        retryable_exceptions = (TimeoutError,)

        async def __call__(self) -> None:
            nonlocal calls
            calls += 1
            raise TimeoutError(f"failure {calls}")

    with patch("dobby.executor.asyncio.sleep", new_callable=AsyncMock):
        events, _ = asyncio.run(
            _run(
                [TimeoutTool()],
                [ToolUsePart(id="call-timeout", name="timeout", inputs={})],
            )
        )

    details = _results(events)[0].error_details
    assert details is not None
    assert calls == 2
    assert details.attempt == 2
    assert details.max_attempts == 2
    assert details.message == "failure 2"


def test_non_invocation_corrections_have_no_attempt_metadata() -> None:
    events, _ = asyncio.run(
        _run(
            [],
            [ToolUsePart(id="call-missing", name="missing", inputs={})],
            max_model_corrections=1,
        )
    )

    details = _results(events)[0].error_details
    assert details is not None
    assert details.error_code == "tool_not_found"
    assert details.tool_name == "missing"
    assert details.tool_call_id == "call-missing"
    assert details.attempt is None
    assert details.max_attempts is None


def test_final_result_validation_has_no_invocation_attempts() -> None:
    class Output(BaseModel):
        value: int

    provider = _RecordingProvider(
        [ToolUsePart(id="call-final", name="final_result", inputs={"value": "bad"})]
    )
    executor = AgentExecutor(provider="openai", llm=provider, output_type=Output)
    events = asyncio.run(_collect(executor.run_stream(messages=[], max_model_corrections=1)))

    details = _results(events)[0].error_details
    assert details is not None
    assert details.error_code == "final_result_invalid"
    assert details.attempt is None
    assert details.max_attempts is None


async def _collect(iterator):
    return [event async for event in iterator]


def _fast_provider_retry_config(**kwargs):
    return {
        "reraise": True,
        "stop": stop_after_attempt(kwargs["max_retries"]),
        "wait": wait_none(),
        "retry": retry_if_exception_type(tuple(kwargs["errors"])),
        "before_sleep": lambda retry_state: None,
    }


def test_provider_retry_does_not_change_tool_attempt_metadata() -> None:
    @dataclass
    class BrokenTool(Tool):
        name = "broken"
        description = "Fail once invoked."

        async def __call__(self) -> None:
            raise ValueError("tool failure")

    class RetryingProvider(_RecordingProvider):
        name = "retrying"
        max_retries = 2

        def __init__(self, tool_calls):
            super().__init__(tool_calls)
            self.provider_attempts = 0

        @with_retries
        async def chat(self, messages, **kwargs):
            self.provider_attempts += 1
            if self.provider_attempts == 1:
                raise RateLimitError("provider retry", provider=self.name)
            return await super().chat(messages, **kwargs)

    provider = RetryingProvider([ToolUsePart(id="call-broken", name="broken", inputs={})])
    executor = AgentExecutor(provider="openai", llm=provider, tools=[BrokenTool()])
    with patch(
        "dobby.providers._retry.create_retry_config",
        side_effect=_fast_provider_retry_config,
    ):
        events = asyncio.run(_collect(executor.run_stream(messages=[])))

    details = _results(events)[0].error_details
    assert details is not None
    assert details.attempt == 1
    assert provider.provider_attempts == 3
    assert provider.calls == 2


def test_exhausted_provider_error_remains_provider_error_and_is_correlated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingProvider:
        name = "failing"
        model = "mock"
        max_retries = 0

        async def chat(self, *args, **kwargs):
            del args, kwargs
            raise ProviderError("provider failed", provider=self.name)

    executor = AgentExecutor(provider="openai", llm=FailingProvider())

    async def run() -> None:
        with pytest.raises(ProviderError, match="provider failed"):
            async for _ in executor.run_stream(messages=[]):
                pass

    with caplog.at_level(logging.ERROR, logger="dobby"):
        asyncio.run(run())

    record = next(record for record in caplog.records if record.message.startswith("Provider"))
    assert record.layer == "provider"
    assert record.run_id
    assert record.provider == "failing"
