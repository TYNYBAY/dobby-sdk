"""Executable demonstration of Dobby's error-handling behavior.

This is not a test suite. Existing tests prove correctness. This script drives
the real AgentExecutor with deterministic fake tools and a scripted model so
you can inspect model-facing results, host diagnostics, and logs side by side.

No provider API keys are required.

Usage:
    uv run python examples/error_handling_demo.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dobby import AgentExecutor
from dobby.exceptions import ApprovalRequired, ModelRetry
from dobby.tools import Tool
from dobby.types import (
    StreamEndEvent,
    ToolResultEvent,
    ToolResultPart,
    ToolUsePart,
    Usage,
    UserMessagePart,
)

SEPARATOR = "=" * 60


class ScriptedProvider:
    """Deterministic stand-in for an LLM: one scripted tool-call batch per turn.

    This is the only fake needed. Tools, classification, retry, validation,
    logging, and control-flow assembly all run through AgentExecutor.
    """

    name = "scripted"
    model = "demo-model"

    def __init__(self, turns: list[list[ToolUsePart]]) -> None:
        self.turns = turns
        self.chat_messages: list[list[Any]] = []

    async def chat(self, messages, **kwargs):
        """Record history and stream the next scripted assistant turn."""
        del kwargs
        self.chat_messages.append(list(messages))
        call_index = len(self.chat_messages) - 1
        parts = self.turns[call_index] if call_index < len(self.turns) else []
        stop_reason = "tool_use" if parts else "end_turn"

        async def stream():
            yield StreamEndEvent(
                type="stream_end",
                model=self.model,
                parts=parts,
                stop_reason=stop_reason,
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()


@dataclass
class ScenarioRun:
    """Captured output from one AgentExecutor.run_stream() pass."""

    events: list[Any]
    provider: ScriptedProvider
    logs: list[logging.LogRecord]
    log_text: list[str]
    host_exception: BaseException | None = None


def _configure_demo_logging() -> logging.Logger:
    """Attach a capturing handler to Dobby's logger for this process."""
    logger = logging.getLogger("dobby")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def _history_tool_results(messages: list[Any]) -> list[ToolResultPart]:
    """Return tool-result parts the model would see on a later chat turn."""
    return [
        part
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]


def _result_events(events: list[Any]) -> list[ToolResultEvent]:
    """Return tool-result events yielded by the executor."""
    return [event for event in events if isinstance(event, ToolResultEvent)]


def _requested_tool_calls(events: list[Any]) -> list[ToolUsePart]:
    """Return tool calls the scripted model asked for."""
    calls: list[ToolUsePart] = []
    for event in events:
        if isinstance(event, StreamEndEvent):
            for part in event.parts:
                if isinstance(part, ToolUsePart):
                    calls.append(part)
    return calls


async def _run_executor(
    tools: list[Tool],
    turns: list[list[ToolUsePart]],
    logger: logging.Logger,
    **run_kwargs: Any,
) -> ScenarioRun:
    """Run AgentExecutor and collect events, logs, and any host exception."""
    provider = ScriptedProvider(turns)
    executor = AgentExecutor(provider="openai", llm=provider, tools=tools)

    records: list[logging.LogRecord] = []
    lines: list[str] = []
    handler = logging.Handler()
    handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    def emit(record: logging.LogRecord) -> None:
        records.append(record)
        lines.append(handler.format(record))

    handler.emit = emit  # type: ignore[method-assign]
    logger.addHandler(handler)

    events: list[Any] = []
    host_exception: BaseException | None = None
    try:
        async for event in executor.run_stream(messages=[], **run_kwargs):
            events.append(event)
    except Exception as exception:
        host_exception = exception
    finally:
        logger.removeHandler(handler)

    return ScenarioRun(
        events=events,
        provider=provider,
        logs=records,
        log_text=lines,
        host_exception=host_exception,
    )


def _print_header(title: str) -> None:
    """Print a scenario banner."""
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


def _print_flow(steps: list[str]) -> None:
    """Print an execution-flow diagram built from the run."""
    print("EXECUTION FLOW")
    for index, step in enumerate(steps):
        print(f"  {step}")
        if index < len(steps) - 1:
            print("    ↓")
    print()


def _print_block(label: str, lines: list[str]) -> None:
    """Print a labeled section."""
    print(label)
    if not lines:
        print("  (none)")
    else:
        for line in lines:
            for wrapped in line.splitlines() or [""]:
                print(f"  {wrapped}")
    print()


def _model_lines(run: ScenarioRun) -> list[str]:
    """Model-facing content from yielded results and later chat history."""
    lines: list[str] = ["Yielded ToolResultEvent (this is what the runtime exposes as result):"]
    results = _result_events(run.events)
    if not results:
        lines.append("  (no tool results)")
    for event in results:
        lines.append(
            f"  {event.name} id={event.tool_use_id} is_error={event.is_error} "
            f"is_terminal={event.is_terminal} -> {event.result!r}"
        )

    later_turns = run.provider.chat_messages[1:]
    if not later_turns:
        lines.append("No later model turn (the run ended before another chat call).")
        return lines
    for turn_index, messages in enumerate(later_turns, start=2):
        lines.append(f"Content sent to the model on chat turn {turn_index}:")
        history = _history_tool_results(messages)
        if not history:
            lines.append("  (no new tool-result history parts on this turn)")
            continue
        for part in history:
            text = part.parts[0].text if part.parts else ""
            lines.append(
                f"  {part.name} id={part.tool_use_id} is_error={part.is_error} -> {text!r}"
            )
    return lines


def _host_lines_from_result(event: ToolResultEvent) -> list[str]:
    """Host diagnostics from a ToolResultEvent.error_details payload."""
    details = event.error_details
    if details is None:
        return [
            f"{event.name} id={event.tool_use_id}: error_details=None "
            "(no classified host diagnostics on this result)."
        ]
    lines = [
        f"{event.name} id={event.tool_use_id}",
        f"exception_type={details.exception_type}",
        f"exception_module={details.exception_module}",
        f"message={details.message!r}",
        f"error_code={details.error_code}",
        f"run_id={details.run_id}",
        f"tool_call_id={details.tool_call_id}",
        f"attempt={details.attempt}/{details.max_attempts}",
        "traceback:",
        details.traceback.rstrip() or "(empty)",
    ]
    return lines


def _log_lines(run: ScenarioRun) -> list[str]:
    """Logger output captured from the dobby logger."""
    if not run.log_text:
        return ["(dobby logger emitted nothing at DEBUG+)"]
    return list(run.log_text)


def _traceback_in_model_content(run: ScenarioRun) -> bool:
    """Return whether a CPython traceback dump is present in model-facing text."""
    needles = ("Traceback (most recent call last):", "traceback (most recent call last):")
    blobs: list[str] = []
    for event in _result_events(run.events):
        blobs.append(str(event.result))
    for messages in run.provider.chat_messages[1:]:
        for part in _history_tool_results(messages):
            if part.parts:
                blobs.append(part.parts[0].text)
    return any(needle in blob for blob in blobs for needle in needles)


# --- Scenario tools -----------------------------------------------------------------


@dataclass
class FlakyLookupTool(Tool):
    """Fail once with a listed retryable exception, then succeed."""

    name = "lookup_account"
    description = "Look up an account by id."
    retryable_exceptions = (TimeoutError,)
    invocations: list[str] = field(default_factory=list)

    async def __call__(self, account_id: str) -> dict[str, str]:
        self.invocations.append(account_id)
        if len(self.invocations) == 1:
            raise TimeoutError("account service timed out")
        return {"account_id": account_id, "status": "active"}


@dataclass
class AlwaysTimeoutTool(Tool):
    """Keep raising a listed retryable exception until the retry budget ends."""

    name = "lookup_account"
    description = "Look up an account by id."
    retryable_exceptions = (TimeoutError,)
    invocations: list[str] = field(default_factory=list)

    async def __call__(self, account_id: str) -> dict[str, str]:
        self.invocations.append(account_id)
        raise TimeoutError(f"account service timed out on attempt {len(self.invocations)}")


@dataclass
class TypedLookupTool(Tool):
    """Require a string account_id; records only successful body entry."""

    name = "lookup_account"
    description = "Look up an account by id."
    invocations: list[str] = field(default_factory=list)

    async def __call__(self, account_id: str) -> dict[str, str]:
        self.invocations.append(account_id)
        return {"account_id": account_id, "status": "active"}


@dataclass
class NeedsCorrectionTool(Tool):
    """Execute, then ask the model to correct the call via ModelRetry."""

    name = "refund"
    description = "Issue a refund for an order."
    invocations: list[str] = field(default_factory=list)

    async def __call__(self, order_id: str) -> dict[str, str]:
        self.invocations.append(order_id)
        if order_id == "draft":
            raise ModelRetry("order_id must be a confirmed order, not a draft")
        return {"order_id": order_id, "refund": "issued"}


@dataclass
class BrokenLookupTool(Tool):
    """Raise an ordinary unexpected exception."""

    name = "lookup_account"
    description = "Look up an account by id."
    invocations: list[str] = field(default_factory=list)

    async def __call__(self, account_id: str) -> dict[str, str]:
        self.invocations.append(account_id)
        raise RuntimeError("secret host diagnostic: connection reset on shard-7")


@dataclass
class DeleteRecordTool(Tool):
    """Regular tool that requires host approval before the body runs."""

    name = "delete_record"
    description = "Delete a record. Requires approval."
    requires_approval = True
    invocations: list[str] = field(default_factory=list)

    async def __call__(self, record_id: str) -> dict[str, str]:
        self.invocations.append(record_id)
        return {"record_id": record_id, "deleted": "true"}


@dataclass
class HangUpTool(Tool):
    """Terminal tool that must not run after a prior ApprovalRequired."""

    name = "hang_up"
    description = "End the conversation."
    terminal = True
    invocations: list[str] = field(default_factory=list)

    async def __call__(self) -> str:
        self.invocations.append("ran")
        return "ended"


# --- Scenarios ----------------------------------------------------------------------


async def scenario_retry_success(logger: logging.Logger) -> None:
    """Demonstrate host-side ToolRetryPolicy retry that later succeeds."""
    tool = FlakyLookupTool()
    run = await _run_executor(
        [tool],
        [
            [ToolUsePart(id="call-lookup", name="lookup_account", inputs={"account_id": "A-100"})],
            [],
        ],
        logger,
    )
    results = _result_events(run.events)
    retry_logs = [record for record in run.logs if "Retrying tool" in record.getMessage()]

    _print_header("SCENARIO 1 — TOOL RETRY → SUCCESS")
    _print_flow(
        [
            "Model requested lookup_account(account_id='A-100')",
            f"Tool invocation #{len(tool.invocations) and 1} (body entered)",
            "TimeoutError (listed in retryable_exceptions)",
            f"Host ToolRetryPolicy retry ({len(retry_logs)} retry log line(s))",
            f"Tool invocation #{len(tool.invocations)} (same call, same arguments)",
            "Success",
        ]
    )
    _print_block("MODEL", _model_lines(run))
    host = [
        f"tool body invocations={len(tool.invocations)} args={tool.invocations!r}",
        "This is host-side tool retry, not model correction: the model was not "
        "asked to change arguments between attempts.",
    ]
    if results:
        host.extend(_host_lines_from_result(results[0]))
    _print_block("HOST", host)
    _print_block("LOGS", _log_lines(run))
    _print_block(
        "OUTCOME",
        [
            f"Final result={results[0].result!r}" if results else "No result event.",
            f"is_error={results[0].is_error}" if results else "",
            f"host_exception={type(run.host_exception).__name__ if run.host_exception else None}",
            "The first failure never became a model-facing error because retry succeeded.",
        ],
    )


async def scenario_retry_exhaustion(logger: logging.Logger) -> None:
    """Demonstrate retry budget exhaustion with a classified execution error."""
    tool = AlwaysTimeoutTool()
    run = await _run_executor(
        [tool],
        [
            [ToolUsePart(id="call-lookup", name="lookup_account", inputs={"account_id": "A-100"})],
            [],
        ],
        logger,
    )
    results = _result_events(run.events)
    result = results[0] if results else None

    _print_header("SCENARIO 2 — TOOL RETRY → EXHAUSTION")
    _print_flow(
        [
            "Model requested lookup_account(account_id='A-100')",
            f"Tool invocations={len(tool.invocations)} (all failed)",
            "Retry budget exhausted",
            "Classified tool_execution_error returned to the model; traceback stays on the host",
        ]
    )
    _print_block("MODEL", _model_lines(run))
    host = [
        f"tool body invocations={len(tool.invocations)}",
        f"last tool-raised message={tool.invocations and f'account service timed out on attempt {len(tool.invocations)}'!r}",
    ]
    if result is not None:
        host.extend(_host_lines_from_result(result))
    _print_block("HOST", host)
    _print_block("LOGS", _log_lines(run))
    outcome = []
    if result is not None:
        model_text = str(result.result)
        host_message = result.error_details.message if result.error_details else ""
        outcome.extend(
            [
                f"MODEL-FACING result={model_text!r}",
                f"HOST last exception message={host_message!r}",
                f"Raw traceback in model-facing content: {_traceback_in_model_content(run)}",
            ]
        )
    _print_block("OUTCOME", outcome)


async def scenario_invalid_input_correction(logger: logging.Logger) -> None:
    """Demonstrate validation rejection before execution, then a corrected call."""
    tool = TypedLookupTool()
    run = await _run_executor(
        [tool],
        [
            [ToolUsePart(id="call-bad", name="lookup_account", inputs={"account_id": 123})],
            [ToolUsePart(id="call-good", name="lookup_account", inputs={"account_id": "A-100"})],
            [],
        ],
        logger,
    )
    results = _result_events(run.events)
    retry_logs = [record for record in run.logs if "Retrying tool" in record.getMessage()]
    requested = _requested_tool_calls(run.events)

    _print_header("SCENARIO 3 — INVALID TOOL INPUT → MODEL CORRECTION")
    _print_flow(
        [
            f"Model turn 1: {requested[0].name}({requested[0].inputs})"
            if requested
            else "Model turn 1",
            "validate_inputs() rejected the arguments (tool body did not run)",
            "Model received [tool_input_invalid] correction feedback",
            f"Model turn 2: {requested[1].name}({requested[1].inputs})"
            if len(requested) > 1
            else "Model turn 2",
            "Corrected call succeeded",
        ]
    )
    _print_block("MODEL", _model_lines(run))
    _print_block(
        "HOST",
        [
            f"tool body invocations={len(tool.invocations)} args={tool.invocations!r}",
            "Only the corrected call entered the tool body.",
            f"Host ToolRetryPolicy retry logs={len(retry_logs)} (expected 0).",
            *(
                _host_lines_from_result(results[0])
                if results
                else ["No result events."]
            ),
        ],
    )
    _print_block("LOGS", _log_lines(run))
    _print_block(
        "OUTCOME",
        [
            "This is model correction after input validation, not ToolRetryPolicy.",
            "The invalid call never executed; the model changed its arguments.",
            f"chat turns={len(run.provider.chat_messages)}",
        ],
    )


async def scenario_model_retry_correction(logger: logging.Logger) -> None:
    """Demonstrate ModelRetry from a successful validation / executed tool."""
    tool = NeedsCorrectionTool()
    run = await _run_executor(
        [tool],
        [
            [ToolUsePart(id="call-draft", name="refund", inputs={"order_id": "draft"})],
            [ToolUsePart(id="call-ok", name="refund", inputs={"order_id": "ORD-9"})],
            [],
        ],
        logger,
    )
    results = _result_events(run.events)
    retry_logs = [record for record in run.logs if "Retrying tool" in record.getMessage()]
    warning_logs = [
        record
        for record in run.logs
        if record.levelno == logging.WARNING and "model correction" in record.getMessage()
    ]

    _print_header("SCENARIO 4 — TOOL ModelRetry → MODEL CORRECTION")
    _print_flow(
        [
            "Model called refund(order_id='draft') with valid types",
            "Tool body executed and raised ModelRetry",
            "Model received [tool_retry] correction feedback",
            "Model called refund(order_id='ORD-9')",
            "Corrected tool call succeeded",
        ]
    )
    _print_block("MODEL", _model_lines(run))
    _print_block(
        "HOST",
        [
            f"tool body invocations={len(tool.invocations)} args={tool.invocations!r}",
            f"Host ToolRetryPolicy retry logs={len(retry_logs)} (expected 0).",
            "ModelRetry is never retried by ToolRetryPolicy (see tools.retry._NEVER_RETRY).",
            f"model-correction warning logs={len(warning_logs)}",
            *(
                _host_lines_from_result(results[0])
                if results
                else ["No result events."]
            ),
        ],
    )
    _print_block("LOGS", _log_lines(run))
    _print_block(
        "OUTCOME",
        [
            "The tool ran, then asked the model to correct the call.",
            "That is model correction, not a host-side ToolRetryPolicy retry.",
            f"Final successful result={results[-1].result!r}" if results else "",
        ],
    )


async def scenario_unexpected_exception(logger: logging.Logger) -> None:
    """Demonstrate classification of an unexpected tool exception."""
    tool = BrokenLookupTool()
    run = await _run_executor(
        [tool],
        [
            [ToolUsePart(id="call-broken", name="lookup_account", inputs={"account_id": "A-100"})],
            [],
        ],
        logger,
    )
    results = _result_events(run.events)
    result = results[0] if results else None
    secret = "secret host diagnostic: connection reset on shard-7"

    _print_header("SCENARIO 5 — UNEXPECTED TOOL EXCEPTION")
    _print_flow(
        [
            "Model called lookup_account",
            "Tool raised RuntimeError (not listed, not ModelRetry/ToolFailure)",
            "Executor classified it as tool_execution_error",
            "Model received [tool_execution_error] The tool failed unexpectedly.",
            "Host error_details keep the original exception message and traceback",
        ]
    )
    _print_block("MODEL", _model_lines(run))
    host = [f"tool body invocations={len(tool.invocations)}"]
    if result is not None:
        host.extend(_host_lines_from_result(result))
    _print_block("HOST", host)
    _print_block("LOGS", _log_lines(run))

    model_text = str(result.result) if result is not None else ""
    _print_block(
        "OUTCOME",
        [
            f"MODEL classified result={model_text!r}",
            f"Exception message present in MODEL text: {secret in model_text}",
            f"CPython traceback present in MODEL text: {_traceback_in_model_content(run)}",
            f"Exception message present in HOST error_details.message: "
            f"{bool(result and result.error_details and secret in result.error_details.message)}",
            f"Traceback present on HOST error_details: "
            f"{bool(result and result.error_details and result.error_details.traceback)}",
        ],
    )


async def scenario_approval_control_flow(logger: logging.Logger) -> None:
    """Demonstrate ApprovalRequired assembly of a later unexecuted terminal tool."""
    delete_tool = DeleteRecordTool()
    hang_up = HangUpTool()
    run = await _run_executor(
        [delete_tool, hang_up],
        [
            [
                ToolUsePart(
                    id="call-delete",
                    name="delete_record",
                    inputs={"record_id": "rec-1"},
                ),
                ToolUsePart(id="call-hang-up", name="hang_up", inputs={}),
            ]
        ],
        logger,
    )
    results = _result_events(run.events)
    error = run.host_exception

    _print_header("SCENARIO 6 — APPROVAL / CONTROL FLOW")
    _print_flow(
        [
            "Model requested delete_record then hang_up (terminal) in one batch",
            "delete_record requires_approval=True and was not pre-approved",
            "ApprovalRequired raised before the delete body ran",
            "hang_up was not executed",
            "A control-flow placeholder was still assembled for hang_up",
            "ApprovalRequired propagated to the host",
        ]
    )
    _print_block("MODEL", _model_lines(run))
    host = [
        f"delete_record body invocations={len(delete_tool.invocations)} (expected 0)",
        f"hang_up body invocations={len(hang_up.invocations)} (expected 0)",
        f"host_exception type={type(error).__name__ if error else None}",
        f"host_exception={error!r}",
    ]
    if isinstance(error, ApprovalRequired):
        host.extend(
            [
                f"ApprovalRequired.tool_name={error.tool_name!r}",
                f"ApprovalRequired.tool_call_id={error.tool_call_id!r}",
                f"ApprovalRequired.tool_args={error.tool_args!r}",
            ]
        )
    for event in results:
        host.append(
            f"assembled {event.name} id={event.tool_use_id} result={event.result!r} "
            f"is_error={event.is_error} error_details={event.error_details}"
        )
    _print_block("HOST", host)
    _print_block("LOGS", _log_lines(run))
    _print_block(
        "OUTCOME",
        [
            "Approval is host control flow, not a classified tool error.",
            f"Later terminal tool ran: {bool(hang_up.invocations)}",
            "Later tool placeholder assembled: "
            + (
                str(any(event.name == "hang_up" for event in results))
            ),
            f"ApprovalRequired reached the host: {isinstance(error, ApprovalRequired)}",
        ],
    )


async def main() -> None:
    """Run all six demonstrations in order."""
    logger = _configure_demo_logging()
    print("Dobby error-handling demonstration")
    print("Uses AgentExecutor with a scripted model. No API keys required.")
    await scenario_retry_success(logger)
    await scenario_retry_exhaustion(logger)
    await scenario_invalid_input_correction(logger)
    await scenario_model_retry_correction(logger)
    await scenario_unexpected_exception(logger)
    await scenario_approval_control_flow(logger)


if __name__ == "__main__":
    asyncio.run(main())
