"""Run and tool-call correlation context for host-side observability."""

from contextvars import ContextVar

run_id_var: ContextVar[str | None] = ContextVar("dobby_run_id", default=None)
tool_name_var: ContextVar[str | None] = ContextVar("dobby_tool_name", default=None)
tool_call_id_var: ContextVar[str | None] = ContextVar("dobby_tool_call_id", default=None)
tool_attempt_var: ContextVar[int | None] = ContextVar("dobby_tool_attempt", default=None)
tool_max_attempts_var: ContextVar[int | None] = ContextVar(
    "dobby_tool_max_attempts",
    default=None,
)


def correlation_fields() -> dict[str, str | int]:
    """Return non-empty correlation fields for structured log extras."""
    fields: dict[str, str | int] = {}
    for key, variable in (
        ("run_id", run_id_var),
        ("tool_name", tool_name_var),
        ("tool_call_id", tool_call_id_var),
        ("attempt", tool_attempt_var),
        ("max_attempts", tool_max_attempts_var),
    ):
        value = variable.get()
        if value is not None:
            fields[key] = value
    return fields
