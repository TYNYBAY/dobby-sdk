"""Tool-related exceptions."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ApprovalRequired(Exception):
    """Raised when a tool requires human approval before execution.

    This exception is raised during tool execution when:
    1. The tool has `requires_approval=True`
    2. The tool_call_id is not in the `approved_tool_calls` set

    The caller should catch this exception, present the tool call to the user,
    and re-run with the tool_call_id added to approved_tool_calls.

    Attributes:
        tool_call_id: Unique identifier for this tool call
        tool_name: Name of the tool that requires approval
        tool_args: Arguments that would be passed to the tool
    """

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]

    def __str__(self) -> str:
        return f"Tool '{self.tool_name}' requires approval (call_id: {self.tool_call_id})"
