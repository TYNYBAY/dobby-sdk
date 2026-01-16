"""Tests for terminal tools feature."""

from dataclasses import dataclass
from typing import Any

from dobby.tools import Tool
from dobby.types import ToolResultEvent


class TestToolTerminalAttribute:
    """Test cases for Tool.terminal class attribute."""

    def test_terminal_tool_attribute(self) -> None:
        """Terminal tools have terminal=True."""

        @dataclass
        class MyTerminalTool(Tool):
            description = "Test terminal tool"
            terminal = True

            def __call__(self) -> dict[str, str]:
                return {"status": "done"}

        assert MyTerminalTool.terminal is True

    def test_non_terminal_tool_default(self) -> None:
        """Tools are non-terminal by default."""

        @dataclass
        class MyTool(Tool):
            description = "Test tool"

            def __call__(self) -> dict[str, str]:
                return {"status": "done"}

        assert MyTool.terminal is False

    def test_terminal_tool_instance(self) -> None:
        """Terminal attribute is accessible on instances."""

        @dataclass
        class EndCallTool(Tool):
            name = "end_call"
            description = "End the call"
            terminal = True

            def __call__(self, reason: str) -> dict[str, str]:
                return {"reason": reason}

        tool = EndCallTool()
        assert tool.terminal is True
        assert tool.name == "end_call"


class TestToolResultEventIsTerminal:
    """Test cases for ToolResultEvent.is_terminal flag."""

    def test_tool_result_event_default_not_terminal(self) -> None:
        """ToolResultEvent.is_terminal defaults to False."""
        event = ToolResultEvent(
            tool_use_id="test-id",
            name="test_tool",
            result={"data": "test"},
        )
        assert event.is_terminal is False

    def test_tool_result_event_terminal_true(self) -> None:
        """ToolResultEvent.is_terminal can be set to True."""
        event = ToolResultEvent(
            tool_use_id="test-id",
            name="end_call",
            result={"reason": "completed"},
            is_terminal=True,
        )
        assert event.is_terminal is True

    def test_tool_result_event_terminal_with_error(self) -> None:
        """ToolResultEvent can be both terminal and error."""
        event = ToolResultEvent(
            tool_use_id="test-id",
            name="end_call",
            result={"error": "failed"},
            is_error=True,
            is_terminal=True,
        )
        assert event.is_terminal is True
        assert event.is_error is True


class TestTerminalToolExamples:
    """Example terminal tools for documentation."""

    def test_end_interview_tool(self) -> None:
        """Example: End interview terminal tool."""

        @dataclass
        class EndInterviewTool(Tool):
            name = "end_interview"
            description = "End the screening interview"
            terminal = True

            async def __call__(self, reason: str) -> dict[str, str]:
                return {"reason": reason}

        tool = EndInterviewTool()
        assert tool.terminal is True
        assert tool.name == "end_interview"

    def test_transfer_to_human_tool(self) -> None:
        """Example: Transfer to human terminal tool."""

        @dataclass
        class TransferToHumanTool(Tool):
            name = "transfer_to_human"
            description = "Transfer the conversation to a human agent"
            terminal = True

            async def __call__(self, department: str, summary: str) -> dict[str, Any]:
                return {"department": department, "summary": summary}

        tool = TransferToHumanTool()
        assert tool.terminal is True

    def test_mixed_tools(self) -> None:
        """Non-terminal and terminal tools can coexist."""

        @dataclass
        class SearchTool(Tool):
            name = "search"
            description = "Search knowledge base"
            # terminal = False (default)

            def __call__(self, query: str) -> list[str]:
                return ["result1", "result2"]

        @dataclass
        class EscalateTool(Tool):
            name = "escalate"
            description = "Escalate to supervisor"
            terminal = True

            def __call__(self, reason: str) -> dict[str, Any]:
                return {"escalated": True, "reason": reason}

        search = SearchTool()
        escalate = EscalateTool()

        assert search.terminal is False
        assert escalate.terminal is True
