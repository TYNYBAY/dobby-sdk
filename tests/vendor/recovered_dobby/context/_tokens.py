"""Shared token/text helpers for context compaction.

Both the trigger's token estimate (executor) and the summarizer's span rendering
(summarize) need to flatten message parts to text; this is the one place that
logic lives.
"""

from ..types import MessagePart, TextPart, ToolResultPart, ToolUsePart


def part_to_text(part: object, *, labeled: bool = False) -> str:
    """Flatten a message part to text.

    Args:
        part: A message part (TextPart, ToolUsePart, ToolResultPart, or other).
        labeled: When True, tool calls/results are wrapped in human-readable
            ``[tool call ...]`` / ``[tool result ...]`` markers for a summarizer.
            When False, raw text is returned for token estimation.

    Returns:
        The flattened text.
    """
    if isinstance(part, TextPart):
        return part.text
    if isinstance(part, ToolUsePart):
        return (
            f"[tool call {part.name}({part.inputs})]" if labeled else f"{part.name}{part.inputs}"
        )
    if isinstance(part, ToolResultPart):
        inner = "".join(part_to_text(p, labeled=labeled) for p in part.parts)
        return f"[tool result {part.name}: {inner}]" if labeled else inner
    return str(part)


def estimate_input_tokens(messages: list[MessagePart]) -> int:
    """Rough input-token estimate (~chars / 4) for when provider usage is absent.

    Keeps the percentage trigger engaged on providers/turns that omit usage
    metadata (e.g. Gemini without ``usage_metadata``), per KTD-1.
    """
    chars = sum(len(part_to_text(part)) for msg in messages for part in msg.parts)
    return chars // 4
