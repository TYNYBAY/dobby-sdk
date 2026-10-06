"""Shared token/text helpers for context compaction.

Both the trigger's token estimate (executor) and the summarizer's span rendering
(summarize) need to flatten message parts to text; this is the one place that
logic lives.
"""

from ..types import (
    Base64ImageSource,
    DocumentPart,
    FileDocumentSource,
    ImagePart,
    MessagePart,
    PlainTextSource,
    ReasoningPart,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    URLImageSource,
    URLSource,
)


def _image_to_text(part: ImagePart) -> str:
    """Describe an image without its raw bytes or data URL."""
    source = part.source
    if isinstance(source, URLImageSource) and not source.url.startswith("data:"):
        return f"[image {source.url}]"
    if isinstance(source, Base64ImageSource):
        return f"[image {source.media_type}]"
    return "[image]"


def _document_to_text(part: DocumentPart, *, labeled: bool) -> str:
    """Describe a document without embedded PDF bytes.

    Plain-text document bodies are real model input, so they are kept.
    Base64 PDF payloads and ``data:`` URLs are not.
    """
    source = part.source
    if isinstance(source, PlainTextSource):
        if labeled:
            return f"[document {part.filename}] {source.data}"
        return source.data
    if isinstance(source, URLSource) and not source.url.startswith("data:"):
        return f"[document {part.filename} {source.url}]"
    if isinstance(source, FileDocumentSource):
        return f"[document {part.filename} {source.file_id}]"
    return f"[document {part.filename}]"


def _reasoning_to_text(part: ReasoningPart, *, labeled: bool) -> str:
    """Use readable reasoning text, never signatures or redacted payloads."""
    if part.redacted:
        return "[redacted reasoning]"
    if not part.text:
        return ""
    if labeled:
        return f"[reasoning] {part.text}"
    return part.text


def part_to_text(part: object, *, labeled: bool = False) -> str:
    """Flatten a message part to text.

    Args:
        part: A message part (TextPart, ToolUsePart, ToolResultPart, or other).
        labeled: When True, tool calls/results are wrapped in human-readable
            ``[tool call ...]`` / ``[tool result ...]`` markers for a summarizer.
            When False, raw text is returned for token estimation.

    Returns:
        The flattened text. Image bytes, PDF bytes, ``data:`` URLs, reasoning
        signatures, and redacted reasoning payloads are omitted; the part
        objects themselves are not modified.
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
    if isinstance(part, ImagePart):
        return _image_to_text(part)
    if isinstance(part, DocumentPart):
        return _document_to_text(part, labeled=labeled)
    if isinstance(part, ReasoningPart):
        return _reasoning_to_text(part, labeled=labeled)
    return f"[{type(part).__name__}]"


def estimate_input_tokens(messages: list[MessagePart]) -> int:
    """Rough input-token estimate (~chars / 4) for when provider usage is absent.

    Keeps the percentage trigger engaged on providers/turns that omit usage
    metadata (e.g. Gemini without ``usage_metadata``).
    """
    chars = sum(len(part_to_text(part)) for msg in messages for part in msg.parts)
    return chars // 4


def compaction_trigger_basis(
    last_input_tokens: int | None,
    outgoing_messages: list[MessagePart],
) -> int | None:
    """Token count used to decide whether to compact before the next model call.

    Provider-reported input from the previous turn is combined with a char
    estimate of the live outgoing message list so tool results appended after
    that turn still count toward the trigger. When there is no previous usage
    (the first model call), the outgoing estimate is used alone so an already
    oversized initial history can compact before that call.
    """
    estimated = estimate_input_tokens(outgoing_messages)
    if last_input_tokens is None:
        return estimated
    return max(last_input_tokens, estimated)
