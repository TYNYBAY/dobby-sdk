"""LLM-backed summarize: replace an old span of turns with one ``<summary>`` turn.

Unlike trim (deterministic, recomputed each turn), summarize issues one model
call and writes a non-empty digest *back* into the live message list. The
executor watermarks that edit, and also a completed attempt whose digest is
empty, so the same combined token basis does not summarize again. A later
change in that basis can summarize again. A provider failure is not watermarked.
"""

from typing import NamedTuple

from ..types import (
    AppliedEdit,
    MessagePart,
    StreamEndEvent,
    TextPart,
    UserMessagePart,
)
from ._tokens import part_to_text
from .edit import _find_tool_pairs
from .policy import ContextPolicy

SUMMARIZE_PROMPT = (
    "You are compacting an AI agent's tool-interaction history to save context. "
    "Compress the following interactions into a concise factual digest. Preserve "
    "identifiers, concrete values, decisions made, file paths, and any result the "
    "agent may still need later. Drop redundancy and chatter. Return only the digest."
)


class _SummarizeAttempt(NamedTuple):
    """Outcome of one summarize pass.

    ``blank_digest`` is true only when the model call finished and produced no
    usable text. It stays false when nothing was eligible to summarize, when an
    edit was written, and when ``llm.chat`` raises (the exception propagates
    and this result is never returned).
    """

    applied: AppliedEdit | None
    blank_digest: bool


def _span_text(messages: list[MessagePart]) -> str:
    """Render a span of messages as plain role-tagged text for the summarizer."""
    lines = []
    for msg in messages:
        body = " ".join(part_to_text(p, labeled=True) for p in msg.parts)
        lines.append(f"{msg.role}: {body}")
    return "\n".join(lines)


async def summarize_context(
    messages: list[MessagePart],
    policy: ContextPolicy,
    llm,
    *,
    extra_instructions: str | None = None,
) -> AppliedEdit | None:
    """Summarize the old span of ``messages`` in place, write-back style.

    Selects tool round-trips older than ``policy.keep_last_n`` (never an
    in-flight tool use). Only those pair messages are summarized and removed;
    user instructions and assistant text between pairs stay in place. One
    non-streaming model call reusing ``llm`` summarizes the pairs, then a
    single new ``<summary>`` user turn is inserted at the first removed pair.
    Only the list entries are mutated — fresh dataclass instances are created,
    so the caller's part objects are never touched. The replaced originals are
    stashed on the returned edit. The executor, not this function, suppresses
    another attempt at the same combined token basis.

    Args:
        messages: The live working message list (mutated in place).
        policy: Compaction policy (uses ``keep_last_n``).
        llm: A provider exposing ``chat(..., stream=False)`` (the parent model).
        extra_instructions: Optional text appended to the summarizer prompt
            (used by the agent-invoked compaction tool).

    Returns:
        An :class:`AppliedEdit` describing the summarize, or ``None`` when there
        is nothing older than ``keep_last_n`` to summarize or the model returns
        an empty digest. An empty digest does not mutate ``messages``. Errors
        from ``llm.chat``, including :class:`~dobby.providers.base.ProviderError`,
        propagate to the caller.
    """
    return (
        await _summarize_attempt(messages, policy, llm, extra_instructions=extra_instructions)
    ).applied


async def _summarize_attempt(
    messages: list[MessagePart],
    policy: ContextPolicy,
    llm,
    *,
    extra_instructions: str | None = None,
) -> _SummarizeAttempt:
    """Shared implementation of :func:`summarize_context`.

    The executor uses ``blank_digest`` to advance its watermark without
    emitting a :class:`~dobby.types.tool_events.ContextEditEvent`.
    """
    pairs = _find_tool_pairs(messages)
    if len(pairs) <= policy.keep_last_n:
        return _SummarizeAttempt(None, False)

    clearable = pairs[: len(pairs) - policy.keep_last_n]
    clear_indices: set[int] = set()
    span: list[MessagePart] = []
    for use_idx, result_idx in clearable:
        clear_indices.update((use_idx, result_idx))
        span.append(messages[use_idx])
        span.append(messages[result_idx])

    prompt = SUMMARIZE_PROMPT
    if extra_instructions:
        prompt = f"{SUMMARIZE_PROMPT}\n\nAdditional instructions: {extra_instructions}"

    # stream=False returns a single StreamEndEvent value to await.
    # ProviderError and other chat failures propagate; they are not blank digests.
    result: StreamEndEvent = await llm.chat(
        [UserMessagePart(parts=[TextPart(text=_span_text(span))])],
        system_prompt=prompt,
        stream=False,
        tools=None,
    )
    summary_text = "".join(p.text for p in result.parts if isinstance(p, TextPart)).strip()
    if not summary_text:
        return _SummarizeAttempt(None, True)

    summary_msg = UserMessagePart(parts=[TextPart(text=f"<summary>{summary_text}</summary>")])
    replaced_originals = list(span)
    rebuilt: list[MessagePart] = []
    inserted = False
    for idx, msg in enumerate(messages):
        if idx in clear_indices:
            if not inserted:
                rebuilt.append(summary_msg)
                inserted = True
            continue
        rebuilt.append(msg)
    messages[:] = rebuilt

    return _SummarizeAttempt(
        AppliedEdit(
            type="summarize",
            cleared_tool_uses=len(clearable),
            summary_text=summary_text,
            replaced_originals=replaced_originals,
        ),
        False,
    )
