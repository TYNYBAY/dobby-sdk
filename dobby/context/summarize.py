"""LLM-backed summarize: replace an old span of turns with one ``<summary>`` turn.

Unlike trim (deterministic, recomputed each turn), summarize issues one model
call and writes a non-empty digest *back* into the live message list. The
executor watermarks that edit so a later combined token basis at or below that
count does not summarize again. An empty digest does not advance the watermark,
so a later turn can still compact. A provider failure is not watermarked.

A later summarize folds earlier ``<summary>`` turns that sit in the compacted
prefix into the new digest and removes them, so summary messages do not
accumulate. Other user and assistant text stays in place.
"""

import dataclasses
from typing import NamedTuple

from ..providers.base import Provider
from ..types import (
    AppliedEdit,
    MessagePart,
    StreamEndEvent,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    UserMessagePart,
)
from ._tokens import part_to_text
from .edit import _find_tool_round_trips
from .policy import ContextPolicy

_SUMMARY_PREFIX = "<summary>"
_SUMMARY_SUFFIX = "</summary>"

SUMMARIZE_PROMPT = (
    "You are compacting an AI agent's tool-interaction history to save context. "
    "Compress the following interactions into a concise factual digest. Preserve "
    "identifiers, concrete values, decisions made, file paths, and any result the "
    "agent may still need later. If an earlier <summary> digest appears in the "
    "transcript, carry its facts forward into the new digest. Drop redundancy and "
    "chatter. Return only the digest."
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


def _prior_summary_text(message: MessagePart) -> str | None:
    """Return the wrapped summary text when ``message`` is one compaction summary turn."""
    if not isinstance(message, UserMessagePart) or len(message.parts) != 1:
        return None
    part = message.parts[0]
    if not isinstance(part, TextPart):
        return None
    text = part.text.strip()
    if (
        text.startswith(_SUMMARY_PREFIX)
        and text.endswith(_SUMMARY_SUFFIX)
        and len(text) > len(_SUMMARY_PREFIX) + len(_SUMMARY_SUFFIX)
    ):
        return text
    return None


def _summary_fold_indices(
    messages: list[MessagePart],
    candidates: list[tuple[int, int]],
    keep_last_n: int,
    protect_after: int | None,
) -> set[int]:
    """Indexes of prior summaries that this pass should absorb.

    Only summaries before the kept tail are folded. User instructions and
    assistant text are left alone, including text that merely mentions a summary.
    """
    if keep_last_n > 0:
        boundary = candidates[-keep_last_n][0]
    else:
        boundary = protect_after if protect_after is not None else len(messages)
    return {
        index
        for index, message in enumerate(messages)
        if index < boundary and _prior_summary_text(message) is not None
    }


def _span_text(messages: list[MessagePart]) -> str:
    """Render a span of messages as plain role-tagged text for the summarizer."""
    lines = []
    for msg in messages:
        body = " ".join(part_to_text(p, labeled=True) for p in msg.parts)
        lines.append(f"{msg.role}: {body}")
    return "\n".join(lines)


def _is_clearable_tool_part(part: object, clearable_ids: set[str]) -> bool:
    """True when ``part`` is a tool use or result belonging to a clearable pair."""
    if isinstance(part, ToolUsePart):
        return part.id in clearable_ids
    if isinstance(part, ToolResultPart):
        return part.tool_use_id in clearable_ids
    return False


def _message_without_clearable_parts(
    message: MessagePart, clearable_ids: set[str]
) -> MessagePart | None:
    """Drop clearable tool parts. ``None`` when nothing remains."""
    kept = [part for part in message.parts if not _is_clearable_tool_part(part, clearable_ids)]
    if not kept:
        return None
    if len(kept) == len(message.parts):
        return message
    return dataclasses.replace(message, parts=kept)


def _message_clearable_span(message: MessagePart, clearable_ids: set[str]) -> MessagePart | None:
    """Message containing only the clearable tool parts, if any."""
    clearable = [part for part in message.parts if _is_clearable_tool_part(part, clearable_ids)]
    if not clearable:
        return None
    if len(clearable) == len(message.parts):
        return message
    return dataclasses.replace(message, parts=clearable)


def _summary_insert_index(
    messages: list[MessagePart],
    *,
    clearable_ids: set[str],
    fold_indices: set[int],
    kept_pairs: list[tuple[int, int]],
) -> int:
    """Index at which to insert the new ``<summary>`` turn.

    Starts at the first folded summary or message that holds a clearable part
    (the previous "first removed entry" rule). If that point sits strictly
    inside a kept pair, it moves to that pair's tool-use so the kept use and
    result stay together.
    """
    touched = [
        index
        for index, message in enumerate(messages)
        if index in fold_indices
        or any(_is_clearable_tool_part(part, clearable_ids) for part in message.parts)
    ]
    insert_at = min(touched)
    for use_idx, result_idx in kept_pairs:
        if use_idx < insert_at < result_idx:
            insert_at = use_idx
    return insert_at


async def summarize_context(
    messages: list[MessagePart],
    policy: ContextPolicy,
    llm: Provider,
    *,
    extra_instructions: str | None = None,
) -> AppliedEdit | None:
    """Summarize the old span of ``messages`` in place, write-back style.

    Selects tool round-trips older than ``policy.keep_last_n`` (never an
    in-flight tool use). Those tool-use and tool-result *parts* are summarized
    and removed; a message that still has other parts (kept tool parts or
    normal text) stays. Earlier ``<summary>`` turns in that same prefix are
    included in the digest and removed so repeated summarizes collapse to one
    summary message. User instructions and assistant text between pairs stay
    in place. One non-streaming model call reusing ``llm`` summarizes the
    span, then a single new ``<summary>`` user turn is inserted at the first
    removed entry, or before a kept pair that insertion would otherwise split.
    Only the list entries are mutated — fresh dataclass instances are created,
    so the caller's part objects are never touched. The replaced originals are
    stashed on the returned edit. The executor, not this function, suppresses
    another attempt until the combined token basis grows past that edit.

    Args:
        messages: The live working message list (mutated in place).
        policy: Compaction policy (uses ``keep_last_n``).
        llm: The parent :class:`~dobby.providers.base.Provider`, reused for the
            non-streaming summarize call.
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
    llm: Provider,
    *,
    extra_instructions: str | None = None,
    protect_after: int | None = None,
) -> _SummarizeAttempt:
    """Shared implementation of :func:`summarize_context`.

    The executor uses ``blank_digest`` to skip a :class:`~dobby.types.tool_events.ContextEditEvent`
    without advancing the summarize watermark.

    ``protect_after`` is the length of ``messages`` before the current tool
    batch was appended. Pairs whose tool-use index is at or after that point
    (this turn's sibling tools and ``compact_context`` itself) are not
    clearable, so ``keep_last_n`` still applies only to older history.
    """
    trips = _find_tool_round_trips(messages)
    if protect_after is None:
        candidates = trips
    else:
        candidates = [trip for trip in trips if trip[0] < protect_after]
    if len(candidates) <= policy.keep_last_n:
        return _SummarizeAttempt(None, False)

    keep_last_n = policy.keep_last_n
    clearable = candidates[: len(candidates) - keep_last_n]
    kept_pairs = (
        [(use_idx, result_idx) for use_idx, result_idx, _ in candidates[-keep_last_n:]]
        if keep_last_n
        else []
    )
    clearable_ids = {tool_id for _, _, tool_id in clearable}
    fold_indices = _summary_fold_indices(
        messages,
        [(use_idx, result_idx) for use_idx, result_idx, _ in candidates],
        keep_last_n,
        protect_after,
    )
    span: list[MessagePart] = []
    for index, message in enumerate(messages):
        if index in fold_indices:
            span.append(message)
            continue
        piece = _message_clearable_span(message, clearable_ids)
        if piece is not None:
            span.append(piece)
    # Count the result parts whose ids are actually removed, not every
    # ToolResultPart sharing a message with a clearable pair.
    cleared_tool_uses = sum(
        1
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart) and part.tool_use_id in clearable_ids
    )

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
    insert_at = _summary_insert_index(
        messages,
        clearable_ids=clearable_ids,
        fold_indices=fold_indices,
        kept_pairs=kept_pairs,
    )
    rebuilt: list[MessagePart] = []
    inserted = False
    for idx, msg in enumerate(messages):
        if idx == insert_at and not inserted:
            rebuilt.append(summary_msg)
            inserted = True
        if idx in fold_indices:
            continue
        stripped = _message_without_clearable_parts(msg, clearable_ids)
        if stripped is None:
            continue
        rebuilt.append(stripped)
    if not inserted:
        rebuilt.append(summary_msg)
    messages[:] = rebuilt

    return _SummarizeAttempt(
        AppliedEdit(
            type="summarize",
            cleared_tool_uses=cleared_tool_uses,
            summary_text=summary_text,
            replaced_originals=replaced_originals,
        ),
        False,
    )
