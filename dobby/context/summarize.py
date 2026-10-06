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

from typing import NamedTuple

from ..providers.base import Provider
from ..types import (
    AppliedEdit,
    MessagePart,
    StreamEndEvent,
    TextPart,
    UserMessagePart,
)
from ._tokens import part_to_text
from .edit import _cleared_tool_result_count, _find_tool_pairs
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


async def summarize_context(
    messages: list[MessagePart],
    policy: ContextPolicy,
    llm: Provider,
    *,
    extra_instructions: str | None = None,
) -> AppliedEdit | None:
    """Summarize the old span of ``messages`` in place, write-back style.

    Selects tool round-trips older than ``policy.keep_last_n`` (never an
    in-flight tool use). Those pair messages are summarized and removed.
    Earlier ``<summary>`` turns in that same prefix are included in the digest
    and removed so repeated summarizes collapse to one summary message. User
    instructions and assistant text between pairs stay in place. One
    non-streaming model call reusing ``llm`` summarizes the span, then a
    single new ``<summary>`` user turn is inserted at the first removed entry.
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
    pairs = _find_tool_pairs(messages)
    if protect_after is None:
        candidates = pairs
    else:
        candidates = [
            (use_idx, result_idx) for use_idx, result_idx in pairs if use_idx < protect_after
        ]
    if len(candidates) <= policy.keep_last_n:
        return _SummarizeAttempt(None, False)

    clearable = candidates[: len(candidates) - policy.keep_last_n]
    clear_indices: set[int] = set()
    span_entries: list[tuple[int, MessagePart]] = []
    for use_idx, result_idx in clearable:
        clear_indices.update((use_idx, result_idx))
        span_entries.append((use_idx, messages[use_idx]))
        span_entries.append((result_idx, messages[result_idx]))
    fold_indices = _summary_fold_indices(messages, candidates, policy.keep_last_n, protect_after)
    for index in sorted(fold_indices):
        span_entries.append((index, messages[index]))
    if fold_indices:
        span_entries.sort(key=lambda item: item[0])
    span = [message for _, message in span_entries]
    remove_indices = clear_indices | fold_indices
    # Count before write-back. Shared result messages contribute every
    # ToolResultPart on them, matching trim, not len(clearable).
    cleared_tool_uses = _cleared_tool_result_count(
        messages, {result_idx for _, result_idx in clearable}
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
    rebuilt: list[MessagePart] = []
    inserted = False
    for idx, msg in enumerate(messages):
        if idx in remove_indices:
            if not inserted:
                rebuilt.append(summary_msg)
                inserted = True
            continue
        rebuilt.append(msg)
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
