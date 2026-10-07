"""Deterministic, turn-aware context trimming (no LLM).

:func:`edit_context` replaces stale tool-result payloads with a small placeholder
on a *fresh* message list, leaving the full record passed in untouched.
"""

import dataclasses

from ..types import (
    AppliedEdit,
    AssistantMessagePart,
    MessagePart,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    UserMessagePart,
)
from .policy import ContextPolicy


def _tool_use_ids(msg: MessagePart) -> list[str]:
    """Tool-call ids on an assistant message, otherwise empty."""
    if not isinstance(msg, AssistantMessagePart):
        return []
    return [part.id for part in msg.parts if isinstance(part, ToolUsePart)]


def _tool_result_ids(msg: MessagePart) -> list[str]:
    """Tool-result ids on a user message, otherwise empty."""
    if not isinstance(msg, UserMessagePart):
        return []
    return [part.tool_use_id for part in msg.parts if isinstance(part, ToolResultPart)]


def _find_tool_round_trips(messages: list[MessagePart]) -> list[tuple[int, int, str]]:
    """Complete tool round-trips as ``(tool_use_index, tool_result_index, tool_use_id)``.

    Pairs a ``ToolUsePart`` with the later ``ToolResultPart`` that shares its
    ``tool_use_id``. Adjacent executor-shaped use/result messages still pair;
    interleaved uses then results (A, B, result-A, result-B) pair by id rather
    than by neighbor. Unrelated ids are not paired. A trailing tool-use with no
    matching result (in-flight) is excluded, so the current/unanswered turn is
    never a compaction candidate. Shared by the trim and summarize paths.
    """
    unmatched: dict[str, int] = {}
    pairs: list[tuple[int, int, str]] = []
    for index, message in enumerate(messages):
        for use_id in _tool_use_ids(message):
            unmatched.setdefault(use_id, index)
        for result_id in _tool_result_ids(message):
            use_index = unmatched.pop(result_id, None)
            if use_index is not None:
                pairs.append((use_index, index, result_id))
    return pairs


def _find_tool_pairs(messages: list[MessagePart]) -> list[tuple[int, int]]:
    """Index complete tool round-trips as ``(tool_use_index, tool_result_index)``."""
    return [
        (use_index, result_index)
        for use_index, result_index, _ in _find_tool_round_trips(messages)
    ]


def _cleared_tool_result_count(messages: list[MessagePart], result_indices: set[int]) -> int:
    """Count tool-result payloads on messages that compaction actually edits.

    Trim reports this as ``cleared_tool_uses``. Executor-shaped history has one
    result per round-trip, so the count matches the number of cleared pairs.
    When several results share one user message, every result on that message
    is cleared together, and each one counts.
    """
    total = 0
    for index in result_indices:
        message = messages[index]
        if not isinstance(message, UserMessagePart):
            continue
        total += sum(isinstance(part, ToolResultPart) for part in message.parts)
    return total


def edit_context(
    messages: list[MessagePart], policy: ContextPolicy
) -> tuple[list[MessagePart], AppliedEdit | None]:
    """Trim stale tool results into a placeholder on a fresh message list.

    Identifies tool round-trips by matching ``ToolUsePart.id`` to
    ``ToolResultPart.tool_use_id`` (adjacent executor-shaped pairs still match),
    keeps the most recent ``policy.keep_last_n`` of them plus any in-flight
    (unanswered) tool use verbatim, and replaces the tool-result payloads of
    older round-trips with ``policy.placeholder``.

    The round-trip skeleton is preserved: the assistant tool-use message (and any
    Gemini thought-signature in its metadata) is reused by identity, and only the
    *result* payload is swapped, so every ``ToolResultPart`` keeps its matching
    ``ToolUsePart`` and ``tool_use_id`` — no provider pairing ever breaks.

    The input ``messages`` list and its part objects are never mutated; kept
    entries are reused by identity, and only cleared entries are new instances.

    Args:
        messages: The full conversation record.
        policy: Compaction policy (uses ``keep_last_n`` and ``placeholder``).

    Returns:
        ``(new_messages, applied_edit)`` when something was cleared, otherwise
        ``(messages, None)``.
    """
    pairs = _find_tool_pairs(messages)
    if len(pairs) <= policy.keep_last_n:
        return messages, None  # nothing older than keep_last_n to clear

    # Clear every round-trip except the most recent keep_last_n.
    clear_cutoff = len(pairs) - policy.keep_last_n
    indices_to_clear = {result_idx for _, result_idx in pairs[:clear_cutoff]}
    cleared_tool_uses = _cleared_tool_result_count(messages, indices_to_clear)
    if cleared_tool_uses == 0:
        return messages, None

    new_messages: list[MessagePart] = []
    cleared_chars = 0
    for idx, msg in enumerate(messages):
        if idx not in indices_to_clear:
            new_messages.append(msg)  # kept verbatim (object-identical)
            continue

        assert isinstance(msg, UserMessagePart)
        new_parts: list = []
        for part in msg.parts:
            if isinstance(part, ToolResultPart):
                cleared_chars += sum(len(p.text) for p in part.parts if isinstance(p, TextPart))
                new_parts.append(
                    dataclasses.replace(part, parts=[TextPart(text=policy.placeholder)])
                )
            else:
                new_parts.append(part)
        new_messages.append(dataclasses.replace(msg, parts=new_parts))

    applied = AppliedEdit(
        type="clear_tool_uses",
        cleared_tool_uses=cleared_tool_uses,
        cleared_input_tokens_estimate=cleared_chars // 4,
    )
    return new_messages, applied
