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


def _is_tool_use_message(msg: MessagePart) -> bool:
    """True when ``msg`` is an assistant message carrying a tool call."""
    return isinstance(msg, AssistantMessagePart) and any(
        isinstance(p, ToolUsePart) for p in msg.parts
    )


def _is_tool_result_message(msg: MessagePart) -> bool:
    """True when ``msg`` is a user message carrying a tool result."""
    return isinstance(msg, UserMessagePart) and any(
        isinstance(p, ToolResultPart) for p in msg.parts
    )


def _find_tool_pairs(messages: list[MessagePart]) -> list[tuple[int, int]]:
    """Index complete tool round-trips as ``(tool_use_index, tool_result_index)``.

    A trailing tool-use with no following tool-result (in-flight) is excluded, so
    the current/unanswered turn is never a compaction candidate. Shared by the
    trim and summarize paths.
    """
    pairs: list[tuple[int, int]] = []
    i = 0
    n = len(messages)
    while i < n:
        if (
            _is_tool_use_message(messages[i])
            and i + 1 < n
            and _is_tool_result_message(messages[i + 1])
        ):
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    return pairs


def edit_context(
    messages: list[MessagePart], policy: ContextPolicy
) -> tuple[list[MessagePart], AppliedEdit | None]:
    """Trim stale tool results into a placeholder on a fresh message list.

    Identifies tool round-trips (an assistant tool-use message immediately
    followed by its user tool-result message), keeps the most recent
    ``policy.keep_last_n`` of them plus any in-flight (unanswered) tool use
    verbatim, and replaces the tool-result payloads of older round-trips with
    ``policy.placeholder``.

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

    new_messages: list[MessagePart] = []
    cleared_tool_uses = 0
    cleared_chars = 0
    for idx, msg in enumerate(messages):
        if idx not in indices_to_clear:
            new_messages.append(msg)  # kept verbatim (object-identical)
            continue

        assert isinstance(msg, UserMessagePart)
        new_parts: list = []
        for part in msg.parts:
            if isinstance(part, ToolResultPart):
                cleared_tool_uses += 1
                cleared_chars += sum(len(p.text) for p in part.parts if isinstance(p, TextPart))
                new_parts.append(
                    dataclasses.replace(part, parts=[TextPart(text=policy.placeholder)])
                )
            else:
                new_parts.append(part)
        new_messages.append(dataclasses.replace(msg, parts=new_parts))

    if cleared_tool_uses == 0:
        return messages, None

    applied = AppliedEdit(
        type="clear_tool_uses",
        cleared_tool_uses=cleared_tool_uses,
        cleared_input_tokens_estimate=cleared_chars // 4,
    )
    return new_messages, applied
