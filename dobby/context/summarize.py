"""LLM-backed summarize: replace an old span of turns with one ``<summary>`` turn.

Unlike trim (deterministic, recomputed each turn), summarize issues one model
call and writes the result *back* into the live message list so it is computed
once per growth episode (gated by a watermark in the executor).
"""

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

    Selects the tool round-trips older than ``policy.keep_last_n`` (whole turns,
    never the in-flight pair), summarizes that span with one non-streaming model
    call reusing ``llm``, and **replaces** the span in the ``messages`` list with a
    single new ``<summary>`` user turn. Only the list entries are mutated — fresh
    dataclass instances are created, so the caller's part objects are never
    touched (KTD-3). The replaced originals are stashed on the returned edit.

    Args:
        messages: The live working message list (mutated in place).
        policy: Compaction policy (uses ``keep_last_n``).
        llm: A provider exposing ``chat(..., stream=False)`` (the parent model).
        extra_instructions: Optional text appended to the summarizer prompt
            (used by the agent-invoked compaction tool).

    Returns:
        An :class:`AppliedEdit` describing the summarize, or ``None`` when there
        is nothing older than ``keep_last_n`` to summarize.
    """
    pairs = _find_tool_pairs(messages)
    if len(pairs) <= policy.keep_last_n:
        return None

    clearable = pairs[: len(pairs) - policy.keep_last_n]
    span_start = clearable[0][0]  # first tool-use index
    span_end = clearable[-1][1]  # last tool-result index
    span = messages[span_start : span_end + 1]

    prompt = SUMMARIZE_PROMPT
    if extra_instructions:
        prompt = f"{SUMMARIZE_PROMPT}\n\nAdditional instructions: {extra_instructions}"

    # stream=False returns a single StreamEndEvent value to await (KTD-2).
    result: StreamEndEvent = await llm.chat(
        [UserMessagePart(parts=[TextPart(text=_span_text(span))])],
        system_prompt=prompt,
        stream=False,
        tools=None,
    )
    summary_text = "".join(p.text for p in result.parts if isinstance(p, TextPart)).strip()

    summary_msg = UserMessagePart(parts=[TextPart(text=f"<summary>{summary_text}</summary>")])
    replaced_originals = list(span)
    messages[span_start : span_end + 1] = [summary_msg]

    return AppliedEdit(
        type="summarize",
        cleared_tool_uses=len(clearable),
        summary_text=summary_text,
        replaced_originals=replaced_originals,
    )
