"""CompactContextTool — lets an agent compact its own context on demand.

When the model calls this tool, the executor records the normal tool result and
then routes the call through summarize because ``edits_context`` is true.
"""

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from .tool import Tool


@dataclass
class CompactContextTool(Tool):
    """Agent-invoked context compaction (summarize older tool history).

    Not registered automatically — add an instance to ``AgentExecutor(tools=[...])``.
    After a successful call, the executor patches this tool's result to match
    whether summarize actually edited the working copy:

    * ``context_compacted`` — a digest was written back.
    * ``context_unchanged`` / ``no_policy`` — no ``context_policy``; no summarize.
    * ``context_unchanged`` / ``already_compacted`` — an automatic compaction
      already ran this turn.
    * ``context_unchanged`` / ``nothing_to_compact`` — nothing older than
      ``keep_last_n`` was eligible.
    * ``context_unchanged`` / ``empty_summary`` — summarizer returned only
      whitespace; no edit.

    When a policy is set, the executor records the tool result first, then runs
    summarize (never trim) with ``instructions`` and optional ``keep_last_n``
    overriding the policy default. A non-retryable tool error does not
    summarize. At most one compaction edit runs per agent turn; an automatic
    compaction earlier in the same turn skips the summarize step for
    ``compact_context``.
    """

    name = "compact_context"
    description = (
        "Compact your own conversation context when it has grown large. Summarizes "
        "older tool interactions into a concise digest, freeing input context while "
        "keeping recent turns verbatim. Call this when the history is long and you no "
        "longer need older tool results word-for-word."
    )
    edits_context = True

    def __call__(
        self,
        instructions: Annotated[
            str,
            "What the summary must preserve (IDs, concrete values, decisions, file paths).",
        ],
        keep_last_n: Annotated[
            int | None,
            Field(
                ge=0,
                description=(
                    "How many recent tool turns to keep verbatim. Omit to use the policy default."
                ),
            ),
        ] = None,
    ) -> dict[str, str]:
        """Return a small directive; the executor performs the actual compaction.

        Args:
            instructions: Guidance appended to the summarizer prompt.
            keep_last_n: Optional override for how many recent turns to keep.

        Returns:
            A short confirmation surfaced to the model as the tool result.
        """
        return {
            "status": "context_compacted",
            "detail": "Older tool history has been summarized into a digest above.",
        }
