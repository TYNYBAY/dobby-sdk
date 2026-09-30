"""CompactContextTool — lets an agent compact its own context on demand.

When the model calls this tool, the executor records the normal tool result and
then routes the call through summarize because ``edits_context`` is true.
"""

from dataclasses import dataclass
from typing import Annotated

from .tool import Tool


@dataclass
class CompactContextTool(Tool):
    """Agent-invoked context compaction (summarize older tool history).

    Requires the executor to be constructed with a ``context_policy``; otherwise
    the call falls through as a normal tool result and does not summarize.
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
            "How many recent tool turns to keep verbatim. Omit to use the policy default.",
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
