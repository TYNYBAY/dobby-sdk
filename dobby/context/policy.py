"""Context compaction policy configuration.

Defines :class:`ContextPolicy`, the opt-in config object that controls when and
how the agentic loop reduces stale tool history before each model call.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ContextPolicy(BaseModel):
    """Configuration for automatic context compaction.

    Constructing a policy is the BETA gate: passing one to ``AgentExecutor``
    enables compaction, while ``None`` (the default) preserves today's behavior
    exactly.

    The trigger fires when the previous turn's input tokens cross
    :attr:`trigger_tokens` (``trigger_pct`` of ``context_window``). The window is
    supplied here rather than looked up, since providers expose only ``model``.

    Attributes:
        context_window: Total context-window size, in tokens, for the model.
        trigger_pct: Fraction of ``context_window`` that triggers compaction.
        keep_last_n: Number of most-recent tool turns kept verbatim.
        mode: ``"trim"`` (deterministic, no LLM) or ``"summarize"`` (one LLM call).
        placeholder: Text substituted for cleared tool-result payloads in trim mode.
    """

    context_window: int = Field(default=128_000, gt=0)
    trigger_pct: float = Field(default=0.8, gt=0, le=1)
    keep_last_n: int = Field(default=3, ge=0)
    mode: Literal["trim", "summarize"] = "trim"
    placeholder: str = "[Tool result cleared to save context.]"

    @property
    def trigger_tokens(self) -> int:
        """Input-token count at which compaction fires.

        Returns:
            ``int(trigger_pct * context_window)``.
        """
        return int(self.trigger_pct * self.context_window)
