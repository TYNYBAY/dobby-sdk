"""Context compaction policy configuration.

Defines :class:`ContextPolicy`, the opt-in config object that controls when and
how the agentic loop reduces stale tool history before each model call.
"""

from decimal import ROUND_CEILING, Decimal
from typing import Literal

from pydantic import BaseModel, Field


class ContextPolicy(BaseModel):
    """Configuration for automatic context compaction.

    Passing an instance to ``AgentExecutor(context_policy=...)`` opts in to
    compaction; ``None`` (default) leaves the agent loop unchanged.

    The trigger fires between turns when the combined token basis (previous
    turn's input usage plus a live estimate of the outgoing message list)
    reaches :attr:`trigger_tokens` (``ceil(trigger_pct * context_window)``).
    The window size is configured here rather than looked up from the provider.

    Attributes:
        context_window: Total context-window size, in tokens, for the model.
        trigger_pct: Fraction of ``context_window`` that triggers compaction (0.5–1.0).
        keep_last_n: Number of most-recent complete tool round-trips kept verbatim.
        mode: ``"trim"`` (deterministic placeholder on the send view) or
            ``"summarize"`` (one LLM call, write-back into the working list).
        placeholder: Text substituted for cleared tool-result payloads in trim mode.
    """

    context_window: int = Field(default=128_000, gt=0)
    trigger_pct: float = Field(default=0.8, ge=0.5, le=1)
    keep_last_n: int = Field(default=3, ge=0)
    mode: Literal["trim", "summarize"] = "trim"
    placeholder: str = "[Tool result cleared to save context.]"

    @property
    def trigger_tokens(self) -> int:
        """Threshold at which automatic compaction may fire (with usage + estimate).

        Returns:
            ``ceil(trigger_pct * context_window)`` using exact decimal arithmetic.
        """
        line = Decimal(str(self.trigger_pct)) * self.context_window
        return int(line.to_integral_value(rounding=ROUND_CEILING))
