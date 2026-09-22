"""Host-side retry policy for tool invocations.

This is separate from provider/LLM retry. It re-invokes the same tool call
after a transient, author-listed exception. It never asks the model to
correct arguments.
"""

from dataclasses import dataclass
import random

from ..exceptions import ApprovalRequired, ModelRetry, ToolFailure

_NEVER_RETRY: tuple[type[BaseException], ...] = (
    ApprovalRequired,
    ModelRetry,
    ToolFailure,
)


@dataclass(frozen=True, slots=True)
class ToolRetryPolicy:
    """Retry budget and backoff for a single tool invocation.

    Attributes:
        max_retries: Extra retries after the first attempt. ``0`` or less
            means one invocation only. ``1`` means two invocations.
        retryable_exceptions: Exception types that may be retried. An empty
            tuple disables retry even when ``max_retries`` is positive.
        min_backoff_seconds: Lower bound for exponential backoff with jitter.
        max_backoff_seconds: Upper bound for exponential backoff with jitter.
    """

    max_retries: int
    retryable_exceptions: tuple[type[BaseException], ...] = ()
    min_backoff_seconds: float = 0.1
    max_backoff_seconds: float = 2.0


def max_tool_invocations(policy: ToolRetryPolicy) -> int:
    """Return the maximum number of invocations for this policy.

    When ``retryable_exceptions`` is empty, retry is impossible, so the
    effective budget is always one invocation regardless of ``max_retries``.
    """
    if policy.max_retries <= 0 or not policy.retryable_exceptions:
        return 1
    return policy.max_retries + 1


def is_retryable_tool_exception(exception: BaseException, policy: ToolRetryPolicy) -> bool:
    """Return whether a failed invocation may be retried under ``policy``."""
    if not isinstance(exception, Exception) or isinstance(exception, _NEVER_RETRY):
        return False
    if not policy.retryable_exceptions:
        return False
    return isinstance(exception, policy.retryable_exceptions)


def retry_backoff_seconds(policy: ToolRetryPolicy, failed_attempt: int) -> float:
    """Compute jittered exponential backoff after a failed retryable attempt.

    Args:
        policy: Tool retry policy.
        failed_attempt: 1-based attempt that just failed (1 after the first try).
    """
    exponential = policy.min_backoff_seconds * (2 ** (failed_attempt - 1))
    high = min(policy.max_backoff_seconds, exponential)
    low = min(policy.min_backoff_seconds, high)
    if high <= 0:
        return 0.0
    return random.uniform(low, high)
