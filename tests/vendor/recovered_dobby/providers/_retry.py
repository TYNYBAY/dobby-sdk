"""Retry utilities for LLM providers."""

from collections.abc import AsyncIterator, Callable, Sequence
import functools
import inspect
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)
from tenacity.stop import stop_base

from .._logging import logger
from .base import RETRYABLE_ERRORS


def create_retry_config(
    max_retries: int = 6,
    min_seconds: float = 1,
    max_seconds: float = 60,
    stop_after_delay_seconds: float | None = None,
    errors: Sequence[type[BaseException]] = (),
    func_name: str = "LLM call",
) -> dict[str, Any]:
    """Create tenacity retry configuration.

    Args:
        max_retries: Maximum number of retry attempts.
        min_seconds: Minimum wait time between retries.
        max_seconds: Maximum wait time between retries.
        stop_after_delay_seconds: Optional total timeout for all retries.
        errors: Tuple of exception types to retry on.
        func_name: Name of the function for logging purposes.

    Returns:
        Dictionary of tenacity configuration options.
    """
    wait_strategy = wait_random_exponential(min=min_seconds, max=max_seconds)

    stop_strategy: stop_base = stop_after_attempt(max_retries)
    if stop_after_delay_seconds is not None:
        stop_strategy = stop_strategy | stop_after_delay(stop_after_delay_seconds)

    def before_sleep_callback(retry_state: Any) -> None:
        """Log retry attempts with function name."""
        exc = retry_state.outcome.exception()
        exc_name = type(exc).__name__ if exc else "unknown error"
        logger.warning(
            f"Retrying {func_name} in {retry_state.next_action.sleep:.1f}s "
            f"(attempt {retry_state.attempt_number}/{max_retries}) "
            f"after {exc_name}: {exc}"
        )

    return {
        "reraise": True,
        "stop": stop_strategy,
        "wait": wait_strategy,
        "retry": retry_if_exception_type(tuple(errors)),
        "before_sleep": before_sleep_callback,
    }


def with_retries[F: Callable[..., Any]](f: F) -> F:
    """Decorator that applies retry logic to async LLM API calls.

    For async generators, wraps only the initial API call that may fail.
    Reads `max_retries` from `self` if available, otherwise skips retry.
    Retries on unified RETRYABLE_ERRORS from base module.
    """

    @functools.wraps(f)
    async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        max_retries = getattr(self, "max_retries", 0)
        if max_retries <= 0:
            return await f(self, *args, **kwargs)

        config = create_retry_config(
            max_retries=max_retries,
            min_seconds=4,
            max_seconds=60,
            stop_after_delay_seconds=120,
            errors=RETRYABLE_ERRORS,
            func_name=f.__name__,
        )

        async for attempt in AsyncRetrying(**config):
            with attempt:
                return await f(self, *args, **kwargs)

    @functools.wraps(f)
    async def async_gen_wrapper(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        max_retries = getattr(self, "max_retries", 0)

        if max_retries <= 0:
            async for item in f(self, *args, **kwargs):
                yield item
            return

        config = create_retry_config(
            max_retries=max_retries,
            min_seconds=4,
            max_seconds=60,
            stop_after_delay_seconds=120,
            errors=RETRYABLE_ERRORS,
            func_name=f.__name__,
        )

        # Retry getting the async generator (where the API call happens)
        gen: AsyncIterator[Any] = f(self, *args, **kwargs)  # overwritten on first attempt
        first_item: Any = None
        async for attempt in AsyncRetrying(**config):
            with attempt:
                gen = f(self, *args, **kwargs)
                # Get first item to trigger any connection errors
                first_item = await anext(gen)

        # If we get here, first_item was successful
        if first_item is not None:
            yield first_item
        async for item in gen:
            yield item

    # Detect if it's an async generator
    if inspect.isasyncgenfunction(f):
        return async_gen_wrapper  # type: ignore[return-value]
    return async_wrapper  # type: ignore[return-value]
