
from pydantic import BaseModel


class Usage(BaseModel):
    """Token usage statistics from the LLM provider.

    Total input tokens in a request is the summation of `input_tokens`,
    `cache_creation_input_tokens`, and `cache_read_input_tokens`.
    """

    input_tokens: int
    """Number of input tokens in the request."""

    output_tokens: int
    """Number of output tokens generated."""

    total_tokens: int
    """Total tokens (input + output)."""

    cache_creation_input_tokens: int | None = None
    """The number of input tokens used to create the cache entry."""

    cache_read_input_tokens: int | None = None
    """The number of input tokens read from the cache."""
