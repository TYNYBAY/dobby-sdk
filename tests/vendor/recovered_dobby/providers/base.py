"""Base provider abstraction for LLM providers.

This module provides the abstract base class that all LLM providers must inherit from.
It defines the common interface for chat completions with streaming support.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Any, Literal, overload

from ..types import MessagePart, StreamEndEvent, StreamEvent


class ProviderError(Exception):
    """Base exception for all provider errors.

    Non-retryable by default. Covers auth errors, bad requests, etc.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
    ):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(ProviderError):
    """HTTP 429 or equivalent. Retryable with backoff."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after: float | None = None,
    ):
        self.retry_after = retry_after  # seconds, from Retry-After header if available
        super().__init__(message, provider=provider, status_code=429)


class APIConnectionError(ProviderError):
    """Network/connection failure. Retryable."""

    def __init__(self, message: str, *, provider: str | None = None):
        super().__init__(message, provider=provider)


class APITimeoutError(APIConnectionError):
    """Request timed out. Retryable. Subclass of APIConnectionError."""

    def __init__(self, message: str, *, provider: str | None = None):
        super().__init__(message, provider=provider)


class InternalServerError(ProviderError):
    """HTTP 5xx from provider. Retryable."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int = 500,
    ):
        super().__init__(message, provider=provider, status_code=status_code)


RETRYABLE_ERRORS: tuple[type[ProviderError], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


class Provider[ClientT](ABC):
    """Abstract base class for LLM providers.

    Each provider wraps a specific LLM API and converts between
    Dobby's provider-agnostic types and the provider's native types.

    Type Parameters:
        ClientT: The type of the underlying API client (e.g., AsyncOpenAI, genai.Client)

    Attributes:
        _client: The underlying API client instance
        max_retries: Maximum number of retry attempts for transient errors

    Example:
        ```python
        class OpenAIProvider(Provider[AsyncOpenAI]):
            @property
            def name(self) -> str:
                return "openai"

            @property
            def model(self) -> str:
                return self._model

            @property
            def client(self) -> AsyncOpenAI:
                return self._client
        ```
    """

    _client: ClientT
    max_retries: int

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'gemini', 'anthropic').

        Used for logging and provider-specific logic in AgentExecutor.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier (e.g., 'gpt-4o', 'gemini-2.5-flash').

        Returns the model name or deployment ID being used.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def client(self) -> ClientT:
        """Authenticated client instance for this provider.

        Returns the underlying API client for direct access if needed.
        """
        raise NotImplementedError()

    @overload
    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: Literal[False] = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> StreamEndEvent: ...

    @overload
    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: Literal[True],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]: ...

    @abstractmethod
    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> StreamEndEvent | AsyncIterator[StreamEvent]:
        """Generate a response from conversation messages.

        Converts provider-agnostic messages to the provider's native format,
        makes the API call, and converts responses back to Dobby types.

        Args:
            messages: Conversation history as Dobby MessagePart objects
            stream: Whether to stream response chunks (default: False)
            system_prompt: Optional system instructions for the model
            temperature: Sampling temperature 0.0-2.0 (default: 0.0 for deterministic)
            tools: Tool definitions in provider-specific format (optional)
            **kwargs: Provider-specific parameters (e.g., reasoning_effort for OpenAI)

        Returns:
            StreamEndEvent: When stream=False, contains complete response with parts and usage
            AsyncIterator[StreamEvent]: When stream=True, yields events as they arrive

        Raises:
            ProviderError: Base class for all provider-specific errors
            RateLimitError: When rate limit is exceeded (retryable)
            APIConnectionError: When connection fails (retryable)
            APITimeoutError: When request times out (retryable)
            InternalServerError: When provider returns 5xx (retryable)
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"
