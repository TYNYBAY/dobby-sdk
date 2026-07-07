"""Vertex AI provider for Google Cloud's OpenAI-compatible Model-as-a-Service endpoint.

This module provides the VertexAIProvider class for interacting with Vertex AI's
Model Garden catalog (e.g. Llama, self-deployed containers) via its OpenAI-compatible
Chat Completions endpoint. Authentication uses Application Default Credentials (or an
explicitly supplied `google.auth.credentials.Credentials` object) and refreshes the
bearer token transparently on every request via the OpenAI SDK's native async-callable
`api_key` hook.
"""

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, Literal, overload

import google.auth
import google.auth.credentials
import google.auth.transport.requests
from openai import AsyncOpenAI

from ...types import MessagePart, StreamEndEvent, StreamEvent
from ..base import Provider

__all__ = ["VertexAIProvider"]


class VertexAIProvider(Provider[AsyncOpenAI]):
    """Provider for Vertex AI's OpenAI-compatible Model-as-a-Service (MaaS) endpoint.

    Targets Vertex AI's Model Garden catalog (Llama, self-deployed containers, and any
    other serving container that speaks OpenAI's Chat Completions wire format) through
    `POST https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/
    {location}/endpoints/openapi/chat/completions`.

    Auth is via Application Default Credentials (ADC) by default, or an explicitly
    supplied `google.auth.credentials.Credentials` object. The bearer token is kept
    fresh across calls by passing a bound async method as the OpenAI SDK's `api_key`
    parameter — the SDK invokes it before every request (including retries, streaming
    and non-streaming), so no manual header injection or eager refresh is needed.

    Attributes:
        project: GCP project ID.
        location: GCP location (default: us-central1).
        scopes: OAuth scopes forwarded to `google.auth.default()` (only used when no
            explicit `credentials` is supplied).
        max_retries: Maximum retry attempts for transient errors.

    Example:
        ```python
        provider = VertexAIProvider(
            model="meta/llama-3.1-405b-instruct-maas",
            project="my-project",
        )
        ```
    """

    project: str
    location: str
    scopes: Sequence[str] | None
    _model: str
    _credentials: google.auth.credentials.Credentials
    _client: AsyncOpenAI
    max_retries: int

    def __init__(
        self,
        model: str,
        project: str,
        location: str = "us-central1",
        credentials: google.auth.credentials.Credentials | None = None,
        scopes: Sequence[str] | None = None,
        max_retries: int = 3,
    ):
        """Initialize Vertex AI provider.

        Args:
            model: Publisher-qualified model id (e.g. "meta/llama-3.1-405b-instruct-maas"),
                forwarded verbatim — no hardcoded allow-list.
            project: GCP project ID.
            location: GCP location (default: "us-central1").
            credentials: Pre-built credentials object. When omitted, resolved via
                `google.auth.default(scopes=scopes)` at construction time.
            scopes: OAuth scopes forwarded to `google.auth.default()`. Only used when
                `credentials` is not supplied. Defaults to google-auth's own default
                resolution (typically the broad `cloud-platform` scope) — pass a
                narrower list for least-privilege where the ADC source supports it.
            max_retries: Maximum retry attempts for transient errors (default: 3).
        """
        self._model = model
        self.project = project
        self.location = location
        self.scopes = scopes
        self.max_retries = max_retries

        if credentials is not None:
            self._credentials = credentials
        else:
            self._credentials, _ = google.auth.default(scopes=scopes)

        self._refresh_lock = asyncio.Lock()

        self._client = AsyncOpenAI(
            base_url=(
                f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
                f"/locations/{location}/endpoints/openapi"
            ),
            api_key=self._bearer_token,
        )

    @property
    def name(self) -> str:
        """Provider name (distinct from GeminiProvider's "gemini-vertexai")."""
        return "vertexai"

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    @property
    def client(self) -> AsyncOpenAI:
        """Authenticated client instance.

        Fully self-sufficient: the callable `api_key` is invoked by the SDK's own
        request-preparation hook on every request, so direct use of
        `provider.client.chat.completions.create(...)` also gets a fresh token.
        """
        return self._client

    async def _bearer_token(self) -> str:
        """Return a valid bearer token, refreshing it first if necessary.

        Passed as the `api_key` callable to `AsyncOpenAI` — invoked automatically by
        the SDK before every request. Guarded by `self._refresh_lock` so concurrent
        requests racing an expired token only trigger one refresh.
        """
        async with self._refresh_lock:
            if not self._credentials.valid:
                await asyncio.to_thread(
                    self._credentials.refresh, google.auth.transport.requests.Request()
                )
            return self._credentials.token

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

        Not yet implemented — non-streaming and streaming chat completions land in
        later implementation units (U3/U4) of the Vertex AI provider plan. This stub
        exists only so `VertexAIProvider` satisfies `Provider`'s abstract interface
        and can be instantiated by this unit's tests.

        Raises:
            NotImplementedError: Always, until U3/U4 land.
        """
        raise NotImplementedError(
            "VertexAIProvider.chat() is not yet implemented (lands in a later unit)."
        )
