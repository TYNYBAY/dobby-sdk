"""Vertex AI provider that routes each model to the API its publisher speaks.

This module provides the VertexAIProvider class. With the default
`routing="native"`, `google/gemini-*` models go to Gemini's native API (through
`GeminiProvider` on a `genai.Client(vertexai=True)`) and `anthropic/*` models go
to the Anthropic Messages API on Vertex (through `AnthropicProvider` on an
`AsyncAnthropicVertex`). Every other publisher, every self-deployed endpoint, and
every model under `routing="openai"` uses Vertex's OpenAI-compatible Chat
Completions endpoint. The route is fixed at construction from the instance model.

All routes share one set of credentials. They come from an explicitly supplied
`google.auth.credentials.Credentials` object or from Application Default
Credentials. On the OpenAI-compatible route the bearer token is refreshed before
every request via the OpenAI SDK's async-callable `api_key` hook; the native SDK
clients refresh the same credentials object themselves.
"""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
import json
import os
from typing import Any, Literal, NoReturn, overload

from anthropic import AsyncAnthropicVertex
from google import genai
import google.auth
import google.auth.credentials
import google.auth.transport.requests
from google.oauth2 import service_account
import openai
from openai import AsyncOpenAI

from ..._logging import logger
from ...tools.tool import Tool
from ...types import (
    MessagePart,
    ResponsePart,
    StopReason,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolUseEvent,
    ToolUsePart,
    Usage,
)
from .._retry import with_retries
from ..anthropic.adapter import AnthropicProvider
from ..base import (
    APIConnectionError as DobbyAPIConnectionError,
    APITimeoutError as DobbyAPITimeoutError,
    InternalServerError as DobbyInternalServerError,
    Provider,
    ProviderError as DobbyProviderError,
    RateLimitError as DobbyRateLimitError,
    _iter_translated,
)
from ..gemini.adapter import GeminiProvider
from .converters import EXTRA_CONTENT_KEY, to_vertexai_messages, to_vertexai_tool

__all__ = ["VertexAIProvider"]


def _require_non_empty_model(model: str | None) -> str:
    """Validate and return a model id.

    Plain input validation, deliberately with no model-family matching: this
    provider forwards every well-formed id verbatim. Without this check a `None`
    or empty id constructs successfully and reaches the wire as a null model
    field, turning a config typo into an opaque server-side error.

    Returns:
        The validated model id, so callers get a narrowed `str`.

    Raises:
        ValueError: `model` is `None`, empty, or whitespace-only.
    """
    if not model or not model.strip():
        raise ValueError("VertexAIProvider requires a non-empty model id.")
    return model


class _EndpointTarget(ABC):
    """Where a request is routed, and what the body's `model` field carries.

    Vertex exposes the same OpenAI-compatible Chat Completions surface at two
    different paths whose contracts differ in exactly two ways: the endpoints
    path segment, and whether the body's `model` field means anything. Both are
    answered here so neither `chat()` nor `_build_kwargs()` has to branch.
    """

    @property
    @abstractmethod
    def path_segment(self) -> str:
        """Value for the `endpoints/{...}` segment of the base URL."""
        raise NotImplementedError()

    @abstractmethod
    def body_model(self, model: str) -> str:
        """Value to send as the request body's `model` field."""
        raise NotImplementedError()


class _ModelGardenTarget(_EndpointTarget):
    """Model Garden / MaaS publisher models, e.g. `meta/llama-3.3-70b-instruct-maas`.

    Routed through the literal `openapi` path segment. The model id is required
    and is what selects the model, so it is forwarded verbatim.
    """

    @property
    def path_segment(self) -> str:
        return "openapi"

    def body_model(self, model: str) -> str:
        return model


class _DeployedEndpointTarget(_EndpointTarget):
    """A self-deployed / custom-deployed endpoint, addressed by its numeric id.

    The endpoint id alone selects the model, so Vertex ignores the body's
    `model` field entirely -- Google's REST samples omit it and their
    OpenAI-SDK samples send `""`. We send `""` because the OpenAI SDK requires
    the argument. Sending anything else is harmless but misleading.
    """

    def __init__(self, endpoint_id: str) -> None:
        self._endpoint_id = endpoint_id

    @property
    def path_segment(self) -> str:
        return self._endpoint_id

    def body_model(self, model: str) -> str:
        return ""


def _resolve_host(location: str, endpoint_host: str | None) -> str:
    """Resolve the API host for a location, honouring an explicit override.

    Three cases, in order:

    1. `endpoint_host` given -- a dedicated endpoint's DNS. Used verbatim. Once
       an endpoint has `dedicatedEndpointEnabled`, the shared regional DNS stops
       serving it, so this cannot be derived and must be supplied.
    2. `location == "global"` -- the host carries **no** region prefix. Building
       `global-aiplatform.googleapis.com` is the classic URL bug here.
    3. Otherwise the regional host.
    """
    if endpoint_host:
        # The Endpoint resource's `dedicatedEndpointDns` is documented with a
        # scheme but returned by the API without one, and the official Python
        # client prepends `https://` unconditionally. Accept either.
        return endpoint_host.removeprefix("https://").removeprefix("http://").rstrip("/")
    if location == "global":
        return "aiplatform.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


_Route = Literal["gemini", "anthropic", "openai"]

# Native routes: route -> (publisher, required model-name prefix). A model id
# `{publisher}/{name}` takes the route when `name` starts with the prefix.
# Anything that matches no row stays on the OpenAI-compatible route, which is
# what serves it today (e.g. `google/gemma-*` Model Garden MaaS).
_NATIVE_ROUTES: dict[_Route, tuple[str, str]] = {
    "gemini": ("google", "gemini"),
    "anthropic": ("anthropic", ""),
}

# Chat Completions parameters with no meaning on a native route, per route.
# Gemini would raise a raw Pydantic error for them, and on Claude `extra_body`
# is a real SDK option that silently rewrites the request body, so they are
# rejected up front. `seed` and the two penalties are native
# `GenerateContentConfig` fields with the same meaning, so the Gemini route
# passes them through. `logprobs` stays rejected there: Gemini's field is a
# top-N count, not Chat Completions' on/off flag.
_SHARED_CHAT_COMPLETIONS_ONLY_KWARGS = frozenset(
    {
        "extra_body",
        "stop",
        "n",
        "logprobs",
        "top_logprobs",
        "response_format",
        # Gemini has no field of these names and Claude's `tool_choice` takes a
        # different shape, so they would fail with a raw SDK error or, for a
        # `tool_choice` dict, be misread.
        "max_completion_tokens",
        "tool_choice",
        "parallel_tool_calls",
        "user",
        "stream_options",
    }
)
_CHAT_COMPLETIONS_ONLY_KWARGS: dict[_Route, frozenset[str]] = {
    "gemini": _SHARED_CHAT_COMPLETIONS_ONLY_KWARGS,
    "anthropic": _SHARED_CHAT_COMPLETIONS_ONLY_KWARGS
    | {"seed", "frequency_penalty", "presence_penalty"},
}


def _resolve_route(
    model: str, *, routing: Literal["native", "openai"], endpoint_id: str | None
) -> _Route:
    """Pick the API a provider instance talks to. Reads the publisher only.

    A self-deployed endpoint selects its own model and speaks Chat Completions,
    so `endpoint_id` always wins. `routing="openai"` opts every model out of
    the native routes.

    Args:
        model: The instance model id, e.g. `google/gemini-2.5-flash`.
        routing: `"native"` to route by publisher, `"openai"` to opt out.
        endpoint_id: Self-deployed endpoint id, or None for Model Garden.

    Returns:
        `"gemini"`, `"anthropic"` or `"openai"`.
    """
    if endpoint_id is not None or routing == "openai":
        return "openai"
    publisher, _, name = model.partition("/")
    for route, (route_publisher, name_prefix) in _NATIVE_ROUTES.items():
        if publisher == route_publisher and name.startswith(name_prefix):
            return route
    return "openai"


# Vertex Chat Completions finish_reason values that map 1:1 (via this table)
# onto Dobby's StopReason. "tool_calls" is included for completeness, but
# `_non_stream_chat_completion` always prefers the presence of `tool_calls` on
# the message itself over this table (some OpenAI-compatible servers report a
# different finish_reason alongside a populated `tool_calls` list).
_FINISH_REASON_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "content_filter",
}


def _extra_content_metadata(tool_call: Any) -> dict[str, Any] | None:
    """Return ToolUsePart metadata for a tool call's `extra_content`, if usable.

    Vertex's OpenAI-compatible endpoint returns Gemini thought signatures on
    `tool_calls[].extra_content`, a field the openai SDK keeps as a pydantic
    extra. Omitting a required signature on a Gemini 3+ function call returns
    400, so it must be replayed unchanged. Values that are missing, empty or
    not a dict are dropped rather than replayed.

    Args:
        tool_call: A response tool call or streaming tool-call delta.

    Returns:
        `{"extra_content": value}`, or None when there is nothing to replay.
    """
    value = getattr(tool_call, "extra_content", None)
    if isinstance(value, dict) and value:
        return {EXTRA_CONTENT_KEY: value}
    return None


# Bounds for the streaming tool-call accumulator (see _stream_chat_completion).
# The target of this provider includes self-deployed/third-party Model Garden
# containers (per the origin document's scope) — a less-trusted boundary than
# native OpenAI — so an unbounded per-`index` accumulator keyed by
# backend-supplied data is a resource-exhaustion vector worth closing.
#
# 64 distinct tool-call indices comfortably covers any realistic parallel
# tool-call fan-out (real-world usage rarely exceeds single digits) while
# still bounding worst-case dict growth.
_MAX_TOOL_CALL_INDICES = 64
# ~1MB across all indices combined comfortably covers realistic tool-call
# argument payloads (JSON objects with many/large string fields) while
# bounding worst-case memory growth from a misbehaving or malicious stream.
_MAX_ACCUMULATED_TOOL_CALL_ARGUMENTS_LENGTH = 1_000_000


def _usage_from(raw: Any) -> Usage | None:
    """Build a `Usage` from a Chat Completions usage payload, or `None`.

    Presence of the usage object is not enough: real OpenAI-compatible backends
    send one whose token counts are all `None` (observed live from Vertex's
    `openai/gpt-oss-20b-maas`). Constructing `Usage` from those raises a
    `ValidationError` that kills the stream mid-flight, so every field is checked
    rather than just the container.

    Returns `None` when any count is missing, which is the documented contract:
    streaming usage is best-effort and never normalised. Substituting zeros would
    misreport a real token spend as free.
    """
    if raw is None:
        return None
    prompt = getattr(raw, "prompt_tokens", None)
    completion = getattr(raw, "completion_tokens", None)
    total = getattr(raw, "total_tokens", None)
    if prompt is None or completion is None or total is None:
        logger.debug(f"Ignoring incomplete usage payload from Vertex: {raw!r}")
        return None
    return Usage(input_tokens=prompt, output_tokens=completion, total_tokens=total)


def _map_finish_reason(reason: str | None) -> StopReason:
    """Map a Vertex AI Chat Completions finish_reason onto Dobby's StopReason.

    Unknown or absent reasons collapse to ``"end_turn"`` so a future/unexpected
    value never produces an out-of-contract StopReason.
    """
    mapped = _FINISH_REASON_MAP.get(reason) if reason is not None else None
    if mapped is not None:
        return mapped
    if reason is not None:
        logger.debug(f"Unhandled Vertex AI finish_reason: {reason}")
    return "end_turn"


class VertexAIProvider(Provider[AsyncOpenAI]):
    """Provider for models on Vertex AI, routed to the API each publisher speaks.

    The route is decided once, at construction, from the instance model:

    | Instance                                         | Route                       |
    |--------------------------------------------------|-----------------------------|
    | `google/gemini-*`                                | Native Gemini API           |
    | `anthropic/*`                                    | Anthropic Messages API      |
    | any other publisher (incl. `google/gemma-*`)     | OpenAI-compatible endpoint  |
    | `endpoint_id=` set                               | OpenAI-compatible endpoint  |
    | `routing="openai"`                               | OpenAI-compatible endpoint  |

    The native routes delegate to `GeminiProvider` and `AnthropicProvider`, built
    around Vertex-backed SDK clients (`genai.Client(vertexai=True)` and
    `AsyncAnthropicVertex`). The publisher prefix is removed before the native
    SDK sees the id. `name` stays `"vertexai"` and `model` keeps the id the caller
    passed, on every route. Use `format_tools()` to get tools in the active
    route's format.

    The OpenAI-compatible route serves both of Vertex's Chat Completions surfaces:

    - **Model Garden / MaaS** — publisher models such as
      `meta/llama-3.3-70b-instruct-maas`, routed through the shared `endpoints/openapi`
      path. The model id selects the model and is sent in the request body.
    - **Self-deployed endpoints** (`endpoint_id=`) — a model you deployed yourself,
      routed through `endpoints/{endpoint_id}`. The endpoint selects the model, so
      Vertex ignores the body's `model` field and this provider sends `""`, matching
      Google's own OpenAI-SDK samples.

    ```
    POST https://{host}/{api_version}/projects/{project}/locations/{location}
         /endpoints/{openapi|endpoint_id}/chat/completions
    ```

    `host` is `{location}-aiplatform.googleapis.com`, or bare
    `aiplatform.googleapis.com` when `location="global"`, or an explicit
    `endpoint_host` for a dedicated endpoint's DNS.

    On that route model ids are forwarded verbatim — no allow-list, no family
    validation.

    Auth resolves, in order: an explicitly supplied `google.auth.credentials.Credentials`
    object; a stringified service-account key in the `GOOGLE_APPLICATION_CREDENTIALS_JSON`
    env var (no key file needed — suits secret-manager-style injection in prod); or
    Application Default Credentials (ADC) via `google.auth.default()`. All routes share
    the resolved credentials, project and location. On the OpenAI-compatible route the
    bearer token is kept fresh by passing a bound async method as the OpenAI SDK's
    `api_key` parameter — the SDK invokes it before every request (including retries,
    streaming and non-streaming). The native SDK clients refresh the same credentials
    object themselves.

    Attributes:
        project: GCP project ID.
        location: GCP location (default: us-central1).
        endpoint_id: Self-deployed endpoint id, or None for Model Garden.
        api_version: API version path segment (default: "v1").
        scopes: OAuth scopes forwarded to `google.auth.default()` (only used when no
            explicit `credentials` is supplied).
        max_retries: Maximum retry attempts for transient errors. The native SDK
            clients' own retries are off, so this is the only retry layer.

    Example:
        ```python
        provider = VertexAIProvider(
            model="google/gemini-2.5-flash",  # native Gemini route
            project="my-project",
        )
        ```
    """

    project: str
    location: str
    scopes: Sequence[str] | None
    endpoint_id: str | None
    api_version: str
    _target: _EndpointTarget
    _model: str
    _credentials: google.auth.credentials.Credentials
    _client: AsyncOpenAI | None
    _route: _Route
    _native: GeminiProvider | AnthropicProvider | None
    max_retries: int

    # Model Garden: `model` is required, endpoint parameters are not accepted.
    @overload
    def __init__(
        self,
        model: str,
        project: str | None = ...,
        location: str = ...,
        credentials: google.auth.credentials.Credentials | None = ...,
        scopes: Sequence[str] | None = ...,
        max_retries: int = ...,
        *,
        api_version: str = ...,
        routing: Literal["native", "openai"] = ...,
    ) -> None: ...

    # Self-deployed: `endpoint_id` is required, `model` is an optional label.
    @overload
    def __init__(
        self,
        model: str | None = ...,
        project: str | None = ...,
        location: str = ...,
        credentials: google.auth.credentials.Credentials | None = ...,
        scopes: Sequence[str] | None = ...,
        max_retries: int = ...,
        *,
        endpoint_id: str,
        endpoint_host: str | None = ...,
        api_version: str = ...,
    ) -> None: ...

    def __init__(
        self,
        model: str | None = None,
        project: str | None = None,
        location: str = "us-central1",
        credentials: google.auth.credentials.Credentials | None = None,
        scopes: Sequence[str] | None = None,
        max_retries: int = 3,
        *,
        endpoint_id: str | None = None,
        endpoint_host: str | None = None,
        api_version: str = "v1",
        routing: Literal["native", "openai"] = "native",
    ):
        """Initialize Vertex AI provider.

        The two endpoint types are separate overloads, so a type checker rejects
        `endpoint_host` without `endpoint_id`, a missing `model` without an
        `endpoint_id`, and `routing` with an `endpoint_id`, before the code runs.
        The runtime checks below still fire for untyped callers.

        Args:
            model: Publisher-qualified model id (e.g. "google/gemini-2.5-flash",
                "meta/llama-3.1-405b-instruct-maas"). Its publisher picks the route (see
                the class docstring); no hardcoded allow-list. Native routes send it
                without the publisher prefix, the OpenAI-compatible route verbatim.
                **Required for Model Garden.** For a self-deployed endpoint (`endpoint_id` set) the endpoint
                itself selects the model, so this is optional and used only as a display
                label for `provider.model` and `StreamEndEvent.model`; it is never sent
                on the wire. Defaults to `"endpoint-{endpoint_id}"` in that mode.
            project: GCP project ID. When omitted, derived from the resolved credentials'
                `project_id` (service-account credentials only — from `GOOGLE_APPLICATION_CREDENTIALS_JSON`,
                a key file, or a directly-supplied service-account `credentials` object) or
                from ADC's own project resolution. Required explicitly for credential types
                that carry no project (e.g. user ADC without a set quota project).
            location: GCP location (default: "us-central1").
            credentials: Pre-built credentials object. When omitted, resolution falls
                back to the `GOOGLE_APPLICATION_CREDENTIALS_JSON` env var (a stringified
                service-account key, for secret-manager-style injection with no key file
                on disk) and finally to `google.auth.default(scopes=scopes)` (ADC).
            scopes: OAuth scopes forwarded to credential construction. Only used when
                `credentials` is not supplied. Defaults to google-auth's own default
                resolution (typically the broad `cloud-platform` scope) — pass a
                narrower list for least-privilege where the ADC source supports it.
                Required (non-None) when falling back to `GOOGLE_APPLICATION_CREDENTIALS_JSON`
                or a service-account key file — those credential types have no
                implicit scope and requests fail with `invalid_scope` otherwise.
            max_retries: Maximum retry attempts for transient errors (default: 3).
            endpoint_id: Keyword-only. Numeric id of a self-deployed / custom-deployed
                endpoint. Switches routing from the shared `endpoints/openapi` path to
                `endpoints/{endpoint_id}`, and stops sending `model` on the wire because
                the endpoint selects the model itself. Omit for Model Garden.
            endpoint_host: Keyword-only. A dedicated endpoint's DNS, as returned in the
                Endpoint resource's `dedicatedEndpointDns` (e.g.
                "1234567890.us-central1-987654321.prediction.vertexai.goog"). Required
                once an endpoint has `dedicatedEndpointEnabled`, because the shared
                regional DNS stops serving it. Read this value from the API — the uid
                segment is not always the project number. A leading scheme is stripped
                if present. Only valid alongside `endpoint_id`.
            api_version: Keyword-only. API version path segment (default: "v1").
                Google registers both "v1" and "v1beta1" for this route and documents
                neither as preferred; their dedicated-endpoint samples use "v1beta1",
                and "v1beta1" is also required for Gemini preview fields. Applies to the
                OpenAI-compatible route only; the native SDKs pick their own version.
            routing: Keyword-only. `"native"` (default) sends `google/gemini-*` and
                `anthropic/*` models to their native APIs. `"openai"` keeps every model
                on the OpenAI-compatible Chat Completions endpoint. Ignored with
                `endpoint_id`, which always uses Chat Completions.

        Raises:
            ValueError: `routing` is not `"native"` or `"openai"`; `model` is
                empty/`None`/whitespace-only without an `endpoint_id`;
                `endpoint_host` is given without `endpoint_id`; `endpoint_id` is empty or
                whitespace-only; `api_version` is empty; or `project` is omitted and can't
                be derived from the resolved credentials.

        Note:
            Chat Completions on a self-deployed endpoint requires a serving container
            that implements it — Google's prebuilt vLLM and HF TGI containers do, an
            arbitrary custom container may only support `:rawPredict`, which this
            provider does not speak.
        """
        if routing not in ("native", "openai"):
            raise ValueError(f'routing must be "native" or "openai", got {routing!r}.')

        self._target: _EndpointTarget
        if endpoint_id is not None:
            if not endpoint_id.strip():
                raise ValueError(
                    "VertexAIProvider requires a non-empty endpoint_id when one is given."
                )
            self._target = _DeployedEndpointTarget(endpoint_id)
            # The endpoint selects the model, so `model` is a display label only.
            resolved_model = model or f"endpoint-{endpoint_id}"
        else:
            if endpoint_host is not None:
                raise ValueError(
                    "endpoint_host is only valid together with endpoint_id — a dedicated "
                    "DNS belongs to a deployed endpoint, not to the shared Model Garden path."
                )
            self._target = _ModelGardenTarget()
            resolved_model = _require_non_empty_model(model)

        if not api_version.strip():
            raise ValueError("VertexAIProvider requires a non-empty api_version.")

        self._model = resolved_model
        self.location = location
        self.scopes = scopes
        self.max_retries = max_retries
        self.endpoint_id = endpoint_id
        self.api_version = api_version
        self._endpoint_host = endpoint_host

        resolved_project = project
        if credentials is not None:
            self._credentials = credentials
            if resolved_project is None:
                resolved_project = getattr(credentials, "project_id", None)
        elif credentials_json := os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
            info = json.loads(credentials_json)
            self._credentials = service_account.Credentials.from_service_account_info(
                info, scopes=scopes
            )
            if resolved_project is None:
                resolved_project = info.get("project_id")
        else:
            self._credentials, adc_project = google.auth.default(scopes=scopes)
            if resolved_project is None:
                resolved_project = adc_project

        if resolved_project is None:
            raise ValueError(
                "project could not be determined automatically for the resolved "
                "credentials — pass `project=` explicitly."
            )
        self.project = resolved_project

        self._refresh_lock = asyncio.Lock()

        host = _resolve_host(location, endpoint_host)
        self._openai_base_url = (
            f"https://{host}/{api_version}/projects/{self.project}"
            f"/locations/{location}/endpoints/{self._target.path_segment}"
        )

        # Native routes run on their own SDK clients around the same credentials.
        # The Chat Completions client is built now only for the OpenAI-compatible
        # route; on a native route it is built on first access to `client`.
        self._route = _resolve_route(resolved_model, routing=routing, endpoint_id=endpoint_id)
        self._client: AsyncOpenAI | None = (
            self._build_openai_client() if self._route == "openai" else None
        )
        self._native = self._build_native_provider()

    @property
    def name(self) -> str:
        """Provider name."""
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

        This is always the OpenAI-compatible Chat Completions client, whatever the
        route; native routes send `chat()` through their own SDK clients. On a
        native route it is built on first access.
        """
        if self._client is None:
            self._client = self._build_openai_client()
        return self._client

    def _build_openai_client(self) -> AsyncOpenAI:
        """Build the Chat Completions client on the shared bearer-token auth."""
        return AsyncOpenAI(base_url=self._openai_base_url, api_key=self._bearer_token)

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
            token = self._credentials.token
            if token is None:
                raise DobbyProviderError(
                    "Vertex AI credentials resolved to no access token after refresh.",
                    provider="vertexai",
                )
            return token

    def _native_model_id(self, model: str) -> str:
        """Remove the active native route's publisher prefix from a model id.

        Args:
            model: A model id, with or without the publisher prefix.

        Returns:
            The id the native SDK expects, e.g. `gemini-2.5-flash`.
        """
        publisher, _ = _NATIVE_ROUTES[self._route]
        return model.removeprefix(f"{publisher}/")

    def _build_native_provider(self) -> GeminiProvider | AnthropicProvider | None:
        """Build the adapter a native route delegates to, or None for Chat Completions.

        Both SDK clients get the credentials, project and location this provider
        already resolved, so callers configure nothing new. `AsyncAnthropicVertex`
        is built with its own retries off so `with_retries` stays the only retry
        layer. `region="global"` needs no `base_url`; the SDK builds that host.

        Returns:
            The internal adapter for a native route, or None on the OpenAI route.
        """
        if self._route == "gemini":
            client = genai.Client(
                vertexai=True,
                credentials=self._credentials,
                project=self.project,
                location=self.location,
            )
            return GeminiProvider._from_client(
                client,
                self._native_model_id(self._model),
                max_retries=self.max_retries,
                provider_name=self.name,
            )
        if self._route == "anthropic":
            client = AsyncAnthropicVertex(
                region=self.location,
                project_id=self.project,
                credentials=self._credentials,
                max_retries=0,
            )
            return AnthropicProvider._from_client(
                client,
                self._native_model_id(self._model),
                max_retries=self.max_retries,
                provider_name=self.name,
            )
        return None

    def format_tools(self, tools: list[Tool]) -> list[Any]:
        """Format tools for the route this instance talks to.

        `AgentExecutor` calls this, so `AgentExecutor("vertexai", provider)`
        sends tools in the active route's format. Callers of `chat()` should use
        it too, in place of `to_vertexai_tool()`, which fits the
        OpenAI-compatible route only.

        Args:
            tools: Tools to format.

        Returns:
            Gemini `Tool` objects on the Gemini route, Anthropic tool dicts on
            the Anthropic route, or nested Chat Completions tool dicts otherwise.
        """
        if self._route == "gemini":
            return [tool.to_gemini_format() for tool in tools]
        if self._route == "anthropic":
            return [tool.to_anthropic_format() for tool in tools]
        return [to_vertexai_tool(tool) for tool in tools]

    def _reject_chat_completions_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Raise before any network call for kwargs the native route cannot honour.

        Args:
            kwargs: The passthrough kwargs given to `chat()`.

        Raises:
            TypeError: A Chat-Completions-only kwarg was given, or a `str`
                `reasoning_effort` was given on the Anthropic route (which takes
                an `int` thinking budget).
        """
        rejected = _CHAT_COMPLETIONS_ONLY_KWARGS.get(self._route, frozenset())
        for key in kwargs:
            if key in rejected:
                raise TypeError(
                    f"{key!r} is a Chat Completions parameter and is not supported on "
                    f"VertexAIProvider's native {self._route} route. Construct the "
                    f'provider with routing="openai" to send it.'
                )
        reasoning_effort = kwargs.get("reasoning_effort")
        if self._route == "anthropic" and isinstance(reasoning_effort, str):
            raise TypeError(
                "VertexAIProvider's native anthropic route requires reasoning_effort "
                f"as an int thinking budget (budget_tokens), got {reasoning_effort!r}. "
                'Construct the provider with routing="openai" to send a string effort.'
            )

    @overload
    async def chat(
        self,
        messages: Iterable[MessagePart],
        *,
        stream: Literal[False] = False,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        model: str | None = None,
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
        model: str | None = None,
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
        model: str | None = None,
        **kwargs: Any,
    ) -> StreamEndEvent | AsyncIterator[StreamEvent]:
        """Generate a response from conversation messages.

        On a native route, delegates to the internal Gemini or Anthropic adapter
        after rejecting Chat-Completions-only kwargs. Otherwise converts the
        messages to Chat Completions format and calls the OpenAI-compatible
        endpoint.

        Args:
            messages: Conversation history with user/assistant/tool messages.
            stream: Whether to stream response chunks.
            system_prompt: Optional system message to guide behavior.
            temperature: Controls randomness (0.0-2.0, default 0.0).
            tools: Tool definitions in the active route's format. Use
                `format_tools()` to build them; `to_vertexai_tool()` output fits
                the OpenAI-compatible route only.
            model: Per-call model override. Falls back to the instance model. It is
                always sent over the instance's route: a native route removes the
                route's publisher prefix, the OpenAI-compatible route forwards it
                verbatim. An override meant for another route fails with that
                backend's own error. On a self-deployed endpoint the endpoint
                selects the model, so an override changes only the reported
                `StreamEndEvent.model`, not what is served.
            **kwargs: Provider-native passthrough params. On the OpenAI-compatible
                route they are forwarded verbatim to `chat.completions.create()`
                (e.g. `max_tokens`, `top_p`, `stop`, `extra_body`); Vertex ignores
                parameters a model does not support, and third-party model support
                varies -- consult the model's own documentation. On a native route
                they go to the native adapter's `chat()` (e.g. `reasoning_effort`,
                `max_tokens`, `top_p`), and Chat-Completions-only kwargs are rejected.

        Returns:
            StreamEndEvent for non-streaming, AsyncIterator[StreamEvent] for streaming.

        Raises:
            TypeError: On a native route, a Chat-Completions-only kwarg (`extra_body`,
                `stop`, `n`, `logprobs`, `top_logprobs`, `response_format`, `seed`,
                `frequency_penalty`, `presence_penalty`) was given, or a `str`
                `reasoning_effort` on the Anthropic route. Raised before any
                network call; `routing="openai"` accepts them.
        """
        # A whitespace-only override is a typo, not a fallback signal; an empty
        # string still falls through to the instance model as before.
        if model:
            _require_non_empty_model(model)

        if self._native is not None:
            self._reject_chat_completions_kwargs(kwargs)
            return await self._native.chat(
                messages,
                stream=stream,
                system_prompt=system_prompt,
                temperature=temperature,
                tools=tools,
                model=self._native_model_id(model) if model else None,
                **kwargs,
            )

        target_model = model or self._model

        vertexai_messages = to_vertexai_messages(messages)
        if system_prompt is not None:
            vertexai_messages.insert(0, {"role": "system", "content": system_prompt})

        if stream:
            return self._stream_chat_completion(
                vertexai_messages, target_model, temperature, tools, kwargs
            )

        return await self._non_stream_chat_completion(
            vertexai_messages, target_model, temperature, tools, kwargs
        )

    def _build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for chat.completions.create(), excluding unset optionals.

        The `model` field is delegated to the endpoint target: Model Garden needs
        it to select the model, a self-deployed endpoint ignores it entirely and
        receives `""`.

        Args:
            model: Model id (per-call override or instance model).
            messages: Chat-Completions-formatted messages.
            temperature: Sampling temperature.
            tools: Optional, already Chat-Completions-formatted tool schemas.
            extra: Provider-native passthrough params, applied first so the
                fields this provider owns -- `model` above all, which must stay
                `""` on a self-deployed endpoint -- cannot be overwritten.

        Returns:
            Dictionary of kwargs to pass to chat.completions.create().
        """
        kwargs: dict[str, Any] = dict(extra or {})
        kwargs.update(
            {
                "model": self._target.body_model(model),
                "messages": messages,
                "temperature": temperature,
            }
        )
        if tools is not None:
            kwargs["tools"] = tools
        return kwargs

    def _dedicated_endpoint_hint(self) -> str:
        """Return a remediation hint for the dedicated-endpoint misconfiguration, if it fits.

        Enabling `dedicatedEndpointEnabled` on an endpoint stops the shared regional
        DNS from serving it, which surfaces as a 404 or a connection failure rather
        than anything self-describing. Nothing client-side can detect the flag without
        an admin API call this provider deliberately does not make, so instead of
        guessing we name the one setting that fixes it — and only when the shape of
        the failure actually matches (a deployed endpoint reached over the shared host).

        Returns:
            A hint to append to the error message, or `""` when not applicable.
        """
        if self.endpoint_id is None or self._endpoint_host is not None:
            return ""
        return (
            f" — if endpoint {self.endpoint_id!r} has a dedicated DNS "
            "(`dedicatedEndpointEnabled`), the shared regional host no longer serves it. "
            "Pass the endpoint's `dedicatedEndpointDns` as `endpoint_host=`."
        )

    def _translate_error(self, e: Exception) -> NoReturn:
        """Map OpenAI SDK exceptions to unified dobby errors.

        Same exception hierarchy as `OpenAIProvider._translate_error` since
        `AsyncOpenAI` raises `openai.*` errors regardless of which endpoint or
        sub-resource is hit.

        Always raises — never returns normally.

        Args:
            e: The original OpenAI SDK exception.

        Raises:
            DobbyRateLimitError: For rate limit errors (429).
            DobbyAPIConnectionError: For connection failures.
            DobbyAPITimeoutError: For request timeouts.
            DobbyInternalServerError: For server errors (5xx).
            DobbyProviderError: For all other API errors.
        """
        match e:
            case openai.APIStatusError() if e.status_code == 404:
                raise DobbyProviderError(
                    f"{e}{self._dedicated_endpoint_hint()}",
                    provider=self.name,
                    status_code=404,
                ) from e
            case openai.RateLimitError():
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    raw = e.response.headers.get("retry-after")
                    if raw is not None:
                        try:
                            retry_after = float(raw)
                        except (ValueError, TypeError):
                            pass
                raise DobbyRateLimitError(
                    str(e), provider=self.name, retry_after=retry_after
                ) from e
            case openai.APITimeoutError():
                raise DobbyAPITimeoutError(str(e), provider=self.name) from e
            case openai.APIConnectionError():
                raise DobbyAPIConnectionError(
                    f"{e}{self._dedicated_endpoint_hint()}", provider=self.name
                ) from e
            case openai.InternalServerError():
                raise DobbyInternalServerError(
                    str(e), provider=self.name, status_code=e.status_code
                ) from e
            case openai.APIStatusError():
                raise DobbyProviderError(
                    str(e), provider=self.name, status_code=e.status_code
                ) from e
            case _:
                raise DobbyProviderError(str(e), provider=self.name) from e

    @with_retries
    async def _non_stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StreamEndEvent:
        """Non-streaming chat completion with retry support.

        No manual auth handling at this call site — the client's callable
        `api_key` (see `_bearer_token`) refreshes transparently via the SDK's
        own request-preparation hook.

        Args:
            messages: Chat-Completions-formatted messages.
            model: Model id (per-call override or instance model).
            temperature: Sampling temperature.
            tools: Optional, already Chat-Completions-formatted tool schemas.
            extra: Provider-native passthrough params forwarded verbatim to
                chat.completions.create().

        Returns:
            StreamEndEvent with the complete response.
        """
        create_kwargs = self._build_kwargs(
            model=model, messages=messages, temperature=temperature, tools=tools, extra=extra
        )

        try:
            response = await self.client.chat.completions.create(**create_kwargs)
        except Exception as e:
            self._translate_error(e)

        message = response.choices[0].message

        parts: list[ResponsePart] = []
        if message.content:
            parts.append(TextPart(text=message.content))

        tool_calls = message.tool_calls or []
        for tool_call in tool_calls:
            parts.append(
                ToolUsePart(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    inputs=json.loads(tool_call.function.arguments),
                    metadata=_extra_content_metadata(tool_call),
                )
            )

        stop_reason: StopReason = (
            "tool_use" if tool_calls else _map_finish_reason(response.choices[0].finish_reason)
        )

        usage = _usage_from(response.usage)

        return StreamEndEvent(
            model=response.model or model,
            parts=parts,
            stop_reason=stop_reason,
            usage=usage,
        )

    @with_retries
    async def _stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming chat completion with retry support.

        Parses Chat Completions SSE delta chunks (`choices[0].delta`) into
        Dobby's discriminated StreamEvent sequence. No manual auth handling at
        this call site — same as `_non_stream_chat_completion`.

        Tool-call fragments are keyed by `index` (the only safe correlation
        key across chunks — `id` may not appear on every fragment) and merged
        idempotently: `id`/`function.name` are set-if-present on *any* chunk
        for that index, not assumed to only arrive on the first one. `usage`
        is captured via `chunk.usage is not None`, independent of whether
        `chunk.choices` is empty — real third-party (often vLLM-backed)
        OpenAI-compatible servers have shipped both deviations from native
        OpenAI's streaming conventions. See the accumulator bound constants
        above for the resource-exhaustion mitigation this implies.

        Args:
            messages: Chat-Completions-formatted messages.
            model: Model id (per-call override or instance model).
            temperature: Sampling temperature.
            tools: Optional, already Chat-Completions-formatted tool schemas.
            extra: Provider-native passthrough params forwarded verbatim to
                chat.completions.create().

        Yields:
            StreamEvent objects: StreamStartEvent, TextDeltaEvent,
            ToolUseEvent, StreamErrorEvent, StreamEndEvent.
        """
        create_kwargs = self._build_kwargs(
            model=model, messages=messages, temperature=temperature, tools=tools, extra=extra
        )
        create_kwargs["stream"] = True
        create_kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = await self.client.chat.completions.create(**create_kwargs)
        except Exception as e:
            self._translate_error(e)

        stream_started = False
        model_name: str = model
        accumulated_text: str = ""
        finish_reason: str | None = None
        usage: Usage | None = None

        # index -> {"id": str | None, "name": str | None, "arguments": str,
        #           "metadata": dict | None}
        tool_call_accumulator: dict[int, dict[str, Any]] = {}
        accumulated_arguments_length = 0

        async for chunk in _iter_translated(stream, self._translate_error):
            if not stream_started:
                yield StreamStartEvent(
                    id=getattr(chunk, "id", None) or f"vertexai_{model}",
                    model=getattr(chunk, "model", None) or model_name,
                )
                stream_started = True

            if getattr(chunk, "model", None):
                model_name = chunk.model

            # Checked independent of `choices` length: some OpenAI-compatible
            # (vLLM-backed) servers don't guarantee a dedicated empty-choices
            # usage chunk the way native OpenAI does.
            chunk_usage = _usage_from(chunk.usage)
            if chunk_usage is not None:
                usage = chunk_usage

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                accumulated_text += delta.content
                yield TextDeltaEvent(delta=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    index = tc_delta.index
                    is_new_index = index not in tool_call_accumulator

                    if is_new_index and len(tool_call_accumulator) >= _MAX_TOOL_CALL_INDICES:
                        yield StreamErrorEvent(
                            error_code="tool_call_accumulator_overflow",
                            error_message=(
                                "Exceeded max distinct tool-call indices "
                                f"({_MAX_TOOL_CALL_INDICES}) in a single stream; "
                                "stopping accumulation."
                            ),
                        )
                        return

                    fragment = ""
                    if tc_delta.function is not None and tc_delta.function.arguments:
                        fragment = tc_delta.function.arguments

                    if (
                        accumulated_arguments_length + len(fragment)
                        > _MAX_ACCUMULATED_TOOL_CALL_ARGUMENTS_LENGTH
                    ):
                        yield StreamErrorEvent(
                            error_code="tool_call_accumulator_overflow",
                            error_message=(
                                "Exceeded max accumulated tool-call arguments length "
                                f"({_MAX_ACCUMULATED_TOOL_CALL_ARGUMENTS_LENGTH} chars) "
                                "in a single stream; stopping accumulation."
                            ),
                        )
                        return

                    if is_new_index:
                        tool_call_accumulator[index] = {
                            "id": None,
                            "name": None,
                            "arguments": "",
                            "metadata": None,
                        }
                    acc = tool_call_accumulator[index]

                    # Idempotent set-if-present merge, not "only first chunk":
                    # real third-party servers have shipped id/name on later
                    # fragments for the same index instead of the first one.
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function is not None and tc_delta.function.name:
                        acc["name"] = tc_delta.function.name
                    if metadata := _extra_content_metadata(tc_delta):
                        acc["metadata"] = metadata
                    if fragment:
                        acc["arguments"] += fragment
                        accumulated_arguments_length += len(fragment)

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        parts: list[ResponsePart] = []
        if accumulated_text:
            parts.append(TextPart(text=accumulated_text))

        # A tool call's arguments are only safe to treat as "fully assembled"
        # once the whole stream has ended — Chat Completions has no per-index
        # completion signal, and third-party servers may still add id/name to
        # an index on a later chunk (see the idempotent merge above).
        for acc in tool_call_accumulator.values():
            tool_event = ToolUseEvent(
                id=acc["id"] or "",
                name=acc["name"] or "",
                inputs=json.loads(acc["arguments"]) if acc["arguments"] else {},
                metadata=acc["metadata"],
            )
            yield tool_event
            parts.append(
                ToolUsePart(
                    id=tool_event.id,
                    name=tool_event.name,
                    inputs=tool_event.inputs,
                    metadata=tool_event.metadata,
                )
            )

        stop_reason: StopReason = (
            "tool_use" if tool_call_accumulator else _map_finish_reason(finish_reason)
        )

        yield StreamEndEvent(
            model=model_name,
            parts=parts,
            stop_reason=stop_reason,
            usage=usage,
        )
