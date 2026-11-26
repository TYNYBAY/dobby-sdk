# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`max_model_corrections` on `AgentExecutor.run_stream()`.** One run-wide model-correction budget, default `3`, shared by tool-call correction and final-result correction. Exhausting it raises `ModelRetryExhaustedError`.
- Host-side `error_details` on tool result events: exception type, message, traceback, and related diagnostics. The model-facing payload does not include the traceback.
- **`VertexAIProvider` routes Gemini and Claude to their native APIs.** With the new default `routing="native"`, `google/gemini*` models go to Gemini's native API through `google-genai` (`Client(vertexai=True)`), and `anthropic/*` models go to the Anthropic Messages API through `AsyncAnthropicVertex`. Every other publisher (including `google/gemma-*` and unprefixed ids) and every `endpoint_id` instance stays on the OpenAI-compatible Chat Completions route. The native routes reuse `GeminiProvider`'s and `AnthropicProvider`'s converters, streaming and error translation, share the provider's credentials, project and location, and run with the SDKs' own retries off so `max_retries` stays the only retry layer. `provider.name` stays `"vertexai"` and `provider.model` returns the id as passed. See [docs/vertex-routing-options.md](./docs/vertex-routing-options.md) for diagrams.
- **`routing=` on `VertexAIProvider`.** `"native"` (default) or `"openai"`; anything else raises `ValueError`. `routing="openai"` keeps every model on the Chat Completions route, as in `0.2.x`.
- **`Provider.format_tools(tools)`.** Returns tools in the provider's wire format, or `None` (the default) to leave the format to `AgentExecutor`'s `provider` literal. `VertexAIProvider` answers per route: Gemini `Tool` objects, Anthropic tool dicts, or nested Chat Completions dicts. `AgentExecutor` uses it automatically; duck-typed providers without the method keep the literal format.

### Changed
- **`AgentExecutor` keeps each tool turn whole in history.** A turn is now one assistant message holding every `StreamEndEvent` part (reasoning, text and all tool calls) followed by one user message holding every result in call order. Before, each call was split into its own assistant/user pair and the turn's reasoning and text were dropped. Events and their order are unchanged, and split-shape histories sent by clients are still accepted. Hosts that persist history should store it the same way; see [docs/executor.md](./docs/executor.md#history-shape).
- **Anthropic streaming emits one `ReasoningPart` per thinking block.** Each part carries its own signature, and `StreamEndEvent.parts` now follows the provider's block order. An interleaved message can carry more than one `TextPart`.

### Fixed
- **`GeminiProvider` works through `AgentExecutor`.** `chat()` now accepts `reasoning_effort`: an `int` sets `thinking_budget` and a `str` sets `thinking_level`.
- **Gemini 429s and network errors are retried.** `google-genai` puts the HTTP code in the int `code` and the gRPC status string in `status`; error translation now keys on `code`, so a 429 becomes `RateLimitError` instead of a plain `ProviderError`. `httpx` timeouts and transport errors map to `APITimeoutError` and `APIConnectionError`. Applies to `GeminiProvider` and the native Gemini route.
- **Gemini tool-call IDs are unique.** A call Gemini returns without an ID keeps `call_{name}` on first use and gets `call_{name}_2`, `_3`, … when that ID is already taken in the response or by a completed call in the history. A call answered only by an approval or cancellation placeholder keeps its ID, so a resumed call still matches `approved_tool_calls`, while a later, separate call cannot reuse an earlier approval.
- **Claude thinking blocks keep their own signatures when streaming.** Joining blocks under the last signature broke signature checks once reasoning was replayed.
- **Gemini thought signatures survive the Chat Completions route.** `tool_calls[].extra_content` is stored on `ToolUsePart.metadata["extra_content"]` and sent back unchanged.
- **Parallel tool calls no longer teach the model to stop parallel-calling.** The split history above showed the model one call per turn.
- `httpx` is declared as a direct dependency; the Gemini error translation imports it.

### Breaking
- **Tool error results and history no longer use `{"error": ...}`.** Classified failures are now a string of the form `[error_code] message` (for example `[tool_execution_error] The tool failed unexpectedly.`). Hosts that indexed `event.result["error"]` will raise `TypeError`.
- **`run_stream()` can raise `ModelRetryExhaustedError`.** With the default budget of 3, a fourth correction batch raises instead of continuing until `max_iterations`. `max_iterations` is unchanged: it remains a silent cap.
- **`VertexAIProvider` routes `google/gemini*` and `anthropic/*` natively by default.** Pass `routing="openai"` to keep the `0.2.x` Chat Completions route. History batching and thought-signature replay are fixes and apply on every route. [Migration](./docs/providers/vertexai.md#upgrading-from-02x)
- **A `str` `reasoning_effort` raises `TypeError` on the native Claude route.** Pass an `int` thinking budget. [Migration](./docs/providers/vertexai.md#upgrading-from-02x)
- **Chat-Completions-only kwargs raise `TypeError` on native routes.** `extra_body`, `stop`, `n`, `logprobs`, `top_logprobs` and `response_format` are rejected on both native routes, and `seed`, `frequency_penalty` and `presence_penalty` on the Claude route, before any network call, naming `routing="openai"`. On the Gemini route those three are native `GenerateContentConfig` fields and pass through. [Migration](./docs/providers/vertexai.md#upgrading-from-02x)
- **Direct `chat()` callers must format tools with `provider.format_tools()` on native routes.** `to_vertexai_tool()` output fits the Chat Completions route only. `AgentExecutor` callers need no change. [Migration](./docs/providers/vertexai.md#upgrading-from-02x)
- **A per-call `model=` is no longer forwarded verbatim to another publisher on native instances.** It is sent over the instance's route with that route's publisher prefix removed; a model for another route fails with the backend's own error. This reverses the `0.2.17` entry "forwarded verbatim, at construction and on per-call `model=` overrides alike" for native instances. [Migration](./docs/providers/vertexai.md#upgrading-from-02x)
- **Output differs on native routes.** Claude emits reasoning events and `ReasoningPart`s; stop reasons use each vendor's mapping (Claude adds `stop_sequence`, `refusal`, `pause_turn`); usage comes from the native response; Gemini tool-call IDs are minted as `call_{name}`; Claude sends `max_tokens=8192` when none is given. [Migration](./docs/providers/vertexai.md#upgrading-from-02x)
- **Region availability follows each vendor.** Claude and Gemini are offered in different Vertex regions; an unsupported `location` returns Vertex's own error. [Migration](./docs/providers/vertexai.md#upgrading-from-02x)
- **Exception classes are unchanged.** Errors are still Dobby's `ProviderError` family, reported as `vertexai`; only `__cause__` now comes from the native SDK. Code that inspected `openai` exceptions on `__cause__` must handle `google.genai` and `anthropic` ones. [Migration](./docs/providers/vertexai.md#upgrading-from-02x)
- **Anthropic streaming can return more than one `TextPart`.** Code that read `parts[0]` as the whole text should join every `TextPart`. [Migration](./docs/providers/vertexai.md#upgrading-from-02x)

## [0.2.17] - 2026-07-20

Vertex AI now has one route into this SDK: `VertexAIProvider`, against Vertex's
OpenAI-compatible Model Garden endpoint. The per-vendor Vertex flags are gone.

### Added
- **`VertexAIProvider` supports self-deployed endpoints.** Pass `endpoint_id=` to route through `endpoints/{id}` instead of the shared Model Garden path; the endpoint selects the model, so `model=` becomes an optional display label and is not sent on the wire. `endpoint_host=` carries a dedicated endpoint's DNS, required once an endpoint has `dedicatedEndpointEnabled` because the shared regional DNS stops serving it. `api_version=` defaults to `"v1"`. The docs previously claimed this support without implementing it.
- `examples/vertexai_example.py` — runnable Vertex AI Model Garden example with tool-calling.
- `examples/live_vertex_test.py` — live end-to-end test across Gemini, Claude, gpt-oss, Llama and self-deployed endpoints, printing raw responses.

### Fixed
- **`VertexAIProvider` no longer crashes on a usage payload with null token counts.** Vertex's `openai/gpt-oss-20b-maas` returns a usage object whose `prompt_tokens`/`completion_tokens`/`total_tokens` are all `None`. The guard checked only that the object existed, so those reached `Usage(...)` and raised a `ValidationError` that aborted the stream mid-response. Usage is now omitted when any count is missing rather than zero-filled, which would misreport a real token spend as free. Found by live testing; every unit-test mock supplied integers.
- **`VertexAIProvider.chat(**kwargs)` now forwards to the API.** `max_tokens`, `top_p`, `stop`, `extra_body` and friends were accepted and silently discarded, so a caller migrating from `GeminiProvider` or `OpenAIProvider` lost their token cap with no error and no warning. Passthrough params are applied first, so the fields the provider owns cannot be overwritten.
- `location="global"` built the nonexistent host `global-aiplatform.googleapis.com`. Google documents the global endpoint as the bare `aiplatform.googleapis.com` with no region prefix.

### Removed
- **`GeminiProvider`'s Vertex mode.** The `vertexai`, `project`, and `location` parameters are gone, along with the matching public attributes and the `"gemini-vertexai"` value of `provider.name`. `GeminiProvider` now targets the Gemini Developer API only.
- **`AnthropicProvider`'s Vertex mode.** The `vertex`, `project_id`, `region`, `credentials`, and `access_token` parameters are gone, along with the `"anthropic-vertex"` value of `provider.name`. `AnthropicProvider` now targets the direct Anthropic API and Azure AI Foundry only, and no longer imports `google.auth`.
- **`VertexAIProvider`'s model-family guard.** `google/gemini-*`, `gemini-*`, `claude-*`, and `anthropic/*` model ids are no longer rejected — well-formed ids are forwarded verbatim, at construction and on per-call `model=` overrides alike. The guard also rejected legitimately-named self-deployed containers (e.g. `gemini-finetune-v2`) and matched only unqualified prefixes, so fully-qualified ids bypassed it anyway.

### Breaking
- **`GeminiProvider`'s Vertex mode was published API.** `vertexai`, `project`, and `location` shipped in `0.2.15`, the latest release on PyPI. Three distinct failure modes for existing callers:
  - `GeminiProvider(vertexai=True, ...)` raises `TypeError`.
  - Reading `provider.vertexai`, `provider.project`, or `provider.location` raises `AttributeError`. These were public instance attributes, not only constructor parameters.
  - `provider.name` can no longer return `"gemini-vertexai"`. Code matching on that string does not raise — it silently takes the wrong branch. Audit routing, metrics dimensions, and log filters that key on it.

  There is no equivalent replacement. `VertexAIProvider` reaches Vertex-hosted models through the OpenAI-compatible endpoint, but on a cruder path: no thought-signature handling, coarser finish-reason mapping. Treat it as a behavior change, not a drop-in swap.
- `GeminiProvider.__init__` makes `max_retries` keyword-only. Deliberate: `vertexai` used to be the 3rd positional parameter, so `GeminiProvider("gemini-2.5-flash", None, True)` would otherwise bind `True` to the retry count and silently route traffic to the Developer API when the caller meant Vertex. Keyword-only turns that into a loud `TypeError`.
- `AnthropicProvider(vertex=True, ...)` raises `TypeError` and `provider.name` no longer returns `"anthropic-vertex"`. Unlike the Gemini removal this API was added in the unpublished `0.2.17`, so only callers tracking this branch are affected.
- `AnthropicProvider.__init__` keeps `max_retries` keyword-only. Against published `0.2.15`, where it was the 6th positional parameter, a positional caller now gets a `TypeError`.

`VertexAIProvider` still rejects a `None`, empty, or whitespace-only model id at
construction and on per-call overrides. That check is plain input validation and
survives the family-guard removal.

Version `0.2.16` was never published; the effective upgrade path for users is
`0.2.15 → 0.2.17`.

## [0.2.16] - 2026-06-08

### Added
- Defensive handling of `max_tokens`-truncated tool calls on the OpenAI (Responses API) and Gemini providers. When a tool call's arguments are cut off by the token limit, the SDK now surfaces a clear, catchable signal instead of crashing the stream or silently executing a tool on partial input:
  - `ToolUseErrorEvent` (new `StreamEvent`) is yielded during streaming; the stream still completes with a `StreamEndEvent`, and any valid tool call in the same stream is still delivered.
  - `ToolCallTruncatedError` (new `ProviderError` subclass) is raised on the non-streaming path. It is intentionally **not** retryable — truncation is deterministic, so retrying with the same `max_tokens` reproduces it.
- `examples/tool_call_truncation_demo.py` — live end-to-end demo that forces a real `max_tokens` truncation against OpenAI and Gemini and verifies the streaming/non-streaming behavior.
- Regression tests (`tests/test_tool_call_truncation.py`) covering streaming truncation, mixed valid/invalid tool calls in one stream, the non-streaming raise, and an `response.incomplete`-terminated stream.

### Fixed
- OpenAI Responses streaming now handles the `response.incomplete` terminal event (e.g. a `max_output_tokens` cutoff). Previously only `response.completed` emitted a terminal `StreamEndEvent`, so a truncated stream ended without one. The cutoff is now mapped to `stop_reason="max_tokens"`.

### Note for consumers
- Downstream consumers matching on `event.type` should add a branch for `"tool_use_error"`, or it will fall through their dispatch silently.

## [0.2.15] - 2026-05-20

### Changed
- Relaxed the `google-genai` requirement from `>=2.4.0` to `>=1.68.0,<2`. The 2.4.0 pin made dobby-sdk impossible to install alongside ecosystems that cap `google-genai` below 2.0 (e.g. pipecat's `google-genai>=1.68.0,<2`). `parameters_json_schema` — which the Gemini tool-schema fix relies on — is available across the 1.x line, so no functionality is lost; verified against 1.75.0 including a live structured-output call.

## [0.2.14] - 2026-05-20

### Fixed
- Gemini tools built from nested Pydantic models no longer fail before a request is sent. `Tool.to_gemini_format()` now passes the parameter schema through `parameters_json_schema` instead of the restricted `parameters` field, so `$ref`/`$defs` (and `allOf`, `oneOf`, `const`, `prefixItems`) are accepted and Gemini dereferences them server-side. This fixes structured output with nested output models on Gemini.

### Changed
- Bumped `google-genai` minimum to `>=2.4.0` (required for `parameters_json_schema`).

### Added
- `examples/gemini_structured_output.py` demonstrating nested-model structured output against the Gemini API end-to-end.
- Regression tests (`tests/test_gemini_schema.py`) covering the full rejected-construct family, no-arg and non-model tools, a mutual-exclusivity guard, and a credential-gated live smoke test.
