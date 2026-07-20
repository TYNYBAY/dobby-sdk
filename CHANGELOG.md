# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.17] - 2026-07-20

Vertex AI now has one route into this SDK: `VertexAIProvider`, against Vertex's
OpenAI-compatible Model Garden endpoint. The per-vendor Vertex flags are gone.

### Added
- `examples/vertexai_example.py` — runnable Vertex AI Model Garden example with tool-calling.

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
