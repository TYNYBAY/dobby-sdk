# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.17] - 2026-07-17

### Added
- `AnthropicProvider` gains a `vertex=True` mode for Claude-on-Vertex, constructing `anthropic.lib.vertex.AsyncAnthropicVertex` under Google ADC (or explicit `credentials`/`access_token` for least-privilege scoping). Mutually exclusive with Azure/direct params (`resource`, `base_url`, `azure_ad_token_provider`, `api_key`). `provider.name` reports `"anthropic-vertex"`.
- `examples/vertexai_example.py` — runnable Vertex AI Model Garden example with tool-calling.

### Fixed
- `VertexAIProvider` now rejects `google/gemini-*`, `gemini-*`, `claude-*`, and `anthropic/*` model ids client-side (at construction and on any per-call `model=` override), instead of silently routing them through its degraded Model Garden path. Input is normalized (whitespace-stripped, case-insensitive) before matching, and an empty/`None`/whitespace-only model id raises a clear error instead of crashing.

### Breaking
- Any existing caller passing a native-Gemini or native-Claude model id to `VertexAIProvider` will now get a `ValueError` at construction (or on a per-call `model=` override) instead of the previous degraded-but-working behavior. Affected callers should switch to `GeminiProvider(vertexai=True)` or `AnthropicProvider(vertex=True)` respectively.
- `AnthropicProvider.__init__`'s parameters from `vertex` onward (`vertex`, `project_id`, `region`, `credentials`, `access_token`, `max_retries`) are now keyword-only. A caller passing `max_retries` positionally (6th positional argument) will now get a `TypeError` instead of silently binding that value to `vertex`.

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
