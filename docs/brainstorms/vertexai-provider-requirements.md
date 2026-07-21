# Vertex AI Provider — Requirements

> **Historical record.** `VertexAIProvider` shipped and most of this document still
> holds. One premise no longer does: the references below to Gemini-on-Vertex being
> covered by `GeminiProvider(vertexai=True)` were accurate when written. Those
> parameters were removed in `0.2.17`, so `VertexAIProvider` is now the only Vertex
> route in the SDK. See the `0.2.17` CHANGELOG entry.

**Date:** 2026-07-07
**Status:** Ready for planning
**Scope tier:** Deep — feature (existing product shape; extends the established Provider abstraction)

## Problem / Motivation

Dobby has provider parity with OpenAI, Anthropic (incl. Azure AI Foundry), and Gemini (incl. Gemini-via-Vertex through `google-genai`'s `vertexai=True` flag). It has no path to Google Cloud Vertex AI's **Model Garden** — the catalog of non-Gemini foundation models (Llama, Mistral, Claude-on-Vertex, self-deployed endpoints) hosted on Vertex infrastructure.

This is **general capability-parity work**, not driven by a specific blocked project: no single Model Garden model is required today. The goal is for dobby to offer a Vertex AI provider option the way its peer SDKs (pydantic-ai, LiveKit Agents) do.

## Research Findings (grounding for scope)

Confirmed via direct research against official docs and the actual source of pydantic-ai and LiveKit Agents (not guessed):

- **No unified Python client exists for Model Garden.** Each model family requires a different transport:
  - **Gemini** — `google-genai` SDK, `vertexai=True` (already covered by dobby's `GeminiProvider`).
  - **Claude-on-Vertex** — official `anthropic[vertex]` extra (`AsyncAnthropicVertex`), calling `.../publishers/anthropic/models/{model}:streamRawPredict` with Anthropic's own Messages API shape.
  - **Llama (GA MaaS) and select partner/self-deployed OpenAI-compatible containers** — a genuinely unified, OpenAI-compatible REST surface: `POST https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions`. Google's own docs construct `openai.AsyncOpenAI(base_url=...)` against this endpoint, using a live ADC-derived bearer token as the `api_key`.
  - **Mistral** — Mistral's own SDK (`mistralai` + `MistralGCP`), not the OpenAI-compatible endpoint.
  - **Self-deployed custom containers not speaking OpenAI's format** — raw `aiplatform.Endpoint.raw_predict` / `.stream_raw_predict`, arbitrary request/response shape per container.
  - Auth for all paths above resolves through **Application Default Credentials (ADC)** via `google-auth`.

- **Neither reference SDK unifies this either:**
  - **pydantic-ai**: `GoogleModel` + `GoogleCloudProvider` is Gemini-client-based; it *can* address other publisher-qualified model strings (e.g. `meta/llama-3.3-70b-instruct-maas`) only insofar as `google-genai`'s Vertex client happens to accept them — not a deliberate Model Garden feature. Claude-on-Vertex is not handled by this class at all.
  - **LiveKit Agents**: keeps a **separate class**, `AIPlatformLLM`, specifically for self-deployed Vertex endpoints speaking an OpenAI-compatible wire format — distinct from its Gemini-only `google.LLM` (`vertexai=True`) class. Claude-on-Vertex is unsupported by any current livekit-agents plugin.

This confirms: building one dobby provider that tries to wrap every Model Garden model family in a single class is not how either reference ecosystem does it, and would multiply auth paths, error shapes, and test surface for low marginal value. Scoping to the OpenAI-compatible MaaS surface mirrors LiveKit's `AIPlatformLLM` precedent and reuses dobby's existing OpenAI-shaped conversion logic.

## Decisions Made

1. **Scope: standalone provider, not a Gemini/Anthropic extension.** New package `dobby/providers/vertexai/` (adapter.py, converters.py, __init__.py — same shape as existing providers), distinct from `GeminiProvider`'s existing `vertexai=True` mode.

2. **V1 target: OpenAI-compatible MaaS endpoint only.** Covers:
   - Llama models (GA on Vertex MaaS)
   - Self-deployed Model Garden endpoints whose serving container speaks OpenAI's chat-completions format
   Explicitly **out of scope for v1** (deferred, not rejected):
   - Claude-on-Vertex — candidate for a future `AnthropicProvider` extension (mirroring how Azure AI Foundry was recently added as a backend option), not this provider.
   - Mistral-on-Vertex — requires its own dedicated SDK; low value to wrap for one model family.
   - Raw non-OpenAI-compatible custom endpoints (`rawPredict`/`streamRawPredict` with arbitrary payloads) — no generic `chat()` translation is possible without assuming a wire format.

3. **Auth: ADC by default, plus optional pre-built credentials.** Resolves via `google.auth.default()` (service account file, gcloud user ADC, workload identity, metadata server) unless the caller passes a pre-built `google.auth.credentials.Credentials` object. No custom async token-provider callable in v1 (unlike Anthropic's `azure_ad_token_provider`) — ADC + explicit credentials object covers the realistic use cases without adding a second auth surface to test and document.

4. **Feature parity target: match `OpenAIProvider`.** Since the wire format is OpenAI's chat-completions shape, streaming, tool/function calling, and error-hierarchy mapping should reach the same level of support as the existing OpenAI provider — this is the parity bar, not a reduced "best effort" integration.

## Functional Requirements

- Implements dobby's `Provider[ClientT]` abstract contract (`name`, `model`, `client` properties; `chat()` with the same `stream: Literal[False/True]` overload shape as every other provider).
- Constructor accepts: model id (publisher-qualified string, e.g. `meta/llama-3.3-70b-instruct-maas`, passed through without a hardcoded allow-list), GCP project, location/region (sensible default, override per instance), optional pre-built credentials.
- Non-streaming and streaming chat completions against the MaaS endpoint.
- Tool/function calling support translated to/from dobby's `MessagePart`/`ResponsePart`/`StreamEvent` types, consistent with how `OpenAIProvider` and `AnthropicProvider` do it today.
- Errors translated into dobby's existing `ProviderError` hierarchy (`RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError`) using the same retryable/non-retryable classification as other providers.
- Bearer token handling: since this is not a static API key, the provider must keep the ADC-derived token fresh across calls (refresh before expiry), not just mint it once at construction.
- `provider.name` should clearly distinguish this from `GeminiProvider`'s existing `vertexai=True` mode (e.g. `"vertexai"`) so logs/telemetry aren't ambiguous between the two Vertex-touching providers.

## Non-Goals

- Wrapping Claude-on-Vertex, Mistral-on-Vertex, or raw custom-container endpoints in this provider (see Decisions #2).
- Building a cross-model-family "unified Vertex Model Garden" abstraction — no such thing exists in the ecosystem today; don't invent one speculatively.
- Changing `GeminiProvider`'s existing Vertex support.

## Assumptions / Dependencies

- New dependency: `google-auth` (ADC resolution) is required; `google-cloud-aiplatform` is **not** required for v1 since the OpenAI-compat endpoint is reached via a plain HTTP/OpenAI-SDK-shaped client, not the `aiplatform` package.
- Google's Vertex AI docs tree was mid-rebrand to "Gemini Enterprise Agent Platform" at research time; endpoint hostnames/paths are unchanged, only doc URLs moved — flagged here so planning doesn't get tripped up by stale-looking links.
- Region/model availability for MaaS models (e.g. Llama) is currently limited to specific locations (e.g. `us-central1`) — planning should confirm current availability rather than assuming all regions work.

## Open Questions for Planning

- Whether to reuse dobby's existing OpenAI converters (`to_openai_messages` et al.) given the wire format is OpenAI-compatible, versus writing dedicated Vertex converters — an implementation/reuse decision, not a product one.
- Exact token-refresh mechanism (sync vs async credential refresh, caching strategy) — implementation detail for planning.
- Whether `docs/providers/index.md` (currently stale — lists Anthropic as "🚧 Planned" though it already shipped) should be corrected as part of this work or filed separately.

## Success Criteria

- A `VertexAIProvider` importable from `dobby.providers` alongside the existing four, passing the same category of tests (unit + contract tests) that exist for `OpenAIProvider`/`AnthropicProvider`/`GeminiProvider`.
- Streaming and non-streaming chat, tool calls, and error mapping all demonstrated working against a real (or realistically mocked) Vertex MaaS endpoint.
- Documented in `docs/providers/` following the existing per-provider doc pattern.
