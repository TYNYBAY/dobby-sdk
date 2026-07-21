---
date: 2026-07-08
topic: anthropic-vertex-and-vertexai-guard
---

# Anthropic-on-Vertex Support + VertexAIProvider Model-Family Guard

> **Superseded in 0.3.0.** The Vertex modes and the model-family guard described here were
> removed. Retained as a record of the original reasoning, not as current requirements.

## Summary

Add Vertex AI backend support to `AnthropicProvider` (mirroring the multi-backend pattern it already uses for Azure AI Foundry, and the `vertexai` flag `GeminiProvider` already uses for Vertex-hosted Gemini), and add a client-side guard to `VertexAIProvider` that rejects `gemini-*`/`claude-*` model ids with an error pointing to the correct native provider.

---

## Problem Frame

Google Cloud Vertex AI is not one wire protocol — it's three, each requiring a different client:

1. **Gemini native** (`google-genai` SDK) — already handled: `GeminiProvider(vertexai=True)`.
2. **Claude-on-Vertex** (Anthropic's own `AsyncAnthropicVertex` client, same Messages API shape as direct Anthropic) — **not handled today**. `dobby/providers/anthropic/adapter.py` supports direct Anthropic and Azure AI Foundry only.
3. **Model Garden OpenAI-compatible MaaS** (`openai.AsyncOpenAI` pointed at Vertex's `openapi/chat/completions` endpoint) — handled by `VertexAIProvider`, scoped at build time to third-party/self-deployed models (Llama, Mistral, etc.) with Claude-on-Vertex and Gemini explicitly called out as deferred (see `docs/brainstorms/vertexai-provider-requirements.md`, Decisions #2).

Two gaps follow from this:

- **Missing coverage**: dobby has no way to call Claude models hosted on Vertex, even though Anthropic ships an official client for exactly that (`anthropic[vertex]` extra, same request/response schema as the direct client — only auth and transport differ).
- **Silent wrong-path risk**: Vertex's OpenAI-compatible endpoint also happens to serve native Gemini models (model ids like `google/gemini-2.5-flash`). Nothing currently stops a caller from passing such an id into `VertexAIProvider` — and unlike a typical "wrong client" mistake, the request **doesn't error**, it succeeds through a strictly worse path (no thought-signature handling, cruder finish-reason mapping) than `GeminiProvider(vertexai=True)` already provides. The same risk applies to `claude-*` ids once Claude-on-Vertex has its own proper home.

Verified against official docs and source (not guessed): the Anthropic Python SDK ships `AnthropicVertex`/`AsyncAnthropicVertex` in `anthropic.lib.vertex`, documented as the standard client interface with only auth/routing differing from the direct client. Vertex's OpenAI-compatible Chat Completions endpoint is confirmed (Google Cloud docs) to also serve native Gemini models under `google/gemini-*` ids, not just Model Garden third-party models.

---

## Requirements

**AnthropicProvider Vertex support**
- R1. `AnthropicProvider` gains a way to select the Vertex AI backend, following the same "detect/select backend" shape it already uses for Azure AI Foundry (explicit flag and/or parameter-based detection — exact mechanism is a planning decision).
- R2. When targeting Vertex, the provider constructs `AsyncAnthropicVertex` instead of `AsyncAnthropic`/`AsyncAnthropicFoundry`. Auth resolves via Google ADC (`google.auth.default()`), consistent with how `VertexAIProvider` and `GeminiProvider(vertexai=True)` already resolve GCP credentials.
- R3. Message conversion, streaming/non-streaming chat, and error translation reuse the existing Anthropic Messages API code paths unchanged — Claude-on-Vertex has the same wire schema as direct Anthropic, so no new converters are needed.
- R4. `provider.name` distinguishes the Vertex-backed instance from both direct Anthropic and Azure AI Foundry (consistent with how `GeminiProvider` already reports `"gemini-vertexai"` vs `"gemini"`), so logs/telemetry aren't ambiguous across the three backends.

**VertexAIProvider model-family guard**
- R5. `VertexAIProvider` rejects (raises, does not warn-and-continue) when given a model id belonging to a native-Gemini or native-Claude family (e.g. `google/gemini-*`, `claude-*`/`anthropic/*` prefixes), rather than silently routing the call through the OpenAI-compatible MaaS path.
- R6. The rejection error names the correct provider to use instead (`GeminiProvider(vertexai=True)` for Gemini ids, the new Vertex-backed `AnthropicProvider` mode for Claude ids), so the failure is immediately actionable.
- R7. The guard fires client-side, before any network call is made.

---

## Acceptance Examples

- AE1. **Covers R5, R6, R7.** Given a `VertexAIProvider` instance, when `chat()` (or construction, per planning's chosen check point) is called with model id `"google/gemini-2.5-flash"`, then it raises an error identifying that `GeminiProvider(vertexai=True)` should be used instead — no request is sent to Vertex.
- AE2. **Covers R5, R6, R7.** Given a `VertexAIProvider` instance, when called with a `claude-*`/`anthropic/*` model id, then it raises an error identifying the Vertex-backed `AnthropicProvider` mode instead — no request is sent to Vertex.
- AE3. **Covers R1, R2, R4.** Given `AnthropicProvider` constructed in Vertex mode, when `chat()` is called, then it uses `AsyncAnthropicVertex` under ADC auth and `provider.name` reflects the Vertex backend distinctly from `"anthropic"` and `"azure-anthropic"`.

---

## Success Criteria

- A Vertex-backed `AnthropicProvider` mode is usable end-to-end (streaming + non-streaming + tool calls + error mapping), tested to the same bar as the existing Azure AI Foundry mode.
- `VertexAIProvider` cannot silently serve a `gemini-*` or `claude-*` model id — attempting to do so fails fast with a clear, actionable error, covered by a test for each rejected family.
- No regression to `GeminiProvider(vertexai=True)`, `VertexAIProvider`'s existing Model Garden behavior, or `AnthropicProvider`'s direct/Azure modes.

---

## Scope Boundaries

- Renaming `dobby/providers/vertexai/` to something less confusable with "Vertex-hosted Gemini" (e.g. `vertex_maas/`) — raised during discussion, deliberately left out of this doc's scope. Separate decision, not required for the guard or the Anthropic extension to work correctly.
- Mistral-on-Vertex and raw non-OpenAI-compatible custom endpoints — already out of scope per `docs/brainstorms/vertexai-provider-requirements.md`; unaffected by this work.
- Any change to `VertexAIProvider`'s existing Model Garden (Llama/self-deployed) behavior beyond the new guard.
- Any change to `GeminiProvider`'s existing Vertex-hosted Gemini support.

---

## Key Decisions

- **Hard reject, not warn-and-continue, for the guard**: Industry norm (OpenAI/Azure, Anthropic-on-Bedrock) is to let the server's own error surface for "wrong model on this client" — client-side guarding is uncommon and considered unnecessary work by most major SDKs. This case is the exception: Vertex's OpenAI-compatible endpoint does **not** error on `gemini-*` ids, it succeeds through a degraded path. Silent degraded-success is worse than the clean 404s that let other SDKs skip this guard, so it's worth the (low) cost here specifically.
- **Extend `AnthropicProvider` in place, not a new module**: Matches the precedent already set twice in this codebase (`AnthropicProvider`'s own Azure AI Foundry mode, `GeminiProvider`'s `vertexai` flag) — one class, backend selected via constructor, same wire format. A third parallel provider module for "Anthropic but on Vertex" would duplicate conversion/error-translation logic for no benefit, since the wire schema is identical to direct Anthropic.
- **No new converters**: Claude-on-Vertex uses the same Messages API request/response shape as direct Anthropic (confirmed via Anthropic SDK docs) — existing `to_anthropic_messages` and response parsing carry over unchanged.

---

## Dependencies / Assumptions

- New dependency: `anthropic[vertex]` extra (adds `google-auth` transitively — already a dobby dependency via `VertexAIProvider`/`GeminiProvider`'s Vertex mode, so no net-new credential-resolution surface).
- Assumes Google ADC resolution (service account, gcloud user credentials, workload identity) is an acceptable auth story for Claude-on-Vertex, consistent with how dobby's other Vertex-touching providers already authenticate.

---

## Outstanding Questions

### Deferred to Planning

- **[Affects R1][Technical]** Exact mechanism for backend selection on `AnthropicProvider` (explicit `vertex: bool` flag vs. Azure-style implicit detection from which params are passed) — implementation decision, should follow whichever precedent (Azure's implicit detection or Gemini's explicit flag) reads more clearly with a third backend added.
- **[Affects R5–R7][Technical]** Exact check point for the guard (constructor-time vs. per-`chat()`-call) and exact model-id prefix matching rules (e.g. handling of unprefixed Gemini ids, if any exist on the MaaS endpoint) — needs verification against Vertex's actual id conventions during planning.
- **[Affects R4]** Naming convention for the Vertex-backed Anthropic provider's `name` property (e.g. `"anthropic-vertex"`, mirroring `"gemini-vertexai"` and `"azure-anthropic"`) — small decision, left to planning for consistency-checking against the other two.
