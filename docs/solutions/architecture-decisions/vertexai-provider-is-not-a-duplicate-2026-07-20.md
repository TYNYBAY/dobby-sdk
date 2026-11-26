---
title: VertexAIProvider looks like a duplicate of OpenAIProvider but cannot be deleted
date: 2026-07-20
category: architecture-decisions
module: dobby/providers
problem_type: architecture_decision
component: providers
symptoms:
  - "dobby/providers/vertexai/adapter.py (691 LOC) and dobby/providers/openai/adapter.py (725 LOC) both wrap AsyncOpenAI and look near-identical at a glance"
  - "Both packages carry their own converters.py, error translation, retry wiring, and streaming accumulator"
  - "Reviewers repeatedly propose deleting dobby/providers/vertexai/ as dead duplication"
  - "The package is named after a cloud vendor, so its real identity (a wire protocol) is invisible from the name"
root_cause: naming_obscures_identity
resolution_type: keep_and_restructure
severity: medium
tags:
  - vertexai
  - openai
  - chat-completions
  - responses-api
  - provider-architecture
  - model-garden
  - duplication
  - pydantic-ai
  - native-routing
related_components:
  - executor
  - tooling
---

# VertexAIProvider looks like a duplicate of OpenAIProvider but cannot be deleted

> **Update 2026-11-26.** `VertexAIProvider` is now **an auth strategy plus a router over
> three protocols**: native Gemini (`google-genai`, `Client(vertexai=True)`), native
> Anthropic (`AsyncAnthropicVertex`), and OpenAI-compatible Chat Completions. See
> [the update below](#update-2026-11-26-a-router-over-three-protocols). The reasoning in the
> rest of this document still holds for its Chat Completions route, which is the only
> route for every publisher other than Google's Gemini and Anthropic.

## Problem

`dobby/providers/vertexai/` and `dobby/providers/openai/` are similarly sized, both
wrap `AsyncOpenAI`, and both carry their own `converters.py`, error translation,
retry wiring, and streaming accumulator. The natural read is that one is redundant
and `vertexai/` should be deleted.

It cannot be. The two speak **different wire protocols**, and `vertexai/` is the only
implementation of its one. This question has been raised repeatedly, so the evidence
is recorded here rather than re-derived each time.

## The four tests that settle it

### 1. Could `OpenAIProvider` serve the Vertex endpoint instead?

No. The two use disjoint API surfaces:

```
dobby/providers/openai/adapter.py    ->  client.responses.create(...)         (Responses API)
dobby/providers/vertexai/adapter.py  ->  client.chat.completions.create(...)  (Chat Completions)
```

`openai/adapter.py` has zero `chat.completions` call sites; `vertexai/adapter.py` has
zero `responses` call sites. The Vertex endpoint is
`https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi`
— an OpenAI-**compatible** surface that serves `/chat/completions` only. It does not
implement the Responses API, so `OpenAIProvider` has no code path that can reach it.

### 2. Could a GCP token just be passed to `OpenAIProvider`?

No. `OpenAIProvider.__init__` accepts `api_key: str | None` and forwards the string
verbatim to `AsyncOpenAI`. `VertexAIProvider` passes something structurally different:

```python
api_key=self._bearer_token   # an async callable, not a string
```

The OpenAI SDK invokes that callable during request preparation, so every request gets
a fresh token; the refresh is guarded by `self._refresh_lock` so concurrent requests
racing an expired token trigger exactly one refresh. GCP access tokens expire in about
an hour, so a static `str` breaks any session outliving one token. A plain string
parameter cannot express this.

### 3. Are the tool schemas genuinely different?

Yes — and the code already shares everything shareable. `to_vertexai_tool()`
(`dobby/providers/vertexai/converters.py`) reuses the existing schema builder and only
re-nests the result:

```python
flat = tool.to_openai_format()          # shared schema construction
return {"type": "function", "function": {...}}   # Chat Completions' nested shape
```

The Responses API wants a flat `{name, description, parameters}`; Chat Completions
wants `{type: "function", function: {...}}`. That is 24 lines of re-nesting on top of
shared logic, not a duplicated schema builder.

### 4. What do comparable frameworks do?

This is the decisive corroboration. **pydantic-ai ships two OpenAI-wire model classes**:

- `OpenAIResponsesModel` — Responses API
- `OpenAIChatModel` — Chat Completions

For Vertex AI Model Garden it uses **`OpenAIChatModel`** — explicitly the Chat
Completions one. LiveKit Agents reaches the same place from the other direction: its
`openai.LLM.with_vertex()` (a Chat Completions path) was removed and regrown as
`google.AIPlatformLLM`, a separate class in a different package, still Chat Completions.

Mapped onto dobby: `openai/adapter.py` is the equivalent of `OpenAIResponsesModel`, and
**`vertexai/adapter.py` is dobby's `OpenAIChatModel`** — the missing half of the pair,
wearing a cloud-vendor name.

## What deletion would cost

Dependents (verified): `dobby/executor.py` (imports `to_vertexai_tool`),
`dobby/providers/__init__.py` (public export), `tests/test_vertexai_provider.py`,
`tests/test_gemini_schema.py`, `examples/vertexai_example.py`, the README setup
section, `docs/providers/vertexai.md`, `docs/providers/index.md`.

Capability lost: **all** Vertex AI Model Garden support — Llama, and any self-deployed
serving container speaking the OpenAI Chat Completions wire format. No replacement path
exists anywhere in the codebase.

## Why it reads as a duplicate anyway

The package is named after a **cloud vendor**, while its actual identity is a **wire
protocol plus an auth strategy**:

| Named | Actually is (2026-07-20) |
|---|---|
| `VertexAIProvider` | OpenAI Chat Completions client + GCP bearer-token auth |

As of 2026-11-26 it is GCP auth plus a router over three protocols; see the update below.

That mismatch is the root cause of the recurring question — it *is* another OpenAI
provider, for a different dialect, and nothing in the name says so.

The same mismatch once justified a `_reject_native_model_family()` guard: the name
promises "everything on Vertex", so callers reach for it with `gemini-*` and `claude-*`
ids, which this endpoint serves through a cruder path. That guard was removed in
`0.2.17` — well-formed ids are now forwarded verbatim — so the naming mismatch is
unmitigated and the case for renaming is stronger, not weaker.

## Resolution: restructure, do not delete

The duplication is real but lives in the **scaffolding** — client construction, retry
wiring, streaming accumulator, error translation, `Provider` boilerplate — not in the
wire logic. The fix is extraction:

```
providers/_openai_chat.py      # Chat Completions wire, extracted and shared
    \_ VertexAIProvider        # GCP auth + base_url (thin)
providers/openai/adapter.py    # Responses wire, unchanged
```

This removes roughly 400 duplicated lines while preserving the ~250 lines that are
genuinely protocol-specific. Precedent: LiveKit keeps one shared OpenAI-wire streaming
engine in core (`livekit-agents/livekit/agents/inference/llm.py`) that its OpenAI plugin
subclasses in 24 lines; its newer vendor plugins subclass `OpenAILLM` (perplexity, 86
LOC) rather than duplicating it.

Renaming the package to describe the protocol rather than the cloud would prevent the
question recurring. That was already flagged as an open decision in
`docs/plans/2026-07-07-001-feat-vertexai-provider-plan.md` and remains unresolved.

Tracked as U2.2 in
`docs/plans/2026-07-20-001-refactor-provider-layer-pydantic-livekit-alignment-plan.md`.

## The file that *is* safe to delete

`dobby/providers/openai/completions.py` (522 LOC) is a **third** Chat Completions
implementation, and this one is genuinely dead:

- Exported nowhere — absent from `dobby/providers/openai/__init__.py`,
  `dobby/providers/__init__.py`, and `dobby/__init__.py`.
- Imported by nothing across `dobby/`, `tests/`, and `examples/`.
- **Unimportable.** Its relative imports are one level short of correct:
  `from ..tools import BaseTool` resolves to `dobby.providers.tools`, which does not
  exist (the working `adapter.py` alongside it correctly uses `from ...tools`). It
  raises `ModuleNotFoundError` on import.
- Its own docstring says it is legacy and may be removed.

It rotted precisely because nothing depended on it — the inverse of `vertexai/`, which
is load-bearing.

## Prevention

- Name provider packages after the **wire protocol** they speak, not the cloud that
  hosts it. A cloud can host several protocols, and one protocol spans several clouds.
- Before proposing deletion of a provider package, run the four tests above: does
  another provider speak its protocol; can its auth model be expressed by the
  alternative; are the wire formats actually identical; and what do comparable
  frameworks do.
- Similar file size and a shared SDK import are not evidence of duplication. Check the
  API surface actually called — here, `responses.create` versus
  `chat.completions.create` — before concluding anything.

## Update 2026-11-26: a router over three protocols

`VertexAIProvider` now routes by publisher, with `routing="native"` as the default:

| Instance model | Route |
|---|---|
| `google/gemini*` | Native Gemini API through an internal `GeminiProvider` on `genai.Client(vertexai=True)` |
| `anthropic/*` | Anthropic Messages API through an internal `AnthropicProvider` on `AsyncAnthropicVertex` |
| anything else, or `endpoint_id=` set | OpenAI-compatible Chat Completions, as described above |

`routing="openai"` keeps every model on Chat Completions. Its identity is now **an auth
strategy plus a router over three protocols**, not one Chat Completions client.
`provider.client` is still that Chat Completions `AsyncOpenAI` client on every route, with
the same bearer-token callable from test 2; on a native route it is built only when
`provider.client` is first read, since `chat()` never uses it there.

Why it changed:

- **Comparable libraries route Gemini and Claude natively.** LiteLLM picks
  `generateContent` for `gemini-*` and Anthropic's `rawPredict` for `claude-*` from one
  entry point. pydantic-ai pairs `GoogleModel` with a Vertex provider and `AnthropicModel`
  with `AsyncAnthropicVertex`. LangChain ships `ChatGoogleGenerativeAI(vertexai=True)` and
  `ChatAnthropicVertex`. Only the remaining Model Garden publishers go through
  `endpoints/openapi` in any of them.
- **The extra translation lost data.** On the Chat Completions route Google translates
  each request a second time. Gemini thought signatures were dropped (the Chat Completions
  route now replays them through `tool_calls[].extra_content`), and Claude's thinking blocks
  and their signatures had no place in the Chat Completions shape.

What did not change, and why the 2026-07-20 decision still stands:

- The Chat Completions route is still the only implementation of that protocol in the
  codebase and still serves Llama, gpt-oss, Gemma and self-deployed endpoints. Test 1
  above still holds for it.
- No converters were copied. The native routes delegate to the existing Gemini and
  Anthropic adapters through an internal `_from_client` constructor, so there is one
  converter per protocol (avoiding LangChain's copied-converter drift).
- `provider.name` stays `"vertexai"` on every route. The route is a private attribute,
  and the tool format reaches the executor through the public `Provider.format_tools()`
  rather than through a new executor literal.

The rename question from 2026-07-20 changes shape: the package is now named after the
cloud because it genuinely spans several protocols on that cloud. Naming by wire protocol
still applies to the protocol modules underneath it.

Diagrams: `docs/vertex-routing-options.md`. Migration: the "Upgrading from 0.2.x" section
of `docs/providers/vertexai.md`.

## Related Issues

- `docs/plans/2026-07-07-001-feat-vertexai-provider-plan.md` — original provider plan;
  the package rename was left open there.
- `CHANGELOG.md`, `0.2.17` — removed the per-vendor Vertex flags on `AnthropicProvider`
  and `GeminiProvider`, and the `VertexAIProvider` model-family guard. `VertexAIProvider`
  is now the only Vertex route, which makes the naming mismatch described above the
  single remaining source of confusion.
