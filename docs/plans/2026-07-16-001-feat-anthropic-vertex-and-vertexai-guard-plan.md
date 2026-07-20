---
title: Anthropic-on-Vertex Support + VertexAIProvider Model-Family Guard
type: feat
status: superseded
superseded_by: CHANGELOG.md 0.3.0
date: 2026-07-16
origin: docs/brainstorms/anthropic-vertex-and-vertexai-guard-requirements.md
---

# Anthropic-on-Vertex Support + VertexAIProvider Model-Family Guard

> **Superseded in 0.3.0.** Every requirement below (R1–R7) shipped and was then removed.
> `AnthropicProvider(vertex=True)` and the `VertexAIProvider` model-family guard no longer
> exist, and `GeminiProvider`'s Vertex mode was removed alongside them. Kept as a record of
> the original reasoning, not as guidance. See the `0.3.0` CHANGELOG entry.

## Summary

Give `AnthropicProvider` a Vertex-backed mode (`vertex=True`, constructing `anthropic.lib.vertex.AsyncAnthropicVertex`) mirroring its existing Azure AI Foundry mode, and give `VertexAIProvider` a client-side guard that rejects native-Gemini/native-Claude model ids instead of silently routing them through its degraded Model Garden path. This work is already implemented and passing the full test suite in the current working tree (uncommitted); this plan documents the approach taken, the decisions behind it, and its impact across the provider layer so it can be reviewed before committing.

---

## Problem Frame

Vertex AI is three different wire protocols wearing one brand name: native Gemini (already handled by `GeminiProvider(vertexai=True)`), Claude-on-Vertex (Anthropic's own `AsyncAnthropicVertex` client — not handled before this work), and the OpenAI-compatible Model Garden MaaS endpoint (`VertexAIProvider`, scoped to Llama/third-party models). Because Vertex's Model Garden endpoint also happens to accept native Gemini and Claude model ids without erroring, nothing previously stopped a caller from passing `google/gemini-2.5-flash` into `VertexAIProvider` and getting a request that succeeds through a strictly worse path (no thought-signature handling, cruder finish-reason mapping) instead of routing to the correct native provider. See origin document for full analysis.

---

## Requirements

**AnthropicProvider Vertex support**
- R1. `AnthropicProvider` gains a way to select the Vertex AI backend, following the same detect/select shape it already uses for Azure AI Foundry.
- R2. When targeting Vertex, the provider constructs `AsyncAnthropicVertex` under Google ADC, consistent with how `VertexAIProvider`/`GeminiProvider(vertexai=True)` resolve GCP credentials.
- R3. Message conversion, streaming/non-streaming chat, and error translation reuse the existing Anthropic Messages API code paths unchanged.
- R4. `provider.name` distinguishes the Vertex-backed instance from both direct Anthropic and Azure AI Foundry.

**VertexAIProvider model-family guard**
- R5. `VertexAIProvider` rejects model ids belonging to a native-Gemini or native-Claude family rather than silently routing through the MaaS path.
- R6. The rejection error names the correct provider to use instead.
- R7. The guard fires client-side, before any network call is made.

---

## Scope Boundaries

- Renaming `dobby/providers/vertexai/` to something less confusable with Vertex-hosted Gemini (carried from origin — separate decision, not required here).
- Mistral-on-Vertex and raw non-OpenAI-compatible custom endpoints (carried from origin — out of scope).
- Any change to `VertexAIProvider`'s existing Model Garden (Llama/self-deployed) behavior beyond the new guard.
- Any change to `GeminiProvider`'s existing Vertex-hosted Gemini support.

### Deferred to Follow-Up Work

- Live-network integration test against a real Vertex-hosted Claude or Gemini endpoint: not attempted — all new test coverage is unit-level against mocked SDK boundaries, matching the existing test posture for `VertexAIProvider` and `GeminiProvider(vertexai=True)`.
- Capturing this work as an institutional learning via `/mt-compound` once it lands — `docs/solutions/` currently has no entry covering Vertex backend-detection patterns or cross-provider auth conventions.

---

## Context & Research

### Relevant Code and Patterns

- `dobby/providers/gemini/adapter.py:81-123` — `GeminiProvider`'s explicit `vertexai: bool` flag and `"gemini-vertexai"` vs `"gemini"` naming; the pattern `AnthropicProvider(vertex=True)` mirrors.
- `dobby/providers/anthropic/adapter.py` (pre-change) — existing implicit Azure detection (`_is_azure` from `resource`/`azure_ad_token_provider`/`"azure"` in `base_url`) that the new `vertex` flag must stay mutually exclusive with.
- `dobby/providers/vertexai/adapter.py` — `VertexAIProvider`'s existing auth-resolution chain (explicit credentials -> `GOOGLE_APPLICATION_CREDENTIALS_JSON` -> ADC) and `_FINISH_REASON_MAP`, cited in the guard's rejection message as evidence of the degraded path.
- `dobby/executor.py:86` — `AgentExecutor`'s `provider: Literal["openai", "azure-openai", "gemini", "anthropic", "vertexai"]` param, used only for tool-schema-format dispatch. Confirmed by repo research: it intentionally has no `"gemini-vertexai"` entry today (Gemini-on-Vertex already uses plain `"gemini"`, per `tests/test_gemini_schema.py:328-333`, `test_gemini_dispatch_unaffected`), so `AnthropicProvider(vertex=True)` correctly pairs with the existing `"anthropic"` dispatch value — same Messages API tool format regardless of backend. No executor change needed, but the symmetry wasn't tested until this plan's U5.
- `docs/providers/vertexai.md`'s `## Known Limitations` section — the established place in this repo for capability-boundary notes; followed rather than inventing a new section shape.

### Institutional Learnings

- `docs/solutions/integration-issues/gemini-function-declaration-nested-schema-2026-05-20.md` — a different Gemini issue (nested Pydantic schema rejection via `parameters=`, fixed by `parameters_json_schema=`), tangential here but confirms two working conventions worth keeping in mind: (1) Gemini/Vertex schema-shape mismatches surface as an instant local error before any network call — the same posture this plan's guard adopts for model-family mismatches; (2) `parameters_json_schema` already works identically across `GEMINI_API` and `VERTEX_AI` backends, an existing (if narrow) data point on backend-detection handling.
- No documented learning exists for the PR #9/#10 cross-integration break (`383d46a`) or for Vertex auth/credential conventions generally — confirmed via direct search of `docs/solutions/`. This plan's work is genuinely new ground for the repo's knowledge base.

### External References

- Anthropic Python SDK: `anthropic.lib.vertex.AsyncAnthropicVertex` — confirmed importable with the existing `anthropic>=0.75.0` dependency already in `pyproject.toml` (no `anthropic[vertex]` extra needed; contradicts the origin doc's initial assumption that a new extra dependency was required). Constructor signature: `region`, `project_id`, `access_token`, `credentials`, plus standard HTTP client params — `region` required (falls back to `CLOUD_ML_REGION` env var, raises `ValueError` if neither given). `.messages` is a fully-fledged `AsyncMessages` resource identical in shape to direct `AsyncAnthropic`, confirming R3 (no new converters needed).

---

## Key Technical Decisions

- **Explicit `vertex: bool` flag, not implicit detection**: origin left this as an open question between mirroring Azure's implicit param-based detection or Gemini's explicit flag. Chose explicit — with three backends now selectable on `AnthropicProvider`, an implicit heuristic risks ambiguous overlap with Azure's detection; Gemini's precedent (`vertexai: bool`) is the cleaner mirror and origin's own reasoning favored it.
- **`region`/`project_id` passed through only when explicitly given**: `AsyncAnthropicVertex` treats these as `NotGiven`-by-default and resolves `region` from `CLOUD_ML_REGION` and project from its own auth resolution when omitted — matching the existing Azure mode's "let the SDK read its own env vars when params aren't passed" documented behavior.
- **`credentials`/`access_token` exposed on `AnthropicProvider` for Vertex mode, mutually exclusive with each other**: ADC-only auth would leave Vertex-backed Claude without the least-privilege scoping `VertexAIProvider` already offers via its own `credentials` param — added the same escape hatch for parity. When neither is given, `AsyncAnthropicVertex` still resolves ADC internally and lazily (on first request), matching how `GeminiProvider(vertexai=True)` delegates ADC resolution to its underlying SDK.
- **`provider.name` returns `"anthropic-vertex"`**: resolves origin's open naming question, mirroring the `"gemini-vertexai"` / `"azure-anthropic"` convention already established by the other two providers.
- **Guard checks both construction time and per-call `model=` override**: origin's R7 requires client-side rejection "before any network call," but `chat()` accepts a per-call model override independent of the constructor's model — checking only at construction would miss an override to a rejected id on an otherwise-valid instance.
- **Prefix-based matching (`google/gemini-`, `gemini-`, `claude-`, `anthropic/`), not substring matching**: origin flagged unprefixed Gemini id handling as an open question; prefix matching (vs. a broader substring check) avoids false-positive rejection of a hypothetical third-party model whose id happens to contain "gemini" or "claude" mid-string, while still catching the known real-world id shapes. Matching normalizes with `.strip().lower()` first, so a leading/trailing whitespace or newline (plausible from an env var or config value) can't slip past the check.
- **`vertex=True` mutually exclusive with every Azure/direct param, not just Azure-detection params**: an initial version only rejected `vertex=True` combined with Azure-shaped params (`resource`, `azure_ad_token_provider`, an `"azure"`-containing `base_url`) — a caller passing `api_key` or a non-Azure `base_url` alongside `vertex=True` got no error, with both values silently unused. Broadened the check to cover `api_key`/`base_url` unconditionally, since Vertex auth and routing is governed entirely by its own params (`credentials`/`access_token`/`region`/`project_id`) and any of the direct/Azure params being set alongside `vertex=True` signals caller confusion worth failing fast on.
- **Empty/`None` model id rejected with a clear `ValueError` before the prefix check**: `_reject_native_model_family` would otherwise call `.lower()` on `None` and raise an unhelpful `AttributeError` from inside the guard instead of a clear validation error — an SDK boundary should reject unmistakably, not crash on a malformed input that's plausible from a factory function or config layer that forgot a required field.

---

## Open Questions

### Resolved During Planning

- Backend-selection mechanism for `AnthropicProvider`: explicit `vertex: bool` flag (see Key Technical Decisions).
- Guard check point and prefix rules: both construction and per-call override; prefix-based matching on four known id shapes (see Key Technical Decisions).
- Vertex-backed provider naming: `"anthropic-vertex"` (see Key Technical Decisions).
- Whether `anthropic[vertex]` is a new dependency: no — `anthropic.lib.vertex.AsyncAnthropicVertex` imports cleanly with the existing `anthropic>=0.75.0` dependency already declared.

### Deferred to Implementation

- Real end-to-end behavior against a live GCP Vertex Claude/Gemini endpoint (region availability, actual auth flow under ADC) — untested here; all current coverage is unit-level against mocked SDK boundaries.

---

## Implementation Units

### U1. AnthropicProvider Claude-on-Vertex mode

**Goal:** Add a Vertex-backed construction path to `AnthropicProvider`, selectable via an explicit flag, mutually exclusive with Azure.

**Requirements:** R1, R2, R3, R4

**Dependencies:** None

**Files:**
- Modify: `dobby/providers/anthropic/adapter.py`
- Test: `tests/test_anthropic_provider.py`

**Approach:**
- Add `vertex: bool = False`, `project_id: str | None = None`, `region: str | None = None`, `credentials: google.auth.credentials.Credentials | None = None`, `access_token: str | None = None` constructor params.
- Raise `ValueError` if `vertex=True` and any Azure/direct param (`resource`, `azure_ad_token_provider`, `base_url`, `api_key`) is also present — Vertex auth is governed entirely by its own params.
- Raise `ValueError` if both `credentials` and `access_token` are given (mutually exclusive, mirroring `AsyncAnthropicVertex`'s own contract).
- When `vertex=True`, construct `AsyncAnthropicVertex`, passing `region`/`project_id`/`credentials`/`access_token` only when not `None`.
- `provider.name` returns `"anthropic-vertex"` when in Vertex mode (via a `getattr(self, "_is_vertex", False)` fallback, matching the existing Azure name-property robustness pattern for instances built via `__new__` in tests).
- No changes to message conversion, streaming, non-streaming, or error translation — same `anthropic.*` exception hierarchy and `.messages.create()` call shape.

**Patterns to follow:**
- `dobby/providers/gemini/adapter.py`'s `vertexai: bool` flag and `name` property (explicit-flag precedent).
- The existing Azure branch in the same file (mutual-exclusivity validation shape, foundry_kwargs conditional-build pattern).
- `VertexAIProvider`'s `credentials` param (`dobby/providers/vertexai/adapter.py`) for the least-privilege passthrough shape.

**Test scenarios:**
- Happy path: constructing with `vertex=True, region=..., project_id=...` calls `AsyncAnthropicVertex(region=..., project_id=...)` exactly once.
- Happy path: constructing with `vertex=True` and no region/project omits both kwargs, letting the SDK's own `NotGiven` defaults and env-var fallback apply.
- Happy path: `provider.name == "anthropic-vertex"` after Vertex-mode construction.
- Happy path: constructing with `vertex=True, credentials=<obj>` forwards `credentials=<obj>` to `AsyncAnthropicVertex`.
- Happy path: constructing with `vertex=True, access_token="..."` forwards `access_token="..."` to `AsyncAnthropicVertex`.
- Error path: `vertex=True` combined with `resource=...` (or other Azure param) raises `ValueError` naming the conflict.
- Error path: `vertex=True` combined with `api_key=...` raises the same.
- Error path: `vertex=True` combined with a non-Azure `base_url=...` raises the same.
- Error path: `vertex=True, credentials=<obj>, access_token="..."` (both given) raises `ValueError`.

**Verification:**
- All nine scenarios pass; no existing Azure or direct-Anthropic test regresses.

---

### U2. VertexAIProvider model-family guard

**Goal:** Reject native-Gemini and native-Claude model ids client-side, before any network call, at both construction and per-call override.

**Requirements:** R5, R6, R7

**Dependencies:** None

**Files:**
- Modify: `dobby/providers/vertexai/adapter.py`
- Test: `tests/test_vertexai_provider.py`

**Approach:**
- Add a module-level `_reject_native_model_family(model: str) -> None` that first rejects an empty/`None` model with a clear `ValueError`, then checks `model.strip().lower()` against `("google/gemini-", "gemini-")` and `("claude-", "anthropic/")` prefix tuples, raising `ValueError` naming `GeminiProvider(vertexai=True)` or `AnthropicProvider(vertex=True)` respectively.
- Call it first in `__init__`, before any credential resolution.
- Call it again in `chat()` when a per-call `model` override is supplied (skip re-checking the instance model on every call — already validated at construction).

**Patterns to follow:**
- Existing `_map_finish_reason`/`_FINISH_REASON_MAP` module-level helper-plus-constant shape in the same file.

**Test scenarios:**
- Happy path: `model="meta/llama-3.1-405b-instruct-maas"` (a real Model Garden id) constructs without error.
- Error path: `model="google/gemini-2.5-flash"` raises `ValueError` mentioning `GeminiProvider(vertexai=True)`.
- Error path: `model="gemini-2.5-flash"` (unprefixed) raises the same.
- Error path: `model="claude-sonnet-4-5"` raises `ValueError` mentioning `AnthropicProvider(vertex=True)`.
- Error path: `model="anthropic/claude-sonnet-4-5"` raises the same.
- Error path: a per-call `chat(model="google/gemini-2.5-flash")` override raises before `client.chat.completions.create` is invoked (assert not called).
- Edge case: `model=" google/gemini-2.5-flash\n"` (surrounding whitespace) still raises the Gemini rejection — confirms normalization happens before the prefix check.
- Edge case: `model=""` raises a clear `ValueError` naming the empty-model condition, not a prefix-mismatch message.
- Edge case: `model=None` raises the same clear `ValueError`, not an `AttributeError`.

**Verification:**
- All nine scenarios pass; existing Model Garden construction/chat tests are unaffected.

---

### U3. Example script cleanup

**Goal:** Replace the stray root-level debug script with a proper example that demonstrates correct provider usage.

**Requirements:** none directly (hygiene follow-through: a debug script was committed at repo root without following the `examples/` convention)

**Dependencies:** U2 (the example's model id must not trip the new guard)

**Files:**
- Delete: `testVertex.py`
- Create: `examples/vertexai_example.py`

**Approach:**
- Move to `examples/`, matching the `<provider>_example.py` naming convention already used by `examples/anthropic_example.py`.
- Add `if __name__ == "__main__":` guard (previously missing).
- Use a real Model Garden model id instead of a native-Gemini id (previously would now be rejected by U2's guard).
- Drop the manual `service_account.Credentials` construction in favor of `VertexAIProvider`'s own built-in auth resolution.
- Check whether the tool was actually invoked before declaring success, rather than printing a clean exit regardless.

**Test expectation:** none -- example script, not covered by the test suite by design (same as other `examples/*.py` files).

**Verification:**
- Script constructs without raising (guard doesn't reject its model id); no automated test required.

---

### U4. Documentation updates

**Goal:** Reflect the new guard and the Claude-on-Vertex capability in provider docs.

**Requirements:** none directly (R6 is fully satisfied by U2's runtime error text alone; this unit is documentation-discoverability follow-through, not a distinct requirement)

**Dependencies:** U1, U2

**Files:**
- Modify: `docs/providers/vertexai.md`
- Modify: `docs/providers/index.md`

**Approach:**
- Add a "Model-Family Guard" subsection to `docs/providers/vertexai.md` showing the rejection error shape.
- Update the "Known Limitations" section to point to `AnthropicProvider(vertex=True)` for Claude-on-Vertex instead of listing it as unsupported.
- Update the provider table in `docs/providers/index.md` to note Anthropic's and Gemini's Vertex-backed modes.

**Test expectation:** none -- documentation only.

**Verification:**
- Docs accurately describe current behavior; no stale claims about Claude-on-Vertex being unsupported remain in `docs/providers/vertexai.md`.

---

### U5. Repo-research follow-through

**Goal:** Close the two gaps repo research surfaced: a broken README reference and an untested dispatch-symmetry assumption.

**Requirements:** none directly (consistency/completeness follow-through)

**Dependencies:** U1, U3

**Files:**
- Modify: `README.md`
- Modify: `tests/test_anthropic_provider.py` (or a new focused test module, implementer's choice)

**Approach:**
- `README.md:74` still references `uv run --env-file .env python testVertex.py`, which U3 deletes — update to point at `examples/vertexai_example.py`.
- `README.md:85`'s provider summary line predates Gemini/Vertex support entirely (pre-existing gap, not introduced by this plan) — extend it while touching this area.
- Add a test proving `AnthropicProvider(vertex=True)` produces the same tool-schema shape through `AgentExecutor(provider="anthropic", ...)` dispatch as the direct-Anthropic mode, closing the symmetry gap `tests/test_gemini_schema.py:267-273` already closes for the Gemini/Vertex pair.

**Test scenarios:**
- Integration: `AgentExecutor(provider="anthropic", llm=<AnthropicProvider(vertex=True) instance>, tools=[...])` produces the same tool-schema dict shape as the direct-Anthropic instance for the same tool.

**Verification:**
- README no longer references a deleted file; the dispatch-symmetry test passes.

---

## System-Wide Impact

- **Interaction graph:** `AgentExecutor`'s tool-schema dispatch (`dobby/executor.py:86`, `get_tools_schema()`) is unaffected — `AnthropicProvider(vertex=True)` reuses the existing `"anthropic"` dispatch value since it's the same class and wire format. No new dispatch branch needed.
- **API surface parity:** `provider.name` naming is now consistent across all three Vertex-touching providers (`"gemini-vertexai"`, `"anthropic-vertex"`, `"vertexai"`), and the mutual-exclusivity validation pattern on `AnthropicProvider` (Vertex vs. Azure) mirrors the existing Azure-internal validation (resource vs. base_url, api_key vs. token_provider).
- **Integration coverage:** No live-network test exists for either the Vertex-backed Anthropic path or the guard's real-world prefix coverage against Vertex's actual Model Garden catalog — both are unit-tested against mocked SDK/client boundaries only, consistent with the rest of this repo's Vertex-related test suite.
- **Unchanged invariants:** Direct Anthropic and Azure AI Foundry modes are untouched (verified by the full existing Azure/direct test suite passing unmodified). `VertexAIProvider`'s existing Model Garden behavior for legitimate ids (e.g. `meta/llama-*`) is untouched — the guard only intercepts the four rejected id shapes. `GeminiProvider`'s existing Vertex-hosted Gemini support is untouched.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| `anthropic.lib.vertex` is a less-traveled corner of the Anthropic SDK than the direct client; a future SDK version could change its constructor shape | Confirmed importable and structurally sound (via direct inspection) against the pinned `anthropic>=0.75.0`; `.messages` resource shape identical to direct client, so any future break would surface as an import or construction error, not a silent behavior change |
| Prefix-based guard matching could theoretically false-positive on a legitimate third-party Model Garden id starting with `claude-` or `gemini-` | Considered unlikely (these are Anthropic/Google trademark-adjacent prefixes unlikely to be reused by unrelated third-party models); error message is specific and immediately actionable if it ever fires incorrectly |
| No live-network validation for either new path | Explicitly logged as Deferred to Follow-Up Work; matches this repo's existing test posture for other Vertex-touching providers |
| `AnthropicProvider(vertex=True)` doesn't eagerly validate `project_id` resolvability when explicit `credentials`/`access_token` is given (unlike `VertexAIProvider`'s construction-time check) — a caller who omits `project_id` with no `ANTHROPIC_VERTEX_PROJECT_ID` env var gets a `RuntimeError` from the vendored SDK on the first `chat()` call instead of at construction | Accepted deliberately: replicating the SDK's own credential→project resolution in `dobby` would duplicate logic that can drift from the SDK's actual behavior. The failure is not silent — it surfaces as a wrapped `DobbyProviderError` on first use, just later than ideal |

---

## Sources & References

- **Origin document:** [docs/brainstorms/anthropic-vertex-and-vertexai-guard-requirements.md](../brainstorms/anthropic-vertex-and-vertexai-guard-requirements.md)
- Related code: `dobby/providers/anthropic/adapter.py`, `dobby/providers/vertexai/adapter.py`, `dobby/providers/gemini/adapter.py`, `dobby/executor.py`
- Related docs: `docs/providers/vertexai.md`, `docs/providers/index.md`
- Related learning: `docs/solutions/integration-issues/gemini-function-declaration-nested-schema-2026-05-20.md`
