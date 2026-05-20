---
title: "fix: Gemini tool schema rejects rich Pydantic constructs ($ref/$defs, allOf, oneOf, const)"
type: fix
status: completed
created: 2026-05-20
depth: standard
---

# fix: Gemini tool schema rejects rich Pydantic constructs

## Problem Frame

`Tool.to_gemini_format()` (`dobby/tools/tool.py:223`) builds a `genai_types.FunctionDeclaration` by passing the tool's parameter schema to the SDK's **`parameters=`** field. That field is validated against Gemini's `genai_types.Schema` — a restricted OpenAPI 3.0.3 subset. When the schema comes from a Pydantic model with nested sub-models (the structured-output path: `Tool.from_model(output_type, ...)` in `dobby/executor.py:96`), `model_json_schema()` emits `$ref` pointers plus a `$defs` block, which `Schema` forbids. Validation fails locally, before any request is sent — instant failure, `$0.00` cost, on every Gemini call that uses a nested output model.

This is **not** a `$ref`/`$defs`-only problem. Probing `google.genai==1.57.0`'s `Schema` validator shows the `parameters=` path rejects an entire family of constructs that ordinary Pydantic models emit:

| Construct | Pydantic source that emits it | `parameters=` |
|---|---|---|
| `$ref` + `$defs` | nested models (the reported bug) | rejected |
| `allOf` | nested model carrying a description/default; some discriminated unions | rejected |
| `oneOf` | certain unions | rejected |
| `const` | single-member `Literal["x"]` | rejected |
| `prefixItems` | typed `tuple[str, int]` | rejected |
| `patternProperties` | constrained `dict` | rejected |

(camelCase constraint keys like `maxItems`, `anyOf`, `maxLength`, `enum`, `additionalProperties` are *accepted* because `Schema` aliases them — so the failure surface is specifically the constructs above, not all of JSON Schema.)

The SDK exposes a purpose-built field for exactly this: **`FunctionDeclaration(parameters_json_schema=...)`** accepts raw JSON Schema unchanged (verified: `$ref`/`$defs`/`allOf`/`oneOf`/`const`/`prefixItems`/`patternProperties` all pass), serializes to the wire as `parametersJsonSchema` with `$defs` preserved, and lets Gemini dereference/normalize server-side. It is **mutually exclusive** with `parameters=`.

**Approach decision (confirmed with user):** route dobby's Gemini tool schema through `parameters_json_schema` instead of writing and maintaining an in-dobby Pydantic→Gemini-Schema converter. A converter would reimplement the dereferencing + construct-normalization that the SDK and Gemini already perform server-side, and would drift each time Pydantic changes its schema output.

---

## Scope

**In scope**
- Fix `Tool.to_gemini_format()` so nested/rich Pydantic models produce a valid `FunctionDeclaration`.
- Unify both branches of `to_gemini_format()` (the `_model` path and the hand-built-params path) onto `parameters_json_schema`, removing the `Schema`-validation foot-gun for every Gemini tool.
- Regression tests enumerating the rejected-construct family plus no-arg and non-model tools.

**Out of scope / non-goals**
- `to_openai_format()` and `to_anthropic_format()` — their validators accept `$ref`/`$defs`; unaffected.
- Building a standalone JSON-Schema→Gemini-Schema converter (explicitly rejected in favor of `parameters_json_schema`).
- Changing dobby's structured-output mechanism (it stays tool-call based via `OUTPUT_TOOL_NAME`).

### Deferred to Follow-Up Work
- Live end-to-end smoke test against the real Gemini API / Vertex AI backend (needs credentials). Captured as an optional, credential-gated test in U3 — not part of the core fix's required verification.

---

## Key Technical Decisions

1. **Use `parameters_json_schema`, not a converter.** One field change covers the whole construct family; Gemini owns dereferencing/normalization. (See Problem Frame table for the evidence.)
2. **Unify both `to_gemini_format()` branches onto it.** The non-`_model` branch currently builds a hand-shaped `{"type":"object","properties":...,"required":...}` dict that happens to satisfy `Schema`. That dict is also valid JSON Schema, so routing it through `parameters_json_schema` keeps one construction path and removes any future risk of a hand-built param tripping `Schema` validation. Set **only** `parameters_json_schema` (never alongside `parameters` — server enforces exclusivity).
3. **No backend branching.** `parameters_json_schema` is accepted by the SDK for both `GEMINI_API` and `VERTEX_AI`; `to_gemini_format()` has no backend context and does not need any. The provider (`dobby/providers/gemini/adapter.py`) continues to receive a ready `genai_types.Tool` from the executor unchanged.
4. **Keep `model_json_schema()` raw.** Do not strip `title`/`default`/`$defs` — Gemini tolerates them and server-side handling is the point.

---

## Implementation Units

### U1. Route `to_gemini_format()` through `parameters_json_schema`

**Goal:** Build the `FunctionDeclaration` with `parameters_json_schema` for both the model and hand-built branches so nested/rich schemas are accepted.

**Requirements:** Fixes the reported Gemini structured-output failure; advances "rich Pydantic models work across all providers."

**Dependencies:** none.

**Files:**
- `dobby/tools/tool.py` (modify `to_gemini_format()`, ~lines 223-248)

**Approach:**
- Compute the params dict exactly as today: `self._model.model_json_schema()` when `_model` is set, else the hand-built `{"type": "object", "properties": {...}, "required": [...]}`.
- Construct a single `genai_types.FunctionDeclaration(name=..., description=..., parameters_json_schema=<params>)` — drop the `parameters=` argument entirely. Do not set both fields.
- Return `genai_types.Tool(function_declarations=[func_decl])` as before.
- Update the method docstring to note schemas are passed as raw JSON Schema (Gemini dereferences server-side).

**Technical design** (directional, not implementation spec):
```python
# parameters dict computed identically to current code (model schema OR built dict)
func_decl = genai_types.FunctionDeclaration(
    name=self.name,
    description=self.description,
    parameters_json_schema=parameters,   # was: parameters=parameters
)
```

**Patterns to follow:** mirror the existing branch structure in `to_openai_format()`/`to_anthropic_format()` for computing the params dict; only the `FunctionDeclaration` field changes.

**Test scenarios:** covered by U2 (this unit has no standalone test file; its behavior is exercised through `to_gemini_format()` tests).

**Verification:** `to_gemini_format()` returns a `genai_types.Tool` without raising for a nested-model tool; the reproduction (`Tool.from_model(ParsedResume, ...)`) that previously raised `ValidationError` now succeeds.

---

### U2. Regression tests for the Gemini schema construct family

**Goal:** Lock in that `to_gemini_format()` accepts every construct the old `parameters=` path rejected, plus no-arg and non-model tools, and that the schema serializes to the wire correctly.

**Requirements:** Prevents recurrence of the reported bug and the wider construct-family failures; this is the "add test cases" deliverable.

**Dependencies:** U1.

**Files:**
- `tests/test_gemini_schema.py` (new)

**Approach:** Table-driven tests over Pydantic models that exercise each rejected construct. Each test calls `Tool.from_model(Model, name=..., description=...).to_gemini_format()` and asserts it returns a `genai_types.Tool` without raising, then asserts the serialized declaration carries `parametersJsonSchema` (not `parameters`) and preserves nesting. Include a guard test that the *old* path would have failed, to document intent.

**Test scenarios** (each: build model → `to_gemini_format()` → assert no raise + correct serialization):
- **Reported bug — nested models in lists:** `ParsedResume` with `list[SocialLinkItem]`, `list[SkillData]`. Assert no `ValidationError`; assert serialized `FunctionDeclaration.model_dump(by_alias=True, exclude_none=True)` contains key `parametersJsonSchema` and that `$defs` is preserved inside it.
- **Nested model as object property:** model with a single nested `Address` field (emits `$ref` in a property, not just in `items`).
- **Deep nesting — def referencing def:** `Resume` → `Experience` → `Company` → `Address`. Assert no raise and no loss of nested structure.
- **`allOf`:** nested model field carrying a `Field(description=...)`/default that makes Pydantic wrap the `$ref` in `allOf`. Assert accepted.
- **`oneOf`/union:** field typed as a union that emits `oneOf`. Assert accepted.
- **`const`:** field typed `Literal["fixed"]` (single-member). Assert accepted.
- **`prefixItems`:** field typed `tuple[str, int]`. Assert accepted.
- **Flat model with constraints:** `Field(ge=0, le=120)`, `max_length`, `Literal[...]` enum, `str | None` optional, defaults. Assert accepted and that constraint keys survive in `parametersJsonSchema`.
- **No-argument tool:** a `Tool` subclass whose `__call__` takes no params (non-`_model` branch → `{"type":"object","properties":{},"required":[]}`). Assert `to_gemini_format()` succeeds and serializes `parametersJsonSchema` with empty `properties`.
- **Non-model tool with typed params:** a `Tool` subclass with annotated scalar params (non-`_model` branch). Assert accepted and serialized via `parametersJsonSchema`.
- **Mutual exclusivity guard:** assert the produced `FunctionDeclaration` sets `parameters_json_schema` and leaves `parameters` unset (`None`), so we never send both.

**Verification:** `tests/test_gemini_schema.py` passes; the nested-model case fails if U1 is reverted (sanity that the test actually guards the fix).

---

### U3. Optional credential-gated live smoke test

**Goal:** Prove the schema is accepted *server-side*, not just constructed client-side, on at least the Gemini Developer API backend.

**Requirements:** De-risks the assumption that Gemini honors `parametersJsonSchema` at runtime; optional because it needs credentials.

**Dependencies:** U1.

**Files:**
- `tests/test_gemini_schema.py` (add a `@pytest.mark.skipif(no GEMINI_API_KEY)` test, or a separate `tests/integration/` module if the repo separates integration tests)

**Approach:** With a real `GeminiProvider`, register a nested-model output tool via the executor's structured-output path and issue one minimal `chat()` call, asserting the request is accepted (no 4xx schema-rejection) and a function call / structured result comes back. Skip cleanly when `GEMINI_API_KEY` is unset so CI without secrets stays green.

**Test scenarios:**
- Covers the nested-model output tool end-to-end against the live Gemini API; asserts no schema-related API error and that a structured tool-call response is returned.
- Skips with a clear reason when credentials are absent.

**Execution note:** keep this isolated and skip-by-default; it must not break credential-less CI.

**Verification:** runs and passes when `GEMINI_API_KEY` is present; reports skipped otherwise.

---

## System-Wide Impact

- **`dobby/executor.py`** — unchanged. It already calls `tool.to_gemini_format()` for every registered tool (including the `OUTPUT_TOOL_NAME` structured-output tool); it just starts receiving valid declarations for nested models.
- **`dobby/providers/gemini/adapter.py`** — unchanged. Receives a ready `genai_types.Tool` and attaches it to `config.tools`; no backend branching added.
- **OpenAI / Anthropic paths** — untouched; they were never affected.

---

## Risks & Mitigations

- **Server-side support on Vertex AI is unverified in this environment** (no credentials). Mitigation: U3 documents the live check; SDK accepts `parameters_json_schema` for both backends client-side, and `google-genai==1.57.0` is recent enough to expose it. If a future Vertex gap surfaces, the fallback-converter option remains available but is explicitly deferred.
- **Accidentally setting both `parameters` and `parameters_json_schema`** would be a server-enforced error. Mitigation: U1 removes the `parameters=` argument entirely; U2's mutual-exclusivity guard test locks it in.

---

## Verification (overall)

- Reproduction script (`Tool.from_model(ParsedResume).to_gemini_format()`) that raised `ValidationError: ... $ref ... $defs` now returns a `genai_types.Tool`.
- `tests/test_gemini_schema.py` passes (U2); reverting U1 makes the nested-model case fail.
- Existing suite (`tests/test_parallel_tools.py`, `tests/test_terminal_tools.py`, `tests/test_provider_errors.py`) stays green.
