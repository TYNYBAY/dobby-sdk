---
title: Gemini FunctionDeclaration rejects nested Pydantic schemas ($ref/$defs)
date: 2026-05-20
category: integration-issues
module: dobby/tools
problem_type: integration_issue
component: tooling
symptoms:
  - "pydantic.ValidationError raised from to_gemini_format() before any request is sent"
  - "model_json_schema() emits $ref/$defs that genai_types.Schema forbids"
  - "Every Gemini structured-output call with a nested output model fails; flat models work"
  - "parameters= also rejects allOf, oneOf, const, and prefixItems"
root_cause: wrong_api
resolution_type: code_fix
severity: high
tags:
  - gemini
  - google-genai
  - json-schema
  - structured-output
  - function-declaration
  - pydantic
  - tool-calling
related_components:
  - assistant
  - testing_framework
---

# Gemini FunctionDeclaration rejects nested Pydantic schemas ($ref/$defs)

## Problem

`Tool.to_gemini_format()` (`dobby/tools/tool.py`) built a Gemini `FunctionDeclaration` using the `parameters=` field. That field is validated against `genai_types.Schema`, a restricted OpenAPI 3.0.3 subset. Pydantic models with nested sub-models emit `$ref` + `$defs` via `model_json_schema()`, which `Schema` forbids — so constructing the declaration raised a `pydantic.ValidationError` locally, before any request was sent, on every Gemini structured-output call backed by a nested model.

## Symptoms

- A local `pydantic.ValidationError` from inside `to_gemini_format()` naming unexpected `$ref` / `$defs` keys.
- Instant, **$0.00** failure: it happens before serialization/send, so there is no API latency, no token spend, and nothing in Gemini's server logs.
- Reproducible on **every** Gemini call using a nested Pydantic output model. Flat single-level models worked, which made it look intermittent or model-specific.
- OpenAI and Anthropic paths for the same model worked fine — their validators already accept `$ref`/`$defs` — so the failure was Gemini-only.
- The same `parameters=` validator also rejects `allOf`, `oneOf`, `const`, and `prefixItems`. camelCase constraint keys (`maxItems`, `anyOf`, `maxLength`, `enum`, `additionalProperties`) ARE accepted because `Schema` aliases them, so the failure surface is specifically the rich constructs.

## What Didn't Work

- **Writing an in-dobby JSON-Schema → Gemini-`Schema` converter** that dereferences `$ref`/`$defs`, flattens `allOf`/`oneOf`, and rewrites `const`/`prefixItems` so the result fits `parameters=`. Deliberately rejected: it reimplements exactly the dereferencing/normalization Gemini already performs server-side, and it would need re-auditing every time Pydantic changed its JSON Schema output. High-maintenance, drift-prone, redundant.
- **Stripping `$defs` / flattening only the nested refs** — a partial version of the converter; same drift problem, plus it silently loses fidelity for `allOf`/`oneOf`/`const`/`prefixItems`.

## Solution

Switch the one field from `parameters=` to `parameters_json_schema=`, which accepts raw JSON Schema unchanged. Both branches of the method (the `self._model` path and the hand-built `{"type": "object", "properties": ...}` path) feed the same dict, so unifying onto the new field covers both.

Before:
```python
func_decl = genai_types.FunctionDeclaration(
    name=self.name, description=self.description, parameters=parameters,
)
```

After (`dobby/tools/tool.py`):
```python
func_decl = genai_types.FunctionDeclaration(
    name=self.name,
    description=self.description,
    parameters_json_schema=parameters,
)
```

Supporting change: bump `google-genai` to `>=2.4.0` (the field requires a recent SDK). No backend branching needed — `parameters_json_schema` works for both `GEMINI_API` and `VERTEX_AI`. OpenAI/Anthropic formatters were untouched.

## Why This Works

`FunctionDeclaration` exposes **two mutually-exclusive schema fields with different validators**:

- `parameters` → validated against `genai_types.Schema`, the **restrictive** OpenAPI-3.0.3 subset. No `$ref`/`$defs`, no `allOf`/`oneOf`/`const`/`prefixItems`. This is what raised the error.
- `parameters_json_schema` → **permissive**: accepts raw JSON Schema as-is, serializes to the wire as `parametersJsonSchema` with `$defs` preserved, and Gemini dereferences/normalizes server-side.

The root cause was using the restrictive field for input it was never meant to hold. Routing the same schema through the permissive field offloads dereferencing to the server — the same work the rejected converter would have duplicated locally. The two fields are mutually exclusive (server-enforced), so the fix sets **only** `parameters_json_schema`.

## Prevention

- **Rule:** for any Pydantic-derived (`model_json_schema()`) schema on Gemini, always use `parameters_json_schema`, never `parameters`. The `parameters` field is only safe for hand-authored, fully-inlined OpenAPI-subset schemas.
- **Mutual-exclusivity guard test:** assert the produced `FunctionDeclaration` sets `parameters_json_schema` and leaves `parameters` unset, so a future edit cannot reintroduce both.
- **Sanity guard test:** assert the old `parameters=` path still rejects a nested schema — if Gemini ever loosens that validator, the test flags it rather than letting the code silently drift back.
- **Detection signature:** a `pydantic.ValidationError` mentioning `$ref`/`$defs` (or `allOf`/`oneOf`/`const`/`prefixItems`) raised *before any API request* — instant, $0.00, no server log — points straight at a restrictive-field schema mismatch.
- **Coverage:** `tests/test_gemini_schema.py` has 16 regression tests (nested-in-list, nested object, deep nesting, `allOf`, `oneOf`, `const`, `prefixItems`, union, constraints, no-arg tool, non-model tool, mutual-exclusivity guard, old-`parameters=`-rejects sanity guard) plus a credential-gated live smoke test. A live e2e run returned and validated a nested `ParsedResume`.

## Verifying the SDK field before relying on it

Before coding the fix, confirm the field exists in the installed SDK (it requires `google-genai>=2.4.0`):

```bash
uv run python -c "from google.genai import types as t; print('parameters_json_schema' in t.FunctionDeclaration.model_fields)"
```

## Related Issues

- Plan: `docs/plans/2026-05-20-001-fix-gemini-tool-schema-parameters-json-schema-plan.md` (the full construct-family table and approach rationale).
- PR: TYNYBAY/dobby-sdk#4 (`fix/gemini-tool-schema-json-schema`, shipped in v0.2.14).
- No related GitHub issues found.
