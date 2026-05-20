# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
