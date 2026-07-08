# Defensive Tool-Call Truncation Handling (All Providers)

**Status:** Requirements (ready for planning)
**Date:** 2026-06-01
**Branch:** `fix/tool-call-truncation` (off `main`; independent of `feat/anthropic-sdk` feature work)
**Related:** [TYNYBAY/TOD#339](https://github.com/TYNYBAY/TOD/issues/339) (origin: OpenAI adapter, prod 2026-05-14), [dobby-sdk#7](https://github.com/TYNYBAY/dobby-sdk/issues/7)

## Problem

When `max_tokens` truncates a tool call's argument payload, every provider
mishandles it — in one of two ways. The goal is **not** to repair the JSON or
retry with a higher budget. The goal is to **surface a clear, catchable error**
instead of crashing the stream or silently executing a tool on garbage.

### Two failure classes

**Class A — loud crash.** The adapter accumulates raw string arguments and
calls unprotected `json.loads(...)`. Truncation → `JSONDecodeError` escapes the
async generator (streaming) or the coroutine (non-streaming), killing the call.

**Class B — silent partial.** The provider SDK pre-parses arguments into a dict.
Truncation yields a partial/empty dict with no error. The caller executes the
tool on wrong/empty inputs. Detectable only via the truncation signal
(`stop_reason`/`finish_reason == max_tokens`) co-occurring with a tool part.

### Bug surface (`main`-relative line numbers — grep if drifted)

| Provider / API | Path | Streaming | Non-streaming | Class |
|---|---|---|---|---|
| Anthropic Messages | `anthropic/adapter.py` | `:484` `json.loads` | `:344` SDK `block.input` | A (stream) / B (non-stream) |
| OpenAI Responses | `openai/adapter.py` | `:494` `json.loads` | `:386` `json.loads` | A |
| OpenAI ChatCompletions | `openai/completions.py` | `:311` `json.loads` | `:205` `json.loads` | A |
| Gemini | `gemini/adapter.py` | `:413` SDK `function_call.args` | `:301` SDK `function_call.args` | B |

**Secondary gap:** `openai/adapter.py` (Responses) never maps the incomplete /
`length` finish signal to `stop_reason="max_tokens"` (stays `end_turn` /
`tool_use`). Its arg path is Class A so it still crashes loud, but the
truncation signal is invisible to callers — fix alongside.

## Users / Value

Consumers of `dobby-sdk` (primarily TOD's live interview agent). Today a
truncated tool call either crashes the stream (Class A) or silently produces
bad tool inputs (Class B), inconsistently per provider. After this change every
provider, both paths, emits a precise catchable signal naming the affected tool
and carrying the raw payload.

## Requirements

### R1 — Streaming: `ToolUseErrorEvent`

Add to the `StreamEvent` union in `dobby/types/stream_events.py`:

```python
class ToolUseErrorEvent(BaseModel):
    type: Literal["tool_use_error"] = "tool_use_error"
    id: str
    name: str
    raw_arguments: str   # raw string for Class A; repr(partial dict) for Class B
    error: str
```

Per provider, at the streaming tool-finalize point:

- **Class A** (Anthropic `:484`, OpenAI Responses `:494`, OpenAI Completions
  `:311`): wrap `json.loads` in `try/except json.JSONDecodeError`; on failure
  `yield ToolUseErrorEvent(...)`, skip appending the broken tool to the
  accumulator, `continue`.
- **Class B** (Gemini `:413`): when `finish_reason == "MAX_TOKENS"` and a
  `function_call` part is present, `yield ToolUseErrorEvent` instead of the
  partial `ToolUseEvent`.

Invariants (all providers):
- Broken tool MUST NOT enter the accumulator → absent from final
  `StreamEndEvent.parts`.
- Stream continues delivering subsequent blocks and the final `StreamEndEvent`.

### R2 — Non-streaming: `ToolCallTruncatedError`

Add a typed exception (subclass of `DobbyProviderError` in
`dobby/providers/base.py`) carrying `tool_name`, `tool_id`, `partial_inputs`,
`raw_arguments`.

- **Class A** (OpenAI Responses `:386`, OpenAI Completions `:205`): wrap
  `json.loads`; on `JSONDecodeError` raise `ToolCallTruncatedError`.
- **Class B** (Anthropic `:344`, Gemini `:301`): after building `parts`, if
  `stop_reason == "max_tokens"` and any `ToolUsePart` is present, raise
  `ToolCallTruncatedError`.

MUST NOT be added to `RETRYABLE_ERRORS` — truncation is deterministic; a retry
with the same `max_tokens` repeats it. (Verified: `with_retries` retries only
the explicit allowlist in `_retry.py:62,86`, so a new `DobbyProviderError`
subclass is not retried by default — keep it out of the allowlist.)

### R3 — OpenAI Responses `stop_reason` coverage

In `openai/adapter.py`, map the Responses incomplete / `max_output_tokens`
signal to `stop_reason="max_tokens"` so the truncation signal is observable
(parity with the other three paths). Required for R2 Class-B-style detection to
be feasible there if its arg path is ever changed; do it regardless for
telemetry parity.

### R4 — Shared helper

Four adapters repeat the same wrap/detect logic. Extract a small helper (e.g.
`dobby/providers/_truncation.py`) so the streaming and non-streaming decision is
written once and imported, not copy-pasted per provider.

### R5 — Tests

Per provider, in each provider's test suite:
1. Truncated tool call in a stream → one `ToolUseErrorEvent`, stream completes.
2. Mixed valid + invalid tool calls in one stream → valid delivered, invalid
   becomes `ToolUseErrorEvent`.
3. Non-streaming truncated tool call → raises `ToolCallTruncatedError`.

Class B providers (Anthropic non-stream, Gemini) drive cases via
`stop_reason/finish_reason == max_tokens` + partial dict; Class A via malformed
argument strings.

## Success Criteria

- No truncated tool JSON escapes any provider as a raw `JSONDecodeError`.
- No provider silently returns partial tool inputs on `max_tokens`.
- A valid tool call in the same stream as a broken one is still delivered.
- `ToolUseErrorEvent` / `ToolCallTruncatedError` behave identically across
  Anthropic, OpenAI (both APIs), and Gemini.
- All R5 tests pass.

## Scope Boundaries

**Out of scope**
- JSON repair / completion of truncated arguments.
- Retry-with-higher-`max_tokens` recovery.
- Any non-tool-call truncation (truncated plain text is already valid).

## Assumptions

- **Conservative non-streaming raise (Class B):** R2 raises whenever
  `max_tokens` + any `ToolUsePart` is present, even if that specific tool block
  is complete and it was trailing text that got cut. Safe default; revisit only
  on false positives.
- **Consumer impact:** adding `tool_use_error` to the `StreamEvent` union means
  downstream consumers (TOD) matching on `event.type` need a branch for it, or
  it falls through their dispatch silently. Flag in release notes; tracked
  downstream by TOD#341 (`agent_error` WS event) / #342 (Brief banner).

## Open Questions

- Should `ToolUseErrorEvent` also fire (Class A) when `stop_reason == max_tokens`
  but the partial JSON coincidentally parses? Rare; out of scope unless observed.
- OpenAI Responses: confirm the exact field carrying the incomplete signal
  (`response.status == "incomplete"` + `incomplete_details.reason`) during R3.
