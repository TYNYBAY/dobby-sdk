# Anthropic Provider — Defensive Tool-Call Truncation Handling

**Status:** Requirements (ready for planning)
**Date:** 2026-06-01
**Branch:** `fix/anthropic-tool-call-truncation` (off `main`; independent of `feat/anthropic-sdk` feature work)
**Related:** [TYNYBAY/TOD#339](https://github.com/TYNYBAY/TOD/issues/339) (same bug class, OpenAI adapter), [dobby-sdk#7](https://github.com/TYNYBAY/dobby-sdk/issues/7)

> Line numbers below are `main`-relative. The streaming bug is at
> `adapter.py:484`, the non-streaming `tool_use` block at `adapter.py:342-344`,
> and the non-streaming `stop_reason` at `adapter.py:350`. Grep
> `json.loads(current_tool_input_json)` if they have drifted.

## Problem

When a tool call's argument JSON is truncated by `max_tokens`, the Anthropic
provider mishandles it in two distinct ways:

- **Streaming** (`dobby/providers/anthropic/adapter.py:511`): the accumulated
  partial JSON is parsed with an unprotected `json.loads(current_tool_input_json)`.
  Truncation raises `JSONDecodeError` out of the async generator, crashing the
  whole stream. This is the loud failure (issue #339 hit the OpenAI equivalent
  3× in prod on 2026-05-14).
- **Non-streaming** (`dobby/providers/anthropic/adapter.py:369-372`): no
  client-side `json.loads` — `block.input` is an SDK-parsed dict. On a
  `max_tokens` cutoff Anthropic returns `stop_reason="max_tokens"` with a
  partial/empty `input`. No crash, but the SDK silently returns a tool call
  with wrong/empty arguments — the caller executes the tool on garbage.

The goal is **not** to repair truncated JSON or retry with a higher token
budget. The goal is to **surface a clear error** instead of crashing or lying.

## Users / Value

Consumers of `dobby-sdk` (primarily TOD's live interview agent). Today a
truncated tool call either crashes the streaming generator or silently produces
bad tool inputs. After this change, both paths emit a precise, catchable signal
naming the affected tool and carrying the raw payload.

## Requirements

### R1 — Streaming: `ToolUseErrorEvent`

Add a new event to the `StreamEvent` union in `dobby/types/stream_events.py`:

```python
class ToolUseErrorEvent(BaseModel):
    type: Literal["tool_use_error"] = "tool_use_error"
    id: str
    name: str
    raw_arguments: str
    error: str
```

At `adapter.py:511`, wrap the parse:

```python
except json.JSONDecodeError as e:
    yield ToolUseErrorEvent(
        id=current_tool_id or "",
        name=current_tool_name or "",
        raw_arguments=current_tool_input_json,
        error=str(e),
    )
    current_block_type = None
    continue  # rest of stream still delivers
```

- The broken tool MUST NOT be appended to `function_calls` (so it is absent
  from the final `StreamEndEvent.parts`).
- The stream continues delivering subsequent blocks and the final
  `StreamEndEvent`.

### R2 — Non-streaming: `ToolCallTruncatedError`

Add a typed exception (subclass of `DobbyProviderError` in
`dobby/providers/base.py`):

```python
class ToolCallTruncatedError(DobbyProviderError):
    # carries tool_name, tool_id, partial_inputs
```

In `_non_stream_chat_completion`, after building `parts`:

```python
if stop_reason == "max_tokens" and any(
    isinstance(p, ToolUsePart) for p in parts
):
    raise ToolCallTruncatedError(...)
```

- MUST NOT be added to `RETRYABLE_ERRORS` — truncation is deterministic, a
  retry with the same `max_tokens` repeats it. (Verified: `with_retries` only
  retries the explicit allowlist in `dobby/providers/_retry.py:62,86`, so a new
  `DobbyProviderError` subclass is not retried by default — no action needed
  beyond keeping it out of the allowlist.)

### R3 — Tests

Add unit tests in the Anthropic provider test suite:
1. Truncated single tool call in a stream → one `ToolUseErrorEvent`, stream
   completes.
2. Mixed valid + invalid tool calls in one stream → valid ones delivered,
   invalid one becomes `ToolUseErrorEvent`.
3. Non-streaming `stop_reason="max_tokens"` with a tool_use block → raises
   `ToolCallTruncatedError`.

## Success Criteria

- Truncated streaming tool JSON never escapes the generator as `JSONDecodeError`.
- Non-streaming truncated tool calls raise `ToolCallTruncatedError` rather than
  returning partial inputs.
- A valid tool call in the same stream as a broken one is still delivered.
- All three tests pass.

## Scope Boundaries

**Out of scope**
- JSON repair / completion of truncated arguments.
- Retry-with-higher-`max_tokens` recovery.
- OpenAI adapter (`dobby/providers/openai/adapter.py:386,494`) — issue #339's own
  target; same pattern but tracked separately.
- Gemini adapter — deferred per #339.

## Assumptions

- **Conservative non-streaming raise:** R2 raises whenever `max_tokens` +
  any `ToolUsePart` is present, even if that specific tool block happens to be
  complete and it was trailing text that got cut. Accepted as a safe default;
  revisit only if false positives appear.
- **Consumer impact:** adding `tool_use_error` to the `StreamEvent` union means
  downstream consumers (TOD) that match on `event.type` need a branch for it,
  or it falls through their dispatch silently. Flag in release notes.

## Open Questions

- Should `ToolUseErrorEvent` also fire in streaming when `stop_reason=="max_tokens"`
  but the partial JSON coincidentally parses? Considered rare; left out of scope
  unless evidence shows it happening.
