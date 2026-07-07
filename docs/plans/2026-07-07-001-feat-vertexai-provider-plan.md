---
title: "Add Vertex AI Provider (OpenAI-Compatible MaaS Endpoint)"
type: feat
status: active
created: 2026-07-07
deepened: 2026-07-07
origin: docs/brainstorms/vertexai-provider-requirements.md
---

# Add Vertex AI Provider (OpenAI-Compatible MaaS Endpoint)

## Problem Frame

Dobby has provider parity with OpenAI, Anthropic (incl. Azure AI Foundry), and Gemini (incl. Gemini-via-Vertex). It has no provider for Google Cloud Vertex AI's **Model Garden** — the catalog of non-Gemini foundation models (Llama, Mistral, Claude-on-Vertex, self-deployed endpoints) hosted on Vertex infrastructure.

Per the origin document, this is general capability-parity work, scoped to a single mechanism: Vertex AI's OpenAI-compatible Model-as-a-Service (MaaS) chat-completions endpoint (`POST https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions`). This covers Llama (GA on Vertex MaaS) and any self-deployed Model Garden endpoint whose serving container speaks OpenAI's chat-completions wire format.

Claude-on-Vertex, Mistral-on-Vertex, and raw non-OpenAI-compatible custom endpoints are explicitly out of scope (see origin document's Decisions #2 and Non-Goals) — each requires a different SDK/wire format with no shared abstraction, and building one would multiply auth paths and test surface for low marginal value in v1.

(see origin: docs/brainstorms/vertexai-provider-requirements.md)

---

## Key Finding That Changes Scope From the Origin Document

The origin document flagged "whether to reuse dobby's existing OpenAI converters" as an open question for planning. Investigation resolves it: **dobby's existing `OpenAIProvider` (dobby/providers/openai/adapter.py) talks OpenAI's Responses API** (`client.responses.create()`, `ResponseInputParam`, `output_text`/`function_call` output items) — **not** the Chat Completions API. Vertex's OpenAI-compatible MaaS endpoint speaks the older **Chat Completions** wire format (`messages` array with `role`/`content`, `tool_calls`, `choices[].delta`, `finish_reason`).

These are structurally different payloads. `to_openai_messages()` and the Responses-API streaming/parsing logic in `dobby/providers/openai/adapter.py` are **not reusable as-is**. This plan writes new Chat-Completions-shaped converters and streaming/parsing logic in the new `dobby/providers/vertexai/` package. What *is* reusable: the `AsyncOpenAI` client class itself (same SDK, different sub-resource — `client.chat.completions.create()` instead of `client.responses.create()`), and the shape of `_translate_error` (same `openai.*` exception types are raised regardless of which endpoint/sub-resource is hit, so the mapping logic is mirrored, not imported, per the codebase's existing per-provider duplication convention — no provider adapter currently imports another's private methods).

This also affects **tool schemas**, not just messages: `dobby/tools/tool.py`'s `Tool.to_openai_format()` returns a **flat** `FunctionToolParam` (`type`/`name`/`description`/`parameters` all top-level) because it targets the Responses API. Chat Completions requires a **nested** shape (`{"type": "function", "function": {"name", "description", "parameters"}}`). This plan reuses `to_openai_format()`'s schema-construction logic (it already handles both the Pydantic-model and parameter-list code paths) and re-nests its output, rather than writing a third independent schema builder (see U2 and U6).

---

## Requirements Traceability

From the origin document:

- **Standalone provider** in `dobby/providers/vertexai/`, implementing the `Provider[ClientT]` contract (dobby/providers/base.py) — same `name`/`model`/`client` properties and `chat()` overload shape as every other provider.
- **V1 scope: OpenAI-compatible MaaS endpoint only.** No Claude-on-Vertex, no Mistral, no raw custom-container `rawPredict`.
- **Auth: ADC by default, plus optional pre-built credentials object.** No custom async token-provider callable exposed to the *caller* in v1 (the provider uses one internally against the OpenAI SDK's own callable-`api_key` hook — see Key Technical Decision #2 — but does not expose a caller-supplied token-provider parameter, matching the origin document's scope).
- **Feature parity target: match `OpenAIProvider`** — streaming, tool/function calling, and error-hierarchy mapping at the same level of support.
- **Token freshness**: the bearer token is not static like an API key; it must be kept valid across calls (refresh before expiry).
- **`provider.name`** must disambiguate from `GeminiProvider`'s existing `"gemini-vertexai"` name (this plan uses `"vertexai"`).
- **Docs**: new `docs/providers/vertexai.md` following the existing per-provider pattern, and correct the stale `docs/providers/index.md` (still lists Anthropic as "🚧 Planned" though it already shipped) while adding the Vertex AI row — confirmed in-scope by user during plan-time synthesis.

---

## Scope Boundaries

### Deferred for later (carried from origin document)
- Claude-on-Vertex support (candidate for a future `AnthropicProvider` extension mirroring the Azure AI Foundry pattern in `dobby/providers/anthropic/adapter.py` — a separate follow-up, not this provider).
- Mistral-on-Vertex support (requires the dedicated `mistralai`/`MistralGCP` SDK).
- Raw non-OpenAI-compatible custom Vertex endpoints (`rawPredict`/`streamRawPredict` with arbitrary, non-chat-completions payloads).
- A custom async token-provider callable *exposed to callers* (the provider's internal auth already uses an async-callable mechanism against the OpenAI SDK — see Decision #2 — but a caller-supplied override is out of scope; ADC + explicit credentials object covers v1).

### Deferred to Follow-Up Work (plan-local)
- The docs/providers/index.md fix was pulled into this plan's scope (see U5) rather than deferred, per user confirmation at plan-time.
- Fixing the same silent-tools-drop gap for the *existing* `"azure-anthropic"` and `"gemini-vertexai"` provider names in `dobby/executor.py`'s `get_tools_schema()` `match` statement (found during plan review — see System-Wide Impact and U6). This plan only ensures `"vertexai"` doesn't join that list; the pre-existing gap for the other two is a separate, unrelated fix.

---

## Key Technical Decisions

1. **Reuse `AsyncOpenAI` as the underlying client, target `Provider[AsyncOpenAI]`.** `VertexAIProvider` constructs a real `openai.AsyncOpenAI(base_url=...)` pointed at the Vertex MaaS endpoint, matching Google's own documented recommendation to use the stock OpenAI SDK against this endpoint. The generic type parameter mirrors `OpenAIProvider(Provider[AsyncOpenAI | AsyncAzureOpenAI])`.

2. **Auth uses `AsyncOpenAI`'s native async-callable `api_key` parameter — no header injection, no client subclassing.** Verified directly against the installed SDK (`openai==2.15.0`, matching `pyproject.toml`'s `openai>=2.14.0` floor): `openai/_client.py`'s constructor accepts `api_key: str | Callable[[], Awaitable[str]] | None`; when callable, `AsyncAPIClient._prepare_options()` (invoked once per request, including retries, for both streaming and non-streaming calls) awaits it via `_refresh_api_key()` before every call and rebuilds the `Authorization` header from the result. This is the exact mechanism `AnthropicProvider` already uses for Azure AD auth (`azure_ad_token_provider: Callable[[], str | Awaitable[str]]`, `anthropic/lib/foundry.py:24`) — just exposed through the `api_key` parameter name instead of a dedicated one. Construct the client as `AsyncOpenAI(base_url=..., api_key=self._bearer_token)`, passing the bound async method itself (not calling it).

   `self._bearer_token()` implementation: under an instance-level `asyncio.Lock` (see concurrency note below), check `self._credentials.valid`; if not valid, refresh via `await asyncio.to_thread(self._credentials.refresh, google.auth.transport.requests.Request())` (refresh is a blocking call in `google-auth` — no public async refresh API exists, confirmed via `google-auth-library-python` issue tracker, where async credential modules are explicitly marked internal/experimental); return `self._credentials.token`.

   **Concurrency note:** `credentials.refresh()` mutates a shared `Credentials` object in place across a real OS thread (via `asyncio.to_thread`). Without a guard, two concurrent requests can both observe `credentials.valid is False` (the SDK invokes `_bearer_token()` independently per request via `_prepare_options`) and both trigger a redundant/racy refresh. `_bearer_token()` must hold an instance-level `asyncio.Lock` around the check-then-refresh section (re-checking `valid` after acquiring the lock, in case another call already refreshed while this one was waiting) to make refresh effectively idempotent under concurrency.

   **`.client` property is fully self-sufficient — no caveat needed.** Because the callable is invoked by the SDK's own `_prepare_options` on *every* request regardless of call path, direct use of `provider.client.chat.completions.create(...)` (the use case `dobby/providers/base.py`'s `client` docstring promises — "for direct access if needed") automatically gets a fresh token through the same mechanism `chat()` uses internally. No placeholder key, no construction-time eager refresh, and no documented deviation from sibling providers' `.client` behavior are needed. (An earlier version of this plan proposed per-call `extra_headers` injection plus an eager-refresh-at-construction workaround for this exact gap; this native-callable design removes the need for both — this is a real simplification found during plan review, not just a style preference.)

   **Secret-handling note:** the callable's returned token flows into the same `self.api_key` attribute every other provider's static API key already occupies — no *new* persistence pattern is introduced. Still, per-call kwargs, `self._client`, and `self._credentials` must never be included in debug/log statements (there is a precedent for ad-hoc kwargs logging in sibling adapters, e.g. `dobby/providers/openai/adapter.py`'s commented-out `# logger.debug(f"kwargs: {kwargs}")` — do not reintroduce that pattern here in a form that would log the client or its resolved auth state).

   **Least-privilege note:** `google.auth.default()` is called with an optional `scopes` constructor parameter (default `None`, meaning google-auth's own default resolution — typically the broad `cloud-platform` scope for most ADC sources). Document in `docs/providers/vertexai.md` that operators wanting least-privilege should pass a narrower scope list where the ADC source (e.g. a dedicated service account) supports it, since a leaked broad-scope token has a larger blast radius than a Vertex-only one.

3. **New Chat-Completions-shaped converters, not extensions of `to_openai_messages`.** New `dobby/providers/vertexai/converters.py` producing/parsing the `messages`/`choices`/`delta`/`tool_calls` shape. Structurally similar in spirit to `dobby/providers/anthropic/converters.py` and `dobby/providers/openai/converters.py` (a `to_vertexai_messages()` function plus a content-part conversion helper), but a new implementation — not a wrapper around the existing OpenAI (Responses API) message converters. Tool *schema* conversion (distinct from message conversion) reuses `Tool.to_openai_format()`'s schema-building logic and re-nests the result (see Key Finding above and U2).

4. **Error translation mirrors `OpenAIProvider._translate_error`.** Since `AsyncOpenAI` raises the same `openai.RateLimitError` / `openai.APITimeoutError` / `openai.APIConnectionError` / `openai.InternalServerError` / `openai.APIStatusError` hierarchy regardless of which endpoint is called, `VertexAIProvider._translate_error` copies the same `match`/`case` structure as `dobby/providers/openai/adapter.py`'s version, substituting `provider=self.name` (`"vertexai"`).

5. **Model id is a pass-through string, no hardcoded allow-list.** Matches every other dobby provider's philosophy (trust the caller) — `model` is a publisher-qualified string (e.g. `"meta/llama-3.1-405b-instruct-maas"`) forwarded verbatim.

6. **`with_retries` decorator reused as-is** (`dobby/providers/_retry.py`) on both chat-completion methods, identical to every other provider.

7. **Auth handling mirrors `AnthropicProvider`'s Azure AD pattern, not diverges from it.** Both providers authenticate via an async callable passed to the underlying SDK's client constructor (`azure_ad_token_provider` for Anthropic, `api_key` accepting a callable for OpenAI) — the parameter name differs but the mechanism (SDK-native, per-request token refresh, no header hacking) is the same. This was corrected during plan review from an earlier draft that incorrectly claimed `AsyncOpenAI` had no equivalent hook and built a header-injection workaround instead.

---

## Output Structure

```
dobby/providers/vertexai/
├── __init__.py       # exports VertexAIProvider, to_vertexai_messages
├── adapter.py         # VertexAIProvider class: auth/token-refresh, chat(), streaming + non-streaming completions, error translation
└── converters.py      # to_vertexai_messages(), content-part conversion helpers, to_vertexai_tool() schema re-nesting (Chat Completions wire shape)

tests/
└── test_vertexai_provider.py   # auth/token-refresh, converters, error translation, streaming + non-streaming chat()

docs/providers/
└── vertexai.md        # new, follows docs/providers/openai.md pattern
```

---

## Implementation Units

### U1. Package scaffold, constructor, auth, and token-refresh

**Goal:** Stand up `VertexAIProvider` with its `Provider[AsyncOpenAI]` contract properties and the async-callable-based auth/token-refresh helper that later units depend on.

**Requirements:** Standalone provider contract; ADC + optional pre-built credentials; token freshness; `provider.name` disambiguation.

**Dependencies:** None.

**Files:**
- `dobby/providers/vertexai/__init__.py` (new)
- `dobby/providers/vertexai/adapter.py` (new)
- `tests/test_vertexai_provider.py` (new)
- `pyproject.toml` (add `google-auth[requests]` to `dependencies` — the `[requests]` extra is required because `google.auth.transport.requests.Request` imports `requests` at module level, which bare `google-auth` does not declare as a base dependency; confirmed via `google_auth`'s own package metadata. It happens to work today only because `requests` is already present transitively via other unrelated dependencies — declare it explicitly rather than relying on that coincidence.)

**Approach:**
- Constructor accepts: `model: str`, `project: str`, `location: str = "us-central1"`, `credentials: google.auth.credentials.Credentials | None = None`, `scopes: Sequence[str] | None = None` (forwarded to `google.auth.default()` only when `credentials` is not supplied — see Key Technical Decision #2's least-privilege note), `max_retries: int = 3`.
- Store constructor inputs as instance attributes, including `self.max_retries = max_retries` (required for `@with_retries`, which reads `getattr(self, "max_retries", 0)` — confirmed in `dobby/providers/_retry.py`).
- If `credentials` is not supplied, resolve via `google.auth.default(scopes=scopes)` at construction time (mirrors `GeminiProvider`'s eager-client-construction style in `dobby/providers/gemini/adapter.py`). No eager token refresh is needed at construction — `google.auth.default()` does not itself perform a network call, and the first real token fetch happens naturally on the first request via the callable below.
- `self._refresh_lock = asyncio.Lock()` — guards the check-then-refresh section against concurrent requests racing a refresh (see Key Technical Decision #2).
- Private async method `async def _bearer_token(self) -> str`: under `self._refresh_lock`, re-check `self._credentials.valid` (another call may have refreshed while this one waited); if still not valid, `await asyncio.to_thread(self._credentials.refresh, google.auth.transport.requests.Request())`; return `self._credentials.token`.
- Construct `self._client = AsyncOpenAI(base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi", api_key=self._bearer_token)` — passing the bound method itself as the callable `api_key`, per Key Technical Decision #2. No placeholder key, no `default_headers` needed.
- `name` property returns `"vertexai"` (distinct from `GeminiProvider`'s `"gemini-vertexai"`).
- `model` property returns the instance model string. `client` property returns the underlying `AsyncOpenAI` instance (fully self-sufficient — see Decision #2).

**Patterns to follow:** `dobby/providers/anthropic/adapter.py`'s constructor for how it wires `azure_ad_token_provider` straight into `AsyncAnthropicFoundry(azure_ad_token_provider=...)` — this plan's `api_key=self._bearer_token` is the direct analog for the OpenAI SDK; `AnthropicProvider.name`'s `getattr(self, "_is_azure", ...)` fallback pattern for defensive `name`/`model` properties when a test constructs the class via `__new__`.

**Test scenarios:**
- Constructing with explicit `credentials` object skips `google.auth.default()` (mock `google.auth.default`, assert not called).
- Constructing without `credentials` calls `google.auth.default(scopes=scopes)` with the constructor's `scopes` value (including the `None` default) and stores the result.
- `name` returns `"vertexai"`.
- `model` returns the constructor-supplied model string.
- `max_retries` is stored as `self.max_retries` on the instance (verifies `@with_retries` can retrieve it).
- The `AsyncOpenAI` client is constructed with `api_key` set to the provider's own `_bearer_token` bound method (not a string), verifying the callable-auth wiring.
- `_bearer_token()` returns the existing token without refreshing when `credentials.valid` is `True`.
- `_bearer_token()` refreshes (via the `asyncio.to_thread` path) when `credentials.valid` is `False`, then returns the refreshed token.
- `_bearer_token()` refresh failure (mocked `credentials.refresh` raises) propagates the exception rather than silently swallowing it.
- Two concurrent `_bearer_token()` calls when `credentials.valid` starts `False`: `credentials.refresh` is invoked once, not twice (verifies the `asyncio.Lock` guard — e.g. run both via `asyncio.gather`, mock `refresh` to flip `valid` to `True` as a side effect, assert call count).

**Verification:** Provider constructs successfully with both auth paths; `_bearer_token()` returns a valid string in both the fresh-token and needs-refresh cases; the constructed `AsyncOpenAI` client's `api_key` is the callable, not a static string; test suite passes.

---

### U2. Chat Completions message and tool-schema converters

**Goal:** Translate dobby's provider-agnostic `MessagePart`/`ResponsePart` types to/from the Chat Completions wire shape Vertex's MaaS endpoint expects, and produce Chat-Completions-shaped (nested) tool schemas.

**Requirements:** Feature parity with `OpenAIProvider` on tool/function calling; new converters per the Key Finding above.

**Dependencies:** None (pure data transformation, independent of U1's auth work).

**Files:**
- `dobby/providers/vertexai/converters.py` (new)
- `tests/test_vertexai_provider.py` (extend)

**Approach:**
- `to_vertexai_messages(messages: Iterable[MessagePart]) -> list[dict[str, Any]]`: produces Chat Completions `messages` array — `{"role": "system"/"user"/"assistant"/"tool", "content": ...}`, with assistant tool calls under `"tool_calls": [{"id", "type": "function", "function": {"name", "arguments"}}]` and tool results as `{"role": "tool", "tool_call_id": ..., "content": ...}` (standard Chat Completions tool-result shape — distinct from Anthropic's `tool_result` content block and OpenAI Responses API's `function_call_output` item type already in the codebase).
- Reasoning parts (`ReasoningPart`) are dropped when converting to Vertex messages — Llama/Model Garden MaaS models do not have a reasoning/thinking channel analogous to Claude's or OpenAI's o-series (mirrors how `to_openai_messages` already drops `ReasoningPart` for the non-reasoning path).
- A small content-part helper (analogous to `content_part_to_openai` in `dobby/providers/openai/converters.py`) for text/image content within a message.
- `to_vertexai_tool(tool: Tool) -> dict[str, Any]`: calls `tool.to_openai_format()` (reusing its existing schema-construction logic — the Pydantic-model and parameter-list code paths in `dobby/tools/tool.py` don't need a third reimplementation) and re-nests the flat result into Chat Completions' nested shape: `{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}`. This function is what U6 calls from the executor's tool-schema dispatch.

**Patterns to follow:** `dobby/providers/openai/converters.py`'s `content_part_to_openai` structural shape (per-part-type `match`/`case` dispatch), even though the target dict shape differs (Chat Completions `content` is a string or a list of `{"type": "text"/"image_url", ...}` parts, not Responses API's `input_text`/`output_text`/`image_url` items); `dobby/tools/tool.py`'s `to_openai_format()`/`to_anthropic_format()` for how tool schema construction already branches on `self._model` vs. `self._parameters` — `to_vertexai_tool()` should call the existing method, not duplicate that branching.

**Test scenarios:**
- Single user text message converts to `{"role": "user", "content": "..."}`.
- Assistant message with a tool call converts to an assistant message with `tool_calls` populated and no bare `content` tool-call leakage.
- `ToolResultPart` converts to a `{"role": "tool", "tool_call_id": ..., "content": ...}` message, including the `is_error` case (verify how errors surface in `content` given Chat Completions has no dedicated error flag — decide and test a concrete convention, e.g. prefixing content, consistent with `to_anthropic_messages`'s error-prefix precedent).
- `ReasoningPart` in an assistant message is dropped from the converted output (no key present, no crash).
- Multi-turn conversation (user → assistant tool call → tool result → assistant text) converts to a complete, correctly ordered `messages` list.
- System prompt (when passed by the adapter) is prepended as a `{"role": "system", ...}` message.
- `to_vertexai_tool()` on a `Tool` built from a Pydantic model produces `{"type": "function", "function": {"name", "description", "parameters"}}` with `parameters` equal to what `to_openai_format()["parameters"]` would produce (i.e., verifies re-nesting, not re-derivation, of the schema).
- `to_vertexai_tool()` on a `Tool` built from a parameter list (non-Pydantic-model path) produces the same nested shape.

**Verification:** Converter output matches the Chat Completions message and tool schema for every dobby `MessagePart`/`ResponsePart`/`Tool` variant currently supported by sibling providers; round-trip through a real (mocked) `chat.completions.create()` call in U3 doesn't error on shape.

---

### U3. Non-streaming chat completion

**Goal:** Implement `chat(stream=False)` end-to-end: convert messages, call `client.chat.completions.create()` (auth handled transparently by U1's callable `api_key`), parse the response into a `StreamEndEvent`, translate errors.

**Requirements:** `chat()` contract parity (`stream: Literal[False]` overload); error-hierarchy mapping; tool/function-call parsing.

**Dependencies:** U1, U2.

**Files:**
- `dobby/providers/vertexai/adapter.py` (extend)
- `tests/test_vertexai_provider.py` (extend)

**Approach:**
- `chat()` public method with the same `@overload` pair (`stream: Literal[False] = False` / `stream: Literal[True]`) as every other provider, delegating to `_non_stream_chat_completion` / `_stream_chat_completion`.
- `_non_stream_chat_completion`, decorated with `@with_retries`, builds kwargs (`model`, `messages`, `temperature`, `tools` when provided — converted via U2's `to_vertexai_tool()`) and calls `await self._client.chat.completions.create(**kwargs)`. No manual auth handling at this call site — the client's callable `api_key` (U1) refreshes transparently via the SDK's own `_prepare_options` hook.
- Parse `response.choices[0].message`: `content` → `TextPart`; `tool_calls` → one `ToolUsePart` per entry (`json.loads(tool_call.function.arguments)` for `inputs`, mirroring `dobby/providers/openai/adapter.py`'s `json.loads(output.arguments)` pattern) with `stop_reason` set to `"tool_use"` when any tool calls are present.
- Map `response.choices[0].finish_reason` (`"stop"`, `"length"`, `"tool_calls"`, `"content_filter"`) onto dobby's `StopReason`, following the same "unknown → safe fallback + debug log" pattern as `_map_stop_reason` in `dobby/providers/anthropic/adapter.py` (a Chat-Completions-specific equivalent, since the value vocabulary differs from both Anthropic's and Gemini's).
- Parse `response.usage` (`prompt_tokens`/`completion_tokens`/`total_tokens`) into dobby's `Usage`.
- `_translate_error` mirrors `OpenAIProvider._translate_error` (see Key Technical Decision #4).

**Patterns to follow:** `dobby/providers/openai/adapter.py`'s `_non_stream_chat_completion` and `_translate_error` for overall shape; note the field names differ from the Responses API (`choices[0].message.tool_calls` vs. Responses API's top-level `function_call` output items) — do not copy field access verbatim.

**Test scenarios:**
- Plain text response → single `TextPart`, `stop_reason="end_turn"`.
- Response with `tool_calls` → one `ToolUsePart` per call, correctly parsed `inputs` from JSON arguments, `stop_reason="tool_use"`.
- `finish_reason="length"` maps to `stop_reason="max_tokens"`.
- Unknown/future `finish_reason` value falls back to `"end_turn"` and logs at debug level (does not raise).
- `usage` fields populate dobby's `Usage` correctly.
- Each `openai.RateLimitError` / `APITimeoutError` / `APIConnectionError` / `InternalServerError` / `APIStatusError` raised by the mocked client translates to the matching `Dobby*Error`, parameterized the same way `tests/test_anthropic_provider.py`'s `TestAnthropicErrorTranslation` parameterizes over Anthropic's error classes.
- `chat.completions.create()` is called without any `extra_headers`/manual auth kwargs — auth flows solely through the client's callable `api_key` (verifies U1/U3 don't reintroduce the header-injection pattern this plan explicitly moved away from).
- Model override: instance model used by default; a per-call `model` kwarg (if the `chat()` signature exposes one, matching sibling providers) overrides it.

**Verification:** Non-streaming chat round-trips through a fully mocked `AsyncOpenAI` client producing dobby-typed output identical in shape to what `OpenAIProvider`/`AnthropicProvider` produce for equivalent inputs.

---

### U4. Streaming chat completion

**Goal:** Implement `chat(stream=True)`, parsing Chat Completions SSE delta chunks into dobby's discriminated `StreamEvent` sequence.

**Requirements:** `chat()` contract parity (`stream: Literal[True]` overload); streaming tool-call accumulation; error-hierarchy mapping mid-stream.

**Dependencies:** U1, U2, U3 (shares kwargs-building and error-translation logic).

**Files:**
- `dobby/providers/vertexai/adapter.py` (extend)
- `tests/test_vertexai_provider.py` (extend)

**Approach:**
- `_stream_chat_completion`, decorated with `@with_retries`, calls `client.chat.completions.create(stream=True, stream_options={"include_usage": True}, ...)` — no manual auth handling, same as U3.
- Chat Completions streaming chunks are `choices[0].delta` fragments: `delta.content` (text), `delta.tool_calls` (each chunk carries a partial `index`/`id`/`function.name`/`function.arguments` fragment that must be accumulated by index across chunks — a materially different accumulation shape than both Anthropic's `content_block_delta`/`input_json_delta` state machine and OpenAI Responses API's `response.output_item.done` per-item events already in the codebase).
- Emit `StreamStartEvent` on the first chunk, `TextDeltaEvent` per content delta, and accumulate tool-call fragments keyed by `index` until each is complete — yield `ToolUseEvent` once each accumulated call's arguments JSON is fully assembled.
- **Bound the accumulator** (found during security review): cap the number of distinct `index` values and the total accumulated `arguments` length per stream (e.g. a small constant multiple of the largest realistic tool-call payload dobby already handles elsewhere, refine the exact bound during implementation). If either bound is exceeded, stop accumulating and yield a `StreamErrorEvent` rather than growing unboundedly — the target here includes self-deployed, potentially untrusted third-party containers (per the origin document's own scope), and an unbounded per-`index` accumulator keyed by backend-supplied data is a resource-exhaustion vector worth closing before shipping, not after.
- On the terminal chunk (`finish_reason` present), emit `StreamEndEvent` with accumulated `parts`, mapped `stop_reason`, and `usage`.

**Technical design:**
```
for chunk in stream:
    if chunk.usage is not None:
        capture usage  # independent of choices length — some OpenAI-compatible
                        # servers (vLLM-backed) don't guarantee an empty-choices
                        # dedicated usage chunk the way native OpenAI does
    if not chunk.choices: continue

    delta = chunk.choices[0].delta
    if delta.content: accumulate text, yield TextDeltaEvent
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            if too_many_distinct_indices or accumulated_length_over_bound:
                yield StreamErrorEvent(...); return
            acc = tool_call_accumulator[tc_delta.index]  # index is the ONLY safe
                                                          # correlation key; id may
                                                          # not appear on every chunk
            if tc_delta.id: acc.id = tc_delta.id                 # idempotent
            if tc_delta.function.name: acc.name = tc_delta.function.name   # set-if-present,
            acc.arguments += tc_delta.function.arguments or ""            # not "only first chunk"
    if chunk.choices[0].finish_reason: capture stop reason
# after loop: finalize accumulated tool calls into ToolUsePart list, yield StreamEndEvent
```
This illustrates the intended accumulation approach and is directional guidance for review, not implementation specification — the implementer should treat it as context, not code to reproduce.

**Third-party compatibility note:** the target is Vertex's OpenAI-compatible endpoint (Llama and self-deployed containers, often vLLM-backed), not native OpenAI. Confirmed via `openai-python`'s own streaming types plus vLLM issue trackers: third-party servers have shipped bugs where a tool call's `id`/`name` are *not* guaranteed to arrive only in the first chunk for that `index`, and `usage` is not guaranteed to arrive in a dedicated empty-`choices` chunk the way native OpenAI does. The accumulation logic above treats `id`/`name`/`type` merges as idempotent ("set if present, on any chunk") rather than "only check the first chunk," and checks `chunk.usage is not None` independently of `choices` length — do not assume native-OpenAI streaming conventions hold exactly on this target.

**Patterns to follow:** `dobby/providers/anthropic/adapter.py`'s `_stream_chat_completion` for the overall manual-iteration-with-try/except-per-`anext()` error-translation pattern (so mid-stream transport errors route through the same `_translate_error`, not just the initial request); note the per-chunk delta *shape* must be freshly designed for Chat Completions' index-based tool-call fragments, since neither existing provider's accumulation logic matches this shape.

**Test scenarios:**
- Plain text stream: sequence of content deltas accumulates into a single final `TextPart`, correct `TextDeltaEvent` sequence yielded in order.
- Streamed tool call: `tool_calls` deltas arriving across multiple chunks (fragmented `arguments` JSON) accumulate correctly by `index` into one complete `ToolUsePart`.
- Two parallel tool calls in one stream (two distinct `index` values) accumulate independently without cross-contamination.
- `finish_reason="tool_calls"` sets `stop_reason="tool_use"` on the final `StreamEndEvent`.
- `usage` present in a dedicated final chunk with empty `choices` (native-OpenAI convention) populates dobby's `Usage` on `StreamEndEvent`.
- `usage` present on a chunk that also has non-empty `choices` (a documented third-party/vLLM-backed deviation from native-OpenAI convention) is still captured correctly — regression test for the idempotent `usage is not None` check.
- A tool call's `id`/`name` arriving on a later chunk instead of the first chunk for that `index` (third-party deviation) still accumulates into a correct, complete `ToolUsePart` — regression test for the idempotent set-if-present merge.
- A malicious/misbehaving stream sending more distinct `index` values (or larger cumulative `arguments`) than the configured bound yields a `StreamErrorEvent` and stops accumulating, rather than growing unboundedly — regression test for the accumulator bound.
- A mid-stream exception from the mocked async iterator translates through `_translate_error` (not an unhandled exception).
- `StreamStartEvent` is yielded exactly once, on the first chunk.

**Verification:** Streaming chat over a mocked async-iterable of Chat Completions chunks yields the same event-type sequence shape (`StreamStartEvent` → deltas/tool-use → `StreamEndEvent`) that `tests/test_openai_provider.py` and `tests/test_anthropic_provider.py` verify for their respective providers, with the accumulator bound holding under an adversarial input.

---

### U5. Package wiring and documentation

**Goal:** Make `VertexAIProvider` importable alongside the other four providers, and document it.

**Requirements:** Exports pattern parity; documentation per success criteria.

**Dependencies:** U1–U4.

**Files:**
- `dobby/providers/vertexai/__init__.py` (finalize exports)
- `dobby/providers/__init__.py` (modify — add `VertexAIProvider`/`to_vertexai_messages` export block)
- `docs/providers/vertexai.md` (new)
- `docs/providers/index.md` (modify — add Vertex AI row, correct stale Anthropic "🚧 Planned" status to reflect it has shipped)

**Approach:**
- `dobby/providers/vertexai/__init__.py` re-exports `VertexAIProvider` and `to_vertexai_messages`, matching the exact re-export style already used in `dobby/providers/gemini/__init__.py` / `dobby/providers/openai/__init__.py`.
- `dobby/providers/__init__.py` gets an alphabetically-placed import block matching the existing `from .anthropic import (...)` / `from .gemini import (...)` / `from .openai import (...)` blocks.
- New `docs/providers/vertexai.md` follows `docs/providers/openai.md`'s structure: Initialization (including the optional `scopes` param and its least-privilege trade-off — see Key Technical Decision #2), Chat Methods (non-streaming/streaming examples), Parameters table, Stream Events table, Message Conversion section — using the actual working import path (`from dobby.providers import VertexAIProvider`), since `docs/providers/openai.md`'s `from dobby import OpenAIProvider` example does not match `dobby/__init__.py`'s actual exports (a pre-existing inconsistency in the docs, not introduced or fixed by this plan — out of scope).
- `docs/providers/index.md` table gets a new `| Vertex AI | ✅ Stable | Chat Completions API (OpenAI-compatible) |` row, and the existing `| Anthropic | 🚧 Planned | Messages API |` row corrected to `✅ Stable` (it shipped per the `feat(anthropic)` and `fix(providers)` commits already on this branch).

**Patterns to follow:** `docs/providers/openai.md` structure; `dobby/providers/__init__.py`'s existing import-block ordering.

**Test scenarios:**
- `from dobby.providers import VertexAIProvider` succeeds and returns the correct class.
- `Test expectation:` beyond the import-succeeds check, this unit is documentation/wiring — no additional behavioral test scenarios apply.

**Verification:** `python -c "from dobby.providers import VertexAIProvider"` succeeds; `docs/providers/index.md` and the new `docs/providers/vertexai.md` render correctly and accurately describe the shipped behavior from U1–U4.

---

### U6. Wire `"vertexai"` into `AgentExecutor`'s tool-schema dispatch

**Goal:** Confirmed during plan review (not a hypothetical risk): `dobby/executor.py`'s `AgentExecutor.get_tools_schema()` (dobby/executor.py:120-132, inside the `if self._formatted_tools is None:` block starting at line 119) is a `match self.provider:` statement with no wildcard/default case, over a `Literal["openai", "azure-openai", "gemini", "anthropic"]`-typed `provider` constructor param (dobby/executor.py:63) that is caller-supplied and independent of `llm.name`. Without an explicit `"vertexai"` case, a caller constructing `AgentExecutor(llm=VertexAIProvider(...), provider="vertexai", ...)` gets `_formatted_tools` silently staying `None` — tools are silently dropped, no error raised. This is the same class of gap `"azure-anthropic"` and `"gemini-vertexai"` already have today (out of scope to fix here — see Deferred to Follow-Up Work), but this plan must not introduce a new instance of it for `"vertexai"`.

**Requirements:** Tool/function-calling parity (a dropped-tools bug would silently violate the parity requirement from the origin document).

**Dependencies:** U1 (needs `VertexAIProvider.name == "vertexai"` to exist), U2 (needs `to_vertexai_tool()`).

**Files:**
- `dobby/executor.py` (modify)
- Existing executor tests (extend whichever test file currently covers `get_tools_schema()` dispatch — locate via the test suite before writing, do not assume a path not yet confirmed)

**Approach:**
- Extend the `Literal` type at `dobby/executor.py:63` to include `"vertexai"`.
- Add a **separate** `case "vertexai":` to the `match` in `get_tools_schema()` (dobby/executor.py:119-133), calling `to_vertexai_tool(tool)` (U2) for each tool — **not** `tool.to_openai_format()` and **not** merged into the existing `case "openai" | "azure-openai":` branch. This was corrected during plan review: `to_openai_format()` returns the Responses API's flat tool shape, which is structurally wrong for a Chat Completions-shaped endpoint (this plan's own Key Finding already establishes that messages need a distinct converter for exactly this reason — the same distinction applies to tool schemas, and an earlier draft of this unit missed it).

**Patterns to follow:** The existing `case "openai" | "azure-openai":` / `case "gemini":` / `case "anthropic":` branches in `dobby/executor.py:119-133` for the overall `match` structure and where `self._formatted_tools` gets assigned — but `"vertexai"` is its own case calling its own converter, not folded into an existing one.

**Test scenarios:**
- `AgentExecutor(provider="vertexai", ...).get_tools_schema()` returns tools formatted via `to_vertexai_tool()` — nested `{"type": "function", "function": {...}}` shape, not the flat Responses-API shape `to_openai_format()` produces.
- Existing `"openai"`, `"azure-openai"`, `"gemini"`, `"anthropic"` cases remain unaffected (no regression in the existing dispatch test coverage).

**Verification:** A `VertexAIProvider`-backed `AgentExecutor` with tools registered produces a non-`None`, correctly Chat-Completions-nested tool schema — both the silent-drop failure mode and the wrong-shape failure mode found during plan review are closed for `"vertexai"`.

---

## System-Wide Impact

- **New dependency**: `google-auth[requests]` added to `pyproject.toml`'s `dependencies` (not optional-extras, matching how `anthropic`, `google-genai`, and `openai` are all unconditional dependencies today rather than extras — consistent with the existing dependency philosophy in this repo). The `[requests]` extra is required, not cosmetic — see U1.
- **Public API surface grows**: `dobby.providers.VertexAIProvider` and `dobby.providers.to_vertexai_messages` become new importable names; no existing exports change shape.
- **No changes to `GeminiProvider`'s existing `vertexai=True` mode** — the two Vertex-touching providers coexist with distinct `name` values (`"gemini-vertexai"` vs. `"vertexai"`).
- **`AgentExecutor` (dobby/executor.py) does need a change — confirmed during plan review, not merely flagged.** Its `get_tools_schema()` special-cases provider name via a `match` statement with no default case; a new provider needs an explicit case (with the correct tool-schema shape) or its tools silently vanish or arrive malformed. See U6. (The rest of `AgentExecutor` — `run_stream`, `_invoke_tool`, `_execute_tool_call`, `_execute_tool_stream` — is fully polymorphic over `Provider[ClientT]` and needs no change.)
- **Pre-existing gap noted, not fixed here:** `"azure-anthropic"` and `"gemini-vertexai"` already hit this same silent-tools-drop gap today (missing cases in the same `match`), since neither the `Literal` type nor the `match` includes them. Fixing that is tangential to this plan's scope — see Deferred to Follow-Up Work.

---

## Risks

| Risk | Mitigation |
|---|---|
| ADC token refresh adds latency/failure surface on every request if not cached correctly | `credentials.valid` check avoids refreshing on every request; only refresh when actually expired, guarded by an `asyncio.Lock` against concurrent refresh races (per Key Technical Decision #2) |
| Chat Completions streaming tool-call accumulation (index-based fragments) is a new, easy-to-get-wrong state machine | Dedicated test scenarios in U4 for fragmented, parallel, and adversarial-order tool-call accumulation |
| Region/model availability for MaaS models is limited (e.g. Llama currently `us-central1`-only per origin document) | Out of this plan's control; document the current known limitation in `docs/providers/vertexai.md` rather than attempting to validate region/model combinations in code |
| Self-deployed/third-party endpoints are a less-trusted boundary than native OpenAI (per origin document scope); an unbounded per-stream tool-call accumulator keyed by backend-supplied `index` is a resource-exhaustion vector | Bounded accumulator with an explicit `StreamErrorEvent` failure path in U4 |
| A leaked bearer token's blast radius depends on ADC scope (commonly broad `cloud-platform` by default) | Optional `scopes` constructor parameter (U1) plus documented least-privilege guidance in `docs/providers/vertexai.md` |

---

## Dependencies / Prerequisites

- `google-auth[requests]` (new dependency, added in U1 — the `requests` extra, not just bare `google-auth`, is required by `google.auth.transport.requests`).
- No new external service accounts needed for implementation/testing — all HTTP calls are mocked per the existing provider test convention (`unittest.mock.MagicMock`/`AsyncMock` on the client, no live network calls in the test suite, matching `tests/test_openai_provider.py` and `tests/test_anthropic_provider.py`).

---

## Documentation Plan

Covered by U5: `docs/providers/vertexai.md` (new, including the `scopes` least-privilege guidance from Key Technical Decision #2) and `docs/providers/index.md` (corrected + extended). No other documentation surfaces (README.md, top-level docs) reference specific providers by name based on the current `docs/providers/index.md` scan.
