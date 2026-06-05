# Dobby SDK — Provider-Agnostic Context Editing (BETA) — Requirements & Design

**Date:** 2026-06-04
**Status:** Brainstorm / requirements capture (local-only; `docs/brainstorms/` is git-ignored). Engineering content — safe to publish later if sanitized.
**Method:** 10-agent research + synthesis + 3-lens adversarial workflow (`w0590ha53`, ~701K tokens), then **direct source verification** of every load-bearing claim against `dobby/` on `main` and the installed SDK versions.
**Pairs with:** `2026-06-04-sdk-evolution-research.md` §4 (this doc supersedes that §4's `ContextManager` sketch and **corrects** its `ToolStreamEvent` claim).

---

## 0. The ask (restated)

Ship **context editing** — automatically pruning/compacting stale tool results (and optionally reasoning) from what the model sees in a long tool-loop — as a **BETA** feature, mirroring Anthropic's context-editing *flow*. We implement it **client-side for OpenAI + Gemini**, which have no equivalent native API. Plus: a demo (following Anthropic's memory cookbook) and "an advanced tool with reasoning capability for context editing."

> **🎯 Scope — OpenAI + Gemini only (decided 2026-06-04).** This feature targets OpenAI and Gemini. We are **not** building out the full Anthropic provider as part of this work, and **Anthropic-native passthrough is explicitly out of scope** — Anthropic's docs are referenced only as the *design template* for the flow we mirror client-side. (If a complete Anthropic provider is ever built independently, the same `ContextEditEvent` shape *could* later forward to native `context_management` — but that is a separate, uncommitted effort, not a deliverable here.)

Reference inputs:
- Context editing: <https://platform.claude.com/docs/en/build-with-claude/context-editing>
- Advanced tool use: <https://www.anthropic.com/engineering/advanced-tool-use>
- Memory cookbook: <https://platform.claude.com/cookbook/tool-use-memory-cookbook>

---

## 1. Verified ground truth (checked against source — not assumed)

| Claim | Verdict | Evidence |
|---|---|---|
| Installed pins: openai **2.15.0**, google-genai **1.75.0**, anthropic **0.75.0**, pydantic **2.12.5** | ✅ | `uv pip list`. **But pyproject declares floors** `openai>=2.14.0`, `google-genai>=1.68,<2`, `anthropic>=0.75` — a fresh resolve can drift. |
| dobby's OpenAI provider uses the **Responses API**, not Chat Completions | ✅ | `responses.create(...)` at `providers/openai/adapter.py:328`. (`completions.py` is the deprecated, unexported path.) |
| **The `chat(**kwargs)` "seam" is a mirage — kwargs are silently dropped** | ✅ **(blocker)** | OpenAI `chat()` accepts `**kwargs` (`adapter.py:257`) but calls `_stream_chat_completion`/`_non_stream_chat_completion` with **fixed positional args** (`adapter.py:286-292`); `_build_kwargs` is a strict whitelist of `model/input/tools/reasoning/max_output_tokens` (`adapter.py:160-169`). Gemini `chat()` only does `kwargs.pop("max_tokens")` (`adapter.py:218`) and ignores the rest. |
| `ToolStreamEvent.type` is **free-form `str` by design** (not a tag to collapse) | ✅ **(corrects prior doc)** | `tool_events.py:12` `type: str`; docs show users yielding `ToolStreamEvent(type="progress", ...)` (`docs/tools/creating-tools.md:138,142`). It **is** a member of the `Field(discriminator="type")` union (`stream_events.py:101,105`), so the union is malformed for `TypeAdapter` validation — but the fix is **not** "make it a `Literal`" (that breaks the documented API). |
| Message parts are plain `@dataclass`, **not Pydantic, not frozen** | ✅ | `message.py:31,40`, `tool_part.py:5`, `tool_result_part.py`. `list(messages)` is a shallow copy sharing instances → in-place edits mutate the originals. |
| `ToolUsePart` already has a `metadata` slot; discriminator is `kind` | ✅ | `tool_part.py:15,17`. |
| Token usage is available per turn (free trigger source) | ✅ | `StreamEndEvent.usage: Usage \| None` (`stream_events.py:90`); `usage.input_tokens` (`usage.py:11`); iterated at the loop's stream-end handling (`executor.py:266`). **Can be `None`** (Gemini sets it only if `usage_metadata` present, `adapter.py:387`). |
| No dobby Anthropic adapter exists | ✅ | `dobby/providers/anthropic/` contains only `.gitkeep`; literal `"anthropic"` is accepted (`executor.py:63`) and `to_anthropic_format()` exists (`tool.py:206`), but no adapter. |
| Gemini: dobby substitutes a **dummy thought-signature** and pairs tool calls **by name/position, not id** | ✅ (reviewer-cited; verify at impl) | dummy `b"skip_thought_signature_validator"` at `gemini/adapter.py:306,420`; `FunctionResponse(name=...)` with no id at `gemini/converters.py:113-118`. |

### The Anthropic flow we are mirroring (from the docs)
- Field `context_management.edits[]`; edit types `clear_tool_uses_20250919` and `clear_thinking_20251015`; beta header `context-management-2025-06-27`; memory tool `memory_20250818`.
- Knobs on `clear_tool_uses`: `trigger` (input-token threshold, default ~100k), `keep` (keep last N tool-use/result pairs, default 3), `clear_at_least` (min tokens to clear, all-or-nothing), `exclude_tools`, `clear_tool_inputs` (default false).
- Behaviour: server-side, **non-destructive** (the stored conversation is untouched; editing is recomputed per request), leaves a **placeholder marker** (not a silent drop), and the response echoes `applied_edits`. Pairs with the memory tool ("clear from context, persist to `/memories` first").
- Published efficacy: ~84% token reduction; +29% (editing) / +39% (editing + memory) on long-horizon tasks. **These are Anthropic's server-side numbers — not a dobby promise.**

---

## 2. Per-provider feasibility (corrected)

| Provider (installed) | Native context editing? | What dobby can actually do | v1 strategy |
|---|---|---|---|
| **OpenAI 2.15** (Responses API) | Partial: `truncation="auto"` (drop-oldest) is *typed*; `responses.compact()` exists but assumes server-side conversation state (`previous_response_id`) dobby doesn't use; inline `context_management` is **not typed** in 2.15. | **Client-side trim** of the `MessagePart` list before `responses.create`. `truncation="auto"` is reachable **only after** an adapter change (kwargs are dropped today). Reasoning is already dropped on serialize, so trimming it is a no-op. | Client-side trim **now**. |
| **Gemini 1.75** (`generateContent`) | **None.** `context_window_compression` is Live-API-only. Context caching lowers cost but removes nothing. `count_tokens` available. | **Client-side trim** of `contents`, at **turn granularity**, never touching in-flight calls. | Client-side trim **now**. |
| **Anthropic 0.75** | **Yes**, natively (under `client.beta.messages`) — **but no dobby adapter exists, and building it is out of scope here.** | n/a for this feature. | **Out of scope.** We are not building the Anthropic provider as part of context editing. |

**Conclusion:** the feature is a **client-side editor on dobby's own `MessagePart` list** — the one place OpenAI and Gemini share state — reachable today with no provider/native work.

---

## 3. Recommended design — minimal v1, with a clear growth path

> The three adversarial lenses converged: a single **"keep last N tool-result pairs, trigger on previous-turn input tokens, replace older results with a placeholder"** function delivers ~90% of the value. The synthesis's own open-decisions already recommended trim-first. So v1 is deliberately small; everything else is staged behind real user pull.

### 3a. The single hook point
The only shared conversation state is `working_messages = list(messages)` (`executor.py:250`), and the only model call is `await self.llm.chat(working_messages, ...)` (`executor.py:256-263`) inside `for _ in range(max_iterations)` (`executor.py:253`). **Apply the edit to a transient copy immediately before that call; never mutate `working_messages`.** This reproduces Anthropic's "non-destructive, recomputed-per-request" semantics and keeps the full record intact for audit/replay.

### 3b. v1 surface (small)
1. **One bug fix (land standalone, independent of this feature):** make the `StreamEvent` union well-formed *without* breaking the free-form `ToolStreamEvent.type`. Two options — **(A, recommended)** add a dedicated discriminator field to every union member (e.g. `kind: Literal[...]`) and switch `Field(discriminator="kind")`, leaving `ToolStreamEvent.type` free-form; **(B)** drop the discriminator and rely on Pydantic smart-union. *Do NOT collapse `ToolStreamEvent.type` to a `Literal` — it breaks the documented `type="progress"` API.*
2. **One event:** `ContextEditEvent(applied_edits: list[AppliedEdit])` + `AppliedEdit(cleared_tool_uses, cleared_input_tokens_estimate, ...)`, added to the union. The `applied_edits[]` shape intentionally mirrors Anthropic's response so the event reads identically across providers (and would map 1:1 if a native path is ever added later — but no `source` discriminator is needed now, since the only path is client-side).
3. **One function**, not a Protocol+manager+5-class policy tree (which would have exactly one impl and one call site in v1):
   ```python
   def edit_context(
       messages: list[MessagePart],
       *,
       keep_pairs: int = 3,
       trigger_tokens: int = 100_000,
       last_input_tokens: int | None,
       exclude_tools: tuple[str, ...] = (),
       placeholder: str = "[Tool result cleared to save context.]",
   ) -> tuple[list[MessagePart], AppliedEdit | None]: ...
   ```
   - **Trigger:** fire only if `last_input_tokens is not None and last_input_tokens >= trigger_tokens`. First turn never trims.
   - **Turn-aware, whole-pair granularity:** identify tool round-trips (each `_emit_tool_result` writes an `AssistantMessagePart([ToolUsePart])` + `UserMessagePart([ToolResultPart])`, `executor.py:182-194`). Keep the most recent `keep_pairs`, keep any `name in exclude_tools`, keep any **in-flight** pair (a `ToolUsePart` with no matching `ToolResultPart`). For older pairs, build **new** dataclass instances (`dataclasses.replace`) with the `ToolResultPart` content replaced by a `TextPart(placeholder)`. **Never mutate inputs in place** (parts are shared dataclasses).
   - **Parallel calls:** treat a contiguous run of tool pairs from one model turn as **one atomic unit** — keep or clear the whole batch (a parallel turn emits multiple `ToolUseParts` in one `StreamEndEvent`, re-synthesized into separate pairs by `_emit_tool_result`; splitting them across the `keep` boundary would orphan/interleave, breaking Gemini).
4. **Executor wiring:** optional `context_manager`/`context_policy` param on `AgentExecutor.__init__` (default `None` → zero behaviour change, mirroring optional `tools`/`output_type`). Track `last_input_tokens`; before `chat`, compute `send_messages` (full history stays in `working_messages`); `yield ContextEditEvent` when an edit applied; pass `send_messages` to `chat`.

### 3c. Non-negotiable invariants (these are the real correctness core)
- **Never trim an unresolved (in-flight) tool call.** Satisfies both OpenAI `call_id` pairing AND Gemini's in-flight thought-signature rule.
- **Never orphan a `tool_result` from its `tool_use`;** trim only whole pairs/turns.
- **Build new dataclass instances;** never mutate shared parts (preserves the non-destructive guarantee).
- **Gemini:** pair by **name + position within the turn**, not by id (ids collide as `call_<name>` for parallel same-name calls). **Leave thought-signatures intact** (the dummy `skip_thought_signature_validator` is known-safe) — do **not** blank them.

---

## 4. Corrections from the adversarial pass (DO NOT skip)

These are wrong-in-the-synthesis (or wrong-in-the-prior-doc) items, all source-verified:

1. **`**kwargs` does not forward.** "Pass `truncation="auto"` as a one-line backstop" and "native config rides the existing seam, no signature change" are **false**. Both require real adapter surgery (thread kwargs through `_build_kwargs` / `GenerateContentConfig`). → All native-passthrough work (OpenAI `truncation`, Anthropic) is **out of v1**.
2. **`ToolStreamEvent.type` must stay free-form.** Collapsing it to `Literal["tool_stream"]` breaks `type="progress"` (documented). Move the discriminator instead (§3b.1).
3. **Parts are dataclasses** → use `dataclasses.replace`, not in-place mutation, or the "non-destructive" property is a lie on iteration 2+.
4. **Gemini reasoning in the synthesis was misread** — dobby already uses a dummy signature and pairs by name/position. Restated correctly in §3c.
5. **`usage` can be `None`** → the lagging trigger silently no-ops on exactly the long runs it targets. Handle it: carry forward the last known value, or fall back to a `len(text)/4` estimate over `send_messages` so the trigger still engages.
6. **Cache invalidation is a real cost tradeoff, not a pure win.** Client-side trimming mutates the prefix → busts OpenAI/Gemini prompt caching from that point. Trim in larger, less-frequent batches to amortize; note OpenAI `prompt_cache_key` exists if stability matters. Document the tradeoff.
7. **`responses.compact()` is incompatible with dobby's stateless full-history-rebuild** (`to_openai_messages` every turn, no `previous_response_id`). It's a larger item than "opt-in later" — leave it out.
8. **Same-turn blow-up:** the one-turn trigger lag means a single huge tool result can exceed the window before the next-turn trigger fires. The synchronous backstop for this is OpenAI `truncation="auto"` — which needs the adapter change. Acknowledge the gap in v1.

---

## 5. The "advanced reasoning tool for context editing" + memory tool (Phase 2 — the user explicitly asked for these)

Anthropic ships **no single packaged "reasoning tool for context editing"** — it's a composite (memory tool + clear strategies + pre-clear warning). For dobby, the most concrete reading the user asked for:

- **`CompactContextTool`** — a tool the agent invokes *on its own transcript* with its reasoning about what to keep/summarize/drop; the executor routes it through a `summarize` edit (an LLM round-trip) and yields a `ContextEditEvent`. The agent's words become the summarizer prompt — that's the "reasoning capability."
  - **Honest cost:** this is the **highest-complexity item** — it needs a new `edits_context: ClassVar[bool]` on the core `Tool` dataclass plus a new branch in the hot loop (mirroring `terminal` at `executor.py:308,379`), and `summarize` mode adds latency + nondeterminism. All three reviewers flagged it as highest-effort / lowest-proven-demand. **Recommend Phase 2, gated on the demo landing first.**
- **`MemoryTool`** — a dobby `Tool` mirroring Anthropic's six-command set (`view/create/str_replace/insert/delete/rename`, asymmetric params, verbatim return strings) over a swappable `MemoryBackend` (default `LocalMemoryBackend`). **It works as an ordinary function tool today with zero context-editing machinery**, so it's independently shippable.
  - **Security is the core, not an add-on:** sandbox = `Path.resolve(strict=False)` then assert the resolved path is under `MEMORIES_ROOT`; reject `..` escapes and **symlinks** whose target escapes. **Do not** URL-decode (`%2e%2e%2f` is a literal dir name at the FS layer; decoding would *create* the hole). Per-command audit log (ties into the governance differentiator).
- **The synergy ("nothing is lost"):** model sees a lean window; memory keeps everything the agent persisted. v1-cheap version: when trim is imminent and `MemoryTool` is registered, inject a one-line "persist to /memories now" note on the prior turn.

---

## 6. Demo (mirror the cookbook, but right-sized)

The existing `examples/` are **`.py` scripts, not notebooks**. Match that for v1:
- **`examples/context_editing.py`** — trim-only on OpenAI + Gemini: drive a ~15-call tool loop with a token-heavy tool, print `input_tokens` per turn, mark turns where `ContextEditEvent` fired, show the placeholder left behind (proving non-silent clearing), and the Gemini in-flight-safety callout.
- A fuller **`examples/context_management_memory.ipynb`** (memory persistence across a fresh run, `CompactContextTool`, audit-log dump) lands **with Phase 2** — once the memory tool exists.

---

## 7. Open product decisions (need your call)

1. **v1 scope — ✅ DECIDED (2026-06-04): trim core + `CompactContextTool` (the reasoning tool).** v1 ships the client-side trimmer + `ContextEditEvent` + `summarize` mode + the agent-invoked `CompactContextTool`, on OpenAI + Gemini. `MemoryTool` is Phase 2. Anthropic-native passthrough is **out of scope** (not building the Anthropic provider here). Accepted tradeoff: `CompactContextTool` adds a new `edits_context` flag on the core `Tool` dataclass and a branch in the hot loop — built knowingly because the demo's wow factor is the agent compacting its own context. *(Trim-only-first was the adversarial recommendation; overridden deliberately for demo value.)*
2. **Opt-in vs always-on.** *Recommend opt-in* (`context_manager=None` default → zero behaviour change; constructing the policy IS the BETA gate, plus a one-time `DobbyBetaWarning`).
3. **trim vs summarize default.** *Recommend `trim`* (deterministic, no extra LLM call, auditable). `summarize` only with `CompactContextTool` / Phase 2.
4. **Token-trigger precision.** *Recommend the free previous-turn-usage trigger for v1* (with the `usage=None` fallback from §4.5). Provider `count_tokens` precision is deferred.
5. **Dependency pins.** Tighten for the BETA (e.g. `openai>=2.15`) **or** add runtime capability probes — the design depends on surfaces the declared floors permit drifting away from.

---

## 8. Phased backlog

**Phase 0 — standalone bug fix (do regardless of this feature):**
- [ ] Make `StreamEvent` union well-formed without breaking `ToolStreamEvent.type` (add a `kind` discriminator field across members, or drop the discriminator). Verify `executor.py:363,387` isinstance checks unaffected.

**Phase 1 — v1 BETA: trim core + reasoning tool (OpenAI + Gemini, the NOW) — ✅ chosen scope:**
- [ ] `dobby/context/edit.py`: `edit_context(...)` function (§3b.3) + invariants (§3c). ~40-60 LOC.
- [ ] `dobby/types/tool_events.py`: `AppliedEdit`, `ContextEditEvent`; register in the (fixed) union.
- [ ] Wire `AgentExecutor.__init__` (`executor.py:61`) optional param + `run_stream` (`executor.py:210-266`): track `last_input_tokens` (handle `None`), compute `send_messages` before `chat`, yield `ContextEditEvent`, keep `working_messages` unmutated.
- [ ] `summarize` mode (synthetic `<summary>` turn; recent window verbatim; one LLM round-trip).
- [ ] `edits_context: ClassVar[bool] = False` on `Tool` (`tool.py:74-77`) + `CompactContextTool` + loop branch routing it through `summarize` (mirror the `terminal` branch at `executor.py:308,379`), yielding `ContextEditEvent`.
- [ ] Tests: keeps last-N; excludes named tools; never trims in-flight pair; parallel same-name batch kept/cleared atomically; OpenAI no orphaned `function_call_output`; Gemini signature/pairing preserved; input objects object-identical when only some pairs edited; trigger no-ops gracefully when `usage=None`; `CompactContextTool` round-trip (agent reasoning → summary turn → `ContextEditEvent`).
- [ ] `examples/context_editing.py` (trim + a `CompactContextTool` call, OpenAI + Gemini).

**Phase 2 — memory tool (governance/audit differentiator; independently shippable):**
- [ ] `dobby/tools/memory.py`: `MemoryTool` (six-command union) + `MemoryBackend`/`LocalMemoryBackend` with the path/symlink sandbox + per-command audit log.
- [ ] Pre-clear "persist now" warning when memory registered + trim imminent.
- [ ] Security tests (traversal, symlink escape, absolute paths).
- [ ] `examples/context_management_memory.ipynb`.

**Out of scope (not part of this feature):**
- ❌ **Anthropic-native passthrough** — depends on building the full Anthropic provider, which we are not doing here. The client-side editor covers OpenAI + Gemini; if an Anthropic provider is ever built independently, the same `ContextEditEvent` could be wired to native `context_management` then.
- ❌ **OpenAI `truncation="auto"` server backstop** — would need adapter kwargs-forwarding surgery (§4.1); left out of the BETA. If the same-turn-blow-up risk (§4.8) bites in practice, revisit as a small standalone adapter change.

---

## 9. Where the raw research lives
- Full workflow output (10 agents, ~701K tokens): task `w0590ha53` — `tasks/w0590ha53.output` (synthesis design + three adversarial reviews with verbatim findings).
- Six research briefs: labels `anthropic:context-editing`, `anthropic:advanced-tool-use`, `anthropic:memory-cookbook`, `dobby:integration-surface`, `openai:2.15-capabilities`, `gemini:1.75-capabilities`.
- All blocker-level claims re-verified directly against source (see §1 evidence column).
