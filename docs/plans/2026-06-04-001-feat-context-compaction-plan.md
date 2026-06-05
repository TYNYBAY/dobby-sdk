---
title: "feat: Context compaction (trim + summarize) for OpenAI & Gemini"
type: feat
status: active
created: 2026-06-04
depth: deep
origin: docs/brainstorms/2026-06-04-context-editing-requirements.md
---

# feat: Context compaction (trim + summarize) for OpenAI & Gemini

> **Origin:** `docs/brainstorms/2026-06-04-context-editing-requirements.md` (requirements + verified integration surface). This plan is the **HOW** for the compaction half of that BETA.
> **Scope reminder:** OpenAI + Gemini only, client-side. Anthropic-native passthrough and the memory tool are explicitly **out of scope** (separate efforts).
> **Research backing:** workflows `w0590ha53` (context-editing design + adversarial verification) and `wphubqbru` (framework comparison + dobby grounding). Every dobby anchor below was verified against source on `main`.

---

## Problem Frame

In dobby's agentic loop, every model call resends the full conversation. Tool calls and their (often large) results accumulate in `working_messages` (`dobby/executor.py:250`) and are re-sent verbatim every iteration — so a long tool-loop grows the input context without bound: it slows down, costs more, and eventually exceeds the model's window.

**Context compaction** keeps the model's *input window* lean by automatically reducing stale tool history before each model call, while preserving the full record for the caller. Two operations share one trigger:

- **trim** — deterministically replace stale tool-result payloads with a small placeholder. No LLM call.
- **summarize** — replace an old span of turns with one LLM-generated `<summary>` turn, keeping recent turns verbatim. One LLM round-trip, computed once and written back.

Neither provider offers this natively (OpenAI's Responses API only exposes a blunt `truncation="auto"`; Gemini's `generateContent` has nothing), so dobby implements it client-side on its own provider-neutral `MessagePart` list — the one place OpenAI and Gemini share state.

---

## Scope

### In scope
- A `ContextPolicy` config object (percentage-of-window trigger, keep-last-N, mode).
- An automatic, between-turns compaction hook in `run_stream` (trim + summarize).
- A `ContextEditEvent` on the typed stream so compaction is observable.
- An agent-invoked `CompactContextTool` (the "reasoning tool") that triggers summarize on demand.
- A concurrency guard so compactions never overlap within a run.
- One `.py` example, **a cookbook-style demo/test notebook (the final unit)**, and tests — all on OpenAI and Gemini.

### Key decisions carried from origin
- v1 = trim core + summarize + `CompactContextTool` (origin §7, decided).
- Non-destructive editing: `working_messages` is the full record; the model sees an edited view (origin §3a).
- Parts are plain `@dataclass` → edits build **new** instances, never mutate in place (origin §1, §4.3).

### Out of scope (see `### Deferred to Follow-Up Work`)
- Anthropic-native `context_management` passthrough (not building the Anthropic provider).
- A separate/cheaper summarizer model (reuse parent model — user decision 2026-06-04).
- A model-window lookup table ("model repository") — the window comes from config instead.
- `MemoryTool` / persist-before-clear (tracked in GitHub issue #8).
- OpenAI `truncation="auto"` server backstop (needs adapter kwargs surgery).
- A full fix of the pre-existing `StreamEvent` discriminated-union malformation.

---

## Key Technical Decisions

### KTD-1 — Trigger: percentage of a *configured* context window
The trigger fires when the previous turn's input tokens cross a percentage of a context-window size **supplied in the policy** (not looked up — dobby has no window table; `Provider` exposes only `model: str`, `dobby/providers/base.py:123`).

```
trigger_tokens = int(trigger_pct * context_window)
fire when:  last_input_tokens is not None and last_input_tokens >= trigger_tokens
```

- `last_input_tokens` is read from `StreamEndEvent.usage.input_tokens` (`dobby/types/stream_events.py:90`, `dobby/types/usage.py:11`) at the loop's stream-end handler (`dobby/executor.py:266`). One-turn lag, free, no tokenizer dependency.
- `usage` can be `None` (Gemini sets it only when `usage_metadata` is present, `dobby/providers/gemini/adapter.py:387`). Fallback: carry forward the last known value; if still unknown, estimate `sum(len(text))/4` over the messages so the trigger still engages.
- **Testability (explicit user requirement):** set a small `context_window` (e.g. `2000`) or low `trigger_pct` (e.g. `0.5`) → compaction fires after ~1000 input tokens, so the path is exercisable without million-token runs.
- Default `trigger_pct = 0.8` (reserve ~20% headroom — aligns with observed Claude Code auto-compact behavior; LlamaIndex uses 0.75). Default `context_window = 128_000`.

### KTD-2 — Summarizer reuses the parent model (no override)
The summarize path issues its **own** model call via the existing provider contract in **non-stream** mode, reusing `self.llm`:

```
result: StreamEndEvent = await self.llm.chat(span, system_prompt=SUMMARIZE_PROMPT, stream=False, tools=None)
summary_text = "".join(p.text for p in result.parts if isinstance(p, TextPart))
```

`stream=False` returns a single `StreamEndEvent` **value** you `await` — do **not** `async for` it (contrast the streaming loop call at `dobby/executor.py:256`; contract at `dobby/providers/base.py:141-151`). No separate provider, no per-call model override (verified dead end — the adapters drop a per-call `model=`: `dobby/providers/openai/adapter.py:283`, `dobby/providers/gemini/adapter.py:257`). A configurable cheaper model is deferred.

### KTD-3 — Persistence: recompute for trim, write-back for summarize
- **trim → recompute.** Each triggered turn builds a transient `send_messages` from `working_messages` and passes it to `chat`; `working_messages` is never mutated. Non-destructive, cheap (no LLM), audit-clean. Matches Anthropic's context-editing template.
- **summarize → write-back.** The synthetic `<summary>` turn is written **back** into `working_messages` (as **new** dataclass instances) and gated by a `last_compacted_at_tokens` watermark so it computes **once**, not every iteration. Matches every summarizing framework (OpenAI cookbook, langmem `RunningSummary`, LlamaIndex `ChatSummaryMemoryBuffer`). The replaced originals are stashed on `AppliedEdit` for audit/replay (dobby's differentiator).
- **Critical gotcha:** `working_messages = list(messages)` (`dobby/executor.py:250`) is a *shallow* copy sharing non-frozen `@dataclass` part instances with the caller. Write-back must replace **list entries** with fresh instances (`dataclasses.replace`, new `UserMessagePart`/`TextPart`) — never edit a part in place, or iteration 2+ corrupts the caller's `messages`.

### KTD-4 — Compaction is between-turns and single-flight
`run_stream` is strictly serial (one in-flight `chat` at `dobby/executor.py:256`, driven to completion inside `for _ in range(max_iterations)` at `:253`; parallel tools gather but join before the next `chat`). The universal cross-framework rule is "compact only between turns, never mid-stream." Guard against the three overlap vectors compaction introduces (auto-trigger + `CompactContextTool` in the same iteration; re-trigger while a summarize is in flight; reused-executor state collisions) with **loop-local closure variables** in `run_stream` — `compaction_in_progress: bool` and `last_compacted_at_tokens: int | None` — shared by the auto-trigger path and the tool branch. Keep this state **off `self`** (dobby has a latent `self.last_output` reuse smell at `dobby/executor.py:82`; do not add to it).

### KTD-5 — Opt-in via an optional policy param (zero behavior change)
Add `context_policy: ContextPolicy | None = None` to `AgentExecutor.__init__` (`dobby/executor.py:61-68`), mirroring the existing optional `tools`/`output_type`. `None` → today's behavior exactly. Constructing a policy IS the BETA gate.

### KTD-6 — `ContextEditEvent` works at runtime despite the pre-existing union bug
The hot loop dispatches by `isinstance` (`dobby/executor.py:266,363,387`), not by validating the `StreamEvent` union, so adding `ContextEditEvent` (with a proper `type: Literal["context_edit"]`) is safe at runtime. The pre-existing malformation (`ToolStreamEvent.type` is free-form `str`, `dobby/types/tool_events.py:12`, making `TypeAdapter(StreamEvent)` invalid) is **not** introduced or worsened here; its proper fix touches every event type and is deferred (do **not** "fix" it by collapsing `ToolStreamEvent.type` to a `Literal` — that breaks the documented `type="progress"` API, `docs/tools/creating-tools.md:138`).

---

## High-Level Technical Design

*This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

The compaction hook sits immediately before the single model call inside the existing loop:

```
run_stream(messages, ...):
    working_messages = list(messages)            # full record (executor.py:250) — never mutated by trim
    last_input_tokens = None
    compaction_in_progress = False               # loop-local guards (KTD-4)
    last_compacted_at_tokens = None

    for _ in range(max_iterations):              # executor.py:253
        send_messages = working_messages

        # --- compaction hook (NEW) ---
        if policy and not compaction_in_progress and _triggered(last_input_tokens, policy, last_compacted_at_tokens):
            compaction_in_progress = True
            if policy.mode == "trim":
                send_messages, applied = edit_context(working_messages, policy)        # recompute, transient
            else:  # summarize
                applied = await summarize_context(working_messages, policy, self.llm)   # write-back, mutates working_messages list
                send_messages = working_messages
                last_compacted_at_tokens = last_input_tokens
            if applied:
                yield ContextEditEvent(applied_edits=[applied])
            compaction_in_progress = False
        # --- end hook ---

        async for event in await self.llm.chat(send_messages, ...):   # executor.py:256
            yield event
            if isinstance(event, StreamEndEvent) and event.usage:     # executor.py:266
                last_input_tokens = event.usage.input_tokens
        ... execute tool calls (CompactContextTool branch can also set the guard + summarize) ...
```

Trigger / mode / persistence interaction:

| mode | LLM call? | Mutates `working_messages`? | Recomputed each turn? | Default |
|---|---|---|---|---|
| `trim` | no | no (transient `send_messages`) | yes (cheap) | ✅ default |
| `summarize` (auto, `mode="summarize"`) | yes, reuse parent | yes (write-back, new instances) | no (watermark-gated) | opt-in |
| `summarize` (via `CompactContextTool`) | yes, reuse parent | yes (write-back) | n/a (agent-driven) | opt-in tool |

---

## Output Structure

New greenfield package + touched files:

```
dobby/
├── context/                      # NEW package
│   ├── __init__.py               # exports: ContextPolicy, edit_context, summarize_context
│   ├── policy.py                 # ContextPolicy (Pydantic)
│   ├── edit.py                   # edit_context() — trim algorithm
│   └── summarize.py              # summarize_context() + SUMMARIZE_PROMPT
├── tools/
│   ├── tool.py                   # + edits_context ClassVar flag
│   ├── compact.py                # NEW: CompactContextTool
│   └── __init__.py               # + CompactContextTool export
├── types/
│   ├── tool_events.py            # + AppliedEdit, ContextEditEvent
│   ├── stream_events.py          # + ContextEditEvent in StreamEvent union
│   └── __init__.py               # + Context events exports
├── executor.py                   # + context_policy param, compaction hook, guards, tool branch
└── __init__.py                   # + public exports (fix stale module docstring)

examples/context_editing.py                 # NEW canonical .py example (OpenAI + Gemini)
examples/context_compaction_cookbook.ipynb  # NEW cookbook-style test/demo notebook (U7, last)
tests/test_context_compaction.py            # NEW (trim + executor wiring + summarize)
tests/test_compact_tool.py                  # NEW (edits_context flag + tool branch)
```

The per-unit `**Files:**` sections are authoritative; the tree is a scope sketch.

---

## Implementation Units

### U1. `ContextPolicy` + `ContextEditEvent` types (foundation)

**Goal:** Land the config object and the observable event so later units have stable types to depend on. No behavior change.

**Requirements:** KTD-1, KTD-5, KTD-6; origin §3b.

**Dependencies:** none.

**Files:**
- `dobby/context/__init__.py` (new — package), `dobby/context/policy.py` (new)
- `dobby/types/tool_events.py` (add `AppliedEdit`, `ContextEditEvent`)
- `dobby/types/stream_events.py` (add `ContextEditEvent` to the `StreamEvent` union, ~line 93-105)
- `dobby/types/__init__.py` (export under a new `# Context events` group, `X as X` alias style)
- `tests/test_context_compaction.py` (new — type/defaults portion)

**Approach:**
- `ContextPolicy(BaseModel)`: `context_window: int = 128_000`, `trigger_pct: float = 0.8`, `keep_last_n: int = 3`, `mode: Literal["trim","summarize"] = "trim"`, `placeholder: str = "[Tool result cleared to save context.]"`. Add a `trigger_tokens` computed property = `int(trigger_pct * context_window)`. Validate `0 < trigger_pct <= 1` and `context_window > 0`, `keep_last_n >= 0`.
- `AppliedEdit(BaseModel)`: `type: Literal["clear_tool_uses","summarize"]`, `cleared_tool_uses: int = 0`, `cleared_input_tokens_estimate: int = 0`, `summary_text: str | None = None`, `replaced_originals: list[Any] | None = None` (audit stash; keep `Any` to avoid a parts-typing cycle).
- `ContextEditEvent(BaseModel)`: `type: Literal["context_edit"] = "context_edit"`, `applied_edits: list[AppliedEdit]`. Pydantic, matching the events convention (`type: Literal[...] = "..."`).

**Patterns to follow:** Pydantic event models in `dobby/types/tool_events.py` and `dobby/types/stream_events.py`; `X as X` re-export style in `dobby/types/__init__.py:43-47`.

**Technical design:** *Directional.* Mirror `ToolResultEvent` shape for `ContextEditEvent`; `ContextPolicy` mirrors a settings-style Pydantic model with field defaults.

**Test scenarios** (`tests/test_context_compaction.py`):
- `ContextPolicy()` defaults: `trigger_tokens == 102_400` (0.8 × 128_000); `mode == "trim"`.
- `ContextPolicy(context_window=2000, trigger_pct=0.5).trigger_tokens == 1000` (the testing knob).
- Invalid `trigger_pct=0` / `trigger_pct=1.5` / `context_window=0` raise `ValidationError`.
- `ContextEditEvent(applied_edits=[AppliedEdit(type="clear_tool_uses", cleared_tool_uses=2)])` constructs; `.type == "context_edit"`.
- `ContextEditEvent` is importable from `dobby.types`.

**Verification:** `uv run ruff check` + `uv run pytest tests/test_context_compaction.py` pass; types import from the public surface.

---

### U2. `edit_context()` — trim algorithm (deterministic, no LLM)

**Goal:** The pure, turn-aware trimmer that replaces stale tool-result payloads with a placeholder on a **fresh** message list.

**Requirements:** KTD-1, KTD-3 (trim/recompute), origin §3b/§3c invariants.

**Dependencies:** U1.

**Files:**
- `dobby/context/edit.py` (new), `dobby/context/__init__.py` (export `edit_context`)
- `tests/test_context_compaction.py` (trim portion)

**Approach:**
- Signature: `edit_context(messages: list[MessagePart], policy: ContextPolicy) -> tuple[list[MessagePart], AppliedEdit | None]`.
- Identify tool round-trips by the shape `_emit_tool_result` writes (`dobby/executor.py:161-208`): an `AssistantMessagePart([ToolUsePart])` followed by `UserMessagePart([ToolResultPart])`.
- **Group by logical turn:** a contiguous run of tool pairs produced by one model turn is one atomic unit (parallel calls emit multiple pairs) — keep or clear a whole batch together.
- Keep the most recent `keep_last_n` turns verbatim; keep any **in-flight** pair (a `ToolUsePart` with no matching `ToolResultPart`).
- For older clearable pairs: build a **new** `ToolResultPart` (via `dataclasses.replace`) whose content is a single `TextPart(policy.placeholder)`, preserving the pair skeleton and ids. Assemble a **new** list; never mutate inputs.
- Return `(new_list, AppliedEdit(...))`, or `(messages, None)` if nothing cleared.

**Patterns to follow:** dataclass parts in `dobby/types/message.py`, `tool_part.py`, `tool_result_part.py`; `_emit_tool_result` pair shape.

**Technical design:** *Directional.* Walk → segment into turns → partition keep/clear → rebuild. Pure function; no I/O.

**Test scenarios:**
- Keeps the last `keep_last_n` tool turns verbatim; older results replaced by placeholder text.
- **Non-destructive:** the input `messages` list and its part objects are object-identical (`is`) before/after; only the returned list holds new instances.
- **In-flight protection:** a trailing `ToolUsePart` with no `ToolResultPart` is never trimmed.
- **Parallel batch atomicity:** 3 parallel same-turn calls straddling the `keep_last_n` boundary are kept/cleared as one unit (none orphaned/interleaved).
- **OpenAI pairing:** no `ToolResultPart` is left without its `ToolUsePart` (would break `call_id` pairing).
- **Gemini:** thought-signature in `ToolUsePart.metadata` on kept pairs is preserved; cleared prior pairs keep their pair skeleton.
- Returns `(messages, None)` when there is nothing older than `keep_last_n`.

**Verification:** `uv run pytest tests/test_context_compaction.py` green; trim is provider-agnostic (operates on `MessagePart`).

---

### U3. Executor wiring — automatic trim (hook + trigger + event)

**Goal:** Wire `context_policy` into `run_stream` so trim fires automatically between turns when the percentage trigger crosses, emitting `ContextEditEvent`.

**Requirements:** KTD-1, KTD-3 (trim), KTD-4 (guard), KTD-5 (opt-in).

**Dependencies:** U1, U2.

**Files:**
- `dobby/executor.py` (`__init__` param + store; `run_stream` hook, trigger, guards, usage read)
- `tests/test_context_compaction.py` (executor portion)

**Approach:**
- `__init__` (`dobby/executor.py:61-68`): add kw-only `context_policy: ContextPolicy | None = None`; store `self._context_policy` near the other `self.*` assignments (`:78-85`).
- `run_stream` (`:249-269`): declare loop-local `last_input_tokens: int | None = None`, `compaction_in_progress = False`, `last_compacted_at_tokens: int | None = None` right after `working_messages = list(messages)` (`:250`).
- Before the `chat` call (`:256`): if policy set, not in progress, and `_triggered(...)`, compute `send_messages = edit_context(working_messages, policy)[0]` and `yield ContextEditEvent(...)`. Pass `send_messages` (not `working_messages`) to `chat`. Trim leaves `working_messages` untouched (recompute next turn).
- At the `StreamEndEvent` handler (`:266`): set `last_input_tokens = event.usage.input_tokens` when `event.usage` is present; else apply the `usage=None` fallback (carry-forward or `len/4` estimate).
- `_triggered(last_input_tokens, policy, last_compacted_at_tokens)`: `last_input_tokens is not None and last_input_tokens >= policy.trigger_tokens` (and, for write-back modes, `!= last_compacted_at_tokens`).

**Patterns to follow:** optional-param style of `tools`/`output_type` (`dobby/executor.py:65-67`); existing `StreamEndEvent` handling at `:266`.

**Test scenarios** (extend the mock provider in `tests/test_parallel_tools.py:22-52`):
- Mock first `StreamEndEvent` with `usage=Usage(input_tokens=150_000,...)` and a `ContextPolicy(context_window=100_000, trigger_pct=0.8)` (trigger 80k) → a `ContextEditEvent` appears in the collected stream; later `chat` receives the trimmed list.
- Below threshold (`input_tokens=10_000`) → **no** `ContextEditEvent`; `chat` receives full history.
- `context_policy=None` → zero `ContextEditEvent`, behavior identical to today (regression guard).
- `usage=None` on the turn → trigger still engages via fallback (set a tiny `context_window` so the estimate crosses).
- First iteration (no prior usage) never trims.
- `_collect_results(executor, ...)`-style helper filters `ContextEditEvent` (mirror `test_parallel_tools.py:55-61`).

**Verification:** Full `uv run pytest` green; running an executor without a policy is byte-identical to current behavior.

---

### U4. `summarize` mode — write-back + Summarizer (reuse parent model)

**Goal:** Add the LLM-backed summarize path that replaces an old span with one `<summary>` turn written back into `working_messages`, computed once.

**Requirements:** KTD-2, KTD-3 (summarize/write-back), KTD-4 (guard/watermark), origin §5.

**Dependencies:** U1, U3.

**Files:**
- `dobby/context/summarize.py` (new — `summarize_context()` + `SUMMARIZE_PROMPT`), `dobby/context/__init__.py` (export)
- `dobby/executor.py` (summarize branch in the hook; set `last_compacted_at_tokens`)
- `tests/test_context_compaction.py` (summarize portion)

**Approach:**
- `summarize_context(messages, policy, llm) -> AppliedEdit | None`: select the clearable span (older than `keep_last_n`, whole turns, never in-flight), concatenate its text, call `await llm.chat(span, system_prompt=SUMMARIZE_PROMPT, stream=False, tools=None)` (returns a `StreamEndEvent` **value**), extract `summary_text` from `TextPart`s, and **replace** the span in `messages` (the live `working_messages` list) with one new `UserMessagePart([TextPart("<summary>...</summary>")])` — new instances only. Stash replaced originals on `AppliedEdit.replaced_originals`.
- Executor hook: when `policy.mode == "summarize"`, set `compaction_in_progress=True`, `await summarize_context(working_messages, ...)`, set `last_compacted_at_tokens=last_input_tokens`, `yield ContextEditEvent`, reset the flag. `send_messages = working_messages` (write-back is already applied).
- `SUMMARIZE_PROMPT`: "Compress these tool interactions into a concise factual digest; preserve IDs, values, decisions, file paths." (constant in `summarize.py`).

**Patterns to follow:** non-stream `chat` contract (`dobby/providers/base.py:141-151`); dataclass write-back rules from KTD-3.

**Technical design:** *Directional.* `select span → llm.chat(stream=False) → build <summary> turn → splice into list → AppliedEdit`.

**Test scenarios:**
- The mock provider branches on `kwargs.get("stream")`: streaming agent calls return an async generator; the summarizer's `stream=False` call returns a `StreamEndEvent` value with a `TextPart` summary. Assert the summarizer was called with `stream=False`.
- After summarize, `working_messages` contains the synthetic `<summary>` `UserMessagePart` and the old span is gone; recent `keep_last_n` turns remain verbatim.
- **Computed once:** with the watermark, a second iteration at the same `last_input_tokens` does **not** re-summarize (no second `stream=False` call).
- **Non-destructive to caller:** the original `messages` argument's objects are unchanged (write-back replaced list entries with new instances).
- In-flight pair is never folded into the summary.
- `AppliedEdit.replaced_originals` holds the pre-summary parts.

**Verification:** `uv run pytest` green; summarize issues exactly one extra model call per growth episode.

---

### U5. `CompactContextTool` + `edits_context` flag (agent-invoked)

**Goal:** Let the agent compact its own context on demand via a tool, routed through the same summarize machinery, coordinated by the guard.

**Requirements:** KTD-4 (guard), origin §5 (the "reasoning tool").

**Dependencies:** U4.

**Files:**
- `dobby/tools/tool.py` (add `edits_context: ClassVar[bool] = False` after `sequential`, `:76`; validate in `__init_subclass__`, `:86`)
- `dobby/tools/compact.py` (new — `CompactContextTool`), `dobby/tools/__init__.py` (export)
- `dobby/executor.py` (new branch mirroring the `terminal` branch — categorize `:308`, execute `:378-399`)
- `tests/test_compact_tool.py` (new)

**Approach:**
- `edits_context: ClassVar[bool] = False` on `Tool`, validated like the other flags in `__init_subclass__`.
- `CompactContextTool(Tool)` sets `edits_context = True`; its `__call__` accepts the agent's reasoning (e.g. `instructions: str`, optional `keep_last_n: int`) and returns a small directive (not a normal string result).
- Executor: when an `edits_context` tool fires, route through `summarize_context(...)` (force summarize), using the tool's `instructions` to augment `SUMMARIZE_PROMPT`; set `compaction_in_progress` + `last_compacted_at_tokens`; `yield ContextEditEvent`. The full pre-compaction transcript stays in the audit stash.
- Guard ensures a same-iteration auto-trigger and a tool call cannot double-compact.

**Patterns to follow:** ClassVar flags + `__init_subclass__` validation (`dobby/tools/tool.py:67-86`); the `terminal` special-branch in the loop (`dobby/executor.py:308,378-399`).

**Test scenarios** (`tests/test_compact_tool.py`):
- Pure-dataclass flag test (no LLM): `CompactContextTool.edits_context is True`; a normal tool defaults `False` (mirror `tests/test_terminal_tools.py:10-52`).
- Subclass declaring a non-bool `edits_context` raises in `__init_subclass__`.
- Invoking `CompactContextTool` in a run triggers `summarize_context` (mock `stream=False`) and emits a `ContextEditEvent`.
- **Guard:** auto-trigger + `CompactContextTool` in the same iteration produces exactly one compaction (one `ContextEditEvent`, one `stream=False` call).
- The agent `instructions` reach the summarizer prompt (assert the summarizer call's `system_prompt` contains the instruction text).

**Verification:** `uv run pytest tests/test_compact_tool.py` green; the flag is part of the public `Tool` contract.

---

### U6. Public exports + example

**Goal:** Expose the new surface and ship a runnable demo on OpenAI + Gemini.

**Requirements:** origin §6; KTD-1 (testable trigger).

**Dependencies:** U1–U5.

**Files:**
- `dobby/__init__.py` (export `ContextPolicy`, `ContextEditEvent`, `CompactContextTool`; fix the stale module docstring)
- `dobby/context/__init__.py`, `dobby/tools/__init__.py`, `dobby/types/__init__.py` (final export pass)
- `examples/context_editing.py` (new)
- `tests/test_context_compaction.py` (import-surface smoke)

**Approach:**
- `examples/context_editing.py`: `async def main()` + `asyncio.run(main())`, a token-heavy tool, a `ContextPolicy(context_window=2000, trigger_pct=0.5)` (so it fires quickly), a `match event:` loop with a `case ContextEditEvent():` arm printing `applied_edits`, run once on OpenAI and once on Gemini. Mirror `examples/web_search_agent.py` / `examples/parallel_tools_example.py` structure.
- Verify `from dobby import ContextPolicy, ContextEditEvent, CompactContextTool` works.

**Patterns to follow:** existing `examples/*.py` (async `main`, `match`/`case` over `StreamEvent`); `X as X` re-exports.

**Test scenarios:**
- Smoke: `from dobby import ContextPolicy, ContextEditEvent, CompactContextTool` imports without error.
- (Example is not asserted in CI — it needs live keys; note it as a manual/demo artifact.)

**Verification:** `uv run ruff check && uv run ruff format && uv run pytest` all green; example runs against live OpenAI and Gemini keys and prints `ContextEditEvent`s with a low `context_window`.

---

### U7. Cookbook-style demo/test notebook (final — after everything else)

**Goal:** A runnable, cell-by-cell notebook mirroring the Anthropic memory/context cookbook that exercises the *entire* feature (trim + summarize + `CompactContextTool` + the % trigger) on OpenAI and Gemini — the interactive way to **test compaction end-to-end** and the demo artifact.

**Requirements:** origin §6 (mirror the Anthropic cookbook); KTD-1 (small `context_window` makes it trigger fast and observably).

**Dependencies:** U1, U2, U3, U4, U5, U6 — everything must be built and exported first (this is the last unit).

**Files:**
- `examples/context_compaction_cookbook.ipynb` (new)

**Approach:** Author cell-by-cell, mirroring the cookbook structure and reusing U6's logic:
1. **Intro (markdown):** what context bloat is; dobby's answer (one `ContextPolicy`, OpenAI + Gemini, client-side).
2. **Setup:** load `.env` (OpenAI + Gemini keys), imports, `PROVIDERS = ["openai", "gemini"]`.
3. **Token-heavy tool:** a tool returning large synthetic payloads (so context grows fast).
4. **Policy:** `ContextPolicy(context_window=2000, trigger_pct=0.5, mode="trim")` — small window so compaction fires within a few turns; explain each knob.
5. **Run a long tool-loop (trim):** per provider, drive `run_stream`, collect every `StreamEndEvent.usage.input_tokens` and every `ContextEditEvent`.
6. **Tokens before/after table:** per-turn `input_tokens`, marking the turns where a `ContextEditEvent` fired (the drop is the money shot) — same code, both providers side by side.
7. **Show the placeholder:** print one trimmed `ToolResultPart` to prove clearing is non-silent.
8. **Summarize mode:** rerun with `mode="summarize"`; show the synthetic `<summary>` turn and that it computes once (watermark).
9. **`CompactContextTool` cell:** register it; show the agent compacting on demand mid-run + the resulting `ContextEditEvent`.
10. **Gemini correctness callout (markdown):** in-flight pairs/signatures are never trimmed — why Gemini doesn't 400.
11. **Conclusion.**

**Patterns to follow:** ruff already lints notebooks (`pyproject.toml` `extend-include = ["*.ipynb"]`); reuse the `match event:` / `case ContextEditEvent():` arms from `examples/context_editing.py` (U6); mirror the Anthropic memory cookbook section order.

**Test scenarios:** `Test expectation: none -- demo/test notebook requiring live API keys; it IS the manual end-to-end harness, not unit-tested in CI.`

**Verification:** notebook runs top-to-bottom on both OpenAI and Gemini; `input_tokens` visibly drops after each `ContextEditEvent`; the trim placeholder and the `<summary>` turn are both shown; `CompactContextTool` fires; the run completes with no provider pairing/signature errors.

---

## System-Wide Impact

- **`AgentExecutor` public API:** one new optional kw-only param (`context_policy`); additive, non-breaking.
- **`StreamEvent` consumers:** a new `ContextEditEvent` variant. Existing `match`/`isinstance` consumers are unaffected (they ignore unknown variants); consumers wanting compaction visibility add a case.
- **`Tool` contract:** new `edits_context` ClassVar (defaults `False`); existing tools unaffected.
- **Providers:** **no adapter changes** — compaction operates entirely on dobby's `MessagePart` list before `chat`. (This is what keeps the feature in scope and OpenAI/Gemini-symmetric.)
- **Cost/latency:** trim adds none; summarize adds one model round-trip per growth episode (write-back makes it once, not per turn).

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Provider prompt-cache invalidation** — mutating the prefix busts OpenAI/Gemini caching from that point | Higher token cost on the tail | Watermark fires compaction once per growth episode (batched), not every turn; document the trade-off |
| **`usage=None`** disables the trigger on long runs (Gemini) | Compaction silently never fires | Carry-forward last known; `len/4` fallback estimate (KTD-1); test covers it |
| **In-flight pair trimmed/orphaned** → provider 400 | Run crashes | Whole-turn granularity + never touch the current/unanswered pair (U2/U4 invariants + tests) |
| **In-place dataclass mutation** corrupts caller `messages` | Audit/replay + caller state corruption | Build new instances only; object-identity test (U2/U4) |
| **Summary quality loss** (load-bearing fact dropped) | Agent forgets context | Keep last `keep_last_n` verbatim; stash originals on `AppliedEdit`; `summarize` is opt-in (default `trim`) |
| **Pre-existing `StreamEvent` union malformation** | `TypeAdapter(StreamEvent)` invalid (not used today) | Out of scope; runtime uses `isinstance`; deferred proper fix (KTD-6) |

---

## Verification

- `uv run ruff check` and `uv run ruff format --check` clean (line-length 99, Google docstrings).
- `uv run pytest` green, including the new `tests/test_context_compaction.py` and `tests/test_compact_tool.py`.
- Regression: an `AgentExecutor` with `context_policy=None` behaves identically to current `main`.
- Manual: `examples/context_editing.py` against live OpenAI + Gemini keys prints `ContextEditEvent`s when a small `context_window` is set, and the run completes without provider pairing/signature errors.
- Non-destructive invariant proven by object-identity assertions in trim and summarize tests.

---

## Scope Boundaries

### Deferred to Follow-Up Work
- **Configurable / cheaper summarizer model** (`summarizer_llm: Provider | None`) — reuse parent for now (user decision). Revisit if cost becomes the headline.
- **`StreamEvent` union-validity fix** (add a `kind` discriminator across all event types, or drop `Field(discriminator=...)`) — pre-existing, touches every event type; land standalone.
- **OpenAI `truncation="auto"` synchronous backstop** — needs adapter kwargs-forwarding surgery.
- **Provider `count_tokens` precision** for the trigger (replace the one-turn-lagged `usage.input_tokens`).
- **A model-window lookup table** ("model repository") so `context_window` need not be supplied per policy.

### Outside this product's identity (origin)
- Anthropic-native `context_management` passthrough / building the Anthropic provider.
- `MemoryTool` + persist-before-clear (GitHub issue #8).
