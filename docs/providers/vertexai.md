# Vertex AI Provider

The `VertexAIProvider` reaches models hosted on Vertex AI with one set of Google credentials. It routes each model to the API its publisher speaks:

| Instance | Route | Wire |
|---|---|---|
| `model="google/gemini-*"` | Native Gemini | `google-genai` `Client(vertexai=True)`, `generateContent` |
| `model="anthropic/*"` | Native Anthropic | `AsyncAnthropicVertex`, Messages API (`rawPredict`) |
| Any other publisher, e.g. `meta/…`, `openai/…`, `google/gemma-*`, or an unprefixed id | OpenAI-compatible | Chat Completions on `endpoints/openapi` (Model Garden / MaaS) |
| `endpoint_id=` set | OpenAI-compatible | Chat Completions on `endpoints/{endpoint_id}` (self-deployed) |
| `routing="openai"` | OpenAI-compatible | Chat Completions for every model |

The native routes reuse `GeminiProvider`'s and `AnthropicProvider`'s converters, streaming and error translation. They keep each vendor's own features, such as Claude thinking blocks with their signatures and Gemini thought signatures on function calls, without the second translation Google performs behind the Chat Completions endpoint. Diagrams of the routing, history batching and ID rules are in [Vertex routing](../vertex-routing-options.md).

The router reads the publisher only. There is no allow-list and no family validation: a model id that matches no native row stays on the Chat Completions route, which serves it as before.

> **Model ids must be publisher-qualified** on the Model Garden path — `anthropic/claude-sonnet-4-5`, not `claude-sonnet-4-5`. Vertex rejects a bare id before it ever looks at the model:
>
> ```
> 400 INVALID_ARGUMENT — Malformed publisher model (`model`: 'claude-sonnet-4-5')
> for the 'openapi' request endpoint ID; expected '<publisher>/<model>'
> ```
>
> This applies to every publisher, including ones Google's own model cards list unqualified (gpt-oss is documented as `gpt-oss-20b-maas` but must be sent as `openai/gpt-oss-20b-maas`). Verified against the live API. Self-deployed endpoints are exempt — they ignore the field entirely. The prefix is also what picks a native route, so a bare `claude-sonnet-4-5` goes to Chat Completions and fails there.

Raw prediction routes (`:predict`, `:rawPredict`, `:streamRawPredict`) are not exposed for arbitrary models. Only the native Claude route uses `rawPredict`, through Anthropic's own SDK. Mistral-on-Vertex and custom containers that only expose those routes are not supported.

## Routing

The route is fixed at construction from the instance model:

```python
from dobby.providers import VertexAIProvider

gemini = VertexAIProvider(model="google/gemini-2.5-flash", project="my-gcp-project")       # native Gemini
claude = VertexAIProvider(model="anthropic/claude-sonnet-4-5", project="my-gcp-project")   # native Anthropic
llama = VertexAIProvider(model="meta/llama-3.3-70b-instruct-maas", project="my-gcp-project")  # Chat Completions

# Keep every model on Chat Completions, as in 0.2.x
pinned = VertexAIProvider(
    model="google/gemini-2.5-flash", project="my-gcp-project", routing="openai"
)
```

- `routing` is keyword-only and takes `"native"` (default) or `"openai"`. Anything else raises `ValueError`. It is ignored with `endpoint_id`, which always uses Chat Completions.
- On a native route the publisher prefix is removed before the SDK sees the id (`google/gemini-2.5-flash` → `gemini-2.5-flash`). `provider.model` still returns the id you passed.
- `provider.name` is `"vertexai"` on every route, and so is `ProviderError.provider`.
- `provider.client` is always the `AsyncOpenAI` client for the Chat Completions route. The native routes run on their own SDK clients, built from the same credentials, project and location.
- The native SDK clients are built with their own retries off, so `max_retries` (Dobby's `with_retries`) is the only retry layer on every route.

**A per-call `model=` is sent over the instance's route.** A native route removes only its own publisher prefix; the Chat Completions route forwards the id verbatim. An override within the same route works. An override for another route is not re-routed: it reaches the instance's backend and fails with that backend's own error. Build a second provider to talk to another publisher.

## Initialization

```python
from dobby.providers import VertexAIProvider

provider = VertexAIProvider(
    model="google/gemini-2.5-flash",
    project="my-gcp-project",
    location="us-central1",  # default
)
```

## Self-deployed endpoints

Deploy a model from Model Garden (or your own container) and address it by endpoint id:

```python
provider = VertexAIProvider(
    endpoint_id="5464397967697903616",
    project="my-gcp-project",
    location="us-central1",
)
```

Three things change versus Model Garden, all handled for you:

1. **Routing** — `endpoints/{endpoint_id}` instead of the shared `endpoints/openapi`.
2. **The body's `model` field is dropped.** The endpoint selects the model, so Vertex ignores it. The provider sends `""`, matching Google's own OpenAI-SDK samples. A per-call `model=` override therefore changes only the reported `StreamEndEvent.model`, not what is served.
3. **`model` becomes optional.** It is a display label for `provider.model` and `StreamEndEvent.model`. Omit it and it defaults to `endpoint-{endpoint_id}`; pass one for nicer logs.

### Dedicated endpoints

Once an endpoint has `dedicatedEndpointEnabled`, **the shared regional DNS stops serving it**, so you must pass its host:

```python
provider = VertexAIProvider(
    endpoint_id="5464397967697903616",
    endpoint_host="5464397967697903616.us-central1-987654321.prediction.vertexai.goog",
    project="my-gcp-project",
    api_version="v1beta1",
)
```

Read `endpoint_host` from the Endpoint resource's `dedicatedEndpointDns` field — **do not construct it.** The uid segment is usually the project number but is documented as possibly "a random number or a string" (for example `fasttryout`). A leading `https://` is stripped if present, since the API returns the value bare while its schema documents a scheme.

> **Container support.** Chat Completions on a self-deployed endpoint requires a serving container that implements it. Google's prebuilt vLLM and HF TGI containers do; an arbitrary custom container may only support `:rawPredict`, which this provider does not speak.

## Initialization reference

### Constructor parameters

| Parameter     | Type                    | Default         | Description                                                                                                                     |
| ------------- | ----------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `model`       | `str \| None`           | `None`          | Publisher-qualified model id. Its publisher picks the route. Native routes send it without the publisher prefix; Chat Completions forwards it verbatim. **Required unless `endpoint_id` is set.** With `endpoint_id` it is an optional display label, never sent on the wire. |
| `routing`     | `"native" \| "openai"`  | `"native"`      | Keyword-only. `"native"` routes `google/gemini*` and `anthropic/*` to their native APIs; `"openai"` keeps every model on Chat Completions. Ignored with `endpoint_id`. See [Routing](#routing). |
| `endpoint_id` | `str \| None`           | `None`          | Keyword-only. Self-deployed endpoint id. Switches routing and stops sending `model`.                                            |
| `endpoint_host` | `str \| None`         | `None`          | Keyword-only. Dedicated endpoint DNS. Only valid with `endpoint_id`.                                                            |
| `api_version` | `str`                   | `"v1"`          | Keyword-only. Path version segment for the Chat Completions route. Google registers both `v1` and `v1beta1`; use `v1beta1` for dedicated endpoints and Gemini preview fields. The native SDKs pick their own version. |
| `project`     | `str \| None`           | `None`          | GCP project ID. Derived from the resolved credentials or from ADC when omitted. Raises `ValueError` if it cannot be determined. |
| `location`    | `str`                   | `"us-central1"` | GCP location, shared by every route. Forms the endpoint host and path. See [Regions](#regions).                                  |
| `credentials` | `Credentials \| None`   | `None`          | Pre-built `google.auth.credentials.Credentials`. Takes precedence over every other auth source.                                 |
| `scopes`      | `Sequence[str] \| None` | `None`          | OAuth scopes. **Ignored when `credentials` is passed.** Required for both service-account paths — see [Scopes](#scopes).        |
| `max_retries` | `int`                   | `3`             | Retry attempts for transient errors. The native SDK clients' own retries are off, so this is the only retry layer.              |

### Authentication

No API key. Auth resolves in this order, first match wins:

| #   | Source                                | Trigger                                                          |
| --- | ------------------------------------- | ---------------------------------------------------------------- |
| 1   | `credentials=`                        | A `google.auth.credentials.Credentials` object passed explicitly |
| 2   | `GOOGLE_APPLICATION_CREDENTIALS_JSON` | Env var holding a service-account key's **contents**             |
| 3   | `google.auth.default()`               | Application Default Credentials (ADC)                            |

Every route uses the same resolved credentials. On the Chat Completions route the bearer token refreshes transparently on every request via the OpenAI SDK's native async-callable `api_key` hook, so both `provider.chat(...)` and direct use of `provider.client.chat.completions.create(...)` always get a fresh token. The native SDK clients refresh the same credentials object themselves. A long-lived provider keeps working past the ~1 hour token lifetime with no manual refresh.

`examples/vertexai_example.py` has all of the below as runnable code in one `build_provider()` function.

**Application Default Credentials.** Least setup, and the right default locally and on GCE / GKE / Cloud Run:

```python
provider = VertexAIProvider(model="meta/llama-3.1-405b-instruct-maas")
```

`project` is derived from ADC when ADC knows it. Pass it explicitly when it doesn't — typically user ADC where `gcloud auth application-default set-quota-project` was never run. Construction raises `ValueError` in that case rather than failing later at request time:

```python
provider = VertexAIProvider(
    model="meta/llama-3.1-405b-instruct-maas",
    project="my-gcp-project",
)
```

**Service-account key file.** Note the scopes go on the credentials object, not on the provider — `scopes=` is ignored once `credentials=` is passed:

```python
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    "/secure/path/to/key.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

provider = VertexAIProvider(
    model="meta/llama-3.1-405b-instruct-maas",
    credentials=credentials,  # project is read from the key
)
```

**Service-account key as a single env var.** Set `GOOGLE_APPLICATION_CREDENTIALS_JSON` to the key file's contents, not a path to one — this suits secret-manager injection with no key file on disk. Read automatically when `credentials=` is not passed. See the README's "Vertex AI Credentials Setup" for the full walkthrough:

```python
provider = VertexAIProvider(
    model="meta/llama-3.1-405b-instruct-maas",
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
```

**Impersonating another service account.** No key material on disk. Your ADC identity needs `roles/iam.serviceAccountTokenCreator` on the target. Impersonated credentials carry no project, so pass `project` explicitly:

```python
import google.auth
from google.auth import impersonated_credentials

source_credentials, _ = google.auth.default()

credentials = impersonated_credentials.Credentials(
    source_credentials=source_credentials,
    target_principal="vertex-caller@my-gcp-project.iam.gserviceaccount.com",
    target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

provider = VertexAIProvider(
    model="meta/llama-3.1-405b-instruct-maas",
    project="my-gcp-project",
    credentials=credentials,
)
```

### Scopes

`scopes` is consulted **only when `credentials` is not supplied**. Once you pass a pre-built credentials object, scope it at construction instead — the provider's `scopes=` argument is ignored on that path.

**`scopes` is required for both service-account paths** — a key loaded from `GOOGLE_APPLICATION_CREDENTIALS_JSON` or from `from_service_account_file` carries no implicit scope, and omitting it fails every request with:

```
invalid_scope: Invalid OAuth scope or ID token audience provided.
```

Left as `None`, google-auth resolves its own default, which for most ADC sources is the broad `cloud-platform` scope. A leaked broad-scope token reaches every Google Cloud API the identity can touch, so set it explicitly to make the blast radius visible:

```python
provider = VertexAIProvider(
    model="meta/llama-3.1-405b-instruct-maas",
    project="my-gcp-project",
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
```

`cloud-platform` is currently the only scope Vertex AI accepts, so this cannot be narrowed further today. Stating it explicitly is still worth it: it documents the blast radius at the call site and gives you one obvious place to tighten if that changes.

## Chat Methods

### Non-Streaming

```python
from dobby.types import UserMessagePart, TextPart

messages = [UserMessagePart(parts=[TextPart(text="Hello!")])]

result = await provider.chat(messages, stream=False)

print(result.parts)        # [TextPart(text="Hello! How can I help?")]
print(result.stop_reason)  # e.g. "end_turn" | "tool_use" | "max_tokens" | "content_filter"
print(result.usage)        # Usage(input_tokens=5, output_tokens=10, ...)
```

### Streaming

```python
async for event in await provider.chat(messages, stream=True):
    match event.type:
        case "stream_start":
            print(f"Model: {event.model}")
        case "reasoning_delta":  # native Claude route only
            print(event.delta, end="")
        case "text_delta":
            print(event.delta, end="")
        case "tool_use":
            print(f"Tool: {event.name}({event.inputs})")
        case "stream_end":
            print(f"\nTokens: {event.usage.total_tokens}")
        case "stream_error":
            print(f"Error: {event.error_message}")
```

---

## Parameters

| Parameter       | Type                    | Description                                                  |
| --------------- | ----------------------- | ------------------------------------------------------------ |
| `messages`      | `Iterable[MessagePart]` | Conversation history                                         |
| `stream`        | `bool`                  | Enable streaming (default: False)                            |
| `system_prompt` | `str \| None`           | System instructions                                          |
| `temperature`   | `float`                 | Randomness 0.0-2.0 (default: 0.0)                            |
| `tools`         | `list[Any] \| None`     | Tool definitions in the active route's format. Build them with `provider.format_tools(tools)` |
| `model`         | `str \| None`           | Per-call model override, sent over the instance's route (see [Routing](#routing)) |
| `**kwargs`      | `Any`                   | Passthrough params. See [Per-route parameters](#per-route-parameters) |

### Tool formatting

`provider.format_tools(tools)` returns tools in the format of the route this instance talks to: Gemini `Tool` objects on the Gemini route, Anthropic tool dicts on the Anthropic route, nested Chat Completions dicts otherwise.

```python
tools = provider.format_tools([MyTool()])
result = await provider.chat(messages, tools=tools)
```

`AgentExecutor("vertexai", provider, tools=[...])` calls it for you. `to_vertexai_tool()` builds the Chat Completions shape only, so its output does not fit a native route.

### Per-route parameters

`reasoning_effort` and the passthrough kwargs go to the adapter of the active route:

| | Gemini route | Claude route | Chat Completions route |
|---|---|---|---|
| `reasoning_effort` | `int` → `thinking_budget`, `str` → `thinking_level` | `int` → `thinking.budget_tokens`, with `temperature` forced to `1.0` as Anthropic requires. A `str` raises `TypeError`. | Forwarded as is |
| `max_tokens` | Sent as `max_output_tokens` | Sent as `max_tokens`; defaults to `8192` when omitted | Forwarded as is |
| Other kwargs | Forwarded to `GenerateContentConfig` (e.g. `top_p`, `top_k`, `stop_sequences`) | Forwarded to `messages.create()` (e.g. `top_p`, `top_k`, `stop_sequences`, `metadata`) | Forwarded to `chat.completions.create()` |

On a native route these Chat Completions kwargs raise `TypeError` before any network call, naming `routing="openai"`: `extra_body`, `stop`, `n`, `logprobs`, `top_logprobs`, `response_format`, `seed`, `frequency_penalty`, `presence_penalty`. Gemini would otherwise fail with a raw Pydantic error, and on Claude `extra_body` is a real SDK option that silently rewrites the request body. The check is by name, so it applies even where `GenerateContentConfig` has a field of the same name (`seed`, `frequency_penalty`, `presence_penalty`, `logprobs`). See [the kwarg mapping](#kwarg-mapping) for the native fields.

---

## Stream Events

| Event              | Description                                                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `StreamStartEvent` | Stream started, includes model ID                                                                                                                                                           |
| `ReasoningStartEvent`, `ReasoningDeltaEvent`, `ReasoningEndEvent` | Claude thinking, on the native Claude route only                                                                                         |
| `TextDeltaEvent`   | Text chunk with `delta` field                                                                                                                                                               |
| `ToolUseEvent`     | Tool call with `id`, `name`, `inputs`                                                                                                                                                       |
| `StreamEndEvent`   | Stream finished, includes `parts`, `usage`                                                                                                                                                  |
| `StreamErrorEvent` | Error occurred, including when the streaming tool-call accumulator's bounds are exceeded (a self-deployed/third-party Model Garden container is a less-trusted boundary than native OpenAI) |

Reasoning output depends on the route:

- **Claude route:** emits reasoning events. `StreamEndEvent.parts` holds one `ReasoningPart` per thinking block, each with its own signature, in the order Claude sent the blocks. A message that interleaves thinking and text can carry more than one `TextPart`.
- **Gemini route:** emits no reasoning events. Gemini thought signatures ride on each `ToolUsePart.metadata["signature"]` and are sent back with the call.
- **Chat Completions route:** emits no reasoning events. Gemini thought signatures returned on `tool_calls[].extra_content` are kept on `ToolUsePart.metadata["extra_content"]` and sent back unchanged.

Send `StreamEndEvent.parts` back whole as one assistant message. `AgentExecutor` does this for every tool turn; see [History shape](../executor.md#history-shape).

---

## Message Conversion

These helpers serve the Chat Completions route. On a native route the internal Gemini or Anthropic adapter converts messages itself.

Use `to_vertexai_messages()` to convert Dobby messages to Vertex's Chat Completions format:

```python
from dobby.providers.vertexai import to_vertexai_messages

vertexai_format = to_vertexai_messages(messages)
# Returns list[dict[str, Any]] (Chat Completions "messages" array)
```

Tool schemas are converted separately with `to_vertexai_tool()`, which reuses `Tool.to_openai_format()`'s schema-construction logic and re-nests the flat Responses API shape into Chat Completions' nested `{"type": "function", "function": {...}}` shape:

```python
from dobby.providers.vertexai import to_vertexai_tool

tool_schema = to_vertexai_tool(my_tool)
```

Prefer `provider.format_tools()`, which picks the right shape for the active route.

---

## Scope and design principles

This provider forwards requests to Vertex faithfully and surfaces Vertex's own responses. Three rules follow from that, and they explain most of the boundaries below:

1. **No stale client-side validation.** Model ids, region availability, and parameter support are Google's to police. A hardcoded allowlist goes out of date the week Google adds a model, and then rejects something the API would have served.
2. **No duplicating server-side policy.** If Vertex would return an error, we let it — a clear server error beats a guessed client-side one.
3. **Backend-dependent behavior is documented as best-effort, never normalized.** On a self-deployed endpoint the serving container decides what comes back. Inventing values to paper over that would be lying about what happened.

## Known Limitations

These are intentional scope boundaries, not missing features.

**Not implemented**

- **Three protocols only.** Native Gemini, native Anthropic, and Chat Completions. `:predict`, `:rawPredict`, and `:streamRawPredict` are not exposed for other models: Mistral-on-Vertex is unreachable, as are custom containers exposing only those routes. Chat Completions on a self-deployed endpoint requires a container that implements it — Google's prebuilt vLLM and HF TGI containers do.
- **`endpoint_host` is not looked up for you.** Reading an endpoint's `dedicatedEndpointDns` would mean depending on the Vertex admin API, making a network call at construction, and requiring `aiplatform.endpoints.get` — permission that inference alone does not need. Every comparable SDK also makes the caller supply it. When a deployed endpoint 404s or fails to connect over the shared host, the raised error names `endpoint_host` and `dedicatedEndpointDns` explicitly.
- **No public native Vertex classes.** Claude and Gemini on Vertex are reached through `VertexAIProvider`'s router only. `AnthropicProvider` and `GeminiProvider` target the direct Anthropic and Gemini Developer APIs.
- **One `location` per instance.** Every route shares it. Use one provider per region when Claude and Gemini live in different regions.

**Deliberately not validated**

- **Region and model availability.** MaaS availability is region-gated and changes as Google's catalog changes (Llama has been `us-central1`-only, for example). This provider does not check the combination — Vertex's own error is authoritative and always current. Check availability in Google's Model Garden documentation.
- **Parameter support.** Google's documented rule is that unsupported parameters are ignored rather than rejected, and support varies per model for third-party models. We forward what you pass.

**Backend-dependent, best-effort**

- **Streaming `usage`.** `stream_options` / `include_usage` is absent from Google's documented parameter list. Usage is emitted when the backend supplies it and omitted otherwise. On self-deployed endpoints this is entirely the container's behavior (vLLM and TGI differ). Never assume `StreamEndEvent.usage` is populated.
- **Error shapes.** On the Chat Completions route, which is typed `GoogleApiHttpBody`, a failure may arrive as a Google API envelope or as an OpenAI-style error depending on where it occurred. Both are mapped to the same dobby error types. On the native routes the `google-genai` or `anthropic` exception is mapped the same way. On every route the original is preserved on `__cause__`.

### Regions

Claude and Gemini are offered in different Vertex regions, and both change as Google's catalog changes. The provider does not check the combination: a model that is not served in `location` returns Vertex's own error. Check the vendor's Vertex model page for regions.

The Claude route builds `AsyncAnthropicVertex`, which reads `ANTHROPIC_VERTEX_BASE_URL` from the environment. When that variable is set, Claude requests go to its host instead of the Vertex host built from `location`. Unset it unless you mean to redirect them.

### Global location

Setting `location="global"` targets the bare `aiplatform.googleapis.com` host (no region prefix), per Google's documentation. Global quotas and model capabilities differ from regional endpoints, and the `constraints/gcp.restrictEndpointUsage` org policy can block it entirely.

## Upgrading from 0.2.x

In `0.2.x` every model went through Chat Completions. Now `google/gemini*` and `anthropic/*` models go to their native APIs by default. Self-deployed endpoints and every other publisher are unaffected.

### Keep the old route

One argument restores the `0.2.x` route for an instance:

```python
provider = VertexAIProvider(model="anthropic/claude-sonnet-4-5", project="my-gcp-project", routing="openai")
```

`AgentExecutor`'s history batching and the Gemini thought-signature replay are fixes, not routing changes, and apply on this route too.

### Kwarg mapping

The Chat-Completions-only kwargs raise `TypeError` on native routes. Use the native field instead, or pin `routing="openai"`:

| Chat Completions kwarg | Gemini route (`GenerateContentConfig` field) | Claude route (`messages.create` field) |
|---|---|---|
| `stop` | `stop_sequences` | `stop_sequences` |
| `n` | none: Dobby returns one candidate | none |
| `logprobs` | `response_logprobs` | none |
| `top_logprobs` | none: Gemini's own field is also called `logprobs`, and that name is rejected | none |
| `response_format` | `response_mime_type` with `response_schema` or `response_json_schema` | none: use a tool, or `AgentExecutor`'s `output_type` |
| `seed` | none: the name is rejected | none |
| `frequency_penalty` | none: the name is rejected | none |
| `presence_penalty` | none: the name is rejected | none |
| `extra_body` | pass `GenerateContentConfig` fields as kwargs | none: pass `messages.create` fields as kwargs |
| `reasoning_effort` (`str`) | `str` → `thinking_level`; `int` → `thinking_budget` | `int` only → `thinking.budget_tokens`; a `str` raises `TypeError` |

### Output differences

On the native routes the response comes from the vendor's own API, so for the same prompt:

- **Reasoning.** The Claude route emits reasoning events and `ReasoningPart`s, one per thinking block. Streaming can return more than one `TextPart`; join them instead of reading `parts[0]`.
- **Stop reasons.** Each vendor's mapping applies. Claude can return `stop_sequence`, `refusal` and `pause_turn`, which the Chat Completions route never does.
- **Usage.** Token counts come from the native response.
- **Tool-call IDs.** Gemini usually returns calls without IDs, so the Gemini route mints `call_{name}`, then `call_{name}_2`, `_3`, … for repeats. Claude returns its own `toolu_…` IDs.
- **Default output cap.** The Claude route sends `max_tokens=8192` when you pass none.
- **Errors.** Exception classes are unchanged: still the `ProviderError` family, reported as `vertexai`. Only `__cause__` changes, to a `google.genai` or `anthropic` exception.
- **Regions.** Each vendor's region list applies. See [Regions](#regions).

### Direct `chat()` callers

Build tools with `provider.format_tools(tools)` instead of `to_vertexai_tool()`. On a native route `to_vertexai_tool()` output is the wrong shape. `AgentExecutor` callers need no change.

### Per-call `model=`

`0.2.17` forwarded a per-call `model=` verbatim. On a native instance it is now sent over the instance's route with that route's prefix removed, and a model for another route fails with the backend's error. Build one provider per publisher instead of switching publishers per call.

### Resuming a conversation with thinking on

A Claude request with thinking enabled must replay the previous turn's thinking blocks. A history stored by `0.2.x` hosts in the split shape, without `ReasoningPart`s, can return a 400 on the native Claude route. Store `StreamEndEvent.parts` whole from now on (see [History shape](../executor.md#history-shape)), or resume old conversations with thinking off or with `routing="openai"`.
