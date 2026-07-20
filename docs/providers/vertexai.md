# Vertex AI Provider

The `VertexAIProvider` speaks Vertex AI's OpenAI-compatible Chat Completions API. It serves **both** of the surfaces Google exposes there:

| Endpoint type | When | Configuration |
|---|---|---|
| **Model Garden / MaaS** | Publisher models: Llama, DeepSeek, Qwen, gpt-oss, … | `model="meta/llama-3.3-70b-instruct-maas"` (default) |
| **Self-deployed endpoint** | A model you deployed yourself, addressed by endpoint id | `endpoint_id="5464397967697903616"` |

Model ids are forwarded verbatim — there is no allow-list and no family validation. Native Gemini and Claude ids are servable through this same endpoint, though through a cruder path than a dedicated native client would take (no thought-signature handling, coarser finish-reason mapping).

Raw non-OpenAI-compatible prediction routes (`:predict`, `:rawPredict`, `:streamRawPredict`) remain out of scope — this provider only speaks Chat Completions. Mistral-on-Vertex uses `:rawPredict` and is therefore not supported.

## Initialization

```python
from dobby.providers import VertexAIProvider

provider = VertexAIProvider(
    model="meta/llama-3.1-405b-instruct-maas",
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
| `model`       | `str \| None`           | `None`          | Publisher-qualified model id, forwarded verbatim. **Required for Model Garden.** With `endpoint_id` it is an optional display label, never sent on the wire. |
| `endpoint_id` | `str \| None`           | `None`          | Keyword-only. Self-deployed endpoint id. Switches routing and stops sending `model`.                                            |
| `endpoint_host` | `str \| None`         | `None`          | Keyword-only. Dedicated endpoint DNS. Only valid with `endpoint_id`.                                                            |
| `api_version` | `str`                   | `"v1"`          | Keyword-only. Path version segment. Google registers both `v1` and `v1beta1`; use `v1beta1` for dedicated endpoints and Gemini preview fields. |
| `project`     | `str \| None`           | `None`          | GCP project ID. Derived from the resolved credentials or from ADC when omitted. Raises `ValueError` if it cannot be determined. |
| `location`    | `str`                   | `"us-central1"` | GCP location. Forms the endpoint host and path.                                                                                 |
| `credentials` | `Credentials \| None`   | `None`          | Pre-built `google.auth.credentials.Credentials`. Takes precedence over every other auth source.                                 |
| `scopes`      | `Sequence[str] \| None` | `None`          | OAuth scopes. **Ignored when `credentials` is passed.** Required for both service-account paths — see [Scopes](#scopes).        |
| `max_retries` | `int`                   | `3`             | Retry attempts for transient errors.                                                                                            |

### Authentication

No API key. Auth resolves in this order, first match wins:

| #   | Source                                | Trigger                                                          |
| --- | ------------------------------------- | ---------------------------------------------------------------- |
| 1   | `credentials=`                        | A `google.auth.credentials.Credentials` object passed explicitly |
| 2   | `GOOGLE_APPLICATION_CREDENTIALS_JSON` | Env var holding a service-account key's **contents**             |
| 3   | `google.auth.default()`               | Application Default Credentials (ADC)                            |

The bearer token refreshes transparently on every request via the OpenAI SDK's native async-callable `api_key` hook, so both `provider.chat(...)` and direct use of `provider.client.chat.completions.create(...)` always get a fresh token. A long-lived provider keeps working past the ~1 hour token lifetime with no manual refresh.

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
print(result.stop_reason)  # "end_turn" | "tool_use" | "max_tokens" | "content_filter"
print(result.usage)        # Usage(input_tokens=5, output_tokens=10, ...)
```

### Streaming

```python
async for event in await provider.chat(messages, stream=True):
    match event.type:
        case "stream_start":
            print(f"Model: {event.model}")
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
| `tools`         | `list[Any] \| None`     | Tool definitions, already formatted via `to_vertexai_tool()` |
| `model`         | `str \| None`           | Per-call model override (falls back to the instance model)   |

---

## Stream Events

| Event              | Description                                                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `StreamStartEvent` | Stream started, includes model ID                                                                                                                                                           |
| `TextDeltaEvent`   | Text chunk with `delta` field                                                                                                                                                               |
| `ToolUseEvent`     | Tool call with `id`, `name`, `inputs`                                                                                                                                                       |
| `StreamEndEvent`   | Stream finished, includes `parts`, `usage`                                                                                                                                                  |
| `StreamErrorEvent` | Error occurred, including when the streaming tool-call accumulator's bounds are exceeded (a self-deployed/third-party Model Garden container is a less-trusted boundary than native OpenAI) |

Reasoning events are not emitted — Model Garden MaaS models targeted by this provider (Llama and OpenAI-compatible self-deployed containers) have no reasoning/thinking channel analogous to Claude's or OpenAI's o-series.

---

## Message Conversion

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

---

## Scope and design principles

This provider forwards requests to Vertex faithfully and surfaces Vertex's own responses. Three rules follow from that, and they explain most of the boundaries below:

1. **No stale client-side validation.** Model ids, region availability, and parameter support are Google's to police. A hardcoded allowlist goes out of date the week Google adds a model, and then rejects something the API would have served.
2. **No duplicating server-side policy.** If Vertex would return an error, we let it — a clear server error beats a guessed client-side one.
3. **Backend-dependent behavior is documented as best-effort, never normalized.** On a self-deployed endpoint the serving container decides what comes back. Inventing values to paper over that would be lying about what happened.

## Known Limitations

These are intentional scope boundaries, not missing features.

**Not implemented**

- **Only Chat Completions.** `:predict`, `:rawPredict`, and `:streamRawPredict` are separate wire formats and are out of scope. Mistral-on-Vertex uses `:rawPredict` and is therefore unreachable, as are custom containers exposing only those routes. Chat Completions on a self-deployed endpoint requires a container that implements it — Google's prebuilt vLLM and HF TGI containers do.
- **`endpoint_host` is not looked up for you.** Reading an endpoint's `dedicatedEndpointDns` would mean depending on the Vertex admin API, making a network call at construction, and requiring `aiplatform.endpoints.get` — permission that inference alone does not need. Every comparable SDK also makes the caller supply it. When a deployed endpoint 404s or fails to connect over the shared host, the raised error names `endpoint_host` and `dedicatedEndpointDns` explicitly.
- **Claude-on-Vertex and Gemini-on-Vertex are not first-class.** Both are reachable here by passing their model id, at the cost of the cruder path described above. `AnthropicProvider` and `GeminiProvider` target the direct Anthropic and Gemini Developer APIs.

**Deliberately not validated**

- **Region and model availability.** MaaS availability is region-gated and changes as Google's catalog changes (Llama has been `us-central1`-only, for example). This provider does not check the combination — Vertex's own error is authoritative and always current. Check availability in Google's Model Garden documentation.
- **Parameter support.** Google's documented rule is that unsupported parameters are ignored rather than rejected, and support varies per model for third-party models. We forward what you pass.

**Backend-dependent, best-effort**

- **Streaming `usage`.** `stream_options` / `include_usage` is absent from Google's documented parameter list. Usage is emitted when the backend supplies it and omitted otherwise. On self-deployed endpoints this is entirely the container's behavior (vLLM and TGI differ). Never assume `StreamEndEvent.usage` is populated.
- **Error shapes.** Because the route is typed `GoogleApiHttpBody`, a failure may arrive as a Google API envelope or as an OpenAI-style error depending on where it occurred. Both are mapped to the same dobby error types; the original is preserved on `__cause__`.

### Global location

Setting `location="global"` targets the bare `aiplatform.googleapis.com` host (no region prefix), per Google's documentation. Global quotas and model capabilities differ from regional endpoints, and the `constraints/gcp.restrictEndpointUsage` org policy can block it entirely.
