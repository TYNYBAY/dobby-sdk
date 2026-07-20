# Vertex AI Provider

The `VertexAIProvider` targets Google Cloud Vertex AI's **Model Garden** catalog (Llama, self-deployed containers, and any other serving container that speaks OpenAI's Chat Completions wire format) through Vertex's OpenAI-compatible Model-as-a-Service (MaaS) endpoint.

Model ids are forwarded verbatim — there is no allow-list and no family validation. Native Gemini and Claude ids are servable through this same endpoint, though through a cruder path than a dedicated native client would take (no thought-signature handling, coarser finish-reason mapping).

Mistral-on-Vertex and raw non-OpenAI-compatible custom endpoints (`rawPredict`/`streamRawPredict`) remain out of scope for this provider — it only speaks Chat Completions.

## Initialization

```python
from dobby.providers import VertexAIProvider

provider = VertexAIProvider(
    model="meta/llama-3.1-405b-instruct-maas",
    project="my-gcp-project",
    location="us-central1",  # default
)
```

### Constructor parameters

| Parameter     | Type                    | Default         | Description                                                                                                                     |
| ------------- | ----------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `model`       | `str`                   | required        | Publisher-qualified model id, forwarded verbatim. Rejects `None`, empty, and whitespace-only ids at construction.               |
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

## Known Limitations

- Region/model availability for MaaS models is limited by Google (e.g. Llama is currently `us-central1`-only). This provider does not validate region/model combinations — check current availability in Google's Model Garden documentation.
- Mistral-on-Vertex is not supported (it requires a different wire format).
- Claude-on-Vertex and Gemini-on-Vertex have no first-class support in this SDK. Both are reachable through this provider's OpenAI-compatible endpoint by passing their model id, at the cost of the cruder path described above. `AnthropicProvider` and `GeminiProvider` target the direct Anthropic and Gemini Developer APIs only.
