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

### Authentication

Auth uses Application Default Credentials (ADC) by default — no API key is needed. The bearer token is refreshed transparently on every request via the OpenAI SDK's native async-callable `api_key` hook, so both `provider.chat(...)` and direct use of `provider.client.chat.completions.create(...)` always get a fresh token.

You can also supply a pre-built credentials object instead of relying on ADC resolution:

```python
import google.auth

credentials, _ = google.auth.default()

provider = VertexAIProvider(
    model="meta/llama-3.1-405b-instruct-maas",
    project="my-gcp-project",
    credentials=credentials,
)
```

### Least-privilege scopes

When `credentials` is not supplied, the optional `scopes` parameter is forwarded to `google.auth.default(scopes=scopes)`:

```python
provider = VertexAIProvider(
    model="meta/llama-3.1-405b-instruct-maas",
    project="my-gcp-project",
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
```

Left unset, `scopes` defaults to `None`, which means google-auth's own default resolution — typically the broad `cloud-platform` scope for most ADC sources. A leaked broad-scope token has a larger blast radius than a Vertex-only one, so operators whose ADC source supports narrower scopes (e.g. a dedicated service account) should pass a tighter `scopes` list here for least-privilege.

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `Iterable[MessagePart]` | Conversation history |
| `stream` | `bool` | Enable streaming (default: False) |
| `system_prompt` | `str \| None` | System instructions |
| `temperature` | `float` | Randomness 0.0-2.0 (default: 0.0) |
| `tools` | `list[Any] \| None` | Tool definitions, already formatted via `to_vertexai_tool()` |
| `model` | `str \| None` | Per-call model override (falls back to the instance model) |

---

## Stream Events

| Event | Description |
|-------|-------------|
| `StreamStartEvent` | Stream started, includes model ID |
| `TextDeltaEvent` | Text chunk with `delta` field |
| `ToolUseEvent` | Tool call with `id`, `name`, `inputs` |
| `StreamEndEvent` | Stream finished, includes `parts`, `usage` |
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
