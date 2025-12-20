# OpenAI Provider

The `OpenAIProvider` supports both OpenAI and Azure OpenAI using the Responses API.

## Initialization

### OpenAI

```python
from dobby import OpenAIProvider

provider = OpenAIProvider(
    model="gpt-4o",
    api_key="sk-...",  # Optional, uses OPENAI_API_KEY env var
)
```

### Azure OpenAI

```python
provider = OpenAIProvider(
    base_url="https://your-resource.openai.azure.com",
    azure_deployment_id="gpt-4o-deployment",
    api_key="your-azure-key",
    azure_api_version="2025-03-01-preview",
)
```

Azure is auto-detected when `base_url` contains "azure".

---

## Chat Methods

### Non-Streaming

```python
from dobby.types import UserMessagePart, TextPart

messages = [UserMessagePart(parts=[TextPart(text="Hello!")])]

result = await provider.chat(messages, stream=False)

print(result.parts)        # [TextPart(text="Hello! How can I help?")]
print(result.stop_reason)  # "end_turn" | "tool_use"
print(result.usage)        # Usage(input_tokens=5, output_tokens=10, ...)
```

### Streaming

```python
async for event in await provider.chat(messages, stream=True):
    match event.type:
        case "stream-start":
            print(f"Model: {event.model}")
        case "text-delta":
            print(event.delta, end="")
        case "reasoning-delta":
            print(f"[Thinking: {event.delta}]")
        case "tool-use":
            print(f"Tool: {event.name}({event.inputs})")
        case "stream-end":
            print(f"\nTokens: {event.usage.total_tokens}")
```

---

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `Iterable[MessagePart]` | Conversation history |
| `stream` | `bool` | Enable streaming (default: False) |
| `system_prompt` | `str \| None` | System instructions |
| `temperature` | `float` | Randomness 0.0-2.0 (default: 0.0) |
| `tools` | `list[ToolParam]` | Tool definitions |
| `reasoning_effort` | `str` | "low", "medium", "high" (o1 models) |

---

## Reasoning Models

For o1/o3 models, enable reasoning with `reasoning_effort`:

```python
async for event in await provider.chat(
    messages,
    stream=True,
    reasoning_effort="medium",  # "low" | "medium" | "high"
):
    match event.type:
        case "reasoning-start":
            print("[Thinking...]", end="")
        case "reasoning-delta":
            print(event.delta, end="")
        case "reasoning-end":
            print()
        case "text-delta":
            print(event.delta, end="")
```

---

## Stream Events

| Event | Description |
|-------|-------------|
| `StreamStartEvent` | Stream started, includes model ID |
| `TextDeltaEvent` | Text chunk with `delta` field |
| `ReasoningStartEvent` | Reasoning started |
| `ReasoningDeltaEvent` | Reasoning chunk |
| `ReasoningEndEvent` | Reasoning finished |
| `ToolUseEvent` | Tool call with `id`, `name`, `inputs` |
| `StreamEndEvent` | Stream finished, includes `parts`, `usage` |
| `StreamErrorEvent` | Error occurred |

---

## Message Conversion

Use `to_openai_messages()` to convert Dobby messages to OpenAI format:

```python
from dobby.providers.openai import to_openai_messages

openai_format = to_openai_messages(messages)
# Returns ResponseInputParam (list of OpenAI message dicts)
```
