# Providers

Providers are the interface between Dobby and LLM APIs. Each provider handles message conversion, streaming, and tool formatting for its specific API.

## Supported Providers

| Provider | Status | API |
|----------|--------|-----|
| [OpenAI](./openai.md) | ✅ Stable | Responses API |
| Azure OpenAI | ✅ Stable | Responses API |
| Anthropic | ✅ Stable | Messages API (direct and Azure AI Foundry) |
| Gemini | ✅ Stable | `google-genai` (Developer API) |
| [Vertex AI](./vertexai.md) | ✅ Stable | Chat Completions API (OpenAI-compatible; Model Garden) |

## Common Interface

All providers implement the `chat()` method:

```python
async def chat(
    messages: Iterable[MessagePart],
    *,
    stream: bool = False,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    tools: list[ToolParam] | None = None,
    reasoning_effort: str | None = None,
) -> StreamEndEvent | AsyncIterator[StreamEvent]
```

## Usage

```python
from dobby import OpenAIProvider
from dobby.types import UserMessagePart, TextPart

provider = OpenAIProvider(model="gpt-4o")

messages = [UserMessagePart(parts=[TextPart(text="Hello!")])]

# Non-streaming
result = await provider.chat(messages, stream=False)
print(result.parts[0].text)

# Streaming
async for event in await provider.chat(messages, stream=True):
    if event.type == "text-delta":
        print(event.delta, end="")
```

## Next

- [OpenAI Provider](./openai.md) - Detailed OpenAI/Azure configuration
- [Vertex AI Provider](./vertexai.md) - Detailed Vertex AI Model Garden configuration
