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

## Mid-Stream Errors

OpenAI and Gemini wrap stream iteration with the same error translation used at request start. A mid-stream SDK exception becomes a Dobby provider exception rather than an untyped SDK error. Anthropic and Vertex AI use the same wrap.

The mapped types differ by provider:

- **OpenAI:** rate limits, timeouts, connection failures, and 5xx become `RateLimitError`, `APITimeoutError`, `APIConnectionError`, or `InternalServerError`. Every other case is `ProviderError`.
- **Gemini:** HTTP 429 becomes `RateLimitError`, HTTP 408 becomes `APITimeoutError`, and `ServerError` becomes `InternalServerError`. Other cases, including connection failures, are `ProviderError`. Gemini has no dedicated `APIConnectionError` mapping.

(`ProviderError` is aliased as `DobbyProviderError` in the adapters.)

These failures do not consume Dobby tool-retry or model-correction budgets. Provider-level Tenacity may still retry applicable mapped errors (`RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`). That retry is independent of tool retry and `max_model_corrections`.

## Next

- [OpenAI Provider](./openai.md) - Detailed OpenAI/Azure configuration
- [Vertex AI Provider](./vertexai.md) - Detailed Vertex AI Model Garden configuration
