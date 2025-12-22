# Dobby SDK

Lightweight multi-provider LLM SDK with streaming and tool support.

## Installation

```bash
pip install dobby-sdk

# Or from GitHub
pip install git+https://github.com/TYNYBAY/dobby-sdk.git

# With uv
uv add dobby-sdk
```

## Quick Start

```python
from dobby import AgentExecutor, OpenAIProvider
from dobby.types import UserMessagePart, TextPart, TextDeltaEvent

provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
executor = AgentExecutor(provider="openai", llm=provider)

messages = [UserMessagePart(parts=[TextPart(text="Hello!")])]

async for event in executor.run_stream(messages):
    match event:
        case TextDeltaEvent(delta=delta):
            print(delta, end="")
```

## Features

- **Multi-provider**: OpenAI, Azure OpenAI, Anthropic
- **Streaming**: Real-time token streaming with typed events
- **Tools**: Dataclass-based tools with auto-generated schemas
- **Context injection**: Pass runtime context to tools via `Injected[T]`
- **Structured output**: Pydantic model validation for agent responses

## Documentation

See [docs/](./docs/) for detailed documentation:

- [Getting Started](./docs/getting-started.md)
- [Message Types](./docs/types/messages.md)
- [Providers](./docs/providers/)
- [Tools](./docs/tools/)
- [AgentExecutor](./docs/executor.md)
- [Vector Stores](./docs/vector-stores/)
- [Retrievers](./docs/retrievers/)

## License

MIT
