# Dobby SDK

Lightweight multi-provider LLM SDK with streaming and tool support.

## Installation

```bash
# From GitHub
pip install git+https://github.com/TYNYBAY/dobby-sdk.git

# With uv
uv add git+https://github.com/TYNYBAY/dobby-sdk.git
```

## Quick Start

```python
from dobby import AgentExecutor, OpenAIProvider

provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
executor = AgentExecutor(provider="openai", llm=provider)

messages = [{"role": "user", "parts": [{"text": "Hello!"}]}]

async for event in executor.run_stream(messages):
    if event.type == "text-delta":
        print(event.delta, end="")
```

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
