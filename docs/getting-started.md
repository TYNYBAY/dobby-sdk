# Getting Started

## Installation

### From GitHub

```bash
pip install git+https://github.com/TYNYBAY/dobby-sdk.git
```

### With uv

```bash
uv add git+https://github.com/TYNYBAY/dobby-sdk.git
```

### From source

```bash
git clone https://github.com/TYNYBAY/dobby-sdk.git
cd dobby-sdk
pip install -e .
```

## Requirements

- Python 3.12+
- OpenAI API key (for OpenAI/Azure provider)

---

## First Example: Streaming Chat

```python
import asyncio
from dobby import OpenAIProvider
from dobby.types import UserMessagePart, TextPart

async def main():
    provider = OpenAIProvider(
        model="gpt-4o",
        api_key="sk-..."
    )
    
    messages = [
        UserMessagePart(parts=[TextPart(text="What is Python?")])
    ]
    
    async for event in await provider.chat(messages, stream=True):
        match event.type:
            case "text-delta":
                print(event.delta, end="", flush=True)
            case "stream-end":
                print(f"\n\nTokens: {event.usage.total_tokens}")

asyncio.run(main())
```

---

## First Example: Tool Calling

```python
from dataclasses import dataclass
from typing import Annotated
from dobby import AgentExecutor, OpenAIProvider, Tool
from dobby.types import UserMessagePart, TextPart

@dataclass
class WeatherTool(Tool):
    name = "get_weather"
    description = "Get current weather for a city"
    
    async def __call__(
        self,
        city: Annotated[str, "City name"],
    ) -> str:
        return f"Weather in {city}: 22°C, sunny"

async def main():
    provider = OpenAIProvider(model="gpt-4o")
    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[WeatherTool()],
    )
    
    messages = [
        UserMessagePart(parts=[TextPart(text="What's the weather in Tokyo?")])
    ]
    
    async for event in executor.run_stream(messages):
        match event.type:
            case "text-delta":
                print(event.delta, end="")
            case "tool-use":
                print(f"\n[Tool: {event.name}({event.inputs})]")

asyncio.run(main())
```

---

## Next Steps

- [Message Types](./types/messages.md) - Learn about message structures
- [Providers](./providers/) - Configure OpenAI/Azure
- [Tools](./tools/) - Create custom tools
- [AgentExecutor](./executor.md) - Full agentic loop
