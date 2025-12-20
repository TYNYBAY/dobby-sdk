# Built-in Tools

Dobby includes common tools ready to use.

## Tavily Web Search

Tavily provides AI-optimized web search.

### Installation

Tavily is included in dobby-sdk dependencies.

### Usage

```python
from dobby.common_tools import TavilySearchTool

# Create tool with API key
search_tool = TavilySearchTool(api_key="tvly-...")

# Use with executor
executor = AgentExecutor(
    provider="openai",
    llm=provider,
    tools=[search_tool],
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Search query |
| `max_results` | `int` | Number of results (default: 5) |
| `search_depth` | `str` | "basic" or "advanced" |

### Example Response

```python
{
    "results": [
        {
            "title": "Python Tutorial",
            "url": "https://...",
            "content": "Python is a programming language...",
            "score": 0.95
        }
    ]
}
```

---

## Creating Custom Built-in Tools

Add your own to `dobby/common_tools/`:

```python
# dobby/common_tools/my_tool.py
from dataclasses import dataclass
from typing import Annotated
from dobby import Tool

@dataclass
class MyCustomTool(Tool):
    name = "my_tool"
    description = "Does something useful"
    
    api_key: str = ""
    
    async def __call__(
        self,
        param: Annotated[str, "Required parameter"],
    ) -> str:
        # Implementation
        return "result"
```
