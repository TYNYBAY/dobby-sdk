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

## Compact Context Tool

`CompactContextTool` lets the model request summarize-mode compaction on demand. It is **not** registered automatically — add it to your tool list like any custom tool.

### Installation

Part of `dobby-sdk` (`dobby.tools`).

### Usage

```python
from dobby.context import ContextPolicy
from dobby.tools import CompactContextTool

# Any ContextPolicy enables the tool. It always summarizes, even if mode="trim".
policy = ContextPolicy(keep_last_n=3)
executor = AgentExecutor(
    provider="openai",
    llm=provider,
    tools=[CompactContextTool(), ...],
    context_policy=policy,
)
```

When a `context_policy` is set, `compact_context` always uses the summarize path, regardless of `ContextPolicy.mode`. Without `context_policy`, it still executes and returns its confirmation dict, but the executor does **not** summarize.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `instructions` | `str` | Extra guidance appended to the summarizer prompt (IDs, paths, decisions to preserve). |
| `keep_last_n` | `int \| None` | Recent tool round-trips to keep verbatim. Omit to use `ContextPolicy.keep_last_n`. |

### Tool result

On success the model sees:

```python
{"status": "context_compacted", "detail": "Older tool history has been summarized into a digest above."}
```

Hosts should also listen for `ContextEditEvent` on the stream when summarize actually applied.

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
