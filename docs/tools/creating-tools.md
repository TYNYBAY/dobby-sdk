# Creating Tools

Tools let your LLM interact with external systems. Dobby auto-generates schemas from Python type hints.

## Basic Tool

```python
from dataclasses import dataclass
from typing import Annotated
from dobby import Tool

@dataclass
class SearchTool(Tool):
    name = "search"
    description = "Search the web for information"
    
    async def __call__(
        self,
        query: Annotated[str, "Search query"],
    ) -> str:
        # Your implementation
        return f"Results for: {query}"
```

## Tool Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Class name | Tool name sent to LLM |
| `description` | `str` | Required | What the tool does |
| `max_retries` | `int` | 1 | Retry attempts on failure |
| `requires_approval` | `bool` | False | Needs human approval |
| `stream_output` | `bool` | False | Yields streaming events |

---

## Parameter Descriptions

Use `Annotated` to add descriptions:

```python
async def __call__(
    self,
    city: Annotated[str, "City name (e.g., 'Tokyo')"],
    units: Annotated[str, "Temperature units"] = "celsius",
) -> dict:
    ...
```

Generated schema:
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "parameters": {
      "properties": {
        "city": {"type": "string", "description": "City name (e.g., 'Tokyo')"},
        "units": {"type": "string", "description": "Temperature units", "default": "celsius"}
      },
      "required": ["city"]
    }
  }
}
```

---

## Context Injection

Inject runtime context (DB, user info, etc.) using `Injected[T]`:

```python
from dobby import Tool, Injected
from dataclasses import dataclass

@dataclass
class MyContext:
    user_id: str
    db: Database

@dataclass
class GetUserTool(Tool):
    name = "get_user"
    description = "Get current user info"
    
    async def __call__(
        self,
        ctx: Injected[MyContext],  # Injected, not sent to LLM
    ) -> dict:
        return await ctx.db.get_user(ctx.user_id)
```

Pass tool and context to executor:

```python
from dobby import AgentExecutor, OpenAIProvider

# 1. Create context
context = MyContext(user_id="123", db=db)

# 2. Instantiate tool
get_user_tool = GetUserTool()

# 3. Create executor with tool
executor = AgentExecutor(
    provider="openai",
    llm=OpenAIProvider(model="gpt-4o"),
    tools=[get_user_tool],  # Pass tool instances
)

# 4. Run with context
async for event in executor.run_stream(
    messages,
    context=context,  # Context injected into tools with Injected[T]
):
    ...
```

---

## Streaming Tools

For long-running tools, yield progress events:

```python
@dataclass
class AnalyzeTool(Tool):
    name = "analyze"
    description = "Analyze large dataset"
    stream_output = True  # Enable streaming
    
    async def __call__(
        self,
        file_path: Annotated[str, "Path to file"],
    ):
        yield ToolStreamEvent(type="progress", data="Starting analysis...")
        
        for i in range(10):
            await asyncio.sleep(1)
            yield ToolStreamEvent(type="progress", data=f"Progress: {i*10}%")
        
        # Return final result
        return {"status": "complete", "output": "Analysis finished"}
```

---

## Approval Flow

For sensitive operations, require human approval:

```python
@dataclass
class DeleteFileTool(Tool):
    name = "delete_file"
    description = "Delete a file"
    requires_approval = True  # Requires approval
    
    async def __call__(self, path: str) -> str:
        os.remove(path)
        return f"Deleted {path}"
```

In executor:

```python
approved_calls = {"call_abc123"}  # Pre-approved IDs

async for event in executor.run_stream(
    messages,
    approved_tool_calls=approved_calls,
):
    if event.type == "tool-use":
        if event.id not in approved_calls:
            # Show approval UI
            pass
```

---

## Using with Executor

```python
from dobby import AgentExecutor, OpenAIProvider

executor = AgentExecutor(
    provider="openai",
    llm=OpenAIProvider(model="gpt-4o"),
    tools=[SearchTool(), GetUserTool()],
)

async for event in executor.run_stream(messages, context=my_context):
    match event.type:
        case "tool-use":
            print(f"Calling {event.name}")
        case "tool-result":
            print(f"Result: {event.output}")
```
