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
| `terminal` | `bool` | False | Exits agent loop when called |

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

## Terminal Tools

Terminal tools exit the agent loop when called. The result is not sent back to the LLM. Use for actions that end the conversation like hanging up calls, transferring to humans, or escalating.

### Defining Terminal Tools

```python
from dataclasses import dataclass
from typing import Annotated
from dobby import Tool

@dataclass
class EndCallTool(Tool):
    name = "end_call"
    description = "End the call and hang up"
    terminal = True  # Exits agent loop

    async def __call__(
        self,
        reason: Annotated[str, "Reason for ending the call"],
    ) -> dict:
        return {"status": "ended", "reason": reason}

@dataclass
class TransferToHumanTool(Tool):
    name = "transfer_to_human"
    description = "Transfer the conversation to a human agent"
    terminal = True

    async def __call__(
        self,
        department: Annotated[str, "Department to transfer to"],
        summary: Annotated[str, "Summary of the conversation"],
    ) -> dict:
        return {"department": department, "summary": summary}
```

### Complete Example with AgentExecutor

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated

from dobby import AgentExecutor, OpenAIProvider, Tool
from dobby.types import (
    TextDeltaEvent,
    ToolResultEvent,
    UserMessagePart,
    TextPart,
)

# Define a regular tool
@dataclass
class SearchKnowledgeBase(Tool):
    name = "search_kb"
    description = "Search the knowledge base for information"

    async def __call__(self, query: Annotated[str, "Search query"]) -> str:
        return f"Found information about: {query}"

# Define a terminal tool
@dataclass
class EndCallTool(Tool):
    name = "end_call"
    description = "End the call when conversation is complete"
    terminal = True

    async def __call__(
        self,
        reason: Annotated[str, "Reason for ending"],
    ) -> dict:
        return {"reason": reason}

async def main():
    provider = OpenAIProvider(model="gpt-4o")

    executor = AgentExecutor(
        provider="openai",
        llm=provider,
        tools=[SearchKnowledgeBase(), EndCallTool()],
    )

    messages = [
        UserMessagePart(parts=[TextPart(text="Help me find info about Python, then end the call.")])
    ]

    async for event in executor.run_stream(
        messages,
        system_prompt="You are a helpful assistant. Use end_call when done.",
    ):
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)

        elif isinstance(event, ToolResultEvent):
            if event.is_terminal:
                # Terminal tool called - loop has exited
                print(f"\n\n[Call ended: {event.result['reason']}]")
                # Handle post-call actions here
                await handle_call_end(event.result)
            else:
                # Regular tool result - loop continues
                print(f"\n[Tool {event.name} returned: {event.result}]\n")

async def handle_call_end(result: dict):
    """Handle actions after terminal tool is called."""
    print(f"Saving transcript...")
    print(f"Call ended with reason: {result['reason']}")

asyncio.run(main())
```

### How It Works

1. **Regular tools** (`search_kb`): Result is sent back to LLM, loop continues
2. **Terminal tools** (`end_call`): Result is yielded with `is_terminal=True`, loop exits immediately

### Common Use Cases

| Tool | Description |
|------|-------------|
| `end_call` / `hang_up` | End voice calls |
| `transfer_to_human` | Hand off to human agent |
| `escalate` | Escalate to supervisor |
| `complete_task` | Mark task as done and exit |

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
