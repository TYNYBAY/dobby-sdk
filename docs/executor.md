# AgentExecutor

`AgentExecutor` manages the agentic loop: send messages → LLM responds with tool calls → execute tools → repeat until done.

## Basic Usage

```python
from dobby import AgentExecutor, OpenAIProvider, Tool

executor = AgentExecutor(
    provider="openai",
    llm=OpenAIProvider(model="gpt-4o"),
    tools=[MyTool()],
)

async for event in executor.run_stream(messages, system_prompt="You are helpful."):
    match event.type:
        case "text-delta":
            print(event.delta, end="")
        case "tool-use":
            print(f"\n[Tool: {event.name}]")
```

---

## Initialization

```python
executor = AgentExecutor(
    provider="openai",           # "openai" | "azure-openai" | "anthropic"
    llm=provider,                # Provider instance
    tools=[Tool1(), Tool2()],    # Optional tools
    output_type=MyOutputModel,   # Optional structured output
    output_mode="tool",          # "tool" | "native"
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | `str` | Provider name for schema formatting |
| `llm` | `OpenAIProvider` | LLM provider instance |
| `tools` | `list[Tool]` | Available tools |
| `output_type` | `type[BaseModel]` | Pydantic model for structured output |
| `output_mode` | `str` | How to get structured output |

### Accessing Registered Tools

```python
# Get all tools as dict[name, Tool]
executor.tools

# Get tool names
list(executor.tools.keys())  # ['search', 'fetch_data', ...]

# Get a specific tool
executor.tools['search']
```

---

## run_stream()

```python
async for event in executor.run_stream(
    messages,                    # Conversation history
    system_prompt="...",         # Optional system prompt
    context=my_context,          # Passed to Injected[T] tools
    max_iterations=10,           # Max tool call loops
    reasoning_effort="medium",   # For o1/o3 models
    approved_tool_calls=set(),   # Pre-approved tool call IDs
    max_model_corrections=3,     # Run-wide model-correction budget
):
    ...
```

`max_model_corrections` is a single run-wide budget (default `3`) shared by tool-call correction and final-result correction. When it is exhausted, `run_stream()` raises `ModelRetryExhaustedError`.

Tool errors in `ToolResultEvent.result` and conversation history are classified strings of the form `[error_code] message`, not `{"error": ...}`. Host-side `error_details` (`ToolErrorDetails`) still holds exception diagnostics, including the **full exception traceback**. Do not send `error_details` (especially `traceback`) to untrusted clients such as browsers.

Approval and cancellation are host control flow, not classified model errors. The executor still emits a `ToolResultEvent` with `is_error=True` and a placeholder result (`{"approval_required": True}` or `{"cancelled": True}`) so history does not look like the tool succeeded, then re-raises `ApprovalRequired` or `asyncio.CancelledError`. Remaining unexecuted calls in that batch get the same placeholder dict. Already-running parallel calls may still complete successfully.

`dobby.exceptions` also exports the classification helpers the executor uses: `classify_tool_error`, `format_model_error`, and `ErrorDecision`. `classify_tool_error(exception)` returns an `ErrorDecision` (`code`, `model_message`, `retry_model`) or `None` for approval, cancellation, and other non-error control flow. `format_model_error(decision)` produces the `[error_code] message` string sent to the model.

---

## Event Types

The executor yields all provider events plus tool events:

| Event | Description |
|-------|-------------|
| `StreamStartEvent` | Stream started |
| `TextDeltaEvent` | Text chunk |
| `ReasoningDeltaEvent` | Reasoning chunk |
| `ToolUseEvent` | Tool call from LLM |
| `ToolStreamEvent` | Progress from streaming tool |
| `ToolResultEvent` | Tool execution result (check `is_terminal` for terminal tools) |
| `ToolUseEndEvent` | Tool finished |
| `StreamEndEvent` | Stream/iteration finished |

> For terminal tools that exit the loop, see [Terminal Tools](./tools/creating-tools.md#terminal-tools).

---

## Agentic Loop

```mermaid
sequenceDiagram
    participant User
    participant Executor
    participant LLM
    participant Tool

    User->>Executor: run_stream(messages)
    loop Until no tool calls
        Executor->>LLM: chat(messages, tools)
        LLM-->>Executor: TextDelta / ToolUse
        Executor-->>User: yield events

        alt Tool calls exist
            Executor->>Tool: execute(inputs, context)
            Tool-->>Executor: ToolResult
            alt Terminal tool (tool.terminal=True)
                Executor-->>User: ToolResultEvent(is_terminal=True)
                Note over Executor: Exit loop immediately
            else Non-terminal tool
                Executor->>Executor: Add to messages
            end
        end
    end
    Executor-->>User: StreamEndEvent
```

---

## Structured Output

Force LLM to return structured data:

```python
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    city: str
    temperature: float
    conditions: str

executor = AgentExecutor(
    provider="openai",
    llm=provider,
    output_type=WeatherResponse,
    output_mode="tool",  # Uses tool call to get structured output
)

async for event in executor.run_stream(messages):
    if event.type == "stream-end":
        # Find the final_result tool call
        for part in event.parts:
            if isinstance(part, ToolUsePart) and part.name == "final_result":
                result = WeatherResponse(**part.inputs)
                print(result)
```

---

## Context Injection

Pass runtime context to tools:

```python
@dataclass
class AppContext:
    db: Database
    user_id: str

context = AppContext(db=db, user_id="123")

async for event in executor.run_stream(
    messages,
    context=context,
):
    ...
```

Tools receive context via `Injected[T]`:

```python
@dataclass
class DBTool(Tool):
    async def __call__(
        self,
        ctx: Injected[AppContext],
        query: str,
    ) -> list:
        return await ctx.db.query(query)
```

---

## Tool Approval

For sensitive tools, require pre-approval:

```python
async for event in executor.run_stream(
    messages,
    approved_tool_calls={"call_abc"},
):
    if event.type == "tool-use":
        if event.id not in approved_tool_calls:
            # Pause and ask user for approval
            pass
```

If the tool is not in `approved_tool_calls`, the executor yields a `ToolResultEvent` with `is_error=True` and `result={"approval_required": True}` for that call, then raises `ApprovalRequired`. Remaining unexecuted calls get the same `{"approval_required": True}` placeholder (`is_error=True`). Already-running parallel calls may still complete successfully. Cancellation uses the same `is_error=True` event shape with `result={"cancelled": True}` for the cancelled call and remaining unexecuted calls.
