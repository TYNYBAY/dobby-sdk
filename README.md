# LLM Tools System

This document provides a comprehensive guide to the LLM tools system, which enables function calling capabilities for Large Language Models (LLMs) in a provider-agnostic way.

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Tool Definition](#tool-definition)
4. [Using Tools with LLMs](#using-tools-with-llms)
5. [Advanced Examples](#advanced-examples)
6. [API Reference](#api-reference)

## Overview

The LLM tools system provides a flexible framework for:
- Defining tools/functions that LLMs can call
- Automatic schema generation from Python functions
- Support for multiple LLM providers (OpenAI, Anthropic)
- Type-safe parameter handling with Pydantic and TypedDict support
- Streaming and non-streaming responses with tool calls

## Quick Start

### 1. Define a Simple Tool

```python
from typing import Annotated
from src.llms.tools import tool

@tool(description="Get current weather for a location")
async def get_weather(
    location: Annotated[str, "City name or coordinates"],
    unit: Annotated[str, "Temperature unit (celsius/fahrenheit)"] = "celsius"
) -> dict:
    # Your implementation here
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "conditions": "sunny"
    }
```

### 2. Register and Use with LLM

```python
from src.llms.tools import ToolRegistry
from src.llms.providers.openai import OpenAIProvider

# Create registry and register tools
registry = ToolRegistry()
registry.register(get_weather)

# Initialize LLM provider
provider = OpenAIProvider(
    model="gpt-4o-mini",
    api_key="your-api-key"
)

# Get tools for LLM
tools = list(registry.get_all().values())

# Chat with tool support
response = await provider.chat(
    messages=[{
        "role": "user",
        "content": "What's the weather in Tokyo?"
    }],
    tools=tools
)

# Handle tool calls - Method 1: Check finish reason (recommended)
if response.stop_reason == "tool_use":
    for block in response.content:
        if block["type"] == "tool_use":
            result = await registry.execute_tool(
                block["name"], 
                **block["inputs"]
            )
            print(f"Tool {block['name']} returned: {result}")

# Method 2: Check content type
# if isinstance(response.content, list) and any(b["type"] == "tool_use" for b in response.content):
```

## Tool Definition

### Using the @tool Decorator

The `@tool` decorator automatically generates schemas from function signatures:

```python
from typing import Annotated, List, Optional
from pydantic import Field
from src.llms.tools import tool

@tool(description="Search for documents")
async def search_documents(
    query: Annotated[str, "Search query"],
    max_results: Annotated[int, Field(default=10, ge=1, le=100, description="Maximum results")] = 10,
    filters: Optional[dict] = None
) -> List[dict]:
    # Implementation
    pass
```

### Parameter Types

#### Simple Types
- `str`, `int`, `float`, `bool`
- `List[T]`, `Dict[K, V]`
- `Optional[T]` for optional parameters

#### Using Annotated for Descriptions

```python
from typing import Annotated

def process_data(
    input_file: Annotated[str, "Path to input file"],
    threshold: Annotated[float, "Processing threshold value"]
) -> dict:
    pass
```

#### Using Pydantic Field for Constraints

```python
from pydantic import Field

def calculate(
    amount: Annotated[float, Field(gt=0, description="Amount to calculate")],
    rate: Annotated[float, Field(ge=0, le=1, description="Rate (0-1)")],
    years: Annotated[int, Field(gt=0, le=50, description="Number of years")]
) -> float:
    pass
```

### Complex Types with Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class Address(BaseModel):
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City name")
    postal_code: str = Field(..., pattern=r"^\d{5}$", description="5-digit postal code")
    country: str = Field(default="US", description="Country code")

class Customer(BaseModel):
    name: str = Field(..., min_length=1, description="Customer name")
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    phone: Optional[str] = Field(None, pattern=r"^\+?\d{10,}$")
    address: Optional[Address] = None
    tags: List[str] = Field(default_factory=list, description="Customer tags")

@tool(description="Register a new customer")
async def register_customer(
    customer: Customer,
    send_welcome_email: bool = True
) -> dict:
    # Process customer registration
    return {
        "customer_id": "CUST-12345",
        "status": "registered",
        "email_sent": send_welcome_email
    }
```

### Complex Types with TypedDict

```python
from typing import TypedDict, List

class OrderItem(TypedDict):
    product_id: str
    quantity: int
    price: float

class PaymentInfo(TypedDict):
    method: str  # "credit_card", "paypal", "bank_transfer"
    amount: float
    currency: str

@tool(description="Process an order")
def process_order(
    items: Annotated[List[OrderItem], "List of items in the order"],
    payment: Annotated[PaymentInfo, "Payment information"],
    shipping_address: Annotated[dict, "Shipping address details"]
) -> dict:
    total = sum(item["quantity"] * item["price"] for item in items)
    return {
        "order_id": "ORD-12345",
        "total": total,
        "status": "processed"
    }
```

## Message Types

The LLM system uses three message types for conversations:

### UserMessagePart
```python
{
    "role": "user",
    "content": str | Iterable[ContentBlock]  # Text or multimodal content
}
```

### AssistantMessagePart
```python
{
    "role": "assistant", 
    "content": str | Iterable[AssistantContentBlock]  # Can include tool calls
}
```

### ToolResultMessagePart
```python
ToolResultMessagePart(
    role="tool_result",
    tool_use_id="call_123",  # ID from the tool call
    name="get_weather",      # Tool name
    content="...",           # Tool result (string or JSON)
    is_error=False          # Whether the tool execution failed
)
```

## Using Tools with LLMs

### Detecting Tool Calls

There are two approaches to detect when the LLM wants to use tools:

1. **Check `stop_reason` (Recommended)**
   ```python
   if response.stop_reason == "tool_use":
       # Handle tool calls
   ```
   This is the most reliable method as it explicitly indicates the LLM stopped to use tools.

2. **Check content type**
   ```python
   if isinstance(response.content, list) and any(b["type"] == "tool_use" for b in response.content):
       # Handle tool calls
   ```
   This checks if the content contains tool use blocks.

### Complete Example with Tool Handling

```python
import asyncio
import json
from typing import List, Annotated
from src.llms.tools import tool, ToolRegistry
from src.llms.providers.openai import OpenAIProvider
from src.llms.types import MessagePart, ToolResultMessagePart

# Define tools
@tool(description="Get user information")
async def get_user_info(
    user_id: Annotated[str, "User ID to look up"]
) -> dict:
    # Mock implementation
    return {
        "user_id": user_id,
        "name": "John Doe",
        "email": "john@example.com",
        "account_type": "premium"
    }

@tool(description="Send email notification")
async def send_email(
    recipient: Annotated[str, "Email recipient"],
    subject: Annotated[str, "Email subject"],
    body: Annotated[str, "Email body content"]
) -> dict:
    # Mock implementation
    return {
        "status": "sent",
        "message_id": "MSG-12345",
        "timestamp": "2024-01-01T12:00:00Z"
    }

async def main():
    # Setup
    registry = ToolRegistry()
    registry.register(get_user_info)
    registry.register(send_email)
    
    provider = OpenAIProvider(model="gpt-4o-mini")
    tools = list(registry.get_all().values())
    
    # Conversation with tool handling
    messages: List[MessagePart] = []
    
    # User message
    user_message = {
        "role": "user",
        "content": "Look up user USER-123 and send them a welcome email"
    }
    messages.append(user_message)
    
    # Get LLM response
    response = await provider.chat(
        messages=messages,
        tools=tools,
        temperature=0.0
    )
    
    # Handle tool calls - Check finish reason (recommended approach)
    if response.stop_reason == "tool_use":
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": response.content
        })
        
        # Execute tools and collect results
        tool_results = []
        for block in response.content:
            if block["type"] == "tool_use":
                try:
                    result = await registry.execute_tool(
                        block["name"],
                        **block["inputs"]
                    )
                    
                    tool_results.append(ToolResultMessagePart(
                        role="tool_result",
                        tool_use_id=block["id"],
                        name=block["name"],
                        content=json.dumps(result),
                        is_error=False
                    ))
                except Exception as e:
                    tool_results.append(ToolResultMessagePart(
                        role="tool_result",
                        tool_use_id=block["id"],
                        name=block["name"],
                        content=str(e),
                        is_error=True
                    ))
        
        # Add tool results to messages
        messages.extend(tool_results)
        
        # Get final response
        final_response = await provider.chat(
            messages=messages,
            tools=tools
        )
        print(f"Assistant: {final_response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming with Tools

```python
async def streaming_example():
    # ... setup as before ...
    
    # Stream response
    final_content = None
    final_stop_reason = None
    async for chunk in await provider.chat(
        messages=messages,
        tools=tools,
        stream=True
    ):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
        final_content = chunk.content
        if chunk.stop_reason:
            final_stop_reason = chunk.stop_reason
    
    # Handle tool calls from final content
    # In streaming, the stop_reason appears in the final chunk
    if final_stop_reason == "tool_use":
        # ... handle tool calls as before ...
```

## Advanced Examples

### Insurance Claim Processing

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class ClaimItem(BaseModel):
    description: str = Field(..., description="Item description")
    value: float = Field(..., gt=0, description="Item value")
    purchase_date: Optional[str] = Field(None, description="Purchase date (YYYY-MM-DD)")

class ClaimSubmission(BaseModel):
    policy_number: str = Field(..., pattern=r"^POL-\d{6}$")
    incident_date: str = Field(..., description="Date of incident (YYYY-MM-DD)")
    incident_type: str = Field(..., enum=["theft", "damage", "loss", "other"])
    description: str = Field(..., min_length=10, max_length=1000)
    items: List[ClaimItem] = Field(..., min_items=1)
    total_amount: float = Field(..., gt=0)

@tool(description="Submit an insurance claim")
async def submit_insurance_claim(
    claim: ClaimSubmission,
    supporting_documents: Annotated[List[str], "List of document URLs"] = []
) -> dict:
    # Validate claim
    total = sum(item.value for item in claim.items)
    if abs(total - claim.total_amount) > 0.01:
        raise ValueError("Total amount doesn't match sum of items")
    
    # Process claim
    claim_id = f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return {
        "claim_id": claim_id,
        "status": "submitted",
        "policy_number": claim.policy_number,
        "amount": claim.total_amount,
        "estimated_processing_days": 5,
        "next_steps": [
            "Document verification in progress",
            "You will receive an email confirmation",
            "An adjuster may contact you if needed"
        ]
    }

@tool(description="Check claim status")
async def check_claim_status(
    claim_id: Annotated[str, "Claim ID to check"],
    include_history: bool = False
) -> dict:
    # Mock implementation
    status = {
        "claim_id": claim_id,
        "status": "under_review",
        "last_updated": "2024-01-02T10:00:00Z",
        "assigned_adjuster": "Jane Smith"
    }
    
    if include_history:
        status["history"] = [
            {"date": "2024-01-01", "status": "submitted", "notes": "Claim received"},
            {"date": "2024-01-02", "status": "under_review", "notes": "Assigned to adjuster"}
        ]
    
    return status
```

### Multi-Step Tool Execution

```python
@tool(description="Analyze customer sentiment from support tickets")
async def analyze_customer_sentiment(
    customer_id: Annotated[str, "Customer ID"],
    days: Annotated[int, Field(default=30, ge=1, le=365, description="Days to analyze")] = 30
) -> dict:
    # This tool might internally call other tools
    # Mock implementation
    return {
        "customer_id": customer_id,
        "period_days": days,
        "overall_sentiment": "positive",
        "score": 0.75,
        "ticket_count": 5,
        "common_issues": ["billing", "feature_request"],
        "recommendation": "Customer is generally satisfied"
    }
```

## API Reference

### Tool Decorator

```python
@tool(
    name: Optional[str] = None,  # Tool name (defaults to function name)
    description: Optional[str] = None,  # Tool description
    version: str = "1.0.0"  # Tool version
)
```

### ToolRegistry

```python
class ToolRegistry:
    def register(self, tool: BaseTool) -> None
        """Register a tool in the registry."""
    
    def unregister(self, tool_name: str) -> None
        """Remove a tool from the registry."""
    
    def get(self, tool_name: str) -> Optional[BaseTool]
        """Get a tool by name."""
    
    def get_all(self) -> Dict[str, BaseTool]
        """Get all registered tools."""
    
    def list_tools(self) -> List[str]
        """List all tool names."""
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any
        """Execute a tool by name with given parameters."""
```

### Provider Integration

```python
# OpenAI Provider
response = await provider.chat(
    messages=messages,
    tools=tools,  # List[BaseTool]
    temperature=0.0,
    stream=False
)

# Check if tools were called
if response.stop_reason == "tool_use":
    # OpenAI finish_reason "tool_calls" maps to stop_reason "tool_use"
    for block in response.content:
        if block["type"] == "tool_use":
            # block["id"] - unique tool call ID
            # block["name"] - tool name
            # block["inputs"] - tool parameters as dict
```

### Schema Formats

Tools automatically generate schemas in provider-specific formats:

```python
# Get OpenAI format
openai_schema = tool.schema.to_openai_format()

# Get Anthropic format  
anthropic_schema = tool.schema.to_anthropic_format()
```

## Best Practices

1. **Type Annotations**: Always use type annotations for better schema generation
2. **Descriptions**: Provide clear descriptions for tools and parameters
3. **Validation**: Use Pydantic Field constraints for input validation
4. **Error Handling**: Tools should raise exceptions with clear error messages
5. **Async Functions**: Prefer async functions for I/O operations
6. **Return Types**: Return JSON-serializable data (dict, list, primitives)

## Testing Tools

Test your tools before integrating with LLMs:

```python
# Direct execution
registry = ToolRegistry()
registry.register(my_tool)

result = await registry.execute_tool(
    "my_tool",
    param1="value1",
    param2=42
)

# Schema validation
schema = my_tool.schema
print(f"Tool name: {schema.name}")
print(f"Parameters: {[p.name for p in schema.parameters]}")
print(f"OpenAI format: {json.dumps(schema.to_openai_format(), indent=2)}")
```