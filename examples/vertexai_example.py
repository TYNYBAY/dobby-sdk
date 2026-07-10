"""Example: Vertex AI provider (Model Garden) - streaming with a tool call.

Requires GCP credentials resolvable by VertexAIProvider's own auth chain:
an explicit `credentials=` object, the GOOGLE_APPLICATION_CREDENTIALS_JSON
env var, or Application Default Credentials (ADC). See docs/providers/vertexai.md.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated

from dobby import AgentExecutor
from dobby.providers import VertexAIProvider
from dobby.tools import Tool
from dobby.types import (
    MessagePart,
    TextDeltaEvent,
    TextPart,
    ToolResultEvent,
    ToolUseEvent,
    UserMessagePart,
)


@dataclass
class AddNumbersTool(Tool):
    name = "add_numbers"
    description = "Add two numbers together and return the sum"

    async def __call__(
        self,
        a: Annotated[int, "First number"],
        b: Annotated[int, "Second number"],
    ) -> int:
        print("add_numbers called with a=", a, "b=", b)
        return a + b


async def main() -> None:
    # Model Garden id — not a native Gemini/Claude id (VertexAIProvider rejects
    # those; use GeminiProvider(vertexai=True) or AnthropicProvider(vertex=True)).
    provider = VertexAIProvider(
        model="meta/llama-3.1-405b-instruct-maas",
        location="us-central1",
    )

    executor = AgentExecutor(provider="vertexai", llm=provider, tools=[AddNumbersTool()])
    messages: list[MessagePart] = [
        UserMessagePart(parts=[TextPart(text="What is 42 plus 17? Use the add_numbers tool.")])
    ]

    tool_called = False
    async for event in executor.run_stream(messages):
        match event:
            case TextDeltaEvent(delta=delta):
                print(delta, end="")
            case ToolUseEvent(name=name, inputs=inputs):
                tool_called = True
                print(f"\n-> tool call: {name}({inputs})")
            case ToolResultEvent(name=name, result=result, is_error=is_error):
                print(f"-> tool result ({name}, error={is_error}): {result}")

    print()
    if not tool_called:
        print("(model answered without calling add_numbers)")


if __name__ == "__main__":
    asyncio.run(main())
