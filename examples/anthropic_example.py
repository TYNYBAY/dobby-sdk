"""Example: Anthropic Claude provider - streaming and non-streaming."""

import asyncio

from dobby.providers.anthropic import AnthropicProvider
from dobby.types import (
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextPart,
    ToolUseEvent,
    UserMessagePart,
)


async def test_non_streaming():
    """Non-streaming chat completion."""
    provider = AnthropicProvider(model="claude-sonnet-4-20250514")

    result = await provider.chat(
        messages=[UserMessagePart(parts=[TextPart(text="What is 2 + 2? Reply in one word.")])],
        system_prompt="You are a helpful assistant.",
    )
    print("=== Non-Streaming ===")
    print(f"Model: {result.model}")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Parts: {result.parts}")
    print(f"Usage: {result.usage}")
    print()


async def test_streaming():
    """Streaming chat completion."""
    provider = AnthropicProvider(model="claude-sonnet-4-20250514")

    print("=== Streaming ===")
    stream = await provider.chat(
        messages=[UserMessagePart(parts=[TextPart(text="Write a haiku about Python.")])],
        stream=True,
    )
    async for event in stream:
        match event:
            case StreamStartEvent():
                print(f"[start] model={event.model}")
            case TextDeltaEvent():
                print(event.delta, end="", flush=True)
            case ReasoningStartEvent():
                print("\n[thinking...]")
            case ReasoningDeltaEvent():
                print(event.delta, end="", flush=True)
            case ReasoningEndEvent():
                print("\n[/thinking]")
            case StreamEndEvent():
                print(f"\n[end] stop_reason={event.stop_reason} usage={event.usage}")
    print()


async def test_with_thinking():
    """Streaming with extended thinking enabled."""
    provider = AnthropicProvider(model="claude-sonnet-4-20250514")

    print("=== Streaming with Extended Thinking ===")
    stream = await provider.chat(
        messages=[UserMessagePart(parts=[TextPart(text="What is 15 * 37?")])],
        stream=True,
        reasoning_effort=5000,
    )
    async for event in stream:
        match event:
            case StreamStartEvent():
                print(f"[start] model={event.model}")
            case ReasoningStartEvent():
                print("[thinking...]")
            case ReasoningDeltaEvent():
                print(f"  {event.delta}", end="", flush=True)
            case ReasoningEndEvent():
                print("\n[/thinking]")
            case TextDeltaEvent():
                print(event.delta, end="", flush=True)
            case StreamEndEvent():
                print(f"\n[end] stop_reason={event.stop_reason} usage={event.usage}")
    print()


if __name__ == "__main__":
    asyncio.run(test_non_streaming())
    asyncio.run(test_streaming())
    asyncio.run(test_with_thinking())
