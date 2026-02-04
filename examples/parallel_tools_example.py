"""Example: Parallel vs Sequential tool execution with a real LLM.

Three tools each sleep 1s to simulate I/O. The same prompt is sent twice:
  1. Default (parallel)  — batch completes in ~1s
  2. sequential=True     — batch completes in ~3s

Usage:
    python examples/parallel_tools_example.py
"""

import asyncio
from dataclasses import dataclass
import os
import time
from typing import Annotated, Any

from dotenv import load_dotenv

from dobby import AgentExecutor
from dobby.providers.gemini import GeminiProvider
from dobby.tools import Tool
from dobby.types import (
    StreamEndEvent,
    TextDeltaEvent,
    TextPart,
    ToolResultEvent,
    ToolUseEvent,
    UserMessagePart,
)

load_dotenv()


# --- Parallel tools (default) ---


@dataclass
class GetWeatherTool(Tool):
    name = "get_weather"
    description = "Get current weather for a city. Returns temperature and conditions."

    async def __call__(
        self, city: Annotated[str, "City name, e.g. 'London'"]
    ) -> dict[str, Any]:
        await asyncio.sleep(1)
        return {"city": city, "temp_c": 18, "condition": "partly cloudy"}


@dataclass
class GetStockPriceTool(Tool):
    name = "get_stock_price"
    description = "Get the current stock price for a ticker symbol."

    async def __call__(
        self, symbol: Annotated[str, "Stock ticker symbol, e.g. 'AAPL'"]
    ) -> dict[str, Any]:
        await asyncio.sleep(1)
        return {"symbol": symbol, "price_usd": 227.50, "change": "+1.2%"}


@dataclass
class GetNewsHeadlinesTool(Tool):
    name = "get_news_headlines"
    description = "Get the top 3 news headlines for a topic."

    async def __call__(
        self, topic: Annotated[str, "News topic, e.g. 'technology'"]
    ) -> dict[str, Any]:
        await asyncio.sleep(1)
        return {
            "topic": topic,
            "headlines": [
                "AI adoption accelerates across industries",
                "New chip architecture breaks performance records",
                "Open source LLM surpasses proprietary benchmarks",
            ],
        }


# --- Sequential variants (same tools, but with sequential=True) ---


@dataclass
class GetWeatherSequentialTool(Tool):
    name = "get_weather"
    description = "Get current weather for a city. Returns temperature and conditions."
    sequential = True

    async def __call__(
        self, city: Annotated[str, "City name, e.g. 'London'"]
    ) -> dict[str, Any]:
        await asyncio.sleep(1)
        return {"city": city, "temp_c": 18, "condition": "partly cloudy"}


@dataclass
class GetStockPriceSequentialTool(Tool):
    name = "get_stock_price"
    description = "Get the current stock price for a ticker symbol."
    sequential = True

    async def __call__(
        self, symbol: Annotated[str, "Stock ticker symbol, e.g. 'AAPL'"]
    ) -> dict[str, Any]:
        await asyncio.sleep(1)
        return {"symbol": symbol, "price_usd": 227.50, "change": "+1.2%"}


@dataclass
class GetNewsHeadlinesSequentialTool(Tool):
    name = "get_news_headlines"
    description = "Get the top 3 news headlines for a topic."
    sequential = True

    async def __call__(
        self, topic: Annotated[str, "News topic, e.g. 'technology'"]
    ) -> dict[str, Any]:
        await asyncio.sleep(1)
        return {
            "topic": topic,
            "headlines": [
                "AI adoption accelerates across industries",
                "New chip architecture breaks performance records",
                "Open source LLM surpasses proprietary benchmarks",
            ],
        }


# --- Shared runner ---

MESSAGES = [
    UserMessagePart(
        parts=[
            TextPart(
                text=(
                    "I need three things at once: "
                    "1) weather in San Francisco, "
                    "2) AAPL stock price, "
                    "3) top tech news headlines. "
                    "Please fetch all three."
                )
            )
        ]
    )
]

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When the user asks for multiple independent pieces of information, "
    "call all relevant tools in parallel in a single response."
)


async def run_with_tools(provider: GeminiProvider, tools: list[Tool], label: str) -> None:
    """Run the agent and print timing info."""
    executor = AgentExecutor(provider="gemini", llm=provider, tools=tools)

    print(f"--- {label} ---\n")

    tool_call_count = 0
    tool_batch_start: float | None = None
    tool_batch_elapsed: float | None = None

    async for event in executor.run_stream(MESSAGES, system_prompt=SYSTEM_PROMPT):
        match event:
            case TextDeltaEvent(delta=delta):
                print(delta, end="", flush=True)
            case ToolUseEvent(name=name, inputs=inputs):
                tool_call_count += 1
                if tool_batch_start is None:
                    tool_batch_start = time.monotonic()
                print(f"  [tool call {tool_call_count}] {name}({inputs})")
            case ToolResultEvent(name=name, result=result):
                print(f"  [result]    {name} -> {result}")
                tool_batch_elapsed = time.monotonic() - (tool_batch_start or 0)
            case StreamEndEvent():
                pass

    print("\n")
    print(f"  Tools called: {tool_call_count}")
    if tool_batch_elapsed is not None:
        print(f"  Batch time:   {tool_batch_elapsed:.2f}s")
    print()


async def main() -> None:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("Set GEMINI_API_KEY in .env")
        return

    provider = GeminiProvider(api_key=gemini_key, model="gemini-2.5-flash")

    # Run 1: parallel (default) — expected ~1s for 3 tools
    await run_with_tools(
        provider,
        [GetWeatherTool(), GetStockPriceTool(), GetNewsHeadlinesTool()],
        "Run 1: Parallel (default)",
    )

    # Run 2: sequential (sequential=True) — expected ~3s for 3 tools
    await run_with_tools(
        provider,
        [
            GetWeatherSequentialTool(),
            GetStockPriceSequentialTool(),
            GetNewsHeadlinesSequentialTool(),
        ],
        "Run 2: Sequential (sequential=True)",
    )


if __name__ == "__main__":
    asyncio.run(main())
