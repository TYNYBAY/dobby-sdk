#!/usr/bin/env python
"""Example: client-side context compaction on OpenAI and Gemini.

A deliberately token-heavy tool makes the conversation grow fast. A small
``ContextPolicy`` (``context_window=2000``) makes compaction fire within a few
turns so you can watch it happen — every ``ContextEditEvent`` is printed, and the
``input_tokens`` reported by the provider drops after a compaction.

Set both keys in ``.env`` (or the environment):

    OPENAI_API_KEY=sk-...
    GEMINI_API_KEY=...

Usage:
    python examples/context_editing.py
"""

import asyncio
from dataclasses import dataclass
import os
from typing import Annotated, Any

from dotenv import load_dotenv

from dobby import AgentExecutor, ContextEditEvent, ContextPolicy
from dobby.providers.gemini import GeminiProvider
from dobby.providers.openai import OpenAIProvider
from dobby.tools import Tool
from dobby.types import (
    StreamEndEvent,
    TextDeltaEvent,
    TextPart,
    ToolUseEvent,
    UserMessagePart,
)

load_dotenv()


@dataclass
class FetchReportTool(Tool):
    """Returns a large synthetic report so the context grows quickly."""

    name = "fetch_report"
    description = "Fetch a detailed report for a given topic. Returns a long document."

    async def __call__(self, topic: Annotated[str, "The report topic"]) -> dict[str, Any]:
        body = " ".join(f"{topic}-finding-{i}: value={i * 7}" for i in range(200))
        return {"topic": topic, "report": body}


MESSAGES = [
    UserMessagePart(
        parts=[
            TextPart(
                text=(
                    "Fetch reports for these topics one at a time and then give me a "
                    "one-line summary of each: alpha, beta, gamma, delta, epsilon."
                )
            )
        ]
    )
]

SYSTEM_PROMPT = (
    "You are a research assistant. Use the fetch_report tool to retrieve each "
    "topic's report before summarizing. Fetch one topic per turn."
)


async def run_provider(label: str, provider: Any, provider_kind: str) -> None:
    """Drive the agent for one provider, printing tokens and compaction events."""
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

    policy = ContextPolicy(context_window=2000, trigger_pct=0.5, keep_last_n=2, mode="trim")
    executor = AgentExecutor(
        provider=provider_kind,
        llm=provider,
        tools=[FetchReportTool()],
        context_policy=policy,
    )

    turn = 0
    async for event in executor.run_stream(
        MESSAGES, system_prompt=SYSTEM_PROMPT, max_iterations=8
    ):
        match event:
            case ContextEditEvent(applied_edits=edits):
                for edit in edits:
                    print(
                        f"  ⟲ context_edit [{edit.type}]: {edit.model_dump(exclude={'replaced_originals'})}"
                    )
            case ToolUseEvent(name=name, inputs=inputs):
                print(f"  → tool {name}({inputs})")
            case TextDeltaEvent(delta=delta):
                print(delta, end="", flush=True)
            case StreamEndEvent(usage=usage):
                turn += 1
                if usage:
                    print(f"  [turn {turn}] input_tokens={usage.input_tokens}")


async def main() -> None:
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if azure_endpoint:
        await run_provider(
            "Azure OpenAI",
            OpenAIProvider(
                base_url=azure_endpoint,
                azure_deployment_id=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            ),
            "azure-openai",
        )
    elif openai_key:
        await run_provider(
            "OpenAI",
            OpenAIProvider(model="gpt-4o-mini", api_key=openai_key),
            "openai",
        )
    else:
        print("Skipping OpenAI (set AZURE_OPENAI_ENDPOINT or OPENAI_API_KEY).")

    if gemini_key:
        await run_provider(
            "Gemini",
            GeminiProvider(model="gemini-2.5-flash", api_key=gemini_key),
            "gemini",
        )
    else:
        print("Skipping Gemini (set GEMINI_API_KEY).")


if __name__ == "__main__":
    asyncio.run(main())
