#!/usr/bin/env python
"""End-to-end Gemini structured-output example.

Exercises the exact path the Gemini schema fix targets: a nested Pydantic model
used as the structured-output type. Before the fix, building the tool schema
raised a ValidationError locally (Gemini's restricted `parameters=` field
rejects $ref/$defs); now the schema goes through `parameters_json_schema` and
Gemini dereferences it server-side.

Usage:
    export GEMINI_API_KEY=...   # or set it in .env
    python examples/gemini_structured_output.py
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dobby import AgentExecutor
from dobby.providers.gemini import GeminiProvider
from dobby.types import StreamEndEvent, TextDeltaEvent, TextPart, UserMessagePart

load_dotenv()


# --- Nested output model (this is what used to break Gemini) -------------------


class SocialLink(BaseModel):
    platform: str = Field(description="Platform name, e.g. GitHub or LinkedIn")
    url: str


class Skill(BaseModel):
    name: str
    years_experience: int = Field(ge=0, le=60)


class ParsedResume(BaseModel):
    """Structured resume extraction with nested sub-models."""

    full_name: str
    headline: str
    social_links: list[SocialLink]
    skills: list[Skill]


RESUME_TEXT = """
Jane Doe — Senior ML Engineer

Find me at GitHub (https://github.com/janedoe) and
LinkedIn (https://linkedin.com/in/janedoe).

Experience: 6 years of Python, 4 years of PyTorch, 3 years of Kubernetes.
"""


async def main() -> None:
    provider = GeminiProvider(model="gemini-2.5-flash")

    executor: AgentExecutor[None, ParsedResume] = AgentExecutor(
        provider="gemini",
        llm=provider,
        output_type=ParsedResume,
    )

    print("→ Sending nested-model structured-output request to Gemini...\n")

    async for event in executor.run_stream(
        messages=[
            UserMessagePart(
                parts=[
                    TextPart(
                        text=(
                            "Extract this resume into the final_result tool.\n\n"
                            + RESUME_TEXT
                        )
                    )
                ]
            )
        ],
        system_prompt="You extract structured data. Always call final_result.",
    ):
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)
        elif isinstance(event, StreamEndEvent):
            print(f"  [stop_reason={event.stop_reason}]")

    result = executor.last_output
    if result is None:
        raise SystemExit("✗ No structured output returned — model did not call final_result")

    print("\n✓ Validated structured output:\n")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
