"""Quick smoke test — query Anthropic adapter directly.

Usage:
    uv run python tests/test_anthropic_provider.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_FOUNDRY_API_KEY = os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
ANTHROPIC_FOUNDRY_RESOURCE = os.getenv("ANTHROPIC_FOUNDRY_RESOURCE")
ANTHROPIC_FOUNDRY_BASE_URL = os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
ANTHROPIC_FOUNDRY_MODEL = os.getenv("ANTHROPIC_FOUNDRY_MODEL")

_USE_AZURE = bool(
    ANTHROPIC_FOUNDRY_API_KEY and (ANTHROPIC_FOUNDRY_RESOURCE or ANTHROPIC_FOUNDRY_BASE_URL)
)

MODEL = ANTHROPIC_FOUNDRY_MODEL or "claude-sonnet-4-6"


def _provider():
    from dobby.providers.anthropic import AnthropicProvider

    if _USE_AZURE:
        return AnthropicProvider(
            model=MODEL,
            api_key=ANTHROPIC_FOUNDRY_API_KEY,
            resource=ANTHROPIC_FOUNDRY_RESOURCE,
            base_url=ANTHROPIC_FOUNDRY_BASE_URL,
        )
    return AnthropicProvider(model=MODEL, api_key=ANTHROPIC_API_KEY)


async def main() -> None:
    from dobby.types import TextPart, UserMessagePart

    provider = _provider()
    print(f"Provider : {provider.name}")
    print(f"Model    : {provider.model}")
    print("Querying...")

    result = await provider.chat(
        messages=[UserMessagePart(parts=[TextPart(text="Say hello in one sentence.")])],
    )

    print(f"Response : {result.parts[0].text!r}")
    if result.usage:
        print(f"Usage    : in={result.usage.input_tokens} out={result.usage.output_tokens}")


if __name__ == "__main__":
    if not (_USE_AZURE or ANTHROPIC_API_KEY):
        print(
            "\nERROR: No API key.\n"
            "Set ANTHROPIC_API_KEY or\n"
            "ANTHROPIC_FOUNDRY_API_KEY + ANTHROPIC_FOUNDRY_RESOURCE/BASE_URL in .env\n"
        )
        raise SystemExit(1)

    asyncio.run(main())