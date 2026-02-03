"""Example: Injected context with type-safe tool access.

Demonstrates that Injected[T] provides full type checking —
hover over ctx.user_name or ctx.preferences in your IDE to verify.

Usage:
    python examples/injected_context_example.py
"""

import asyncio
from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

from dobby import AgentExecutor
from dobby.providers.gemini import GeminiProvider
from dobby.tools import Injected, Tool
from dobby.types import (
    StreamEndEvent,
    TextDeltaEvent,
    TextPart,
    ToolResultEvent,
    ToolUseEvent,
    UserMessagePart,
)

load_dotenv()


# --- Context definition ---


@dataclass
class UserContext:
    """Context injected into tools at runtime. Not visible to the LLM."""

    user_name: str
    user_email: str
    preferences: dict[str, str] = field(default_factory=dict)

    def greeting(self) -> str:
        return f"Hello, {self.user_name}!"


# --- Tools that use the context ---


@dataclass
class GetUserProfileTool(Tool):
    name = "get_user_profile"
    description = "Get the current user's profile information."

    async def __call__(self, ctx: Injected[UserContext]) -> dict:
        # IDE should autocomplete ctx.user_name, ctx.user_email, ctx.preferences
        return {
            "name": ctx.user_name,
            "email": ctx.user_email,
            "preferences": ctx.preferences,
            "greeting": ctx.greeting(),
        }


@dataclass
class UpdatePreferenceTool(Tool):
    name = "update_preference"
    description = "Update a user preference."

    async def __call__(
        self,
        ctx: Injected[UserContext],
        key: str,
        value: str,
    ) -> dict:
        # IDE should resolve ctx.preferences as dict[str, str]
        ctx.preferences[key] = value
        return {"status": "updated", "key": key, "value": value, "user": ctx.user_name}


# --- Run the agent ---


async def main() -> None:
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        print("Set GEMINI_API_KEY in .env")
        return

    provider = GeminiProvider(
        api_key=gemini_key,
        model="gemini-2.5-flash",
    )

    # Create context — this is what gets injected into tools
    context = UserContext(
        user_name="Anant",
        user_email="anant@tynybay.com",
        preferences={"theme": "dark", "language": "en"},
    )

    executor = AgentExecutor(
        provider="gemini",
        llm=provider,
        tools=[GetUserProfileTool(), UpdatePreferenceTool()],
    )

    messages = [
        UserMessagePart(
            parts=[TextPart(text="What's my profile? Then set my theme preference to light.")]
        )
    ]

    system_prompt = (
        "You are a helpful assistant. Use the available tools to fetch and update user info."
    )

    print("\n--- Streaming agent response ---\n")

    async for event in executor.run_stream(
        messages,
        system_prompt=system_prompt,
        context=context,
    ):
        match event:
            case TextDeltaEvent(delta=delta):
                print(delta, end="", flush=True)
            case ToolUseEvent(name=name, inputs=inputs):
                print(f"\n[tool call] {name}({inputs})")
            case ToolResultEvent(name=name, result=result):
                print(f"[tool result] {name} -> {result}\n")
            case StreamEndEvent(usage=usage):
                if usage:
                    print(
                        f"\n\n--- Done (tokens: {usage.input_tokens} in, "
                        f"{usage.output_tokens} out) ---"
                    )

    # Show that context was mutated by the tool
    print(f"\nContext after run: preferences={context.preferences}")


if __name__ == "__main__":
    asyncio.run(main())
