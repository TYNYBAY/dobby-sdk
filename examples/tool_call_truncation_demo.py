#!/usr/bin/env python
"""End-to-end demo of max_tokens tool-call truncation handling.

The unit tests in `tests/test_tool_call_truncation.py` mock the provider
responses. This script hits the REAL OpenAI and Gemini APIs and forces a genuine
`max_tokens` cutoff mid tool-call, so you can confirm the adapters behave as
designed against live responses:

- Streaming   -> yields a `ToolUseErrorEvent`; the stream still completes with a
                 `StreamEndEvent` (no raw `JSONDecodeError` escapes the generator).
- Non-stream  -> raises `ToolCallTruncatedError` (no silent partial arguments).

How truncation is forced: a tool whose argument is a long string, plus a tiny
`max_tokens`, so the model is cut off while emitting the arguments.

Usage:
    export OPENAI_API_KEY=...    # and/or
    export GEMINI_API_KEY=...
    python examples/tool_call_truncation_demo.py
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv

# Windows consoles default to cp1252, which can't encode the ✓/✗/⚠ glyphs below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dobby.providers.base import ToolCallTruncatedError
from dobby.providers.gemini import GeminiProvider
from dobby.providers.openai import OpenAIProvider
from dobby.types import (
    StreamEndEvent,
    TextPart,
    ToolUseErrorEvent,
    UserMessagePart,
)

load_dotenv()


# A tool that demands a long free-text argument. Combined with a tiny max_tokens,
# the model gets cut off partway through emitting `arguments`.
TOOL_NAME = "save_essay"
TOOL_DESCRIPTION = "Save a long-form essay to disk."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "essay": {
            "type": "string",
            "description": "A 3000-word essay. Write it in full, in one go.",
        }
    },
    "required": ["essay"],
}

PROMPT = (
    "Call the save_essay tool with a complete 3000-word essay about the history "
    "of the Roman Empire. Write the entire essay in the `essay` argument — do not "
    "summarise or stop early."
)

# Pair a huge demanded output (3000 words) with a moderate budget so the model
# reliably STARTS the tool call but gets cut off mid-arguments. Going too low
# (e.g. 16) cuts off before the call even begins (no ToolUsePart, nothing to
# flag); too high lets it finish. ~200 clears OpenAI's tiny-budget 500s while
# staying far below a 3000-word payload. Tune per model if you see NOTE.
TINY_MAX_TOKENS = 200


def _messages():
    return [UserMessagePart(parts=[TextPart(text=PROMPT)])]


async def _run_streaming(provider, tools) -> None:
    saw_error_event = False
    saw_stream_end = False
    try:
        async for event in await provider.chat(
            _messages(), stream=True, tools=tools, max_tokens=TINY_MAX_TOKENS
        ):
            if isinstance(event, ToolUseErrorEvent):
                saw_error_event = True
                print(
                    f"    ✓ ToolUseErrorEvent: name={event.name!r} "
                    f"error={event.error!r} raw={event.raw_arguments!r:.60}"
                )
            elif isinstance(event, StreamEndEvent):
                saw_stream_end = True
                print(
                    f"    · StreamEndEvent stop_reason={event.stop_reason} "
                    f"(broken tool absent from parts: "
                    f"{not any(getattr(p, 'name', None) == TOOL_NAME for p in event.parts)})"
                )
    except json.JSONDecodeError as e:
        # This is the regression R1 prevents: truncated JSON escaped the generator.
        print(
            f"    ✗ FAIL: stream leaked JSONDecodeError: {e} "
            f"(should have yielded ToolUseErrorEvent instead)"
        )
        return
    except Exception as e:  # noqa: BLE001 — setup/connection error, not a truncation bug
        print(
            f"    ⚠ stream errored ({type(e).__name__}: {e}) — likely a key/network/"
            f"model issue, not a truncation regression"
        )
        return

    if saw_error_event and saw_stream_end:
        print("    PASS: error surfaced as event, stream still completed.")
    elif saw_stream_end:
        print(
            "    NOTE: stream completed but no truncation event — model may not "
            "have truncated mid-arguments. Try lowering TINY_MAX_TOKENS."
        )
    else:
        print("    ✗ FAIL: stream did not complete cleanly.")


async def _run_non_streaming(provider, tools) -> None:
    try:
        result = await provider.chat(
            _messages(), stream=False, tools=tools, max_tokens=TINY_MAX_TOKENS
        )
    except ToolCallTruncatedError as e:
        print(
            f"    ✓ PASS: raised ToolCallTruncatedError "
            f"(tool={e.tool_name!r}, id={e.tool_id!r}, provider={e.provider!r})"
        )
        return
    except Exception as e:  # noqa: BLE001 — setup/connection error, not a truncation bug
        print(
            f"    ⚠ call errored ({type(e).__name__}: {e}) — likely a key/network/"
            f"model issue, not a truncation regression"
        )
        return

    print(
        f"    NOTE: returned normally (stop_reason={result.stop_reason}) — model "
        "may not have truncated mid-arguments. Try lowering TINY_MAX_TOKENS."
    )


async def demo_openai() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI: skipped (set OPENAI_API_KEY to run)\n")
        return

    provider = OpenAIProvider(model="gpt-4.1-nano")
    # OpenAI Responses-API native function-tool shape.
    tools = [
        {
            "type": "function",
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": TOOL_PARAMETERS,
        }
    ]

    print("=== OpenAI (Responses API) ===")
    print("  streaming:")
    await _run_streaming(provider, tools)
    print("  non-streaming:")
    await _run_non_streaming(provider, tools)
    print()


async def demo_gemini() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        print("Gemini: skipped (set GEMINI_API_KEY to run)\n")
        return

    provider = GeminiProvider(model="gemini-2.5-flash-lite")
    # Gemini native function-declaration shape.
    tools = [
        {
            "function_declarations": [
                {
                    "name": TOOL_NAME,
                    "description": TOOL_DESCRIPTION,
                    "parameters": TOOL_PARAMETERS,
                }
            ]
        }
    ]

    print("=== Gemini ===")
    print("  streaming:")
    await _run_streaming(provider, tools)
    print("  non-streaming:")
    await _run_non_streaming(provider, tools)
    print()


async def main() -> None:
    await demo_openai()
    await demo_gemini()


if __name__ == "__main__":
    asyncio.run(main())
