"""Live integration tests against the Anthropic API (direct or Azure-hosted).

Setup:
    1. Copy .env.example to .env and fill in your keys:
           cp .env.example .env

    2. Run all live tests:
           uv run pytest tests/test_live_anthropic.py -v -s

    3. Run a specific section:
           uv run pytest tests/test_live_anthropic.py -v -s -k "tool"
           uv run pytest tests/test_live_anthropic.py -v -s -k "pdf"
           uv run pytest tests/test_live_anthropic.py -v -s -k "search"

For Azure AI Foundry set ANTHROPIC_FOUNDRY_API_KEY plus one of
ANTHROPIC_FOUNDRY_RESOURCE or ANTHROPIC_FOUNDRY_BASE_URL in your .env.
Optionally set ANTHROPIC_FOUNDRY_MODEL for the deployment name.
Azure takes priority over ANTHROPIC_API_KEY when both are present.

All tests are skipped automatically if neither API key is set.
"""

import asyncio
import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pytest
from dotenv import load_dotenv

load_dotenv()

# Direct Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Azure AI Foundry — SDK reads these automatically when not passed as constructor args.
# Provide either RESOURCE or BASE_URL (not both).
ANTHROPIC_FOUNDRY_API_KEY = os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
ANTHROPIC_FOUNDRY_RESOURCE = os.getenv("ANTHROPIC_FOUNDRY_RESOURCE")
ANTHROPIC_FOUNDRY_BASE_URL = os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
ANTHROPIC_FOUNDRY_MODEL = os.getenv("ANTHROPIC_FOUNDRY_MODEL")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

_USE_AZURE = bool(
    ANTHROPIC_FOUNDRY_API_KEY and (ANTHROPIC_FOUNDRY_RESOURCE or ANTHROPIC_FOUNDRY_BASE_URL)
)
_ANY_ANTHROPIC_KEY = _USE_AZURE or bool(ANTHROPIC_API_KEY)

requires_anthropic = pytest.mark.skipif(
    not _ANY_ANTHROPIC_KEY,
    reason="No Anthropic key — set ANTHROPIC_API_KEY (direct) or "
           "ANTHROPIC_FOUNDRY_API_KEY + ANTHROPIC_FOUNDRY_RESOURCE/BASE_URL (Azure) in .env",
)
requires_tavily = pytest.mark.skipif(
    not TAVILY_API_KEY,
    reason="TAVILY_API_KEY not set — add it to .env",
)

MODEL = ANTHROPIC_FOUNDRY_MODEL or "claude-sonnet-4-6"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _pdf_bytes(title: str, body: str) -> bytes:
    """Return raw PDF bytes from a fixture file, or generate a minimal one."""
    slug = title.lower().replace(" ", "_") + "_claim.pdf"
    path = FIXTURES_DIR / slug
    if path.exists():
        return path.read_bytes()
    # Fallback: minimal inline PDF
    t = title.replace("(", "\\(").replace(")", "\\)")
    b = body.replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 50 750 Td ({t}) Tj 0 -20 Td ({b}) Tj ET".encode("latin-1")
    objs = [
        b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n",
        b"2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n",
        b"3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>\nendobj\n",
        f"4 0 obj\n<</Length {len(stream)}>>\nstream\n".encode() + stream + b"\nendstream\nendobj\n",
        b"5 0 obj\n<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>\nendobj\n",
    ]
    header = b"%PDF-1.4\n"
    body_b, offsets, pos = b"", [], len(header)
    for obj in objs:
        offsets.append(pos)
        body_b += obj
        pos += len(obj)
    xref = f"xref\n0 6\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    trailer = f"trailer\n<</Size 6/Root 1 0 R>>\nstartxref\n{pos}\n%%EOF\n"
    return header + body_b + xref.encode() + trailer.encode()


def _pdf_b64(title: str, body: str) -> str:
    return base64.b64encode(_pdf_bytes(title, body)).decode()


# ---------------------------------------------------------------------------
# 1. Basic chat (non-streaming)
# ---------------------------------------------------------------------------


@requires_anthropic
class TestLiveNonStreaming:
    def test_simple_text_response(self):
        provider = _provider()

        result = asyncio.run(
            provider.chat(
                messages=[],
                system_prompt="Reply with exactly one word.",
            )
        )

        from dobby.types import StreamEndEvent, TextPart

        assert isinstance(result, StreamEndEvent)
        assert result.stop_reason == "end_turn"
        assert any(isinstance(p, TextPart) for p in result.parts)
        assert result.usage is not None
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0
        print(f"\n  response: {result.parts[0].text!r}")
        print(f"  usage: in={result.usage.input_tokens} out={result.usage.output_tokens}")

    def test_system_prompt_respected(self):
        from dobby.types import TextPart, UserMessagePart

        provider = _provider()
        result = asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="What is 2 + 2?")])],
                system_prompt="Always reply in JSON with key 'answer'.",
            )
        )

        text = result.parts[0].text
        print(f"\n  response: {text!r}")
        assert "answer" in text.lower() or "4" in text

    def test_max_tokens_respected(self):
        from dobby.types import TextPart, UserMessagePart

        provider = _provider()
        result = asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="Write a very long essay about AI.")])],
                max_tokens=30,
            )
        )

        assert result.usage.output_tokens <= 35  # small buffer for safety
        print(f"\n  output_tokens: {result.usage.output_tokens}")
        print(f"  stop_reason: {result.stop_reason}")


# ---------------------------------------------------------------------------
# 2. Streaming
# ---------------------------------------------------------------------------


@requires_anthropic
class TestLiveStreaming:
    def test_streaming_yields_text_deltas(self):
        from dobby.types import StreamEndEvent, StreamStartEvent, TextDeltaEvent, TextPart, UserMessagePart

        provider = _provider()
        events = []

        async def run():
            stream = await provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="Say hello in one sentence.")])],
                stream=True,
            )
            async for ev in stream:
                events.append(ev)

        asyncio.run(run())

        assert any(isinstance(e, StreamStartEvent) for e in events)
        deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert len(deltas) > 0
        full_text = "".join(d.delta for d in deltas)
        print(f"\n  streamed text: {full_text!r}")

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        assert end.stop_reason == "end_turn"
        assert end.usage.total_tokens > 0

    def test_streaming_text_matches_non_streaming(self):
        """Streamed text assembled == non-streaming text."""
        from dobby.types import StreamEndEvent, TextDeltaEvent, TextPart, UserMessagePart

        msg = [UserMessagePart(parts=[TextPart(text="What is the capital of France? One word.")])]
        provider = _provider()

        # Non-streaming
        ns_result = asyncio.run(provider.chat(messages=msg))
        ns_text = next(p.text for p in ns_result.parts if isinstance(p, TextPart))

        # Streaming
        stream_events = []

        async def run():
            stream = await provider.chat(messages=msg, stream=True)
            async for ev in stream:
                stream_events.append(ev)

        asyncio.run(run())
        s_text = "".join(e.delta for e in stream_events if isinstance(e, TextDeltaEvent))

        print(f"\n  non-streaming: {ns_text!r}")
        print(f"  streaming:     {s_text!r}")
        # Both should mention Paris
        assert "paris" in ns_text.lower() or "paris" in s_text.lower()


# ---------------------------------------------------------------------------
# 3. Tool calls
# ---------------------------------------------------------------------------


@dataclass
class CalculatorTool:
    """Performs basic arithmetic."""

    name = "calculator"
    description = "Perform arithmetic: add, subtract, multiply, or divide two numbers"

    async def __call__(
        self,
        operation: Annotated[str, "One of: add, subtract, multiply, divide"],
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> float:
        ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
        result = ops[operation]
        print(f"\n  [tool] calculator({operation}, {a}, {b}) = {result}")
        return result


@dataclass
class PolicyLookupTool:
    """Looks up insurance policy details."""

    name = "policy_lookup"
    description = "Look up an insurance policy by policy number and return its details"

    async def __call__(
        self,
        policy_number: Annotated[str, "The policy number to look up (e.g. POL-001)"],
    ) -> dict:
        db = {
            "POL-001": {"holder": "Jane Smith", "type": "Auto", "coverage": "$50,000", "status": "Active"},
            "POL-002": {"holder": "Bob Jones", "type": "Homeowners", "coverage": "$250,000", "status": "Active"},
            "POL-003": {"holder": "Alice Wu", "type": "Workers Comp", "coverage": "$100,000", "status": "Expired"},
        }
        result = db.get(policy_number, {"error": f"Policy {policy_number} not found"})
        print(f"\n  [tool] policy_lookup({policy_number!r}) = {result}")
        return result


from dobby.tools import Tool as _ToolBase


@dataclass
class LiveCalculatorTool(_ToolBase):
    name = "calculator"
    description = "Perform arithmetic: add, subtract, multiply, or divide two numbers"

    async def __call__(
        self,
        operation: Annotated[str, "One of: add, subtract, multiply, divide"],
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> float:
        ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
        result = ops[operation]
        print(f"\n  [tool] calculator({operation}, {a}, {b}) = {result}")
        return result


@dataclass
class LivePolicyLookupTool(_ToolBase):
    name = "policy_lookup"
    description = "Look up an insurance policy by policy number and return its details"

    async def __call__(
        self,
        policy_number: Annotated[str, "The policy number to look up (e.g. POL-001)"],
    ) -> str:
        db = {
            "POL-001": "Jane Smith | Auto | $50,000 coverage | Active",
            "POL-002": "Bob Jones | Homeowners | $250,000 coverage | Active",
            "POL-003": "Alice Wu | Workers Comp | $100,000 coverage | Expired",
        }
        result = db.get(policy_number, f"Policy {policy_number} not found")
        print(f"\n  [tool] policy_lookup({policy_number!r}) -> {result!r}")
        return result


@requires_anthropic
class TestLiveToolCalls:
    def test_single_tool_call_calculator(self):
        """Model calls calculator tool and returns the result."""
        from dobby.executor import AgentExecutor
        from dobby.types import StreamEndEvent, TextPart, UserMessagePart

        provider = _provider()
        executor = AgentExecutor(
            provider="anthropic",
            llm=provider,
            tools=[LiveCalculatorTool()],
        )

        events = []

        async def run():
            async for ev in executor.run_stream(
                messages=[UserMessagePart(parts=[TextPart(text="What is 347 multiplied by 52?")])],
                system_prompt="Use the calculator tool to answer math questions.",
            ):
                events.append(ev)

        asyncio.run(run())

        end_events = [e for e in events if isinstance(e, StreamEndEvent)]
        final_text = ""
        for end in end_events:
            for p in end.parts:
                if isinstance(p, TextPart):
                    final_text += p.text

        print(f"\n  final answer: {final_text!r}")
        # 347 * 52 = 18044
        assert "18044" in final_text or "18,044" in final_text

    def test_policy_lookup_tool(self):
        """Model calls policy lookup and incorporates the result."""
        from dobby.executor import AgentExecutor
        from dobby.types import StreamEndEvent, TextPart, UserMessagePart

        provider = _provider()
        executor = AgentExecutor(
            provider="anthropic",
            llm=provider,
            tools=[LivePolicyLookupTool()],
        )

        events = []

        async def run():
            async for ev in executor.run_stream(
                messages=[UserMessagePart(parts=[TextPart(text="What is the coverage for policy POL-002?")])],
                system_prompt="Use the policy_lookup tool to answer questions about insurance policies.",
            ):
                events.append(ev)

        asyncio.run(run())

        end_events = [e for e in events if isinstance(e, StreamEndEvent)]
        final_text = " ".join(
            p.text
            for end in end_events
            for p in end.parts
            if isinstance(p, TextPart)
        )

        print(f"\n  final answer: {final_text!r}")
        assert "250,000" in final_text or "250000" in final_text

    def test_multi_turn_tool_conversation(self):
        """Two sequential tool calls in the same conversation."""
        from dobby.executor import AgentExecutor
        from dobby.types import StreamEndEvent, TextPart, ToolResultEvent, UserMessagePart

        provider = _provider()
        executor = AgentExecutor(
            provider="anthropic",
            llm=provider,
            tools=[LiveCalculatorTool(), LivePolicyLookupTool()],
        )

        events = []

        async def run():
            async for ev in executor.run_stream(
                messages=[
                    UserMessagePart(
                        parts=[
                            TextPart(
                                text="Look up policy POL-001, then calculate the square of 47. "
                                     "Give me both answers."
                            )
                        ]
                    )
                ],
                system_prompt="Use tools to answer. You can call multiple tools.",
            ):
                events.append(ev)

        asyncio.run(run())

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        print(f"\n  tool calls made: {[e.name for e in tool_results]}")
        assert len(tool_results) >= 1  # at least one tool was called

        end_events = [e for e in events if isinstance(e, StreamEndEvent)]
        final_text = " ".join(
            p.text for end in end_events for p in end.parts if isinstance(p, TextPart)
        )
        print(f"  final answer: {final_text[:200]!r}")

    def test_tool_schema_sent_to_anthropic(self):
        """Verify the tool schema is correctly formatted (input_schema key)."""
        from dobby.executor import AgentExecutor

        provider = _provider()
        executor = AgentExecutor(
            provider="anthropic",
            llm=provider,
            tools=[LiveCalculatorTool(), LivePolicyLookupTool()],
        )

        schemas = executor.get_tools_schema()
        print(f"\n  schemas: {schemas}")
        assert all("input_schema" in s for s in schemas)
        assert all("name" in s and "description" in s for s in schemas)
        names = [s["name"] for s in schemas]
        assert "calculator" in names
        assert "policy_lookup" in names


# ---------------------------------------------------------------------------
# 4. Streaming tool calls
# ---------------------------------------------------------------------------


@requires_anthropic
class TestLiveStreamingToolCalls:
    def test_streaming_tool_call_events(self):
        """ToolUseEvent is yielded during streaming before StreamEndEvent."""
        from dobby.types import StreamEndEvent, TextPart, ToolUseEvent, UserMessagePart

        provider = _provider()

        tool_schema = [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            }
        ]

        events = []

        async def run():
            stream = await provider.chat(
                messages=[
                    UserMessagePart(parts=[TextPart(text="What's the weather in London? Use the get_weather tool.")])
                ],
                tools=tool_schema,
                stream=True,
            )
            async for ev in stream:
                events.append(ev)
                print(f"  [event] {type(ev).__name__}", end="")
                if isinstance(ev, ToolUseEvent):
                    print(f": {ev.name}({ev.inputs})", end="")
                print()

        asyncio.run(run())

        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert len(tool_events) >= 1
        assert tool_events[0].name == "get_weather"
        assert "city" in tool_events[0].inputs
        print(f"\n  tool inputs: {tool_events[0].inputs}")

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        assert end.stop_reason == "tool_use"


# ---------------------------------------------------------------------------
# 5. PDF / document handling (claim underwriting)
# ---------------------------------------------------------------------------


@requires_anthropic
class TestLiveDocumentHandling:
    def test_auto_liability_claim_pdf(self):
        """Send the auto liability claim PDF and ask Claude to extract key fields."""
        from dobby.types import DocumentPart, TextPart, UserMessagePart
        from dobby.types.document_part import Base64PDFSource

        provider = _provider()
        pdf_data = _pdf_b64("auto_liability", "Claim: CLM-2026-AL-00142. Loss: $9,570. Policy: POL-AUTO-8823991.")

        result = asyncio.run(
            provider.chat(
                messages=[
                    UserMessagePart(
                        parts=[
                            TextPart(text="Extract the claim number, policy number, and total loss amount from this claim document. Reply in JSON."),
                            DocumentPart(
                                source=Base64PDFSource(data=pdf_data, media_type="application/pdf"),
                                filename="auto_liability_claim.pdf",
                            ),
                        ]
                    )
                ],
                system_prompt="You are a claims processing assistant. Extract structured data from insurance documents.",
            )
        )

        text = result.parts[0].text
        print(f"\n  extracted: {text!r}")
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens > 0

    def test_property_damage_claim_pdf(self):
        """Send the property damage claim PDF and ask for underwriting recommendation."""
        from dobby.types import DocumentPart, TextPart, UserMessagePart
        from dobby.types.document_part import Base64PDFSource

        provider = _provider()

        # Use the actual fixture if it was generated
        path = FIXTURES_DIR / "property_damage_claim.pdf"
        if path.exists():
            pdf_data = base64.b64encode(path.read_bytes()).decode()
        else:
            pdf_data = _pdf_b64(
                "property_damage",
                "Commercial property water damage. Total estimate: $157,740. ACV: $108,540. Deductible: $5,000.",
            )

        result = asyncio.run(
            provider.chat(
                messages=[
                    UserMessagePart(
                        parts=[
                            TextPart(text="Review this property damage claim. Should we approve or investigate further? Give a brief underwriting decision with reasoning."),
                            DocumentPart(
                                source=Base64PDFSource(data=pdf_data, media_type="application/pdf"),
                                filename="property_damage_claim.pdf",
                            ),
                        ]
                    )
                ],
                system_prompt="You are a senior property underwriter. Provide concise, professional claim decisions.",
                max_tokens=300,
            )
        )

        text = result.parts[0].text
        print(f"\n  underwriting decision:\n  {text!r}")
        assert len(text) > 20

    def test_multi_pdf_comparison(self):
        """Send two claim PDFs and ask Claude to compare them."""
        from dobby.types import DocumentPart, TextPart, UserMessagePart
        from dobby.types.document_part import Base64PDFSource

        provider = _provider()

        pdf1 = _pdf_b64("workers_comp", "Workers comp knee injury. Reserve: $44,704. Status: Surgery scheduled.")
        pdf2 = _pdf_b64("medical_malpractice", "Medical malpractice bile duct injury. Demand: $1,250,000. Reserve: $1,075,000.")

        result = asyncio.run(
            provider.chat(
                messages=[
                    UserMessagePart(
                        parts=[
                            TextPart(text="Compare these two claims. Which has higher financial exposure and why? Be brief."),
                            DocumentPart(
                                source=Base64PDFSource(data=pdf1, media_type="application/pdf"),
                                filename="workers_comp_claim.pdf",
                            ),
                            DocumentPart(
                                source=Base64PDFSource(data=pdf2, media_type="application/pdf"),
                                filename="medical_malpractice_claim.pdf",
                            ),
                        ]
                    )
                ],
                system_prompt="You are a claims triage specialist.",
                max_tokens=200,
            )
        )

        text = result.parts[0].text
        print(f"\n  comparison: {text!r}")
        assert len(text) > 20


# ---------------------------------------------------------------------------
# 6. Extended thinking (reasoning)
# ---------------------------------------------------------------------------


@requires_anthropic
class TestLiveExtendedThinking:
    def test_reasoning_produces_thinking_part(self):
        """With reasoning_effort set, response includes a ReasoningPart."""
        from dobby.types import ReasoningPart, TextPart, UserMessagePart

        provider = _provider()
        result = asyncio.run(
            provider.chat(
                messages=[
                    UserMessagePart(parts=[TextPart(text="A claim has 3 witnesses. Two say the insured was at fault, one says the third party was at fault. What liability split would you recommend and why?")])
                ],
                reasoning_effort=5000,
            )
        )

        reasoning = [p for p in result.parts if isinstance(p, ReasoningPart)]
        text = [p for p in result.parts if isinstance(p, TextPart)]

        print(f"\n  thinking length: {len(reasoning[0].text) if reasoning else 0} chars")
        print(f"  answer: {text[0].text[:200] if text else '(none)'!r}")

        assert len(reasoning) >= 1
        assert len(reasoning[0].text) > 10
        assert len(text) >= 1

    def test_streaming_thinking_events(self):
        """Streaming with extended thinking yields ReasoningStart/Delta/End events."""
        from dobby.types import (
            ReasoningDeltaEvent,
            ReasoningEndEvent,
            ReasoningStartEvent,
            StreamEndEvent,
            TextPart,
            UserMessagePart,
        )

        provider = _provider()
        events = []

        async def run():
            stream = await provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="What is 97 * 83?")])],
                stream=True,
                reasoning_effort=3000,
            )
            async for ev in stream:
                events.append(ev)

        asyncio.run(run())

        assert any(isinstance(e, ReasoningStartEvent) for e in events)
        assert any(isinstance(e, ReasoningEndEvent) for e in events)
        rdelta = [e for e in events if isinstance(e, ReasoningDeltaEvent)]
        thinking_text = "".join(e.delta for e in rdelta)
        print(f"\n  thinking ({len(thinking_text)} chars): {thinking_text[:100]!r}...")

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        from dobby.types import TextPart as TP
        answer = next((p.text for p in end.parts if isinstance(p, TP)), "")
        print(f"  answer: {answer!r}")
        # 97 * 83 = 8051
        assert "8051" in answer or "8,051" in answer


# ---------------------------------------------------------------------------
# 7. Web search (Tavily)
# ---------------------------------------------------------------------------


@requires_anthropic
@requires_tavily
class TestLiveWebSearch:
    def test_tavily_search_tool_with_executor(self):
        """Real web search via Tavily integrated with Anthropic provider."""
        from dobby.common_tools import TavilySearchTool
        from dobby.executor import AgentExecutor
        from dobby.types import StreamEndEvent, TextPart, ToolResultEvent, UserMessagePart

        provider = _provider()
        executor = AgentExecutor(
            provider="anthropic",
            llm=provider,
            tools=[TavilySearchTool(api_key=TAVILY_API_KEY)],
        )

        events = []

        async def run():
            async for ev in executor.run_stream(
                messages=[
                    UserMessagePart(
                        parts=[TextPart(text="What are the latest changes to insurance regulations in 2025? Search the web.")]
                    )
                ],
                system_prompt="Use the tavily_search tool to find current information.",
            ):
                events.append(ev)

        asyncio.run(run())

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        print(f"\n  tools called: {[e.name for e in tool_results]}")
        assert len(tool_results) >= 1

        end_events = [e for e in events if isinstance(e, StreamEndEvent)]
        final_text = " ".join(
            p.text for end in end_events for p in end.parts if isinstance(p, TextPart)
        )
        print(f"  answer (first 300 chars): {final_text[:300]!r}")
        assert len(final_text) > 50


# ---------------------------------------------------------------------------
# 8. Usage / token tracking
# ---------------------------------------------------------------------------


@requires_anthropic
class TestLiveUsageTracking:
    def test_usage_fields_present(self):
        """Input and output tokens are returned and sane."""
        from dobby.types import TextPart, UserMessagePart

        provider = _provider()
        result = asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="Hello.")])],
            )
        )

        u = result.usage
        assert u is not None
        assert u.input_tokens > 0
        assert u.output_tokens > 0
        assert u.total_tokens == u.input_tokens + u.output_tokens
        print(f"\n  in={u.input_tokens} out={u.output_tokens} total={u.total_tokens}")
        print(f"  cache_creation={u.cache_creation_input_tokens} cache_read={u.cache_read_input_tokens}")

    def test_streaming_usage_fields_present(self):
        """Streaming also returns usage in the StreamEndEvent."""
        from dobby.types import StreamEndEvent, TextPart, UserMessagePart

        provider = _provider()
        events = []

        async def run():
            stream = await provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="Hi.")])],
                stream=True,
            )
            async for ev in stream:
                events.append(ev)

        asyncio.run(run())

        end = next(e for e in events if isinstance(e, StreamEndEvent))
        u = end.usage
        assert u is not None
        assert u.input_tokens > 0
        assert u.output_tokens > 0
        assert u.total_tokens == u.input_tokens + u.output_tokens
        print(f"\n  streaming usage: in={u.input_tokens} out={u.output_tokens}")

    def test_larger_prompt_has_more_input_tokens(self):
        """More text in the prompt = more input tokens."""
        from dobby.types import TextPart, UserMessagePart

        provider = _provider()

        short = asyncio.run(
            provider.chat(messages=[UserMessagePart(parts=[TextPart(text="Hi.")])])
        )
        long = asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="Hi. " * 50)])]
            )
        )

        print(f"\n  short prompt: {short.usage.input_tokens} input tokens")
        print(f"  long prompt:  {long.usage.input_tokens} input tokens")
        assert long.usage.input_tokens > short.usage.input_tokens


# ---------------------------------------------------------------------------
# 9. Multi-turn conversation
# ---------------------------------------------------------------------------


@requires_anthropic
class TestLiveMultiTurn:
    def test_conversation_context_preserved(self):
        """Model remembers earlier turns in the conversation."""
        from dobby.types import AssistantMessagePart, TextPart, UserMessagePart

        provider = _provider()

        # Turn 1
        r1 = asyncio.run(
            provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text="My name is Alex. Remember it.")])],
            )
        )
        print(f"\n  turn 1 response: {r1.parts[0].text!r}")

        # Turn 2 — model should remember the name
        r2 = asyncio.run(
            provider.chat(
                messages=[
                    UserMessagePart(parts=[TextPart(text="My name is Alex. Remember it.")]),
                    AssistantMessagePart(parts=r1.parts),
                    UserMessagePart(parts=[TextPart(text="What is my name?")]),
                ],
            )
        )
        print(f"  turn 2 response: {r2.parts[0].text!r}")
        assert "alex" in r2.parts[0].text.lower()

    def test_tool_result_context_in_follow_up(self):
        """Tool result from turn 1 is available in turn 2's reasoning."""
        from dobby.executor import AgentExecutor
        from dobby.types import StreamEndEvent, TextPart, UserMessagePart

        provider = _provider()
        executor = AgentExecutor(
            provider="anthropic",
            llm=provider,
            tools=[LivePolicyLookupTool()],
        )

        events = []

        async def run():
            async for ev in executor.run_stream(
                messages=[
                    UserMessagePart(
                        parts=[
                            TextPart(
                                text="Look up policy POL-001 and tell me if the policyholder's coverage is sufficient for a $30,000 auto claim."
                            )
                        ]
                    )
                ],
                system_prompt="Use tools to look up policy data, then provide underwriting analysis.",
            ):
                events.append(ev)

        asyncio.run(run())

        end_events = [e for e in events if isinstance(e, StreamEndEvent)]
        final_text = " ".join(
            p.text for end in end_events for p in end.parts if isinstance(p, TextPart)
        )
        print(f"\n  analysis: {final_text[:300]!r}")
        # Should mention the policy details and the $50k coverage
        assert len(final_text) > 30
