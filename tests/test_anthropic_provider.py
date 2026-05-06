"""Anthropic adapter smoke test — basic chat + PDF claim extraction.

Usage:
    uv run python tests/test_anthropic_provider.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path

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
FIXTURES = Path(__file__).parent / "fixtures"

CLAIMS = [
    {
        "file": "auto_liability_claim.pdf",
        "label": "Auto Liability",
        "fields": {"claim_number": "CLM-2026-AL-00142", "policy_number": "POL-AUTO-8823991", "total_estimate": "9570"},
    },
    {
        "file": "property_damage_claim.pdf",
        "label": "Property Damage",
        "fields": {"claim_number": "CLM-2026-PD-00308", "business_name": "Hartfield Printing", "deductible": "5000"},
    },
    {
        "file": "workers_comp_claim.pdf",
        "label": "Workers Comp",
        "fields": {"claim_number": "CLM-2026-WC-00091", "employee_name": "Carlos M. Reyes", "total_reserve": "44704"},
    },
    {
        "file": "medical_malpractice_claim.pdf",
        "label": "Medical Malpractice",
        "fields": {"claim_number": "CLM-2026-MM-00017", "demand_amount": "1250000", "total_reserve": "1075000"},
    },
    {
        "file": "homeowners_claim.pdf",
        "label": "Homeowners",
        "fields": {"claim_number": "CLM-2026-HO-00522", "total_rcv": "26490", "deductible": "2500"},
    },
]

EXTRACTION_PROMPT = (
    "Extract all structured fields from this insurance claim document. "
    "Return ONLY a valid JSON object — no markdown fences, no explanation. "
    "Include every claim number, policy number, name, date, and dollar amount you find."
)


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


def _parse_json(raw: str) -> dict:
    clean = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    clean = re.sub(r"\n?```$", "", clean.strip())
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}


def _field_hit(extracted: dict, value: str) -> bool:
    flat = json.dumps(extracted).lower().replace(",", "").replace("$", "")
    return value.lower().replace(",", "").replace("$", "") in flat


async def test_basic_chat(provider) -> None:
    from dobby.types import TextPart, UserMessagePart

    print("\n[Basic Chat]")
    result = await provider.chat(
        messages=[UserMessagePart(parts=[TextPart(text="Say hello in one sentence.")])],
    )
    print(f"  Response : {result.parts[0].text!r}")
    if result.usage:
        print(f"  Usage    : in={result.usage.input_tokens} out={result.usage.output_tokens}")


async def test_claim_extraction(provider, claim: dict) -> None:
    from dobby.types import DocumentPart, TextPart, UserMessagePart
    from dobby.types.document_part import Base64PDFSource

    path = FIXTURES / claim["file"]
    if not path.exists():
        print(f"  SKIP — {claim['file']} not found (run: python tests/fixtures/generate_claim_pdfs.py)")
        return

    pdf_b64 = base64.b64encode(path.read_bytes()).decode()
    result = await provider.chat(
        messages=[
            UserMessagePart(parts=[
                TextPart(text=EXTRACTION_PROMPT),
                DocumentPart(
                    source=Base64PDFSource(data=pdf_b64, media_type="application/pdf"),
                    filename=claim["file"],
                ),
            ])
        ],
        system_prompt="You are a precise claims data extraction system. Output only valid JSON.",
        max_tokens=1024,
        temperature=0.0,
    )

    raw = next((p.text for p in result.parts if hasattr(p, "text")), "")
    extracted = _parse_json(raw)

    passed = sum(_field_hit(extracted, v) for v in claim["fields"].values())
    total = len(claim["fields"])
    pct = passed / total * 100

    print(f"\n[{claim['label']}]  {passed}/{total} fields ({pct:.0f}%)")
    for field, expected in claim["fields"].items():
        hit = _field_hit(extracted, expected)
        print(f"  {'✓' if hit else '✗'} {field:<25} {expected!r}")
    if result.usage:
        print(f"  tokens: in={result.usage.input_tokens} out={result.usage.output_tokens}")

    print(f"  Claude response:")
    if extracted:
        for line in json.dumps(extracted, indent=4).splitlines():
            print(f"    {line}")
    else:
        print(f"    [JSON parse failed] Raw: {raw[:300]}")


async def main() -> None:
    provider = _provider()
    print(f"Provider : {provider.name}")
    print(f"Model    : {provider.model}")

    await test_basic_chat(provider)

    print("\n--- Claim Extraction ---")
    for claim in CLAIMS:
        await test_claim_extraction(provider, claim)


if __name__ == "__main__":
    if not (_USE_AZURE or ANTHROPIC_API_KEY):
        print(
            "\nERROR: No API key.\n"
            "Set ANTHROPIC_API_KEY or\n"
            "ANTHROPIC_FOUNDRY_API_KEY + ANTHROPIC_FOUNDRY_RESOURCE/BASE_URL in .env\n"
        )
        raise SystemExit(1)

    asyncio.run(main())