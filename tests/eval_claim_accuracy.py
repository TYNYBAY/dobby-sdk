"""Claim extraction accuracy eval.

Sends each claim PDF to the provider, asks for structured JSON extraction,
and compares against known ground truth. Reports per-field accuracy and latency.

Usage:
    uv run python tests/eval_claim_accuracy.py
    uv run python tests/eval_claim_accuracy.py --model claude-sonnet-4-6
    uv run python tests/eval_claim_accuracy.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

FIXTURES_DIR = Path(__file__).parent / "fixtures"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_FOUNDRY_API_KEY = os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
ANTHROPIC_FOUNDRY_RESOURCE = os.getenv("ANTHROPIC_FOUNDRY_RESOURCE")
ANTHROPIC_FOUNDRY_BASE_URL = os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
ANTHROPIC_FOUNDRY_MODEL = os.getenv("ANTHROPIC_FOUNDRY_MODEL")

_USE_AZURE = bool(
    ANTHROPIC_FOUNDRY_API_KEY and (ANTHROPIC_FOUNDRY_RESOURCE or ANTHROPIC_FOUNDRY_BASE_URL)
)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

@dataclass
class ClaimGroundTruth:
    pdf_file: str
    description: str
    expected: dict[str, str]


GROUND_TRUTH: list[ClaimGroundTruth] = [
    ClaimGroundTruth(
        pdf_file="auto_liability_claim.pdf",
        description="Auto liability - rear-end collision",
        expected={
            "claim_number": "CLM-2026-AL-00142",
            "policy_number": "POL-AUTO-8823991",
            "insured_name": "Jane R. Hartwell",
            "date_of_loss": "2026-03-15",
            "total_estimate": "$9,570.00",
            "reserve": "$14,500",
        },
    ),
    ClaimGroundTruth(
        pdf_file="property_damage_claim.pdf",
        description="Commercial property - burst water main",
        expected={
            "claim_number": "CLM-2026-PD-00308",
            "policy_number": "POL-COMM-1143772",
            "business_name": "Hartfield Printing & Design LLC",
            "date_of_loss": "2026-02-28",
            "total_rcv": "$157,740",
            "net_payable_acv": "$103,540",
            "deductible": "$5,000",
        },
    ),
    ClaimGroundTruth(
        pdf_file="workers_comp_claim.pdf",
        description="Workers comp - knee injury",
        expected={
            "claim_number": "CLM-2026-WC-00091",
            "policy_number": "POL-WC-5567234",
            "employee_name": "Carlos M. Reyes",
            "date_of_injury": "2026-01-22",
            "body_part": "Left knee",
            "total_reserve": "$44,704",
            "avg_weekly_wage": "$1,142.00",
        },
    ),
    ClaimGroundTruth(
        pdf_file="medical_malpractice_claim.pdf",
        description="Medical malpractice - bile duct injury",
        expected={
            "claim_number": "CLM-2026-MM-00017",
            "policy_number": "POL-MPL-0023491",
            "provider_name": "Dr. Kevin J. Allard",
            "claimant_name": "Patricia L. Vasquez",
            "demand_amount": "$1,250,000",
            "total_reserve": "$1,075,000",
            "coverage_limits": "$1M/$3M",
        },
    ),
    ClaimGroundTruth(
        pdf_file="homeowners_claim.pdf",
        description="Homeowners - wind/hail damage",
        expected={
            "claim_number": "CLM-2026-HO-00522",
            "policy_number": "POL-HO3-7734128",
            "insured_name": "Marcus & Diana Webb",
            "date_of_loss": "2026-04-02",
            "total_rcv": "$26,490",
            "net_acv_payment": "$15,670",
            "deductible": "$2,500",
        },
    ),
]

EXTRACTION_PROMPT = """\
Extract all key fields from this insurance claim document and return ONLY a JSON object.
Include every field you can find: claim numbers, policy numbers, names, dates, dollar amounts,
reserve figures, deductibles, and any other structured data.
Return ONLY valid JSON, no markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# Field matching
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Strip formatting for loose comparison."""
    s = s.lower().strip()
    s = re.sub(r"[\$,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _field_match(extracted: dict[str, Any], expected_value: str) -> bool:
    """Check if expected_value appears anywhere in the extracted JSON (loose match)."""
    norm_expected = _normalize(expected_value)
    extracted_str = _normalize(json.dumps(extracted))
    return norm_expected in extracted_str


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------

def _make_provider(model: str):
    from dobby.providers.anthropic import AnthropicProvider

    if _USE_AZURE:
        return AnthropicProvider(
            model=model,
            api_key=ANTHROPIC_FOUNDRY_API_KEY,
            resource=ANTHROPIC_FOUNDRY_RESOURCE,
            base_url=ANTHROPIC_FOUNDRY_BASE_URL,
        )
    return AnthropicProvider(model=model, api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# Single claim eval
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    field: str
    expected: str
    found: bool
    extracted_json: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimResult:
    description: str
    pdf_file: str
    field_results: list[FieldResult]
    latency_s: float
    error: str | None = None
    raw_response: str = ""

    @property
    def accuracy(self) -> float:
        if not self.field_results:
            return 0.0
        return sum(1 for r in self.field_results if r.found) / len(self.field_results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.field_results if r.found)

    @property
    def total(self) -> int:
        return len(self.field_results)


async def eval_claim(
    provider,
    ground_truth: ClaimGroundTruth,
    verbose: bool = False,
) -> ClaimResult:
    from dobby.types import DocumentPart, StreamEndEvent, TextPart, UserMessagePart
    from dobby.types.document_part import Base64PDFSource

    pdf_path = FIXTURES_DIR / ground_truth.pdf_file
    if not pdf_path.exists():
        return ClaimResult(
            description=ground_truth.description,
            pdf_file=ground_truth.pdf_file,
            field_results=[],
            latency_s=0.0,
            error=f"PDF not found: {pdf_path}",
        )

    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode()

    start = time.perf_counter()
    try:
        result: StreamEndEvent = await provider.chat(
            messages=[
                UserMessagePart(
                    parts=[
                        TextPart(text=EXTRACTION_PROMPT),
                        DocumentPart(
                            source=Base64PDFSource(data=pdf_b64, media_type="application/pdf"),
                            filename=ground_truth.pdf_file,
                        ),
                    ]
                )
            ],
            system_prompt="You are a precise claims data extraction system. Output only valid JSON.",
            max_tokens=1024,
            temperature=0.0,
        )
    except Exception as exc:
        latency = time.perf_counter() - start
        return ClaimResult(
            description=ground_truth.description,
            pdf_file=ground_truth.pdf_file,
            field_results=[],
            latency_s=latency,
            error=str(exc),
        )

    latency = time.perf_counter() - start
    raw_text = next((p.text for p in result.parts if isinstance(p, TextPart)), "")

    # Parse JSON — strip markdown fences if model included them
    json_text = re.sub(r"^```[a-z]*\n?", "", raw_text.strip())
    json_text = re.sub(r"\n?```$", "", json_text.strip())

    try:
        extracted = json.loads(json_text)
    except json.JSONDecodeError:
        extracted = {}
        if verbose:
            print(f"  [warn] JSON parse failed. Raw:\n{raw_text[:500]}")

    field_results = [
        FieldResult(
            field=field_name,
            expected=expected_value,
            found=_field_match(extracted, expected_value),
            extracted_json=extracted,
        )
        for field_name, expected_value in ground_truth.expected.items()
    ]

    return ClaimResult(
        description=ground_truth.description,
        pdf_file=ground_truth.pdf_file,
        field_results=field_results,
        latency_s=latency,
        raw_response=raw_text,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[ClaimResult], verbose: bool = False) -> None:
    print("\n" + "=" * 70)
    print("CLAIM EXTRACTION ACCURACY REPORT")
    print("=" * 70)

    total_fields = 0
    total_passed = 0

    for r in results:
        status = "ERROR" if r.error else f"{r.passed}/{r.total} fields"
        pct = f"{r.accuracy * 100:.0f}%" if not r.error else "N/A"
        latency = f"{r.latency_s:.2f}s"

        print(f"\n{r.description}")
        print(f"  File:     {r.pdf_file}")
        print(f"  Accuracy: {pct}  ({status})  [{latency}]")

        if r.error:
            print(f"  Error:    {r.error}")
            continue

        for fr in r.field_results:
            mark = "✓" if fr.found else "✗"
            print(f"    {mark} {fr.field:<25} expected: {fr.expected}")

        if verbose and r.raw_response:
            print(f"\n  Raw response:\n{r.raw_response[:800]}")

        total_fields += r.total
        total_passed += r.passed

    print("\n" + "-" * 70)
    overall_pct = (total_passed / total_fields * 100) if total_fields else 0
    print(f"OVERALL ACCURACY: {total_passed}/{total_fields} fields  ({overall_pct:.1f}%)")
    avg_latency = sum(r.latency_s for r in results) / len(results) if results else 0
    print(f"AVG LATENCY:      {avg_latency:.2f}s per claim")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(model: str, verbose: bool) -> None:
    if not (_USE_AZURE or ANTHROPIC_API_KEY):
        print(
            "ERROR: No API key configured.\n"
            "Set ANTHROPIC_API_KEY (direct) or\n"
            "ANTHROPIC_FOUNDRY_API_KEY + ANTHROPIC_FOUNDRY_RESOURCE/BASE_URL (Azure) in .env"
        )
        return

    provider = _make_provider(model)
    print(f"Provider: {provider.name}  Model: {model}")
    print(f"Running eval on {len(GROUND_TRUTH)} claim PDFs...\n")

    results = []
    for gt in GROUND_TRUTH:
        print(f"  Evaluating: {gt.description}...", end="", flush=True)
        result = await eval_claim(provider, gt, verbose=verbose)
        status = f"{result.passed}/{result.total}" if not result.error else "ERROR"
        print(f" {status}  ({result.latency_s:.1f}s)")
        results.append(result)

    print_report(results, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim extraction accuracy eval")
    parser.add_argument(
        "--model",
        default=ANTHROPIC_FOUNDRY_MODEL or "claude-sonnet-4-6",
        help="Model name or Azure deployment name",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw model responses",
    )
    args = parser.parse_args()
    asyncio.run(run(model=args.model, verbose=args.verbose))


if __name__ == "__main__":
    main()
