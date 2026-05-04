"""Travel insurance claim extraction — accuracy and performance test.

Usage:
    uv run python tests/test_anthropic_provider.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, computed_field

load_dotenv()

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    api_key: str | None = None
    foundry_api_key: str | None = None
    foundry_resource: str | None = None
    foundry_base_url: str | None = None
    model: str = "claude-sonnet-4-6"

    @computed_field
    @property
    def use_azure(self) -> bool:
        return bool(self.foundry_api_key and (self.foundry_resource or self.foundry_base_url))

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            foundry_api_key=os.getenv("ANTHROPIC_FOUNDRY_API_KEY"),
            foundry_resource=os.getenv("ANTHROPIC_FOUNDRY_RESOURCE"),
            foundry_base_url=os.getenv("ANTHROPIC_FOUNDRY_BASE_URL"),
            model=os.getenv("ANTHROPIC_FOUNDRY_MODEL", "claude-sonnet-4-6"),
        )


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

class ClaimGroundTruth(BaseModel):
    file: str
    label: str
    fields: dict[str, str]


GROUND_TRUTH: list[ClaimGroundTruth] = [
    ClaimGroundTruth(
        file="trip_cancellation_claim.pdf",
        label="Trip Cancellation",
        fields={
            "claim_number": "CLM-2026-TC-00881",
            "policy_number": "POL-TRVL-4492017",
            "traveler_name": "Emily R. Thornton",
            "destination": "Rome",
            "departure_date": "2026-03-20",
            "total_trip_cost": "8450",
            "net_claim_amount": "7540",
            "deductible": "250",
            "cancellation_reason": "appendicitis",
        },
    ),
    ClaimGroundTruth(
        file="trip_interruption_claim.pdf",
        label="Trip Interruption",
        fields={
            "claim_number": "CLM-2026-TI-00334",
            "policy_number": "POL-TRVL-3381092",
            "traveler_name": "Kim",
            "destination": "Bali",
            "total_trip_cost": "12600",
            "net_claim_amount": "6690",
            "deductible": "500",
            "interruption_reason": "stroke",
        },
    ),
    ClaimGroundTruth(
        file="baggage_loss_claim.pdf",
        label="Baggage Loss",
        fields={
            "claim_number": "CLM-2026-BG-01122",
            "policy_number": "POL-TRVL-5510834",
            "traveler_name": "Maria L. Santos",
            "destination": "Tokyo",
            "total_declared_value": "3970",
            "net_claim_amount": "3320",
            "deductible": "100",
            "airline_pir": "JAL-PIR-2026-ORD-00891",
        },
    ),
    ClaimGroundTruth(
        file="emergency_medical_claim.pdf",
        label="Emergency Medical Evacuation",
        fields={
            "claim_number": "CLM-2026-EM-00219",
            "policy_number": "POL-TRVL-7723561",
            "traveler_name": "Daniel J. Owens",
            "destination": "Cusco",
            "diagnosis": "HAPE",
            "total_medical_costs": "48180",
            "net_claim_amount": "47930",
            "deductible": "250",
        },
    ),
    ClaimGroundTruth(
        file="travel_delay_claim.pdf",
        label="Travel Delay",
        fields={
            "claim_number": "CLM-2026-TD-00667",
            "policy_number": "POL-TRVL-6640293",
            "traveler_name": "Russo",
            "destination": "Cancun",
            "total_expenses": "1028",
            "net_claim_amount": "1028",
            "deductible": "0",
            "delay_cause": "thunderstorm",
        },
    ),
]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class FieldResult(BaseModel):
    field: str
    expected: str
    found: bool


class ClaimResult(BaseModel):
    label: str
    file: str
    fields: list[FieldResult]
    latency_s: float
    input_tokens: int
    output_tokens: int
    error: str | None = None

    @computed_field
    @property
    def passed(self) -> int:
        return sum(f.found for f in self.fields)

    @computed_field
    @property
    def total(self) -> int:
        return len(self.fields)

    @computed_field
    @property
    def accuracy_pct(self) -> float:
        return self.passed / self.total * 100 if self.total else 0.0


class AccuracyReport(BaseModel):
    provider_name: str
    model: str
    claims: list[ClaimResult]

    @computed_field
    @property
    def overall_accuracy_pct(self) -> float:
        t = sum(c.total for c in self.claims)
        p = sum(c.passed for c in self.claims)
        return p / t * 100 if t else 0.0

    @computed_field
    @property
    def avg_latency_s(self) -> float:
        return sum(c.latency_s for c in self.claims) / len(self.claims) if self.claims else 0.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = (
    "Extract all structured fields from this travel insurance claim document. "
    "Return ONLY a valid JSON object — no markdown fences, no explanation. "
    "Include every claim number, policy number, traveler name, destination, "
    "travel dates, dollar amounts, deductibles, and reason for claim."
)


def _build_provider(cfg: ProviderConfig):
    from dobby.providers.anthropic import AnthropicProvider
    if cfg.use_azure:
        return AnthropicProvider(
            model=cfg.model,
            api_key=cfg.foundry_api_key,
            resource=cfg.foundry_resource,
            base_url=cfg.foundry_base_url,
        )
    return AnthropicProvider(model=cfg.model, api_key=cfg.api_key)


def _field_match(extracted: dict[str, Any], value: str) -> bool:
    flat = json.dumps(extracted).lower().replace(",", "").replace("$", "")
    return value.lower().replace(",", "").replace("$", "") in flat


def _parse_json(raw: str) -> dict[str, Any]:
    clean = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    clean = re.sub(r"\n?```$", "", clean.strip())
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}


async def analyze_claim(provider, gt: ClaimGroundTruth) -> ClaimResult:
    from dobby.types import DocumentPart, TextPart, UserMessagePart
    from dobby.types.document_part import Base64PDFSource

    pdf_path = FIXTURES / gt.file
    if not pdf_path.exists():
        return ClaimResult(
            label=gt.label, file=gt.file, fields=[], latency_s=0.0,
            input_tokens=0, output_tokens=0,
            error=f"Missing: {pdf_path} — run: python tests/fixtures/generate_claim_pdfs.py",
        )

    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode()

    t0 = time.perf_counter()
    try:
        result = await provider.chat(
            messages=[
                UserMessagePart(parts=[
                    TextPart(text=EXTRACTION_PROMPT),
                    DocumentPart(
                        source=Base64PDFSource(data=pdf_b64, media_type="application/pdf"),
                        filename=gt.file,
                    ),
                ])
            ],
            system_prompt="You are a precise travel insurance data extraction system. Output only valid JSON.",
            max_tokens=1024,
            temperature=0.0,
        )
    except Exception as exc:
        return ClaimResult(
            label=gt.label, file=gt.file, fields=[], latency_s=time.perf_counter() - t0,
            input_tokens=0, output_tokens=0, error=str(exc),
        )

    latency = time.perf_counter() - t0
    raw = next((p.text for p in result.parts if hasattr(p, "text")), "")
    extracted = _parse_json(raw)

    fields = [
        FieldResult(field=k, expected=v, found=_field_match(extracted, v))
        for k, v in gt.fields.items()
    ]

    usage = result.usage
    return ClaimResult(
        label=gt.label,
        file=gt.file,
        fields=fields,
        latency_s=latency,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
    )


async def run_all(cfg: ProviderConfig) -> AccuracyReport:
    provider = _build_provider(cfg)
    claims = []
    for gt in GROUND_TRUTH:
        print(f"  Analyzing: {gt.label}...", end="", flush=True)
        result = await analyze_claim(provider, gt)
        status = f"ERROR: {result.error}" if result.error else f"{result.passed}/{result.total} fields"
        print(f" {status}  ({result.latency_s:.1f}s)")
        claims.append(result)
    return AccuracyReport(provider_name=provider.name, model=cfg.model, claims=claims)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(report: AccuracyReport) -> None:
    print("\n" + "=" * 65)
    print("TRAVEL INSURANCE CLAIM EXTRACTION — ACCURACY REPORT")
    print("=" * 65)
    print(f"  Provider : {report.provider_name}")
    print(f"  Model    : {report.model}\n")

    for c in report.claims:
        if c.error:
            print(f"  [{c.label}]  ERROR: {c.error}")
            continue
        print(f"  [{c.label}]  {c.accuracy_pct:.0f}%  ({c.passed}/{c.total} fields)  "
              f"latency={c.latency_s:.1f}s  tokens={c.input_tokens}in/{c.output_tokens}out")
        for f in c.fields:
            print(f"    {'✓' if f.found else '✗'} {f.field:<28} {f.expected!r}")

    print("\n" + "-" * 65)
    print(f"  OVERALL ACCURACY : {report.overall_accuracy_pct:.1f}%")
    print(f"  AVG LATENCY      : {report.avg_latency_s:.1f}s per claim")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ProviderConfig.from_env()

    if not (cfg.use_azure or cfg.api_key):
        print(
            "\nERROR: No API key found.\n"
            "Set ANTHROPIC_API_KEY (direct) or\n"
            "ANTHROPIC_FOUNDRY_API_KEY + ANTHROPIC_FOUNDRY_RESOURCE/BASE_URL (Azure) in .env\n"
        )
        raise SystemExit(1)

    report = asyncio.run(run_all(cfg))
    print_report(report)
