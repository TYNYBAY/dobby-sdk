"""Multi-model comparison — claim extraction + underwriting analysis.

Runs every claim through all configured models and produces a side-by-side
accuracy, latency, and underwriting comparison.

Models compared by default:
  - claude-sonnet-4-6
  - claude-haiku-4-5-20251001
  - gpt-4o (OpenAI)

Usage:
    uv run python tests/test_anthropic_provider.py
    uv run python tests/test_anthropic_provider.py --models sonnet haiku
    uv run python tests/test_anthropic_provider.py --models sonnet haiku gpt
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Fallback direct Anthropic key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Sonnet — separate Azure deployment
ANTHROPIC_SONNET_FOUNDRY_API_KEY  = os.getenv("ANTHROPIC_SONNET_FOUNDRY_API_KEY")
ANTHROPIC_SONNET_FOUNDRY_RESOURCE = os.getenv("ANTHROPIC_SONNET_FOUNDRY_RESOURCE")
ANTHROPIC_SONNET_FOUNDRY_BASE_URL = os.getenv("ANTHROPIC_SONNET_FOUNDRY_BASE_URL")
ANTHROPIC_SONNET_FOUNDRY_MODEL    = os.getenv("ANTHROPIC_SONNET_FOUNDRY_MODEL", "claude-sonnet-4-6")

# Haiku — separate Azure deployment
ANTHROPIC_HAIKU_FOUNDRY_API_KEY   = os.getenv("ANTHROPIC_HAIKU_FOUNDRY_API_KEY")
ANTHROPIC_HAIKU_FOUNDRY_RESOURCE  = os.getenv("ANTHROPIC_HAIKU_FOUNDRY_RESOURCE")
ANTHROPIC_HAIKU_FOUNDRY_BASE_URL  = os.getenv("ANTHROPIC_HAIKU_FOUNDRY_BASE_URL")
ANTHROPIC_HAIKU_FOUNDRY_MODEL     = os.getenv("ANTHROPIC_HAIKU_FOUNDRY_MODEL", "claude-haiku-4-5-20251001")

# GPT — Azure OpenAI deployment
OPENAI_AZURE_API_KEY       = os.getenv("OPENAI_AZURE_API_KEY")
OPENAI_AZURE_BASE_URL      = os.getenv("OPENAI_AZURE_BASE_URL")
OPENAI_AZURE_DEPLOYMENT_ID = os.getenv("OPENAI_AZURE_DEPLOYMENT_ID", "gpt-4o")
OPENAI_AZURE_API_VERSION   = os.getenv("OPENAI_AZURE_API_VERSION", "2025-03-01-preview")

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Model registry — short alias → (provider_type, model_id)
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "sonnet": ("anthropic", ANTHROPIC_SONNET_FOUNDRY_MODEL),
    "haiku":  ("anthropic", ANTHROPIC_HAIKU_FOUNDRY_MODEL),
    "gpt":    ("openai",    OPENAI_AZURE_DEPLOYMENT_ID),
}

DEFAULT_MODELS = ["sonnet", "haiku", "gpt"]

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

CLAIMS = [
    {
        "file": "auto_liability_claim.pdf",
        "label": "Auto Liability",
        "fields": {
            "claim_number":  "CLM-2026-AL-00142",
            "policy_number": "POL-AUTO-8823991",
            "insured_name":  "Jane R. Hartwell",
            "date_of_loss":  "2026-03-15",
            "total_estimate":"$9,570.00",
            "reserve":       "$14,500",
        },
    },
    {
        "file": "property_damage_claim.pdf",
        "label": "Property Damage",
        "fields": {
            "claim_number":    "CLM-2026-PD-00308",
            "policy_number":   "POL-COMM-1143772",
            "business_name":   "Hartfield Printing & Design LLC",
            "date_of_loss":    "2026-02-28",
            "total_rcv":       "$157,740",
            "net_payable_acv": "$103,540",
            "deductible":      "$5,000",
        },
    },
    {
        "file": "workers_comp_claim.pdf",
        "label": "Workers Comp",
        "fields": {
            "claim_number":   "CLM-2026-WC-00091",
            "policy_number":  "POL-WC-5567234",
            "employee_name":  "Carlos M. Reyes",
            "date_of_injury": "2026-01-22",
            "body_part":      "Left knee",
            "total_reserve":  "$44,704",
            "avg_weekly_wage":"$1,142.00",
        },
    },
    {
        "file": "medical_malpractice_claim.pdf",
        "label": "Medical Malpractice",
        "fields": {
            "claim_number":   "CLM-2026-MM-00017",
            "policy_number":  "POL-MPL-0023491",
            "provider_name":  "Dr. Kevin J. Allard",
            "claimant_name":  "Patricia L. Vasquez",
            "demand_amount":  "$1,250,000",
            "total_reserve":  "$1,075,000",
            "coverage_limits":"$1M/$3M",
        },
    },
    {
        "file": "homeowners_claim.pdf",
        "label": "Homeowners",
        "fields": {
            "claim_number":  "CLM-2026-HO-00522",
            "policy_number": "POL-HO3-7734128",
            "insured_name":  "Marcus & Diana Webb",
            "date_of_loss":  "2026-04-02",
            "total_rcv":     "$26,490",
            "net_acv_payment":"$15,670",
            "deductible":    "$2,500",
        },
    },
]

EXTRACTION_PROMPT = """\
Extract all key fields from this insurance claim document and return ONLY a JSON object.
Include every field you can find: claim numbers, policy numbers, names, dates, dollar amounts,
reserve figures, deductibles, and any other structured data.
Return ONLY valid JSON, no markdown fences, no explanation."""

UNDERWRITING_PROMPT = """\
You are a senior claims underwriter. Review the extracted claim data below and provide a \
structured underwriting analysis as a JSON object with these exact keys:

- coverage_determination: "covered" | "not_covered" | "pending_investigation"
- liability_assessment: one-sentence summary of fault/liability
- reserve_adequacy: "adequate" | "understated" | "overstated"
- reserve_commentary: brief explanation of reserve position
- red_flags: list of strings (fraud indicators, inconsistencies, missing docs) — empty list if none
- recommendation: "approve" | "deny" | "investigate" | "approve_with_conditions"
- recommendation_rationale: one or two sentences explaining the recommendation
- key_issues: list of strings — top items requiring attention before closure
- coverage_gaps: list of strings — exposures not covered by this policy, if any

Return ONLY valid JSON, no markdown fences, no explanation.

Extracted claim data:
{extracted_json}"""


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

_AZURE_CREDS: dict[str, dict] = {
    "sonnet": {
        "api_key":  ANTHROPIC_SONNET_FOUNDRY_API_KEY,
        "resource": ANTHROPIC_SONNET_FOUNDRY_RESOURCE,
        "base_url": ANTHROPIC_SONNET_FOUNDRY_BASE_URL,
    },
    "haiku": {
        "api_key":  ANTHROPIC_HAIKU_FOUNDRY_API_KEY,
        "resource": ANTHROPIC_HAIKU_FOUNDRY_RESOURCE,
        "base_url": ANTHROPIC_HAIKU_FOUNDRY_BASE_URL,
    },
}


def _make_provider(alias: str):
    provider_type, model_id = MODEL_ALIASES[alias]

    if provider_type == "anthropic":
        from dobby.providers.anthropic import AnthropicProvider
        creds = _AZURE_CREDS.get(alias, {})
        if creds.get("api_key") and (creds.get("resource") or creds.get("base_url")):
            # Use this model's dedicated Azure deployment
            return AnthropicProvider(
                model=model_id,
                api_key=creds["api_key"],
                resource=creds.get("resource"),
                base_url=creds.get("base_url"),
            )
        # Fall back to direct Anthropic API
        return AnthropicProvider(model=model_id, api_key=ANTHROPIC_API_KEY)

    if provider_type == "openai":
        from dobby.providers.openai import OpenAIProvider
        return OpenAIProvider(
            api_key=OPENAI_AZURE_API_KEY,
            base_url=OPENAI_AZURE_BASE_URL,
            azure_deployment_id=OPENAI_AZURE_DEPLOYMENT_ID,
            azure_api_version=OPENAI_AZURE_API_VERSION,
        )

    raise ValueError(f"Unknown provider type: {provider_type}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    name: str
    expected: str
    hit: bool
    extracted_json: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimResult:
    label: str
    file: str
    model_alias: str
    model_id: str
    field_results: list[FieldResult]
    extracted_json: dict[str, Any]
    ext_latency_s: float
    ext_tokens_in: int
    ext_tokens_out: int
    underwriting: dict[str, Any]
    uw_latency_s: float
    uw_tokens_in: int
    uw_tokens_out: int
    error: str | None = None

    @property
    def passed(self) -> int:
        return sum(1 for f in self.field_results if f.hit)

    @property
    def total(self) -> int:
        return len(self.field_results)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def total_latency(self) -> float:
        return self.ext_latency_s + self.uw_latency_s

    @property
    def total_tokens(self) -> int:
        return self.ext_tokens_in + self.ext_tokens_out + self.uw_tokens_in + self.uw_tokens_out


# ---------------------------------------------------------------------------
# Per-claim evaluation
# ---------------------------------------------------------------------------

async def eval_claim(provider, alias: str, model_id: str, claim: dict) -> ClaimResult:
    from dobby.types import DocumentPart, TextPart, UserMessagePart
    from dobby.types.document_part import Base64PDFSource

    pdf_path = FIXTURES / claim["file"]
    if not pdf_path.exists():
        return ClaimResult(
            label=claim["label"], file=claim["file"],
            model_alias=alias, model_id=model_id,
            field_results=[], extracted_json={},
            ext_latency_s=0, ext_tokens_in=0, ext_tokens_out=0,
            underwriting={}, uw_latency_s=0, uw_tokens_in=0, uw_tokens_out=0,
            error="PDF not found — run: uv run python tests/fixtures/generate_claim_pdfs.py",
        )

    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode()

    # --- Pass 1: extraction ---
    t0 = time.perf_counter()
    try:
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
    except Exception as exc:
        return ClaimResult(
            label=claim["label"], file=claim["file"],
            model_alias=alias, model_id=model_id,
            field_results=[], extracted_json={},
            ext_latency_s=time.perf_counter() - t0, ext_tokens_in=0, ext_tokens_out=0,
            underwriting={}, uw_latency_s=0, uw_tokens_in=0, uw_tokens_out=0,
            error=str(exc),
        )

    ext_latency = time.perf_counter() - t0
    raw = next((p.text for p in result.parts if hasattr(p, "text")), "")
    extracted = _parse_json(raw)
    ext_in = result.usage.input_tokens if result.usage else 0
    ext_out = result.usage.output_tokens if result.usage else 0

    field_results = [
        FieldResult(name=fname, expected=fval, hit=_field_hit(extracted, fval), extracted_json=extracted)
        for fname, fval in claim["fields"].items()
    ]

    # --- Pass 2: underwriting ---
    uw: dict[str, Any] = {}
    uw_latency = uw_in = uw_out = 0.0

    if extracted:
        uw_prompt = UNDERWRITING_PROMPT.format(extracted_json=json.dumps(extracted, indent=2))
        t1 = time.perf_counter()
        try:
            uw_result = await provider.chat(
                messages=[UserMessagePart(parts=[TextPart(text=uw_prompt)])],
                system_prompt=(
                    "You are a senior insurance claims underwriter with 20 years of experience. "
                    "Provide precise, actionable underwriting assessments. Output only valid JSON."
                ),
                max_tokens=1024,
                temperature=0.0,
            )
            uw_latency = time.perf_counter() - t1
            uw_raw = next((p.text for p in uw_result.parts if hasattr(p, "text")), "")
            uw = _parse_json(uw_raw)
            if uw_result.usage:
                uw_in = uw_result.usage.input_tokens
                uw_out = uw_result.usage.output_tokens
        except Exception as exc:
            uw_latency = time.perf_counter() - t1
            uw = {"error": str(exc)}

    return ClaimResult(
        label=claim["label"], file=claim["file"],
        model_alias=alias, model_id=model_id,
        field_results=field_results, extracted_json=extracted,
        ext_latency_s=ext_latency, ext_tokens_in=ext_in, ext_tokens_out=ext_out,
        underwriting=uw, uw_latency_s=uw_latency,
        uw_tokens_in=int(uw_in), uw_tokens_out=int(uw_out),
    )


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_report(all_results: dict[str, list[ClaimResult]], aliases: list[str]) -> None:
    claim_labels = [c["label"] for c in CLAIMS]

    # --- Per-claim detail ---
    for label in claim_labels:
        print(f"\n{'─' * 70}")
        print(f"CLAIM: {label}")
        for alias in aliases:
            r = next((x for x in all_results[alias] if x.label == label), None)
            if r is None:
                continue
            tag = f"[{alias.upper()}]"
            if r.error:
                print(f"  {tag}  ERROR: {r.error}")
                continue
            pct = f"{r.accuracy * 100:.0f}%"
            print(f"\n  {tag}  {pct} ({r.passed}/{r.total})  "
                  f"ext={r.ext_latency_s:.1f}s  uw={r.uw_latency_s:.1f}s  "
                  f"tokens={r.total_tokens}")

            for fr in r.field_results:
                mark = "✓" if fr.hit else "✗"
                print(f"    {mark} {fr.name:<25} {fr.expected!r}")

            uw = r.underwriting
            if uw and "error" not in uw:
                print(f"    → coverage:    {uw.get('coverage_determination', '—')}")
                print(f"    → reserve:     {uw.get('reserve_adequacy', '—')}")
                print(f"    → recommend:   {uw.get('recommendation', '—')}")
                flags = uw.get("red_flags", [])
                for f in flags:
                    print(f"    ⚑  {f}")

    # --- Side-by-side accuracy table ---
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON — ACCURACY")
    col = 18
    header = f"{'Claim':<22}" + "".join(f"{a.upper():>{col}}" for a in aliases)
    print(header)
    print("─" * (22 + col * len(aliases)))

    for label in claim_labels:
        row = f"{label:<22}"
        for alias in aliases:
            r = next((x for x in all_results[alias] if x.label == label), None)
            if r is None or r.error:
                cell = "ERROR"
            else:
                cell = f"{r.accuracy * 100:.0f}% ({r.passed}/{r.total})"
            row += f"{cell:>{col}}"
        print(row)

    # totals
    print("─" * (22 + col * len(aliases)))
    total_row = f"{'OVERALL':<22}"
    for alias in aliases:
        good = [r for r in all_results[alias] if not r.error]
        tp = sum(r.passed for r in good)
        tt = sum(r.total for r in good)
        pct = tp / tt * 100 if tt else 0
        total_row += f"{f'{pct:.1f}% ({tp}/{tt})':>{col}}"
    print(total_row)

    # --- Side-by-side latency table ---
    print(f"\n{'─' * 70}")
    print("MODEL COMPARISON — LATENCY (total ext+uw per claim, seconds)")
    print(f"{'Claim':<22}" + "".join(f"{a.upper():>{col}}" for a in aliases))
    print("─" * (22 + col * len(aliases)))
    for label in claim_labels:
        row = f"{label:<22}"
        for alias in aliases:
            r = next((x for x in all_results[alias] if x.label == label), None)
            if r is None or r.error:
                cell = "ERROR"
            else:
                cell = f"{r.total_latency:.2f}s"
            row += f"{cell:>{col}}"
        print(row)
    avg_row = f"{'AVG':<22}"
    for alias in aliases:
        good = [r for r in all_results[alias] if not r.error]
        avg = sum(r.total_latency for r in good) / len(good) if good else 0
        avg_row += f"{f'{avg:.2f}s':>{col}}"
    print("─" * (22 + col * len(aliases)))
    print(avg_row)

    # --- Side-by-side tokens table ---
    print(f"\n{'─' * 70}")
    print("MODEL COMPARISON — TOTAL TOKENS (ext+uw)")
    print(f"{'Claim':<22}" + "".join(f"{a.upper():>{col}}" for a in aliases))
    print("─" * (22 + col * len(aliases)))
    for label in claim_labels:
        row = f"{label:<22}"
        for alias in aliases:
            r = next((x for x in all_results[alias] if x.label == label), None)
            if r is None or r.error:
                cell = "ERROR"
            else:
                cell = f"{r.total_tokens:,}"
            row += f"{cell:>{col}}"
        print(row)
    tot_row = f"{'TOTAL':<22}"
    for alias in aliases:
        good = [r for r in all_results[alias] if not r.error]
        tot_row += f"{f'{sum(r.total_tokens for r in good):,}':>{col}}"
    print("─" * (22 + col * len(aliases)))
    print(tot_row)

    # --- Underwriting recommendation comparison ---
    print(f"\n{'─' * 70}")
    print("MODEL COMPARISON — UNDERWRITING RECOMMENDATIONS")
    print(f"{'Claim':<22}" + "".join(f"{a.upper():>{col}}" for a in aliases))
    print("─" * (22 + col * len(aliases)))
    for label in claim_labels:
        row = f"{label:<22}"
        for alias in aliases:
            r = next((x for x in all_results[alias] if x.label == label), None)
            if r is None or r.error:
                cell = "ERROR"
            elif r.underwriting and "error" not in r.underwriting:
                cell = r.underwriting.get("recommendation", "—")
            else:
                cell = "—"
            row += f"{cell:>{col}}"
        print(row)

    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(
    all_results: dict[str, list[ClaimResult]],
    aliases: list[str],
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).parent / "reports" / f"comparison_{timestamp}.md"

    lines: list[str] = []
    w = lines.append
    claim_labels = [c["label"] for c in CLAIMS]

    w("# Multi-Model Claim Analysis Comparison")
    w("")
    w(f"**Generated:** {generated_at}  ")
    w(f"**Models compared:** {', '.join(f'`{MODEL_ALIASES[a][1]}`' for a in aliases)}  ")
    w(f"**Claims evaluated:** {len(CLAIMS)}  ")
    w("")

    # --- Executive Summary table ---
    w("## Executive Summary")
    w("")
    w("| Model | Overall Accuracy | Avg Latency | Total Tokens |")
    w("|-------|-----------------|-------------|--------------|")
    for alias in aliases:
        good = [r for r in all_results[alias] if not r.error]
        tp = sum(r.passed for r in good)
        tt = sum(r.total for r in good)
        pct = tp / tt * 100 if tt else 0
        avg_lat = sum(r.total_latency for r in good) / len(good) if good else 0
        tot_tok = sum(r.total_tokens for r in good)
        _, model_id = MODEL_ALIASES[alias]
        w(f"| `{model_id}` | **{pct:.1f}%** ({tp}/{tt}) | {avg_lat:.2f}s | {tot_tok:,} |")
    w("")

    # --- Accuracy comparison ---
    w("## Accuracy Comparison")
    w("")
    header = "| Claim |" + "".join(f" {a.upper()} |" for a in aliases)
    sep = "|-------|" + "".join("--------|" for _ in aliases)
    w(header)
    w(sep)
    for label in claim_labels:
        row = f"| {label} |"
        for alias in aliases:
            r = next((x for x in all_results[alias] if x.label == label), None)
            if r is None or r.error:
                row += " ERROR |"
            else:
                row += f" {r.accuracy * 100:.0f}% ({r.passed}/{r.total}) |"
        w(row)
    w("")

    # --- Latency comparison ---
    w("## Latency Comparison (seconds, ext + underwriting)")
    w("")
    header = "| Claim |" + "".join(f" {a.upper()} |" for a in aliases)
    w(header)
    w(sep)
    for label in claim_labels:
        row = f"| {label} |"
        for alias in aliases:
            r = next((x for x in all_results[alias] if x.label == label), None)
            if r is None or r.error:
                row += " ERROR |"
            else:
                row += f" {r.ext_latency_s:.2f}s + {r.uw_latency_s:.2f}s |"
        w(row)
    w("")

    # --- Token comparison ---
    w("## Token Usage Comparison")
    w("")
    header = "| Claim |" + "".join(f" {a.upper()} |" for a in aliases)
    w(header)
    w(sep)
    for label in claim_labels:
        row = f"| {label} |"
        for alias in aliases:
            r = next((x for x in all_results[alias] if x.label == label), None)
            if r is None or r.error:
                row += " ERROR |"
            else:
                row += f" {r.total_tokens:,} |"
        w(row)
    w("")

    # --- Underwriting recommendations ---
    w("## Underwriting Recommendations Comparison")
    w("")
    header = "| Claim |" + "".join(f" {a.upper()} |" for a in aliases)
    w(header)
    w(sep)
    for label in claim_labels:
        row = f"| {label} |"
        for alias in aliases:
            r = next((x for x in all_results[alias] if x.label == label), None)
            if r is None or r.error:
                row += " ERROR |"
            elif r.underwriting and "error" not in r.underwriting:
                row += f" `{r.underwriting.get('recommendation', '—')}` |"
            else:
                row += " — |"
        w(row)
    w("")

    # --- Per-claim detail ---
    w("## Per-Claim Detail")
    w("")
    for label in claim_labels:
        w(f"### {label}")
        w("")
        for alias in aliases:
            r = next((x for x in all_results[alias] if x.label == label), None)
            if r is None:
                continue
            _, model_id = MODEL_ALIASES[alias]
            w(f"#### `{model_id}`")
            w("")
            if r.error:
                w(f"> ❌ {r.error}")
                w("")
                continue

            w(f"- **Accuracy:** {r.accuracy * 100:.1f}% ({r.passed}/{r.total} fields)")
            w(f"- **Latency:** ext {r.ext_latency_s:.2f}s + uw {r.uw_latency_s:.2f}s")
            w(f"- **Tokens:** ext {r.ext_tokens_in}/{r.ext_tokens_out}  uw {r.uw_tokens_in}/{r.uw_tokens_out}")
            w("")

            w("**Field results:**")
            w("")
            w("| Status | Field | Expected |")
            w("|--------|-------|----------|")
            for fr in r.field_results:
                w(f"| {'✅' if fr.hit else '❌'} | `{fr.name}` | `{fr.expected}` |")
            w("")

            w("**Extracted JSON:**")
            w("")
            if r.extracted_json:
                w("```json")
                w(json.dumps(r.extracted_json, indent=2))
                w("```")
            else:
                w("> ⚠️ No valid JSON returned.")
            w("")

            uw = r.underwriting
            if uw and "error" not in uw:
                w("**Underwriting analysis:**")
                w("")
                w("| Dimension | Assessment |")
                w("|-----------|------------|")
                w(f"| Coverage | `{uw.get('coverage_determination', '—')}` |")
                w(f"| Reserve adequacy | `{uw.get('reserve_adequacy', '—')}` |")
                w(f"| **Recommendation** | **`{uw.get('recommendation', '—')}`** |")
                w("")
                w(f"*{uw.get('liability_assessment', '')}*")
                w("")
                w(f"**Rationale:** {uw.get('recommendation_rationale', '—')}")
                w("")
                flags = uw.get("red_flags", [])
                if flags:
                    w("**Red flags:** " + " · ".join(f"⚑ {f}" for f in flags))
                    w("")
                issues = uw.get("key_issues", [])
                if issues:
                    w("**Key issues:** " + " · ".join(issues))
                    w("")

    content = "\n".join(lines)

    # Always print to stdout
    print("\n" + "=" * 70)
    print("MARKDOWN REPORT")
    print("=" * 70)
    print(content)
    print("=" * 70 + "\n")

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        print(f"Report also saved → {out}")
    except Exception as exc:
        print(f"[warn] Could not save to disk: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(aliases: list[str]) -> None:
    print(f"\nModels: {', '.join(aliases)}")
    print(f"Claims: {len(CLAIMS)}")
    print(f"Calls per model: {len(CLAIMS) * 2} (extraction + underwriting)")
    print()

    all_results: dict[str, list[ClaimResult]] = {}

    for alias in aliases:
        _, model_id = MODEL_ALIASES[alias]
        print(f"{'─' * 50}")
        print(f"Running {alias.upper()} ({model_id})")
        try:
            provider = _make_provider(alias)
        except Exception as exc:
            print(f"  ERROR creating provider: {exc}")
            all_results[alias] = []
            continue

        results: list[ClaimResult] = []
        for claim in CLAIMS:
            print(f"  {claim['label']}...", end="", flush=True)
            r = await eval_claim(provider, alias, model_id, claim)
            status = f"{r.passed}/{r.total}" if not r.error else "ERROR"
            print(f" {status}  (ext {r.ext_latency_s:.1f}s  uw {r.uw_latency_s:.1f}s)")
            results.append(r)
        all_results[alias] = results

    print_report(all_results, aliases)
    write_report(all_results, aliases)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model claim comparison")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_ALIASES.keys()),
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help=f"Models to compare. Choices: {', '.join(MODEL_ALIASES.keys())} (default: all)",
    )
    args = parser.parse_args()

    missing: list[str] = []
    for alias in args.models:
        ptype, _ = MODEL_ALIASES[alias]
        if ptype == "anthropic":
            creds = _AZURE_CREDS.get(alias, {})
            has_azure = creds.get("api_key") and (creds.get("resource") or creds.get("base_url"))
            if not has_azure and not ANTHROPIC_API_KEY:
                missing.append(
                    f"ANTHROPIC_{alias.upper()}_FOUNDRY_API_KEY + RESOURCE "
                    f"(or ANTHROPIC_API_KEY as fallback) needed for {alias}"
                )
        if ptype == "openai" and not (OPENAI_AZURE_API_KEY and OPENAI_AZURE_BASE_URL):
            missing.append(f"OPENAI_AZURE_API_KEY + OPENAI_AZURE_BASE_URL (needed for {alias})")
    if missing:
        for m in missing:
            print(f"ERROR: missing {m}")
        raise SystemExit(1)

    asyncio.run(run(aliases=args.models))


if __name__ == "__main__":
    main()
