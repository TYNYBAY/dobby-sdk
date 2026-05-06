"""Anthropic provider smoke test — full claim extraction + underwriting analysis.

Runs on every claim PDF, shows inputs, extracted JSON, field accuracy,
underwriting assessment, rankings, field hit rate, failure patterns,
and auto-saves a markdown report.

Usage:
    uv run python tests/test_anthropic_provider.py
    uv run python tests/test_anthropic_provider.py --model claude-sonnet-4-6
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

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

CLAIMS = [
    {
        "file": "auto_liability_claim.pdf",
        "label": "Auto Liability",
        "fields": {
            "claim_number": "CLM-2026-AL-00142",
            "policy_number": "POL-AUTO-8823991",
            "insured_name": "Jane R. Hartwell",
            "date_of_loss": "2026-03-15",
            "total_estimate": "$9,570.00",
            "reserve": "$14,500",
        },
    },
    {
        "file": "property_damage_claim.pdf",
        "label": "Property Damage",
        "fields": {
            "claim_number": "CLM-2026-PD-00308",
            "policy_number": "POL-COMM-1143772",
            "business_name": "Hartfield Printing & Design LLC",
            "date_of_loss": "2026-02-28",
            "total_rcv": "$157,740",
            "net_payable_acv": "$103,540",
            "deductible": "$5,000",
        },
    },
    {
        "file": "workers_comp_claim.pdf",
        "label": "Workers Comp",
        "fields": {
            "claim_number": "CLM-2026-WC-00091",
            "policy_number": "POL-WC-5567234",
            "employee_name": "Carlos M. Reyes",
            "date_of_injury": "2026-01-22",
            "body_part": "Left knee",
            "total_reserve": "$44,704",
            "avg_weekly_wage": "$1,142.00",
        },
    },
    {
        "file": "medical_malpractice_claim.pdf",
        "label": "Medical Malpractice",
        "fields": {
            "claim_number": "CLM-2026-MM-00017",
            "policy_number": "POL-MPL-0023491",
            "provider_name": "Dr. Kevin J. Allard",
            "claimant_name": "Patricia L. Vasquez",
            "demand_amount": "$1,250,000",
            "total_reserve": "$1,075,000",
            "coverage_limits": "$1M/$3M",
        },
    },
    {
        "file": "homeowners_claim.pdf",
        "label": "Homeowners",
        "fields": {
            "claim_number": "CLM-2026-HO-00522",
            "policy_number": "POL-HO3-7734128",
            "insured_name": "Marcus & Diana Webb",
            "date_of_loss": "2026-04-02",
            "total_rcv": "$26,490",
            "net_acv_payment": "$15,670",
            "deductible": "$2,500",
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
# Helpers
# ---------------------------------------------------------------------------

def _provider(model: str):
    from dobby.providers.anthropic import AnthropicProvider

    if _USE_AZURE:
        return AnthropicProvider(
            model=model,
            api_key=ANTHROPIC_FOUNDRY_API_KEY,
            resource=ANTHROPIC_FOUNDRY_RESOURCE,
            base_url=ANTHROPIC_FOUNDRY_BASE_URL,
        )
    return AnthropicProvider(model=model, api_key=ANTHROPIC_API_KEY)


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


# ---------------------------------------------------------------------------
# Per-claim evaluation
# ---------------------------------------------------------------------------

async def eval_claim(provider, claim: dict) -> ClaimResult:
    from dobby.types import DocumentPart, TextPart, UserMessagePart
    from dobby.types.document_part import Base64PDFSource

    pdf_path = FIXTURES / claim["file"]
    if not pdf_path.exists():
        return ClaimResult(
            label=claim["label"], file=claim["file"],
            field_results=[], extracted_json={},
            ext_latency_s=0, ext_tokens_in=0, ext_tokens_out=0,
            underwriting={}, uw_latency_s=0, uw_tokens_in=0, uw_tokens_out=0,
            error=f"PDF not found — run: uv run python tests/fixtures/generate_claim_pdfs.py",
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
        FieldResult(
            name=fname,
            expected=fval,
            hit=_field_hit(extracted, fval),
            extracted_json=extracted,
        )
        for fname, fval in claim["fields"].items()
    ]

    # --- Pass 2: underwriting ---
    uw: dict[str, Any] = {}
    uw_latency = 0.0
    uw_in = uw_out = 0

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
        field_results=field_results, extracted_json=extracted,
        ext_latency_s=ext_latency, ext_tokens_in=ext_in, ext_tokens_out=ext_out,
        underwriting=uw, uw_latency_s=uw_latency, uw_tokens_in=uw_in, uw_tokens_out=uw_out,
    )


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_report(results: list[ClaimResult]) -> None:
    total_fields = total_passed = 0
    total_ext_in = total_ext_out = 0
    total_uw_in = total_uw_out = 0

    for r in results:
        print(f"\n{'─' * 70}")
        print(f"{r.label}  ({r.file})")

        if r.error:
            print(f"  ERROR: {r.error}")
            continue

        pct = f"{r.accuracy * 100:.0f}%"
        print(f"  Accuracy:    {pct}  ({r.passed}/{r.total})  "
              f"[ext {r.ext_latency_s:.2f}s  uw {r.uw_latency_s:.2f}s]")
        print(f"  Tokens ext:  in={r.ext_tokens_in}  out={r.ext_tokens_out}")
        if r.uw_tokens_in or r.uw_tokens_out:
            print(f"  Tokens uw:   in={r.uw_tokens_in}  out={r.uw_tokens_out}")

        print(f"\n  INPUT PROMPT:")
        for line in EXTRACTION_PROMPT.splitlines():
            print(f"    {line}")

        print(f"\n  OUTPUT — extracted JSON:")
        if r.extracted_json:
            for line in json.dumps(r.extracted_json, indent=4).splitlines():
                print(f"    {line}")
        else:
            print(f"    [no valid JSON returned]")

        print(f"\n  FIELD ACCURACY:")
        for fr in r.field_results:
            mark = "✓" if fr.hit else "✗"
            print(f"    {mark} {fr.name:<25} expected: {fr.expected}")

        uw = r.underwriting
        if uw and "error" not in uw:
            print(f"\n  UNDERWRITING ANALYSIS:")
            print(f"    Coverage:      {uw.get('coverage_determination', '—')}")
            print(f"    Liability:     {uw.get('liability_assessment', '—')}")
            print(f"    Reserve:       {uw.get('reserve_adequacy', '—')} — {uw.get('reserve_commentary', '')}")
            print(f"    Recommendation:{uw.get('recommendation', '—')}")
            print(f"    Rationale:     {uw.get('recommendation_rationale', '—')}")
            for flag in uw.get("red_flags", []):
                print(f"    ⚑  {flag}")
            for issue in uw.get("key_issues", []):
                print(f"    •  {issue}")
            for gap in uw.get("coverage_gaps", []):
                print(f"    ○  {gap}")
        elif uw.get("error"):
            print(f"\n  UNDERWRITING: ERROR — {uw['error']}")

        total_fields += r.total
        total_passed += r.passed
        total_ext_in += r.ext_tokens_in
        total_ext_out += r.ext_tokens_out
        total_uw_in += r.uw_tokens_in
        total_uw_out += r.uw_tokens_out

    good = [r for r in results if not r.error]
    overall_pct = total_passed / total_fields * 100 if total_fields else 0

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"  Overall accuracy: {total_passed}/{total_fields} fields  ({overall_pct:.1f}%)")
    avg_ext = sum(r.ext_latency_s for r in good) / len(good) if good else 0
    print(f"  Avg ext latency:  {avg_ext:.2f}s")
    print(f"  Tokens ext:       in={total_ext_in}  out={total_ext_out}")
    print(f"  Tokens uw:        in={total_uw_in}  out={total_uw_out}")
    print(f"  Tokens total:     {total_ext_in + total_ext_out + total_uw_in + total_uw_out}")

    # --- Rankings ---
    print(f"\n{'─' * 70}")
    print("RANKINGS (worst → best)")
    for r in sorted(good, key=lambda x: x.accuracy):
        bar = ("█" * r.passed) + ("░" * (r.total - r.passed))
        uw_rec = r.underwriting.get("recommendation", "—") if r.underwriting else "—"
        print(f"  {r.accuracy * 100:5.1f}%  [{bar}]  {r.label}  → {uw_rec}")

    # --- Field hit rate ---
    field_hits: dict[str, list[bool]] = defaultdict(list)
    for r in good:
        for fr in r.field_results:
            field_hits[fr.name].append(fr.hit)

    if field_hits:
        print(f"\n{'─' * 70}")
        print("FIELD HIT RATE (aggregated)")
        for fname, hits in sorted(field_hits.items(), key=lambda x: sum(x[1]) / len(x[1])):
            n, p = len(hits), sum(hits)
            bar = ("█" * p) + ("░" * (n - p))
            print(f"  {p / n * 100:5.1f}%  [{bar}]  {fname}")

    # --- Failure patterns ---
    failures = [(r.label, fr) for r in good for fr in r.field_results if not fr.hit]
    if failures:
        print(f"\n{'─' * 70}")
        print("FAILURE PATTERNS")
        for label, fr in failures:
            extracted_val = fr.extracted_json.get(fr.name)
            if extracted_val is None:
                candidates = [
                    f"{k}={v!r}" for k, v in fr.extracted_json.items()
                    if fr.name.split("_")[-1] in k.lower()
                ]
                note = f"key absent (similar: {', '.join(candidates[:2])})" if candidates else "key absent"
            else:
                note = repr(extracted_val)
            print(f"  ✗ [{label}] {fr.name}")
            print(f"      expected:  {fr.expected!r}")
            print(f"      extracted: {note}")
    else:
        print(f"\n{'─' * 70}")
        print("FAILURE PATTERNS: none — all fields matched")

    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(results: list[ClaimResult], model: str, provider_name: str) -> Path:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).parent / "reports" / f"smoke_{timestamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    good = [r for r in results if not r.error]
    total_fields = sum(r.total for r in good)
    total_passed = sum(r.passed for r in good)
    overall_pct = total_passed / total_fields * 100 if total_fields else 0
    avg_ext = sum(r.ext_latency_s for r in good) / len(good) if good else 0
    total_ext_in = sum(r.ext_tokens_in for r in good)
    total_ext_out = sum(r.ext_tokens_out for r in good)
    total_uw_in = sum(r.uw_tokens_in for r in good)
    total_uw_out = sum(r.uw_tokens_out for r in good)

    lines: list[str] = []
    w = lines.append

    w("# Claim Extraction & Underwriting Analysis Report")
    w("")
    w(f"**Provider:** {provider_name}  ")
    w(f"**Model:** `{model}`  ")
    w(f"**Generated:** {generated_at}  ")
    w(f"**Claims evaluated:** {len(results)}  ")
    w("")

    # --- Executive Summary ---
    w("## Executive Summary")
    w("")
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Overall field accuracy | **{overall_pct:.1f}%** ({total_passed}/{total_fields} fields) |")
    w(f"| Average extraction latency | {avg_ext:.2f}s per claim |")
    w(f"| Extraction tokens (in / out) | {total_ext_in:,} / {total_ext_out:,} |")
    w(f"| Underwriting tokens (in / out) | {total_uw_in:,} / {total_uw_out:,} |")
    w(f"| **Total tokens** | **{total_ext_in + total_ext_out + total_uw_in + total_uw_out:,}** |")
    w("")

    errors = [r for r in results if r.error]
    if errors:
        w("> **Note:** The following claims could not be evaluated:")
        for r in errors:
            w(f"> - `{r.file}`: {r.error}")
        w("")

    # --- Rankings ---
    w("## Rankings")
    w("")
    w("| Rank | Claim | Accuracy | Fields | Ext Latency | UW Latency | Recommendation |")
    w("|------|-------|----------|--------|-------------|------------|----------------|")
    for i, r in enumerate(sorted(good, key=lambda x: x.accuracy), 1):
        uw_rec = r.underwriting.get("recommendation", "—") if r.underwriting else "—"
        w(f"| {i} | {r.label} | {r.accuracy * 100:.1f}% | {r.passed}/{r.total} | {r.ext_latency_s:.2f}s | {r.uw_latency_s:.2f}s | `{uw_rec}` |")
    w("")

    # --- Per-Claim ---
    w("## Per-Claim Results")
    w("")
    for r in results:
        w(f"### {r.label}")
        w("")
        w(f"- **File:** `{r.file}`")

        if r.error:
            w(f"> ❌ **Error:** {r.error}")
            w("")
            continue

        w(f"- **Extraction accuracy:** {r.accuracy * 100:.1f}% ({r.passed}/{r.total} fields) in {r.ext_latency_s:.2f}s")
        w(f"- **Extraction tokens:** {r.ext_tokens_in:,} in / {r.ext_tokens_out:,} out")
        if r.uw_tokens_in or r.uw_tokens_out:
            w(f"- **Underwriting tokens:** {r.uw_tokens_in:,} in / {r.uw_tokens_out:,} out ({r.uw_latency_s:.2f}s)")
        w("")

        w("#### Input Prompt")
        w("")
        w("```")
        w(EXTRACTION_PROMPT)
        w("```")
        w("")

        w("#### Output: Field Extraction Accuracy")
        w("")
        w("| Status | Field | Expected |")
        w("|--------|-------|----------|")
        for fr in r.field_results:
            w(f"| {'✅' if fr.hit else '❌'} | `{fr.name}` | `{fr.expected}` |")
        w("")

        w("#### Output: Claude Extracted JSON")
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
            w("#### Underwriting Analysis")
            w("")
            w("| Dimension | Assessment |")
            w("|-----------|------------|")
            w(f"| Coverage determination | `{uw.get('coverage_determination', '—')}` |")
            w(f"| Reserve adequacy | `{uw.get('reserve_adequacy', '—')}` |")
            w(f"| **Recommendation** | **`{uw.get('recommendation', '—')}`** |")
            w("")
            w(f"**Liability:** {uw.get('liability_assessment', '—')}")
            w("")
            w(f"**Reserve commentary:** {uw.get('reserve_commentary', '—')}")
            w("")
            w(f"**Recommendation rationale:** {uw.get('recommendation_rationale', '—')}")
            w("")
            flags = uw.get("red_flags", [])
            if flags:
                w("**Red flags:**")
                for f in flags:
                    w(f"- ⚑ {f}")
                w("")
            issues = uw.get("key_issues", [])
            if issues:
                w("**Key issues before closure:**")
                for i in issues:
                    w(f"- {i}")
                w("")
            gaps = uw.get("coverage_gaps", [])
            if gaps:
                w("**Coverage gaps:**")
                for g in gaps:
                    w(f"- {g}")
                w("")
        elif uw.get("error"):
            w("#### Underwriting Analysis")
            w("")
            w(f"> ❌ Analysis failed: {uw['error']}")
            w("")

    # --- Field Hit Rate ---
    field_hits: dict[str, list[bool]] = defaultdict(list)
    for r in good:
        for fr in r.field_results:
            field_hits[fr.name].append(fr.hit)

    if field_hits:
        w("## Field Hit Rate")
        w("")
        w("Aggregated accuracy per field type across all claims, sorted worst → best.")
        w("")
        w("| Field | Hit Rate | Passed / Total |")
        w("|-------|----------|----------------|")
        for fname, hits in sorted(field_hits.items(), key=lambda x: sum(x[1]) / len(x[1])):
            n, p = len(hits), sum(hits)
            w(f"| `{fname}` | {p / n * 100:.1f}% | {p}/{n} |")
        w("")

    # --- Failure Patterns ---
    failures = [(r.label, fr) for r in good for fr in r.field_results if not fr.hit]
    w("## Failure Patterns")
    w("")
    if not failures:
        w("✅ No failures — all fields matched across all claims.")
    else:
        w(f"{len(failures)} failure(s) identified.")
        w("")
        for label, fr in failures:
            extracted_val = fr.extracted_json.get(fr.name)
            if extracted_val is None:
                candidates = [
                    f"`{k}` = `{v}`" for k, v in fr.extracted_json.items()
                    if fr.name.split("_")[-1] in k.lower()
                ]
                note = f"Key absent. Similar: {', '.join(candidates[:3])}" if candidates else "Key absent from output."
            else:
                note = f"Extracted `{extracted_val!r}` — did not match expected."
            w(f"### ❌ `{fr.name}` — {label}")
            w("")
            w(f"- **Expected:** `{fr.expected}`")
            w(f"- **Result:** {note}")
            w("")

    # --- Notes ---
    w("## Notes & Recommendations")
    w("")
    if overall_pct == 100:
        w("- All fields extracted correctly. Model performance is excellent on this eval set.")
    elif overall_pct >= 80:
        w("- Good overall accuracy. Review failure patterns above for targeted prompt improvements.")
    else:
        w("- Accuracy below 80%. Consider prompt refinement or few-shot examples for failing field types.")
    w("")

    hard = [fname for fname, hits in field_hits.items() if hits and sum(hits) / len(hits) < 0.6]
    if hard:
        w(f"- **Consistently difficult fields:** {', '.join(f'`{f}`' for f in hard)}. Consider explicit extraction hints in the prompt.")
        w("")

    approve = [r for r in good if r.underwriting.get("recommendation") == "approve"]
    investigate = [r for r in good if r.underwriting.get("recommendation") == "investigate"]
    if investigate:
        w(f"- **Claims requiring investigation:** {', '.join(r.label for r in investigate)}.")
        w("")
    if approve:
        w(f"- **Claims recommended for approval:** {', '.join(r.label for r in approve)}.")
        w("")

    content = "\n".join(lines)

    # Always print to stdout — works on remote/CI where files aren't accessible
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
        print(f"[warn] Could not save report to disk: {exc}")

    return out


# ---------------------------------------------------------------------------
# Basic chat sanity check
# ---------------------------------------------------------------------------

async def test_basic_chat(provider) -> None:
    from dobby.types import TextPart, UserMessagePart

    print("\n[Basic Chat]")
    result = await provider.chat(
        messages=[UserMessagePart(parts=[TextPart(text="Say hello in one sentence.")])],
    )
    print(f"  Response : {result.parts[0].text!r}")
    if result.usage:
        print(f"  Usage    : in={result.usage.input_tokens} out={result.usage.output_tokens}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(model: str) -> None:
    provider = _provider(model)
    print(f"Provider : {provider.name}")
    print(f"Model    : {model}")

    await test_basic_chat(provider)

    print(f"\n{'=' * 70}")
    print(f"Running {len(CLAIMS)} claims — extraction + underwriting...\n")

    results: list[ClaimResult] = []
    for claim in CLAIMS:
        print(f"  {claim['label']}...", end="", flush=True)
        r = await eval_claim(provider, claim)
        status = f"{r.passed}/{r.total}" if not r.error else "ERROR"
        print(f" {status}  (ext {r.ext_latency_s:.1f}s  uw {r.uw_latency_s:.1f}s)")
        results.append(r)

    print_report(results)

    report_path = write_report(results, model=model, provider_name=provider.name)
    print(f"Markdown report saved → {report_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic provider smoke test")
    parser.add_argument(
        "--model",
        default=ANTHROPIC_FOUNDRY_MODEL or "claude-sonnet-4-6",
        help="Model name or Azure deployment name",
    )
    args = parser.parse_args()

    if not (_USE_AZURE or ANTHROPIC_API_KEY):
        print(
            "\nERROR: No API key.\n"
            "Set ANTHROPIC_API_KEY or\n"
            "ANTHROPIC_FOUNDRY_API_KEY + ANTHROPIC_FOUNDRY_RESOURCE/BASE_URL in .env\n"
        )
        raise SystemExit(1)

    asyncio.run(run(model=args.model))


if __name__ == "__main__":
    main()
