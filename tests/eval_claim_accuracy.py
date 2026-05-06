"""Claim extraction accuracy eval.

Sends each claim PDF to the provider, asks for structured JSON extraction,
and compares against known ground truth. Reports per-field accuracy and latency.

Usage:
    uv run python tests/eval_claim_accuracy.py
    uv run python tests/eval_claim_accuracy.py --model claude-sonnet-4-6
    uv run python tests/eval_claim_accuracy.py --verbose
    uv run python tests/eval_claim_accuracy.py --report reports/eval.md
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
    extracted_json: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    extraction_prompt: str = ""
    underwriting_analysis: dict[str, Any] = field(default_factory=dict)
    underwriting_latency_s: float = 0.0
    underwriting_tokens_in: int = 0
    underwriting_tokens_out: int = 0

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


def _parse_json_response(raw: str) -> dict[str, Any]:
    text = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


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

    # --- Pass 1: field extraction ---
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
            extraction_prompt=EXTRACTION_PROMPT,
        )

    latency = time.perf_counter() - start
    raw_text = next((p.text for p in result.parts if isinstance(p, TextPart)), "")
    extracted = _parse_json_response(raw_text)

    if not extracted and verbose:
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

    usage = result.usage

    # --- Pass 2: underwriting analysis ---
    uw_analysis: dict[str, Any] = {}
    uw_latency = 0.0
    uw_tok_in = 0
    uw_tok_out = 0

    if extracted:
        uw_prompt = UNDERWRITING_PROMPT.format(
            extracted_json=json.dumps(extracted, indent=2)
        )
        uw_start = time.perf_counter()
        try:
            uw_result: StreamEndEvent = await provider.chat(
                messages=[
                    UserMessagePart(parts=[TextPart(text=uw_prompt)])
                ],
                system_prompt=(
                    "You are a senior insurance claims underwriter with 20 years of experience. "
                    "Provide precise, actionable underwriting assessments. Output only valid JSON."
                ),
                max_tokens=1024,
                temperature=0.0,
            )
            uw_latency = time.perf_counter() - uw_start
            uw_raw = next((p.text for p in uw_result.parts if isinstance(p, TextPart)), "")
            uw_analysis = _parse_json_response(uw_raw)
            if uw_result.usage:
                uw_tok_in = uw_result.usage.input_tokens
                uw_tok_out = uw_result.usage.output_tokens
        except Exception as exc:
            uw_latency = time.perf_counter() - uw_start
            uw_analysis = {"error": str(exc)}

    return ClaimResult(
        description=ground_truth.description,
        pdf_file=ground_truth.pdf_file,
        field_results=field_results,
        latency_s=latency,
        raw_response=raw_text,
        extracted_json=extracted,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        extraction_prompt=EXTRACTION_PROMPT,
        underwriting_analysis=uw_analysis,
        underwriting_latency_s=uw_latency,
        underwriting_tokens_in=uw_tok_in,
        underwriting_tokens_out=uw_tok_out,
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
    total_input_tokens = 0
    total_output_tokens = 0

    for r in results:
        status = "ERROR" if r.error else f"{r.passed}/{r.total} fields"
        pct = f"{r.accuracy * 100:.0f}%" if not r.error else "N/A"

        print(f"\n{'─' * 70}")
        print(f"{r.description}")
        print(f"  File:         {r.pdf_file}")
        print(f"  Accuracy:     {pct}  ({status})  [{r.latency_s:.2f}s extraction]")

        if r.error:
            print(f"  Error:        {r.error}")
            continue

        if r.input_tokens or r.output_tokens:
            print(f"  Tokens (ext): in={r.input_tokens}  out={r.output_tokens}")

        print(f"\n  INPUT PROMPT (extraction):")
        for line in r.extraction_prompt.splitlines():
            print(f"    {line}")

        print(f"\n  OUTPUT (extracted JSON):")
        if r.extracted_json:
            for line in json.dumps(r.extracted_json, indent=4).splitlines():
                print(f"    {line}")
        elif r.raw_response:
            print(f"    [JSON parse failed] Raw: {r.raw_response[:300]}")
        else:
            print(f"    [no response]")

        print(f"\n  FIELD ACCURACY:")
        for fr in r.field_results:
            mark = "✓" if fr.found else "✗"
            print(f"    {mark} {fr.field:<25} expected: {fr.expected}")

        if r.underwriting_analysis and "error" not in r.underwriting_analysis:
            uw = r.underwriting_analysis
            print(f"\n  UNDERWRITING ANALYSIS  [{r.underwriting_latency_s:.2f}s  "
                  f"in={r.underwriting_tokens_in} out={r.underwriting_tokens_out}]")
            print(f"    Coverage:      {uw.get('coverage_determination', '—')}")
            print(f"    Liability:     {uw.get('liability_assessment', '—')}")
            print(f"    Reserve:       {uw.get('reserve_adequacy', '—')} — {uw.get('reserve_commentary', '')}")
            print(f"    Recommendation:{uw.get('recommendation', '—')}")
            print(f"    Rationale:     {uw.get('recommendation_rationale', '—')}")
            flags = uw.get("red_flags", [])
            if flags:
                print(f"    Red flags:")
                for f in flags:
                    print(f"      ⚑  {f}")
            issues = uw.get("key_issues", [])
            if issues:
                print(f"    Key issues:")
                for i in issues:
                    print(f"      •  {i}")
            gaps = uw.get("coverage_gaps", [])
            if gaps:
                print(f"    Coverage gaps:")
                for g in gaps:
                    print(f"      ○  {g}")
        elif r.underwriting_analysis.get("error"):
            print(f"\n  UNDERWRITING ANALYSIS: ERROR — {r.underwriting_analysis['error']}")

        total_fields += r.total
        total_passed += r.passed
        total_input_tokens += r.input_tokens
        total_output_tokens += r.output_tokens

    # ---- Summary ----
    print("\n" + "-" * 70)
    overall_pct = (total_passed / total_fields * 100) if total_fields else 0
    print(f"OVERALL ACCURACY: {total_passed}/{total_fields} fields  ({overall_pct:.1f}%)")
    avg_latency = sum(r.latency_s for r in results) / len(results) if results else 0
    print(f"AVG LATENCY:      {avg_latency:.2f}s per claim")
    print(f"TOTAL TOKENS:     in={total_input_tokens}  out={total_output_tokens}  total={total_input_tokens + total_output_tokens}")

    good_results = [r for r in results if not r.error]

    # ---- Rankings: worst → best ----
    print("\n" + "-" * 70)
    print("RANKINGS (worst → best)")
    for r in sorted(good_results, key=lambda x: x.accuracy):
        bar = ("█" * r.passed) + ("░" * (r.total - r.passed))
        print(f"  {r.accuracy * 100:5.1f}%  [{bar}]  {r.description}")

    # ---- Field-level hit rate across all claims ----
    field_hits: dict[str, list[bool]] = defaultdict(list)
    for r in good_results:
        for fr in r.field_results:
            field_hits[fr.field].append(fr.found)

    if field_hits:
        print("\n" + "-" * 70)
        print("FIELD HIT RATE (aggregated)")
        for fname, hits in sorted(field_hits.items(), key=lambda x: sum(x[1]) / len(x[1])):
            n = len(hits)
            passed_n = sum(hits)
            pct = passed_n / n * 100
            bar = ("█" * passed_n) + ("░" * (n - passed_n))
            print(f"  {pct:5.1f}%  [{bar}]  {fname}")

    # ---- Failure patterns: failed fields + what Claude extracted ----
    failures = [
        (r.description, fr)
        for r in good_results
        for fr in r.field_results
        if not fr.found
    ]
    if failures:
        print("\n" + "-" * 70)
        print("FAILURE PATTERNS")
        for desc, fr in failures:
            extracted_val = fr.extracted_json.get(fr.field) if fr.extracted_json else None
            if extracted_val is None:
                # field key missing — search for closest key
                candidates = [
                    f"{k}={v!r}"
                    for k, v in fr.extracted_json.items()
                    if fr.field.split("_")[-1] in k.lower()
                ] if fr.extracted_json else []
                extracted_str = f"key absent (similar: {', '.join(candidates[:2])})" if candidates else "key absent"
            else:
                extracted_str = repr(extracted_val)
            print(f"  ✗ [{desc}] {fr.field}")
            print(f"      expected:  {fr.expected!r}")
            print(f"      extracted: {extracted_str}")
    else:
        print("\n" + "-" * 70)
        print("FAILURE PATTERNS: none — all fields matched")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown_report(
    results: list[ClaimResult],
    model: str,
    output_path: Path,
) -> None:
    good = [r for r in results if not r.error]
    total_fields = sum(r.total for r in good)
    total_passed = sum(r.passed for r in good)
    overall_pct = (total_passed / total_fields * 100) if total_fields else 0
    avg_latency = sum(r.latency_s for r in good) / len(good) if good else 0
    total_in = sum(r.input_tokens for r in good)
    total_out = sum(r.output_tokens for r in good)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    w = lines.append

    w("# Claim Extraction Accuracy Report")
    w("")
    w(f"**Model:** `{model}`  ")
    w(f"**Generated:** {generated_at}  ")
    w(f"**Claims evaluated:** {len(results)}  ")
    w("")

    total_uw_in = sum(r.underwriting_tokens_in for r in good)
    total_uw_out = sum(r.underwriting_tokens_out for r in good)
    total_tok_in = total_in + total_uw_in
    total_tok_out = total_out + total_uw_out

    # --- Executive Summary ---
    w("## Executive Summary")
    w("")
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Overall field accuracy | **{overall_pct:.1f}%** ({total_passed}/{total_fields} fields) |")
    w(f"| Average extraction latency | {avg_latency:.2f}s per claim |")
    w(f"| Extraction tokens (in / out) | {total_in:,} / {total_out:,} |")
    w(f"| Underwriting tokens (in / out) | {total_uw_in:,} / {total_uw_out:,} |")
    w(f"| **Total tokens** | **{total_tok_in + total_tok_out:,}** |")
    w("")

    error_results = [r for r in results if r.error]
    if error_results:
        w("> **Note:** The following claims failed to evaluate:")
        for r in error_results:
            w(f"> - `{r.pdf_file}`: {r.error}")
        w("")

    # --- Rankings ---
    w("## Rankings")
    w("")
    w("Claims sorted from lowest to highest extraction accuracy.")
    w("")
    w("| Rank | Claim | Accuracy | Fields | Ext. Latency | UW Recommendation |")
    w("|------|-------|----------|--------|--------------|-------------------|")
    for i, r in enumerate(sorted(good, key=lambda x: x.accuracy), 1):
        uw_rec = r.underwriting_analysis.get("recommendation", "—") if r.underwriting_analysis else "—"
        w(f"| {i} | {r.description} | {r.accuracy * 100:.1f}% | {r.passed}/{r.total} | {r.latency_s:.2f}s | {uw_rec} |")
    w("")

    # --- Per-Claim Results ---
    w("## Per-Claim Results")
    w("")
    for r in results:
        pct = f"{r.accuracy * 100:.1f}%" if not r.error else "ERROR"
        w(f"### {r.description}")
        w("")
        w(f"- **File:** `{r.pdf_file}`")
        w(f"- **Extraction accuracy:** {pct} ({r.passed}/{r.total} fields) in {r.latency_s:.2f}s")
        if r.input_tokens or r.output_tokens:
            w(f"- **Extraction tokens:** {r.input_tokens:,} in / {r.output_tokens:,} out")
        if r.underwriting_tokens_in or r.underwriting_tokens_out:
            w(f"- **Underwriting tokens:** {r.underwriting_tokens_in:,} in / {r.underwriting_tokens_out:,} out ({r.underwriting_latency_s:.2f}s)")
        w("")

        if r.error:
            w(f"> ❌ **Error:** {r.error}")
            w("")
            continue

        w("#### Input Prompt (Extraction)")
        w("")
        w("```")
        w(r.extraction_prompt)
        w("```")
        w("")

        w("#### Output: Extracted Fields")
        w("")
        w("| Status | Field | Expected |")
        w("|--------|-------|----------|")
        for fr in r.field_results:
            mark = "✅" if fr.found else "❌"
            w(f"| {mark} | `{fr.field}` | `{fr.expected}` |")
        w("")

        w("#### Output: Claude Extracted JSON")
        w("")
        if r.extracted_json:
            w("```json")
            w(json.dumps(r.extracted_json, indent=2))
            w("```")
        elif r.raw_response:
            w("> ⚠️ JSON parse failed. Raw output:")
            w("")
            w("```")
            w(r.raw_response[:500])
            w("```")
        else:
            w("> No response captured.")
        w("")

        if r.underwriting_analysis and "error" not in r.underwriting_analysis:
            uw = r.underwriting_analysis
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
        elif r.underwriting_analysis.get("error"):
            w("#### Underwriting Analysis")
            w("")
            w(f"> ❌ Analysis failed: {r.underwriting_analysis['error']}")
            w("")

    # --- Field Hit Rate ---
    field_hits: dict[str, list[bool]] = defaultdict(list)
    for r in good:
        for fr in r.field_results:
            field_hits[fr.field].append(fr.found)

    if field_hits:
        w("## Field Hit Rate")
        w("")
        w("Aggregated accuracy per field type across all claims, sorted worst → best.")
        w("")
        w("| Field | Hit Rate | Passed / Total |")
        w("|-------|----------|----------------|")
        for fname, hits in sorted(field_hits.items(), key=lambda x: sum(x[1]) / len(x[1])):
            n = len(hits)
            p = sum(hits)
            w(f"| `{fname}` | {p / n * 100:.1f}% | {p}/{n} |")
        w("")

    # --- Failure Patterns ---
    failures = [
        (r.description, fr)
        for r in good
        for fr in r.field_results
        if not fr.found
    ]

    w("## Failure Patterns")
    w("")
    if not failures:
        w("✅ No failures — all fields matched across all claims.")
    else:
        w(f"{len(failures)} field extraction failure(s) identified.")
        w("")
        for desc, fr in failures:
            extracted_val = fr.extracted_json.get(fr.field) if fr.extracted_json else None
            if extracted_val is None:
                candidates = [
                    f"`{k}` = `{v}`"
                    for k, v in fr.extracted_json.items()
                    if fr.field.split("_")[-1] in k.lower()
                ] if fr.extracted_json else []
                note = f"Key absent from output. Similar keys: {', '.join(candidates[:3])}" if candidates else "Key absent from output."
            else:
                note = f"Extracted `{extracted_val!r}` — value present but did not match expected."
            w(f"### ❌ `{fr.field}` — {desc}")
            w("")
            w(f"- **Expected:** `{fr.expected}`")
            w(f"- **Result:** {note}")
            w("")

    # --- Notes & Recommendations ---
    w("## Notes & Recommendations")
    w("")
    if overall_pct == 100:
        w("- All fields extracted correctly. Model performance is excellent on this eval set.")
    elif overall_pct >= 80:
        w("- Good overall accuracy. Review failure patterns above for targeted prompt improvements.")
    else:
        w("- Accuracy below 80%. Consider prompt refinement or few-shot examples for failing field types.")
    w("")

    hard_fields = [
        fname for fname, hits in field_hits.items()
        if hits and sum(hits) / len(hits) < 0.6
    ]
    if hard_fields:
        w(f"- **Consistently difficult fields:** {', '.join(f'`{f}`' for f in hard_fields)}.")
        w("  These may benefit from explicit extraction instructions in the prompt.")
        w("")

    slow = [r for r in good if r.latency_s > 5]
    if slow:
        w(f"- **Slow claims (>5s):** {', '.join(r.description for r in slow)}. Consider streaming or async batching.")
        w("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown report saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(model: str, verbose: bool, report: Path | None) -> None:
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

    if report:
        write_markdown_report(results, model=model, output_path=report)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        default_path = Path(__file__).parent / "reports" / f"eval_{timestamp}.md"
        write_markdown_report(results, model=model, output_path=default_path)


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
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write markdown report to this path (default: tests/reports/eval_<timestamp>.md)",
    )
    args = parser.parse_args()
    asyncio.run(run(model=args.model, verbose=args.verbose, report=args.report))


if __name__ == "__main__":
    main()
