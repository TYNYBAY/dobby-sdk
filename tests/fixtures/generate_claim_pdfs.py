"""Generate sample claim underwriting PDFs for use as test fixtures.

Run from the repo root:
    python tests/fixtures/generate_claim_pdfs.py

Outputs:
    tests/fixtures/auto_liability_claim.pdf
    tests/fixtures/property_damage_claim.pdf
    tests/fixtures/workers_comp_claim.pdf
    tests/fixtures/medical_malpractice_claim.pdf
    tests/fixtures/homeowners_claim.pdf

No external libraries required - PDFs are generated using raw PDF spec bytes.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Minimal PDF builder (no external deps)
# ---------------------------------------------------------------------------


def _pdf(pages: list[list[str]]) -> bytes:
    """Build a multi-page PDF from a list of pages, each a list of text lines."""

    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    # Build each page content stream
    page_streams: list[bytes] = []
    for lines in pages:
        ops: list[str] = ["BT", "/F1 11 Tf", "50 780 Td", "14 TL"]
        for line in lines:
            if line.startswith("##"):  # heading
                ops += ["/F2 14 Tf", f"({_esc(line[2:].strip())}) Tj", "/F1 11 Tf", "T*"]
            elif line == "---":
                ops += ["0 -4 Td", "0 -4 Td"]
            else:
                ops.append(f"({_esc(line)}) Tj T*")
        ops.append("ET")
        page_streams.append("\n".join(ops).encode("latin-1"))

    # PDF object table
    objs: list[bytes] = []

    def add(b: bytes) -> int:
        objs.append(b)
        return len(objs)  # 1-indexed

    # catalog → obj 1, pages tree → obj 2
    # fonts → obj 3 (Helvetica), obj 4 (Helvetica-Bold)
    # Each page: page_obj, content_stream

    catalog_idx = 1
    pages_tree_idx = 2
    font_regular_idx = 3
    font_bold_idx = 4

    first_page_obj_idx = 5  # pages start at obj 5 (pairs: page_obj, content_obj)
    page_obj_indices: list[int] = []
    for i in range(len(pages)):
        page_obj_indices.append(first_page_obj_idx + i * 2)

    kids_ref = " ".join(f"{i} 0 R" for i in page_obj_indices)

    obj_bytes: list[bytes] = [b""] * (4 + len(pages) * 2)

    obj_bytes[0] = f"1 0 obj\n<</Type/Catalog/Pages {pages_tree_idx} 0 R>>\nendobj\n".encode()
    obj_bytes[1] = (
        f"2 0 obj\n<</Type/Pages/Kids[{kids_ref}]/Count {len(pages)}>>\nendobj\n".encode()
    )
    obj_bytes[2] = b"3 0 obj\n<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>\nendobj\n"
    obj_bytes[3] = b"4 0 obj\n<</Type/Font/Subtype/Type1/BaseFont/Helvetica-Bold>>\nendobj\n"

    for i, stream in enumerate(page_streams):
        page_obj_num = first_page_obj_idx + i * 2
        content_obj_num = page_obj_num + 1
        obj_bytes[4 + i * 2] = (
            f"{page_obj_num} 0 obj\n"
            f"<</Type/Page/Parent {pages_tree_idx} 0 R/MediaBox[0 0 612 792]"
            f"/Contents {content_obj_num} 0 R"
            f"/Resources<</Font<</F1 {font_regular_idx} 0 R/F2 {font_bold_idx} 0 R>>>>>>\n"
            f"endobj\n"
        ).encode()
        obj_bytes[4 + i * 2 + 1] = (
            f"{content_obj_num} 0 obj\n<</Length {len(stream)}>>\nstream\n".encode()
            + stream
            + b"\nendstream\nendobj\n"
        )

    header = b"%PDF-1.4\n"
    body = b""
    offsets: list[int] = []
    pos = len(header)
    for b in obj_bytes:
        offsets.append(pos)
        body += b
        pos += len(b)

    xref_pos = pos
    n = len(obj_bytes) + 1  # +1 for free entry
    xref = f"xref\n0 {n}\n0000000000 65535 f \n"
    xref += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    trailer = f"trailer\n<</Size {n}/Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF\n"

    return header + body + xref.encode() + trailer.encode()


# ---------------------------------------------------------------------------
# Claim document content
# ---------------------------------------------------------------------------


CLAIMS: list[tuple[str, list[list[str]]]] = [
    (
        "auto_liability_claim.pdf",
        [
            [
                "## AUTO LIABILITY CLAIM FORM",
                "---",
                "Claim Number:     CLM-2026-AL-00142",
                "Date of Loss:     2026-03-15",
                "Date Reported:    2026-03-16",
                "Policy Number:    POL-AUTO-8823991",
                "",
                "## INSURED INFORMATION",
                "Name:             Jane R. Hartwell",
                "Address:          4821 Elm Street, Springfield, IL 62701",
                "Phone:            (217) 555-0183",
                "Email:            j.hartwell@example.com",
                "Driver License:   IL-DL-9923847",
                "",
                "## VEHICLE INFORMATION",
                "Year/Make/Model:  2021 Toyota Camry SE",
                "VIN:              1HGBH41JXMN109186",
                "License Plate:    IL  ABC-4421",
                "Odometer:         34,812 miles",
            ],
            [
                "## ACCIDENT DESCRIPTION",
                "Location:         I-72 WB near Exit 105, Sangamon County",
                "Weather:          Clear, dry road surface",
                "Time:             08:42 AM",
                "",
                "Description:",
                "Insured vehicle was struck from behind at highway speed by a",
                "2019 Ford F-150 (VIN 1FTEW1EP0KFB32881) driven by Mark T. Olsen.",
                "Impact caused rear bumper crush, trunk damage, and frame deformation.",
                "Insured sustained whiplash and lower back strain; treated at",
                "St. John's Hospital ER and referred to orthopedic specialist.",
                "",
                "## DAMAGE ESTIMATE",
                "Repair facility:  Springfield Auto Body & Frame",
                "Estimate date:    2026-03-18",
                "Parts:            $4,210.00",
                "Labor (28 hrs):   $3,360.00",
                "Paint & materials:$1,150.00",
                "Sublet (alignment):$220.00",
                "Rental (14 days): $630.00",
                "Total estimate:   $9,570.00",
                "",
                "## UNDERWRITING RECOMMENDATION",
                "Liability: Clear third-party fault. Pursue subrogation against",
                "Olsen/Progressive Insurance policy #PRG-IL-20019. Reserve: $14,500.",
                "Medical: Open - orthopedic evaluation pending.",
            ],
        ],
    ),
    (
        "property_damage_claim.pdf",
        [
            [
                "## COMMERCIAL PROPERTY DAMAGE CLAIM",
                "---",
                "Claim Number:     CLM-2026-PD-00308",
                "Date of Loss:     2026-02-28",
                "Date Reported:    2026-03-01",
                "Policy Number:    POL-COMM-1143772",
                "Coverage Type:    All-Risk Commercial Property",
                "",
                "## POLICYHOLDER",
                "Business Name:    Hartfield Printing & Design LLC",
                "Address:          9000 Industrial Pkwy, Decatur, IL 62521",
                "Contact:          Robert Hartfield, (217) 555-0294",
                "SIC Code:         2750 (Commercial Printing)",
                "",
                "## CAUSE OF LOSS",
                "Peril:            Burst water main (city infrastructure failure)",
                "Origin:           Basement mechanical room, 2-inch supply line failure",
                "Duration:         Approximately 6 hours before discovery",
                "Affected areas:   Basement (storage), Ground floor (press room)",
            ],
            [
                "## INVENTORY OF LOSSES",
                "Item                          Qty  RCV         ACV",
                "------------------------------------------------------",
                "Heidelberg offset press       1    $82,000     $41,000",
                "Riso inkjet system            1    $18,500     $12,000",
                "Paper stock (warehouse)       --   $6,340      $6,340",
                "Finished goods inventory      --   $11,200     $11,200",
                "Shelving/racking system       --   $3,800      $2,100",
                "Electrical panel (partial)    --   $4,200      $4,200",
                "Drywall / flooring repair     --   $9,700      $9,700",
                "Business interruption (est.)  --   $22,000     $22,000",
                "------------------------------------------------------",
                "TOTAL                              $157,740    $108,540",
                "",
                "Deductible:  $5,000",
                "Net payable (ACV basis):  $103,540",
                "",
                "## UNDERWRITING NOTES",
                "Subrogation potential: Investigate city utility negligence.",
                "Adjuster: Sarah Nguyen, ext. 4412",
                "Status: Field inspection complete. Awaiting contractor bid.",
            ],
        ],
    ),
    (
        "workers_comp_claim.pdf",
        [
            [
                "## WORKERS COMPENSATION CLAIM",
                "---",
                "Claim Number:     CLM-2026-WC-00091",
                "Date of Injury:   2026-01-22",
                "Date Reported:    2026-01-23",
                "Policy Number:    POL-WC-5567234",
                "Jurisdiction:     State of Illinois",
                "",
                "## EMPLOYER INFORMATION",
                "Employer:         Midland Logistics & Warehousing Inc.",
                "FEIN:             37-1923847",
                "Address:          7701 Commerce Dr, Bloomington, IL 61701",
                "Contact:          HR Manager Lisa Park, (309) 555-0147",
                "NAICS:            493110 (General Warehousing)",
                "",
                "## EMPLOYEE INFORMATION",
                "Name:             Carlos M. Reyes",
                "DOB:              1988-07-14",
                "SSN (last 4):     ****-**-7743",
                "Job Title:        Forklift Operator, Grade 3",
                "Hire Date:        2021-04-05",
                "Avg Weekly Wage:  $1,142.00",
            ],
            [
                "## INJURY DESCRIPTION",
                "Body Part:        Left knee (medial meniscus)",
                "Nature:           Tear - acute traumatic",
                "Cause:            Misstep off loading dock; knee hyperextended",
                "Witnesses:        Tom Burrows, Shift Supervisor",
                "",
                "Medical Treatment:",
                "  Initial:        Bloomington OSF ER, 2026-01-22",
                "  Orthopedist:    Dr. Amy Chen, Midwest Ortho Group",
                "  MRI (2026-01-29): Confirmed full-thickness medial meniscus tear",
                "  Surgery:        Arthroscopic repair scheduled 2026-02-14",
                "  Physical therapy: 12 sessions authorized",
                "",
                "## CLAIM RESERVES",
                "Medical:          $28,500 (est. surgery + PT)",
                "Indemnity (TTD):  $13,704 (12 weeks @ 66.67% AWW)",
                "Vocational rehab: $2,500 (contingency)",
                "Total reserve:    $44,704",
                "",
                "## UNDERWRITING FLAGS",
                "- Prior knee injury (right knee, 2019): unrelated, documented",
                "- OSHA 300 log entry filed",
                "- Safety training current (forklift cert 2025-11)",
                "- Return-to-work: light duty available from week 8",
            ],
        ],
    ),
    (
        "medical_malpractice_claim.pdf",
        [
            [
                "## MEDICAL MALPRACTICE CLAIM",
                "---",
                "Claim Number:     CLM-2026-MM-00017",
                "Date of Incident: 2025-11-04",
                "Date Reported:    2026-01-10",
                "Policy Number:    POL-MPL-0023491",
                "Coverage Type:    Claims-Made, $1M/$3M limits",
                "",
                "## INSURED (PROVIDER)",
                "Name:             Dr. Kevin J. Allard, MD",
                "Specialty:        General Surgery",
                "NPI:              1234567890",
                "Facility:         Memorial Medical Center, Champaign, IL",
                "Years in practice:18",
                "",
                "## CLAIMANT",
                "Name:             Patricia L. Vasquez",
                "DOB:              1957-03-22",
                "Procedure:        Laparoscopic cholecystectomy (gallbladder removal)",
                "Date of procedure:2025-11-04",
                "Outcome:          Bile duct injury discovered 2025-11-09",
            ],
            [
                "## ALLEGATION SUMMARY",
                "Claimant alleges negligent dissection causing common bile duct",
                "transection. Required open repair surgery (ERCP + Roux-en-Y",
                "hepaticojejunostomy) on 2025-11-12. Extended hospitalization",
                "of 22 days. Subsequent cholangitis episode (2026-01-03) required",
                "re-hospitalization of 8 days. Chronic pain and digestive issues",
                "ongoing at time of filing.",
                "",
                "Claimant counsel: Harrington & Walsh LLP, Chicago",
                "Demand letter received: 2026-01-08, demand: $1,250,000",
                "",
                "## MEDICAL RECORDS REVIEW",
                "Peer reviewer:    Dr. Susan Hoffmann, MD (General Surgery)",
                "Opinion (prelim): Intraoperative cholangiography not performed;",
                "                  standard of care deviation possible.",
                "Full report due:  2026-03-15",
                "",
                "## RESERVE / EXPOSURE ANALYSIS",
                "Economic damages (est.):   $380,000",
                "Non-economic damages (est.):$600,000",
                "Defense costs (est.):      $95,000",
                "Total reserve:             $1,075,000",
                "",
                "Coverage position: Covered, subject to peer review outcome.",
                "Defense counsel assigned: Morrison & Foerster, Springfield",
            ],
        ],
    ),
    (
        "homeowners_claim.pdf",
        [
            [
                "## HOMEOWNERS CLAIM FORM -HO-3 POLICY",
                "---",
                "Claim Number:     CLM-2026-HO-00522",
                "Date of Loss:     2026-04-02",
                "Date Reported:    2026-04-02",
                "Policy Number:    POL-HO3-7734128",
                "Coverage:         Dwelling $425,000 / Contents $212,500",
                "Deductible:       $2,500",
                "",
                "## POLICYHOLDER",
                "Name:             Marcus & Diana Webb",
                "Address:          2240 Birchwood Ln, Normal, IL 61761",
                "Phone:            (309) 555-0366",
                "Years insured:    7",
                "Prior claims:     1 (water -2022, $4,200)",
                "",
                "## CAUSE OF LOSS",
                "Peril:            Wind/hail -documented storm event",
                "Storm date:       2026-04-02  (NWS confirmation attached)",
                "Affected areas:   Roof, gutters, siding (north elevation),",
                "                  2 windows (master BR, living room)",
            ],
            [
                "## DAMAGE DETAIL",
                "Item                       Est. Cost     Notes",
                "------------------------------------------------------",
                "Roof replacement (sq 28)   $14,980       40-yr arch shingle",
                "Gutter/downspout (110 LF)  $1,870",
                "North siding (480 SF)      $4,320        Vinyl, match req.",
                "Window -master BR         $980",
                "Window -living room       $1,140",
                "Interior water intrusion   $3,200        Drywall, insulation",
                "------------------------------------------------------",
                "Subtotal (RCV):            $26,490",
                "Recoverable depreciation:  $5,820",
                "ACV payment:               $18,170",
                "Less deductible:           $2,500",
                "Net ACV payment:           $15,670",
                "",
                "## UNDERWRITING NOTES",
                "Roof age at loss: 9 years (installed 2017). Condition: fair.",
                "Storm verification: NWS storm report confirms 70 mph gusts,",
                "1.5-inch hail. Two neighboring claims on same street.",
                "Recommendation: Pay ACV now; hold RCV pending completion.",
                "Field adjuster: James Okafor, ext. 3281",
                "Status: Estimate approved. Issuing ACV payment.",
            ],
        ],
    ),
]


# ---------------------------------------------------------------------------
# Write PDFs
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, pages in CLAIMS:
        path = OUTPUT_DIR / filename
        data = _pdf(pages)
        path.write_bytes(data)
        b64 = base64.b64encode(data).decode()
        print(f"  Written: {path}  ({len(data):,} bytes, base64 len={len(b64)})")

    print(f"\nAll {len(CLAIMS)} claim PDFs written to {OUTPUT_DIR.resolve()}")
    print("Use Base64PDFSource(data=...) to pass them to DocumentPart in tests.")


if __name__ == "__main__":
    main()
