"""Generate a deliberately messy demo corpus.

The corpus is engineered so every agent behaviour is demonstrable on stage:

  * ANSWERABLE (clean)      -> handbook.pdf, native text layer
  * ANSWERABLE (OCR only)   -> scanned_memo.png, text exists ONLY as an image
  * CONTRADICTION           -> notice period: handbook says 60 days,
                               the addendum says 30 days
  * GAP / ABSTAIN           -> nothing about salaries, CEO address, or
                               health insurance providers
  * AMBIGUOUS -> CLARIFY    -> "leave policy" differs for FTE vs contractor

Run:  python scripts/generate_corpus.py
"""
from __future__ import annotations

from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

HANDBOOK = """NORTHWIND TECHNOLOGIES PRIVATE LIMITED
EMPLOYEE HANDBOOK - REVISION 4.2
Effective 01 January 2026

SECTION 1. REMOTE WORK POLICY

1.1 Northwind operates a hybrid working model. All full-time employees are
required to be present in the Bengaluru office a minimum of three (3) days
per calendar week. The remaining two days may be worked remotely.

1.2 Core collaboration hours are 11:00 to 17:00 IST. Employees must be
reachable on approved channels during core hours regardless of location.

1.3 Fully remote arrangements require written approval from both the
reporting manager and the Head of People Operations, and are reviewed every
six months.

SECTION 2. LEAVE ENTITLEMENT

2.1 Full-time employees are entitled to eighteen (18) days of casual leave
per calendar year, accrued monthly at 1.5 days.

2.2 Full-time employees are additionally entitled to twelve (12) days of
sick leave per calendar year. Sick leave does not carry forward.

2.3 A maximum of six (6) days of unused casual leave may be carried forward
into the following calendar year. Any excess lapses on 31 December.

2.4 Leave requests exceeding five consecutive working days require approval
at least fourteen days in advance.

SECTION 3. SEPARATION

3.1 Employees who wish to resign must serve a notice period of sixty (60)
calendar days from the date the resignation is formally acknowledged.

3.2 Notice period may be waived only at the sole discretion of the Head of
People Operations.

SECTION 4. EQUIPMENT

4.1 Northwind issues each employee one laptop and one external monitor.
Replacement of damaged equipment is subject to an assessment by the IT team.

4.2 Employees are responsible for equipment assigned to them and must
return all assets on or before their final working day.

SECTION 5. EXPENSE REIMBURSEMENT

5.1 Business travel expenses must be submitted within thirty (30) days of
the date the expense was incurred. Claims submitted after this window will
not be processed.

5.2 Meal expenses during domestic travel are reimbursed up to INR 1,200 per
day. International travel is reimbursed up to USD 60 per day.
"""

ADDENDUM = """NORTHWIND TECHNOLOGIES PRIVATE LIMITED
POLICY ADDENDUM A-7
Issued 15 March 2026
Applies to: all staff

ITEM 1. REVISION TO SEPARATION TERMS

1.1 With immediate effect, the notice period applicable to resigning
employees is thirty (30) calendar days.

1.2 This item supersedes any conflicting provision published in earlier
revisions of the Employee Handbook.

ITEM 2. CONTRACTOR ENGAGEMENT TERMS

2.1 Engaged contractors are not covered by the leave entitlements set out
in Section 2 of the Employee Handbook.

2.2 Contractors accrue leave in accordance with the terms of their
individual statement of work. There is no company-wide contractor leave
entitlement.

ITEM 3. TRAVEL

3.1 The domestic meal reimbursement cap is raised to INR 1,500 per day.
"""

# This content exists ONLY inside the scanned image - proving the OCR path.
SCANNED_MEMO = """INTERNAL MEMORANDUM
NORTHWIND TECHNOLOGIES

TO:      All Engineering Staff
FROM:    Priya Raghavan, VP Engineering
DATE:    04 February 2026
SUBJECT: Production Deployment Freeze

Effective immediately, a deployment freeze applies to all
production systems between 22:00 and 06:00 IST on weekdays,
and for the full duration of every weekend.

Emergency hotfixes are exempt from this freeze but require
verbal approval from the on-call engineering lead and must
be recorded in the incident log within one hour.

The freeze will be reviewed at the end of Q2 2026.

All deployment requests must be raised through the internal
change management portal at least four hours before the
intended deployment window.
"""


def write_pdf(path: Path, title: str, body: str) -> None:
    """Write a native-text-layer PDF using reportlab if available,
    otherwise fall back to a .txt so the corpus still works."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError:
        path.with_suffix(".txt").write_text(body, encoding="utf-8")
        print(f"  reportlab missing -> wrote {path.with_suffix('.txt').name}")
        return

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    margin, leading = 20 * mm, 13
    y = height - margin
    c.setFont("Helvetica", 9.5)

    for line in body.split("\n"):
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 9.5)
            y = height - margin
        c.drawString(margin, y, line[:110])
        y -= leading

    c.save()
    print(f"  wrote {path.name}")


def write_scanned_image(path: Path, body: str) -> None:
    """Render text to an image and degrade it so real OCR is required."""
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
    except ImportError:
        path.with_suffix(".txt").write_text(body, encoding="utf-8")
        print("  Pillow missing -> wrote .txt fallback")
        return

    W, H = 1240, 1754  # A4 at 150 dpi
    img = Image.new("RGB", (W, H), (252, 250, 244))  # off-white "paper"
    draw = ImageDraw.Draw(img)

    font = None
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 26)
            break
    if font is None:
        font = ImageFont.load_default()

    y = 90
    for line in body.split("\n"):
        draw.text((90, y), line, fill=(28, 28, 34), font=font)
        y += 38

    # Degrade: slight rotation + blur + scan noise, so the text layer is
    # genuinely absent and Tesseract has to work.
    img = img.rotate(0.45, expand=False, fillcolor=(252, 250, 244))
    img = img.filter(ImageFilter.GaussianBlur(0.6))

    try:
        import numpy as np
        arr = np.array(img).astype("int16")
        noise = np.random.normal(0, 5, arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype("uint8"))
    except ImportError:
        pass

    img.save(path, quality=88)
    print(f"  wrote {path.name}")


def main() -> None:
    print("Generating messy demo corpus in data/raw/ ...")
    write_pdf(RAW / "employee_handbook.pdf", "Employee Handbook", HANDBOOK)
    write_pdf(RAW / "policy_addendum_a7.pdf", "Policy Addendum", ADDENDUM)
    write_scanned_image(RAW / "scanned_deployment_memo.png", SCANNED_MEMO)
    print("\nCorpus ready. Planted characteristics:")
    print("  * notice period CONTRADICTS between handbook (60d) and addendum (30d)")
    print("  * deployment freeze exists ONLY in the scanned image (OCR required)")
    print("  * contractor leave is deliberately underspecified (-> clarify)")
    print("  * salaries / CEO address / insurance absent entirely (-> abstain)")


if __name__ == "__main__":
    main()
