"""Document loading for a deliberately messy corpus.

Handles three real-world cases the problem statement calls out:
  1. Native, text-layer PDFs           -> pdfplumber (layout aware, tables)
  2. Scanned / image-only PDFs         -> rasterise + Tesseract OCR
  3. Standalone scanned images         -> Tesseract OCR

Every extracted page carries metadata (source, page, extraction method,
ocr_confidence) so downstream nodes can reason about *how reliable* the
text is, not just what it says.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.logging_conf import get_logger

log = get_logger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
TEXT_SUFFIXES = {".txt", ".md"}

# Below this many characters we assume the PDF page has no usable text
# layer and fall back to OCR.
MIN_TEXT_LAYER_CHARS = 40


@dataclass
class ExtractedPage:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_document(path: str | Path) -> list[ExtractedPage]:
    """Dispatch to the right extractor based on file type."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return _load_image(path)
    if suffix in TEXT_SUFFIXES:
        return _load_text(path)

    raise ValueError(f"Unsupported file type: {suffix}")


# ----------------------------------------------------------------------
def _load_text(path: Path) -> list[ExtractedPage]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        ExtractedPage(
            text=text,
            metadata={
                "source": path.name,
                "page": 1,
                "extraction": "native_text",
                "ocr_confidence": None,
            },
        )
    ]


def _load_pdf(path: Path) -> list[ExtractedPage]:
    """Per page: use the text layer if present, else OCR that page."""
    import pdfplumber

    pages: list[ExtractedPage] = []

    with pdfplumber.open(str(path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()

            # Preserve tables as pipe-delimited rows so structure survives
            # chunking instead of collapsing into unreadable prose.
            tables = page.extract_tables() or []
            for table in tables:
                rendered = "\n".join(
                    " | ".join((cell or "").strip() for cell in row)
                    for row in table
                    if row
                )
                if rendered.strip():
                    text += f"\n\n[TABLE]\n{rendered}\n[/TABLE]"

            if len(text) >= MIN_TEXT_LAYER_CHARS:
                pages.append(
                    ExtractedPage(
                        text=text,
                        metadata={
                            "source": path.name,
                            "page": idx,
                            "extraction": "native_pdf",
                            "ocr_confidence": None,
                        },
                    )
                )
            else:
                log.info("ingest.pdf_page_needs_ocr", source=path.name, page=idx)
                ocr_text, conf = _ocr_pdf_page(path, idx)
                if ocr_text.strip():
                    pages.append(
                        ExtractedPage(
                            text=ocr_text,
                            metadata={
                                "source": path.name,
                                "page": idx,
                                "extraction": "ocr",
                                "ocr_confidence": conf,
                            },
                        )
                    )

    return pages


def _ocr_pdf_page(path: Path, page_number: int) -> tuple[str, float | None]:
    """Rasterise a single PDF page and OCR it."""
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(
            str(path), dpi=300, first_page=page_number, last_page=page_number
        )
        if not images:
            return "", None
        return _ocr_image_object(images[0])
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("ingest.ocr_failed", source=path.name,
                    page=page_number, error=str(exc))
        return "", None


def _load_image(path: Path) -> list[ExtractedPage]:
    from PIL import Image

    try:
        with Image.open(path) as img:
            text, conf = _ocr_image_object(img)
    except Exception as exc:
        log.warning("ingest.image_failed", source=path.name, error=str(exc))
        return []

    if not text.strip():
        # Explicit, honest failure state rather than silently indexing noise.
        log.warning("ingest.ocr_empty", source=path.name)
        return []

    return [
        ExtractedPage(
            text=text,
            metadata={
                "source": path.name,
                "page": 1,
                "extraction": "ocr",
                "ocr_confidence": conf,
            },
        )
    ]


def _ocr_image_object(img) -> tuple[str, float | None]:
    """Run Tesseract and compute mean word-level confidence."""
    import pytesseract

    text = pytesseract.image_to_string(img)

    confidence: float | None = None
    try:
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT
        )
        scores = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit()]
        scores = [s for s in scores if s >= 0]
        if scores:
            confidence = round(sum(scores) / len(scores) / 100.0, 3)
    except Exception:
        confidence = None

    return text, confidence
