"""Normalisation for OCR and PDF noise.

OCR output is dirty: broken hyphenation, page furniture, stray glyphs,
inconsistent whitespace. Cleaning here directly improves retrieval quality
downstream, which is why the messy-data requirement is an ingestion problem
before it is a retrieval problem.
"""
from __future__ import annotations

import re

# Common OCR confusions in headings/labels we care about
_OCR_FIXES = {
    r"\bl\/O\b": "I/O",
    r"\brn\b": "m",
}

_PAGE_FURNITURE = re.compile(
    r"^\s*(page\s+\d+(\s+of\s+\d+)?|confidential|internal use only)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Re-join words split across line breaks by hyphenation
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Strip page furniture lines
    text = _PAGE_FURNITURE.sub("", text)

    # Remove control/non-printable characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)

    # Collapse runs of whitespace but preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    for pattern, repl in _OCR_FIXES.items():
        text = re.sub(pattern, repl, text)

    return text.strip()
