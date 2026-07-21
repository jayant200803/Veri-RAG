"""Ingestion orchestration: load -> clean -> chunk -> embed -> store."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.chunk import chunk_text
from app.ingestion.clean import clean_text
from app.ingestion.loader import load_document
from app.logging_conf import get_logger
from app.retrieval.store import get_store

log = get_logger(__name__)


def ingest_file(path: str | Path) -> dict[str, Any]:
    """Ingest one document. Never raises - reports status instead."""
    path = Path(path)
    result: dict[str, Any] = {
        "source": path.name,
        "status": "ok",
        "pages": 0,
        "chunks": 0,
        "ocr_pages": 0,
        "error": None,
    }

    try:
        pages = load_document(path)
    except Exception as exc:
        log.error("ingest.load_failed", source=path.name, error=str(exc))
        result.update(status="failed", error=str(exc))
        return result

    if not pages:
        # e.g. an unreadable scan. Flag it rather than pretending success.
        result.update(status="flagged_for_review",
                      error="no extractable text (possible low-quality scan)")
        return result

    all_chunks: list[dict[str, Any]] = []
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue
        all_chunks.extend(chunk_text(cleaned, page.metadata))
        if page.metadata.get("extraction") == "ocr":
            result["ocr_pages"] += 1

    stored = get_store().upsert_chunks(all_chunks)

    result.update(pages=len(pages), chunks=stored)
    log.info("ingest.completed", **result)
    return result


def ingest_directory(directory: str | Path) -> list[dict[str, Any]]:
    directory = Path(directory)
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".txt", ".md"}
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in supported)
    return [ingest_file(f) for f in files]
