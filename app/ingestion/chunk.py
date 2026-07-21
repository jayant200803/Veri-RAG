"""Structure-aware chunking.

We split on paragraph boundaries first and only hard-split when a single
block exceeds the budget. This keeps clauses, table blocks, and policy
sections intact, which matters because the grader downstream judges whether
retrieved context is *sufficient* - fragmented context reads as insufficient.
"""
from __future__ import annotations

import re
from typing import Any

from app.config import settings


def chunk_text(text: str, metadata: dict[str, Any],
               chunk_size: int | None = None,
               overlap: int | None = None) -> list[dict[str, Any]]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    if not text.strip():
        return []

    blocks = _split_blocks(text)
    chunks: list[str] = []
    current = ""

    for block in blocks:
        if len(block) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(_hard_split(block, chunk_size, overlap))
            continue

        if len(current) + len(block) + 2 <= chunk_size:
            current = f"{current}\n\n{block}" if current else block
        else:
            if current.strip():
                chunks.append(current.strip())
            current = block

    if current.strip():
        chunks.append(current.strip())

    return [
        {
            "text": c,
            "metadata": {**metadata, "chunk_index": i, "char_len": len(c)},
        }
        for i, c in enumerate(chunks)
        if c.strip()
    ]


def _split_blocks(text: str) -> list[str]:
    """Keep [TABLE] blocks atomic; otherwise split on blank lines."""
    parts: list[str] = []
    for segment in re.split(r"(\[TABLE\].*?\[/TABLE\])", text, flags=re.DOTALL):
        if not segment.strip():
            continue
        if segment.startswith("[TABLE]"):
            parts.append(segment.strip())
        else:
            parts.extend(p.strip() for p in re.split(r"\n\s*\n", segment) if p.strip())
    return parts


def _hard_split(block: str, size: int, overlap: int) -> list[str]:
    out: list[str] = []
    start = 0
    while start < len(block):
        end = start + size
        # Prefer to break at a sentence boundary near the limit
        window = block[start:end]
        pivot = max(window.rfind(". "), window.rfind("\n"))
        if pivot > size * 0.6 and end < len(block):
            end = start + pivot + 1
        out.append(block[start:end].strip())
        start = max(end - overlap, end) if overlap >= size else end - overlap
    return [c for c in out if c]
