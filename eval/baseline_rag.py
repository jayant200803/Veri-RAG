"""Baseline RAG - the control group.

Retrieve top-k, generate, return. No grading, no verification, no
abstention. This is what almost every RAG tutorial produces, and it is
the system VeriRAG is measured against.
"""
from __future__ import annotations

import time
from typing import Any

from app.config import settings
from app.retrieval.store import get_store
from app.llm.providers import get_llm

BASELINE_SYSTEM = (
    "You are a helpful assistant answering questions about company documents. "
    "Use the provided context to answer the user's question."
)

BASELINE_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


def run_baseline(question: str) -> dict[str, Any]:
    started = time.perf_counter()

    docs = get_store().search(question, top_k=settings.retrieval_top_k)
    context = "\n\n".join(
        f"[{i}] ({d.get('metadata', {}).get('source')}, "
        f"p{d.get('metadata', {}).get('page')})\n{d['text']}"
        for i, d in enumerate(docs, start=1)
    ) or "(no context)"

    answer = get_llm().complete(
        BASELINE_PROMPT.format(context=context, question=question),
        system=BASELINE_SYSTEM,
        temperature=0.1,
        max_tokens=800,
    )

    return {
        "question": question,
        "answer": answer.strip(),
        # The defining property of the baseline: it always claims success.
        "status": "success",
        "confidence": None,
        "attempts": 0,
        "contradiction": False,
        "sources": [
            {
                "source": d.get("metadata", {}).get("source"),
                "page": d.get("metadata", {}).get("page"),
                "score": round(d.get("score", 0.0), 3),
            }
            for d in docs
        ],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
