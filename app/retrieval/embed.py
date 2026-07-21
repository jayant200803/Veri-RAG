"""Local embeddings via fastembed. No API, no cost, no rate limit.

Running embeddings locally is what lets us re-run the full evaluation
harness as many times as we like while tuning thresholds.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)


@lru_cache
def _model():
    from fastembed import TextEmbedding

    log.info("embed.loading", model=settings.embed_model)
    return TextEmbedding(model_name=settings.embed_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [list(map(float, v)) for v in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
