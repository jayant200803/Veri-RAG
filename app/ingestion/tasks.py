"""Celery tasks - keeps heavy OCR off the request path.

Broker is Redis (rather than RabbitMQ) to keep the container count low;
the decoupling property that matters for scalability is identical.
"""
from __future__ import annotations

from celery import Celery

from app.config import settings
from app.ingestion.service import ingest_file

celery_app = Celery("verirag", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=900,
)


@celery_app.task(name="verirag.ingest_document", bind=True)
def ingest_document_task(self, path: str) -> dict:
    return ingest_file(path)
