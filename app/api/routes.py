from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.api.schemas import (HealthResponse, IngestResponse, IngestResult,
                             QueryRequest, QueryResponse)
from app.config import settings
from app.graph.build import run_agent
from app.ingestion.service import ingest_directory, ingest_file
from app.logging_conf import get_logger
from app.retrieval.store import get_store

log = get_logger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-process job registry. Redis-backed Celery is used when a worker is
# running; this keeps the app usable standalone for the demo.
_JOBS: dict[str, dict] = {}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_store()
    return HealthResponse(
        status="ok",
        qdrant=store.health(),
        chunks_indexed=store.count(),
        llm_provider=settings.llm_provider,
        thresholds={
            "answer": settings.confidence_answer_threshold,
            "abstain": settings.confidence_abstain_threshold,
            "max_requery_attempts": settings.max_requery_attempts,
            "top_k": settings.retrieval_top_k,
        },
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(background: BackgroundTasks,
                 files: list[UploadFile] = File(...)) -> IngestResponse:
    """Upload documents. Processing happens off the request path."""
    saved: list[str] = []
    for f in files:
        dest = UPLOAD_DIR / Path(f.filename or f"upload-{uuid.uuid4().hex}").name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest.name)

    task_id = uuid.uuid4().hex
    _JOBS[task_id] = {"status": "queued", "results": []}

    def _run() -> None:
        _JOBS[task_id]["status"] = "running"
        results = [ingest_file(UPLOAD_DIR / name) for name in saved]
        _JOBS[task_id].update(status="completed", results=results)

    background.add_task(_run)

    return IngestResponse(task_id=task_id, queued=True, files=saved,
                          detail="Ingestion started in background.")


@router.get("/status/{task_id}")
def status(task_id: str) -> dict:
    job = _JOBS.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown task_id")
    return job


@router.post("/ingest/seed", response_model=list[IngestResult])
def seed() -> list[IngestResult]:
    """Ingest everything already sitting in data/raw. Used for demo setup."""
    results = ingest_directory(UPLOAD_DIR)
    return [IngestResult(**r) for r in results]


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run the self-correcting agent."""
    started = time.perf_counter()
    try:
        result = run_agent(req.query)
    except Exception as exc:
        log.error("query.failed", error=str(exc))
        # Fail honestly rather than 500-ing with a hallucinated answer.
        raise HTTPException(status_code=503,
                            detail=f"Agent unavailable: {exc}") from exc

    if not req.include_trace:
        result["trace"] = []

    result["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return QueryResponse(**result)


@router.delete("/index")
def reset_index() -> dict:
    get_store().reset()
    return {"status": "reset"}
