"""VeriRAG API entrypoint."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.logging_conf import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

app = FastAPI(
    title="VeriRAG",
    description=(
        "A self-correcting document intelligence agent. Grades its own "
        "evidence, re-queries when weak, asks for clarification when "
        "ambiguous, and abstains rather than hallucinating."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

WEB_DIR = Path("web")
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))

    @app.get("/app", include_in_schema=False)
    def app_page() -> FileResponse:
        return FileResponse(str(WEB_DIR / "app.html"))


@app.on_event("startup")
def _startup() -> None:
    log.info("verirag.start", provider=settings.llm_provider,
             qdrant=settings.qdrant_url or "embedded",
             answer_threshold=settings.confidence_answer_threshold,
             abstain_threshold=settings.confidence_abstain_threshold,
             max_attempts=settings.max_requery_attempts)

    # Ephemeral cloud hosts start with an empty index; seed the demo corpus
    # once on boot so the public URL is always live. No-op locally.
    if settings.auto_seed:
        try:
            from app.ingestion.service import ingest_directory
            from app.retrieval.store import get_store
            if get_store().count() == 0:
                results = ingest_directory(Path("data/raw"))
                log.info("verirag.auto_seed", docs=len(results))
        except Exception as exc:
            log.error("verirag.auto_seed_failed", error=str(exc))
