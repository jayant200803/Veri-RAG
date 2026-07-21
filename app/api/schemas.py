from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    include_trace: bool = True


class SourceRef(BaseModel):
    source: str | None = None
    page: int | str | None = None
    extraction: str | None = None
    score: float | None = None
    excerpt: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    status: Literal["success", "clarify", "abstain", "pending"]
    confidence: float
    contradiction: bool = False
    attempts: int = 0
    faithful: bool | None = None
    clarifying_question: str | None = None
    sources: list[SourceRef] = []
    reasoning: str = ""
    trace: list[dict[str, Any]] = []
    latency_ms: int | None = None


class IngestResponse(BaseModel):
    task_id: str | None = None
    queued: bool
    files: list[str] = []
    detail: str = ""


class IngestResult(BaseModel):
    source: str
    status: str
    pages: int = 0
    chunks: int = 0
    ocr_pages: int = 0
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    chunks_indexed: int
    llm_provider: str
    thresholds: dict[str, Any]
