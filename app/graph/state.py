"""Agent state.

State is what makes this an agent rather than a chain: `attempts`,
`confidence` and `trace` persist across loop iterations and drive the
next decision.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

Status = Literal["success", "clarify", "abstain", "pending"]


class AgentState(TypedDict, total=False):
    # Inputs
    question: str
    original_question: str

    # Retrieval
    documents: list[dict[str, Any]]
    top_score: float

    # Grading
    confidence: float
    grade_reason: str
    contradiction: bool

    # Loop control
    attempts: int

    # Output
    answer: str
    sources: list[dict[str, Any]]
    status: Status
    clarifying_question: str
    faithful: bool

    # Explainability - every decision the agent made, in order
    trace: list[dict[str, Any]]
