"""The agent's decision function, deliberately isolated.

This module has NO dependency on LangGraph, the LLM, or the vector store.
The most safety-critical logic in the system - deciding whether the agent
is allowed to answer - is therefore pure, trivially unit-testable, and
readable in one screen. That isolation is intentional.
"""
from __future__ import annotations

from typing import Any

from app.config import settings

GENERATE = "generate"
REWRITE = "rewrite_query"
CLARIFY = "clarify"
ABSTAIN = "abstain"


def route_after_grading(state: dict[str, Any]) -> str:
    """Decide what the agent does next, given its own confidence.

        confidence >= answer_threshold   -> generate
        confidence <  abstain_threshold  -> abstain
        otherwise (the middle band):
            attempts < max               -> rewrite the query and retry
            attempts == max              -> stop looping, ask the user

    Fails safe: if confidence is missing entirely we abstain rather than
    answer. Never guess by default.
    """
    confidence = float(state.get("confidence", 0.0) or 0.0)
    attempts = int(state.get("attempts", 0) or 0)

    if confidence >= settings.confidence_answer_threshold:
        return GENERATE

    if confidence < settings.confidence_abstain_threshold:
        return ABSTAIN

    if attempts < settings.max_requery_attempts:
        return REWRITE

    return CLARIFY
