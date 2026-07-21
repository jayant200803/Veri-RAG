"""Tests for the agent's decision logic.

These run without any LLM or vector store - the routing function is pure,
which is deliberate: the most safety-critical logic in the system is the
part that is trivially testable.
"""
from app.config import settings
from app.graph.routing import route_after_grading


def test_high_confidence_generates():
    assert route_after_grading({"confidence": 0.92, "attempts": 0}) == "generate"


def test_low_confidence_abstains():
    assert route_after_grading({"confidence": 0.12, "attempts": 0}) == "abstain"


def test_middle_band_rewrites_first():
    assert route_after_grading({"confidence": 0.55, "attempts": 0}) == "rewrite_query"


def test_loop_is_bounded():
    """After MAX attempts the agent must stop looping and ask the user."""
    state = {"confidence": 0.55, "attempts": settings.max_requery_attempts}
    assert route_after_grading(state) == "clarify"


def test_threshold_boundaries():
    assert route_after_grading(
        {"confidence": settings.confidence_answer_threshold, "attempts": 0}
    ) == "generate"
    assert route_after_grading(
        {"confidence": settings.confidence_abstain_threshold - 0.001, "attempts": 0}
    ) == "abstain"


def test_missing_confidence_defaults_to_abstain():
    """Fail safe: absent a score, never answer."""
    assert route_after_grading({}) == "abstain"
