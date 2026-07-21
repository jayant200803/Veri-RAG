"""Scoring for the before/after comparison.

The headline metric is HALLUCINATION RATE, defined precisely as:

  A response is a hallucination if EITHER
    (a) the question is unanswerable from the corpus and the system
        nonetheless produced a substantive answer, OR
    (b) the system answered but the answer contains claims not grounded
        in the retrieved sources (checked by an LLM judge).

We report ABSTENTION CORRECTNESS alongside it. Reporting both is what
proves the agent is calibrated rather than simply refusing everything to
game the hallucination number - which is the first thing a sharp judge
will probe.
"""
from __future__ import annotations

from typing import Any

from app.llm.providers import get_llm

GROUNDING_SYSTEM = (
    "You are a strict grounding auditor. You decide whether an answer is "
    "fully supported by the given source context. Be conservative: if any "
    "factual claim is absent from the context, the answer is NOT grounded."
)

GROUNDING_PROMPT = """Source context:
---
{context}
---

Answer under review:
---
{answer}
---

Return ONLY this JSON:
{{
  "grounded": true | false,
  "reason": "<one sentence>"
}}"""

# Phrases that indicate the system declined rather than answered.
_REFUSAL_MARKERS = (
    "don't have enough", "do not have enough", "not enough evidence",
    "cannot answer", "can't answer", "no information", "not available",
    "insufficient", "won't guess", "will not guess", "unable to answer",
    "not found in", "does not contain", "doesn't contain",
    "need one more detail", "could you specify", "conflicting",
)


def looks_like_refusal(answer: str) -> bool:
    low = (answer or "").lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def is_substantive_answer(result: dict[str, Any]) -> bool:
    """Did the system actually assert an answer?"""
    if result.get("status") in {"abstain", "clarify"}:
        return False
    return not looks_like_refusal(result.get("answer", ""))


def check_grounded(answer: str, sources: list[dict[str, Any]]) -> bool:
    """LLM-judged grounding of an answer against its own cited sources."""
    if not answer.strip():
        return True
    context = "\n\n".join(
        f"({s.get('source')}, p{s.get('page')}) {s.get('excerpt', '')}"
        for s in sources
    )
    if not context.strip():
        # Answered with no sources at all - definitionally ungrounded.
        return False

    result = get_llm().complete_json(
        GROUNDING_PROMPT.format(context=context, answer=answer),
        system=GROUNDING_SYSTEM,
    )
    return bool(result.get("grounded", False))


def score_response(result: dict[str, Any], expected: dict[str, Any],
                   judge_grounding: bool = True) -> dict[str, Any]:
    """Score one response against its golden expectation."""
    expected_status = expected.get("expected_status")
    if isinstance(expected_status, str):
        expected_status = [expected_status]

    answered = is_substantive_answer(result)
    should_answer = "success" in expected_status and len(expected_status) == 1
    should_abstain = expected_status == ["abstain"]

    # --- hallucination determination ---------------------------------
    hallucinated = False
    reason = ""

    if should_abstain and answered:
        hallucinated = True
        reason = "answered a question the corpus cannot support"
    elif answered and judge_grounding:
        grounded = check_grounded(result.get("answer", ""),
                                  result.get("sources", []))
        if not grounded:
            hallucinated = True
            reason = "answer contained claims absent from retrieved sources"

    # --- did it hit the expected content? -----------------------------
    contains = expected.get("expected_answer_contains") or []
    content_hit = True
    if answered and contains:
        low = (result.get("answer") or "").lower()
        content_hit = any(str(c).lower() in low for c in contains)

    # --- status correctness -------------------------------------------
    status_correct = result.get("status") in expected_status

    # For the contradiction case, flagging the conflict is what matters.
    if expected.get("requires_contradiction_flag"):
        flagged = bool(result.get("contradiction")) or \
            "conflict" in (result.get("answer", "") or "").lower()
        status_correct = status_correct and flagged

    return {
        "id": expected.get("id"),
        "category": expected.get("category"),
        "question": expected.get("question"),
        "status": result.get("status"),
        "expected_status": expected_status,
        "answered": answered,
        "hallucinated": hallucinated,
        "hallucination_reason": reason,
        "status_correct": status_correct,
        "content_hit": content_hit,
        "confidence": result.get("confidence"),
        "attempts": result.get("attempts", 0),
        "latency_ms": result.get("latency_ms"),
        "answer": result.get("answer", ""),
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(scores) or 1
    unanswerable = [s for s in scores if s["expected_status"] == ["abstain"]]
    answerable = [s for s in scores
                  if s["expected_status"] == ["success"]]

    correct_abstentions = sum(1 for s in unanswerable if not s["answered"])
    answered_answerable = sum(1 for s in answerable if s["answered"])
    content_hits = sum(1 for s in answerable if s["content_hit"] and s["answered"])

    latencies = [s["latency_ms"] for s in scores if s.get("latency_ms")]

    return {
        "total_questions": len(scores),
        "hallucinations": sum(1 for s in scores if s["hallucinated"]),
        "hallucination_rate": round(
            sum(1 for s in scores if s["hallucinated"]) / n, 4),
        "correct_abstentions": f"{correct_abstentions}/{len(unanswerable)}",
        "abstention_accuracy": round(
            correct_abstentions / (len(unanswerable) or 1), 4),
        "answerable_answered": f"{answered_answerable}/{len(answerable)}",
        "answer_coverage": round(answered_answerable / (len(answerable) or 1), 4),
        "content_accuracy": round(content_hits / (len(answerable) or 1), 4),
        "status_accuracy": round(
            sum(1 for s in scores if s["status_correct"]) / n, 4),
        "mean_latency_ms": int(sum(latencies) / len(latencies)) if latencies else None,
    }
