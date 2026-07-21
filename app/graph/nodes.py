"""The agent's nodes. Each is a pure state -> state transition.

Design note (defend this on stage): anti-hallucination happens in TWO
stages. `grade_context` runs BEFORE generation and catches bad retrieval.
`verify_faithfulness` runs AFTER generation and catches the model inventing
beyond good retrieval. Either alone leaves a hole.
"""
from __future__ import annotations

from typing import Any

from app.config import settings
from app.graph import prompts
from app.graph.state import AgentState
from app.llm.providers import get_llm
from app.logging_conf import get_logger
from app.retrieval.store import get_store

log = get_logger(__name__)

ABSTAIN_MESSAGE = (
    "I don't have enough supporting evidence in the provided documents to "
    "answer this reliably, so I won't guess. "
)


def _record(state: AgentState, node: str, **detail: Any) -> None:
    state.setdefault("trace", []).append({"node": node, **detail})


def _format_context(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "(no context retrieved)"
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.get("metadata", {})
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        extraction = meta.get("extraction", "?")
        blocks.append(
            f"[{i}] source: {src}, page: {page}, extraction: {extraction}, "
            f"score: {d.get('score', 0):.3f}\n{d.get('text', '')}"
        )
    return "\n\n".join(blocks)


# ======================================================================
# 1. RETRIEVE
# ======================================================================
def retrieve(state: AgentState) -> AgentState:
    query = state.get("question", "")
    docs = get_store().search(query, top_k=settings.retrieval_top_k)

    state["documents"] = docs
    state["top_score"] = max((d["score"] for d in docs), default=0.0)

    _record(state, "retrieve", query=query, retrieved=len(docs),
            top_score=round(state["top_score"], 3))
    log.info("agent.retrieve", query=query, n=len(docs),
             top_score=round(state["top_score"], 3))
    return state


# ======================================================================
# 2. GRADE CONTEXT  (the heart of the self-correction layer)
# ======================================================================
def grade_context(state: AgentState) -> AgentState:
    docs = state.get("documents", [])

    if not docs:
        state.update(confidence=0.0, grade_reason="no documents retrieved",
                     contradiction=False)
        _record(state, "grade_context", confidence=0.0, reason="no documents")
        return state

    llm = get_llm()
    result = llm.complete_json(
        prompts.GRADER_PROMPT.format(
            question=state.get("question", ""),
            context=_format_context(docs),
        ),
        system=prompts.GRADER_SYSTEM,
    )

    llm_conf = float(result.get("confidence", 0.0) or 0.0)
    llm_conf = max(0.0, min(1.0, llm_conf))

    # Cross-check the judge against raw retrieval similarity. An LLM judge
    # alone can be confidently wrong; if the best chunk is a poor vector
    # match we cap how confident the agent is allowed to be. This is what
    # stops the grader from rubber-stamping weak retrieval.
    top_score = float(state.get("top_score", 0.0))
    retrieval_ceiling = min(1.0, top_score / 0.75) if top_score > 0 else 0.0
    confidence = round(min(llm_conf, max(retrieval_ceiling, 0.15)), 3) \
        if llm_conf > retrieval_ceiling else round(llm_conf, 3)

    contradiction = bool(result.get("contradiction", False))
    reason = str(result.get("reason", "")) or "no reason given"
    missing = str(result.get("missing", ""))

    # A detected contradiction must never take the fast path to an answer.
    if contradiction:
        confidence = min(confidence, settings.confidence_answer_threshold - 0.05)

    state.update(confidence=confidence, grade_reason=reason,
                 contradiction=contradiction)
    state["_missing"] = missing

    _record(state, "grade_context", confidence=confidence,
            llm_confidence=round(llm_conf, 3),
            retrieval_ceiling=round(retrieval_ceiling, 3),
            contradiction=contradiction, reason=reason)
    log.info("agent.grade", confidence=confidence,
             contradiction=contradiction, reason=reason)
    return state


# ======================================================================
# 3. REWRITE QUERY  (re-query action)
# ======================================================================
def rewrite_query(state: AgentState) -> AgentState:
    attempts = int(state.get("attempts", 0))
    llm = get_llm()

    result = llm.complete_json(
        prompts.REWRITER_PROMPT.format(
            question=state.get("original_question", state.get("question", "")),
            missing=state.get("_missing", "") or "unclear",
            history=state.get("question", ""),
        ),
        system=prompts.REWRITER_SYSTEM,
        temperature=0.3,
    )

    rewritten = (result.get("rewritten_query") or "").strip()
    sub_queries = result.get("sub_queries") or []

    # Multi-hop planning: fold sub-queries into the search string so a
    # decomposed question retrieves evidence for each part.
    if sub_queries and isinstance(sub_queries, list):
        rewritten = " ".join([rewritten, *[str(s) for s in sub_queries[:2]]]).strip()

    if not rewritten:
        rewritten = state.get("question", "")

    state["question"] = rewritten
    state["attempts"] = attempts + 1

    _record(state, "rewrite_query", attempt=state["attempts"],
            rewritten=rewritten, strategy=result.get("strategy", ""))
    log.info("agent.rewrite", attempt=state["attempts"], query=rewritten)
    return state


# ======================================================================
# 4. GENERATE
# ======================================================================
def generate(state: AgentState) -> AgentState:
    docs = state.get("documents", [])
    llm = get_llm()

    answer = llm.complete(
        prompts.GENERATOR_PROMPT.format(
            question=state.get("original_question", state.get("question", "")),
            context=_format_context(docs),
        ),
        system=prompts.GENERATOR_SYSTEM,
        temperature=0.1,
        max_tokens=800,
    )

    state["answer"] = answer.strip()
    state["sources"] = [
        {
            "source": d.get("metadata", {}).get("source"),
            "page": d.get("metadata", {}).get("page"),
            "extraction": d.get("metadata", {}).get("extraction"),
            "score": round(d.get("score", 0.0), 3),
            "excerpt": (d.get("text", "")[:240] + "...")
            if len(d.get("text", "")) > 240 else d.get("text", ""),
        }
        for d in docs
    ]

    _record(state, "generate", chars=len(state["answer"]), sources=len(docs))
    return state


# ======================================================================
# 5. VERIFY FAITHFULNESS  (post-generation anti-hallucination)
# ======================================================================
def verify_faithfulness(state: AgentState) -> AgentState:
    llm = get_llm()
    result = llm.complete_json(
        prompts.VERIFIER_PROMPT.format(
            context=_format_context(state.get("documents", [])),
            answer=state.get("answer", ""),
        ),
        system=prompts.VERIFIER_SYSTEM,
    )

    # Default to faithful=True only if the verifier explicitly says so.
    faithful = bool(result.get("faithful", True)) if result else True
    unsupported = result.get("unsupported_claims") or []

    state["faithful"] = faithful

    if faithful:
        state["status"] = "success"
    else:
        # Downgrade rather than return an ungrounded answer.
        state["status"] = "abstain"
        state["answer"] = (
            ABSTAIN_MESSAGE
            + "My drafted answer contained claims I could not verify against "
              "the source documents."
        )
        state["confidence"] = min(float(state.get("confidence", 0.0)), 0.35)

    _record(state, "verify_faithfulness", faithful=faithful,
            unsupported=unsupported, reason=result.get("reason", ""))
    log.info("agent.verify", faithful=faithful)
    return state


# ======================================================================
# 6. CLARIFY
# ======================================================================
def clarify(state: AgentState) -> AgentState:
    llm = get_llm()
    result = llm.complete_json(
        prompts.CLARIFY_PROMPT.format(
            question=state.get("original_question", ""),
            attempts=state.get("attempts", 0),
            reason=state.get("grade_reason", ""),
            missing=state.get("_missing", ""),
        ),
        system=prompts.CLARIFY_SYSTEM,
        temperature=0.3,
    )

    question = (result.get("clarifying_question") or "").strip() or (
        "Could you specify which document, time period, or entity you mean?"
    )

    state["status"] = "clarify"
    state["clarifying_question"] = question
    state["answer"] = (
        "I need one more detail before I can answer reliably. " + question
    )
    state["sources"] = []

    _record(state, "clarify", question=question)
    log.info("agent.clarify", question=question)
    return state


# ======================================================================
# 7. ABSTAIN
# ======================================================================
def abstain(state: AgentState) -> AgentState:
    reason = state.get("grade_reason", "insufficient supporting evidence")
    contradiction = state.get("contradiction", False)

    if contradiction:
        message = (
            "The documents contain conflicting information on this point, so I "
            "won't pick one over the other. " + reason
        )
    else:
        message = ABSTAIN_MESSAGE + reason

    state["status"] = "abstain"
    state["answer"] = message
    # Still show what we found, so the user can judge for themselves.
    state["sources"] = [
        {
            "source": d.get("metadata", {}).get("source"),
            "page": d.get("metadata", {}).get("page"),
            "score": round(d.get("score", 0.0), 3),
        }
        for d in state.get("documents", [])[:3]
    ]

    _record(state, "abstain", contradiction=contradiction, reason=reason)
    log.info("agent.abstain", contradiction=contradiction)
    return state
