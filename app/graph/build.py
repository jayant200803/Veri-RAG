"""LangGraph wiring - the agent's control flow.

    retrieve -> grade_context
        confidence > 0.7            -> generate -> verify -> END
        0.4 <= confidence <= 0.7
             attempts < MAX         -> rewrite_query -> retrieve   (loop)
             attempts == MAX        -> clarify -> END
        confidence < 0.4            -> abstain -> END

The loop is BOUNDED at MAX_REQUERY_ATTEMPTS (default 2). This guarantees
termination, caps latency and cost, and is a deliberate engineering
decision rather than an accident.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, StateGraph

from app.graph import nodes
from app.graph.routing import route_after_grading
from app.graph.state import AgentState
from app.logging_conf import get_logger

log = get_logger(__name__)


@lru_cache
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_context", nodes.grade_context)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify_faithfulness", nodes.verify_faithfulness)
    graph.add_node("clarify", nodes.clarify)
    graph.add_node("abstain", nodes.abstain)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_context")

    graph.add_conditional_edges(
        "grade_context",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "clarify": "clarify",
            "abstain": "abstain",
        },
    )

    # The loop: a rewritten query goes back through retrieval.
    graph.add_edge("rewrite_query", "retrieve")

    graph.add_edge("generate", "verify_faithfulness")
    graph.add_edge("verify_faithfulness", END)
    graph.add_edge("clarify", END)
    graph.add_edge("abstain", END)

    return graph.compile()


def run_agent(question: str) -> dict[str, Any]:
    """Execute the agent for one question and return a serialisable result."""
    agent = build_agent()

    initial: AgentState = {
        "question": question,
        "original_question": question,
        "attempts": 0,
        "documents": [],
        "trace": [],
        "status": "pending",
    }

    # recursion_limit guards against pathological loops even if thresholds
    # are misconfigured - belt and braces on top of the attempt counter.
    final = agent.invoke(initial, config={"recursion_limit": 25})

    return {
        "question": final.get("original_question", question),
        "answer": final.get("answer", ""),
        "status": final.get("status", "abstain"),
        "confidence": float(final.get("confidence", 0.0)),
        "contradiction": bool(final.get("contradiction", False)),
        "attempts": int(final.get("attempts", 0)),
        "faithful": final.get("faithful"),
        "clarifying_question": final.get("clarifying_question"),
        "sources": final.get("sources", []),
        "reasoning": final.get("grade_reason", ""),
        "trace": final.get("trace", []),
    }
