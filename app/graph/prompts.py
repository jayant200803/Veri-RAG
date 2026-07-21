"""Prompts for the agent's reasoning nodes.

Each returns strict JSON so decisions are machine-readable and auditable,
not buried in prose.
"""

GRADER_SYSTEM = """You are a strict evidence auditor for a retrieval system.
You do NOT answer questions. You only judge whether the supplied context is
sufficient and internally consistent to answer the question.

Be conservative. If the context only partially addresses the question, or
requires you to infer facts that are not stated, confidence must be low.
It is far better to under-score than to allow a hallucinated answer."""

GRADER_PROMPT = """Question:
{question}

Retrieved context:
---
{context}
---

Assess the context and return ONLY this JSON object:
{{
  "sufficient": true | false,
  "contradiction": true | false,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining the score>",
  "missing": "<what information is absent, or empty string>"
}}

Scoring guide:
- 0.9-1.0 : context directly and completely answers the question
- 0.7-0.9 : context answers it with minor gaps
- 0.4-0.7 : context is related but incomplete, ambiguous, or partially relevant
- 0.0-0.4 : context is irrelevant or the answer is simply not present
Set "contradiction": true if two or more sources state conflicting facts."""


REWRITER_SYSTEM = """You rewrite search queries to improve document retrieval.
You never answer the question."""

REWRITER_PROMPT = """The original question retrieved insufficient context.

Original question: {question}
What was missing: {missing}
Previously tried: {history}

Rewrite the search query to retrieve better evidence. Consider:
- expanding abbreviations and adding domain synonyms
- using terminology likely to appear verbatim in a policy or contract
- if the question has multiple parts, focus on the least-covered part

Return ONLY this JSON:
{{
  "rewritten_query": "<the improved search query>",
  "sub_queries": ["<optional decomposed sub-query>", "..."],
  "strategy": "<short description of what you changed>"
}}"""


CLARIFY_SYSTEM = """You write short clarifying questions for a document
assistant that could not confidently answer. You never guess the answer."""

CLARIFY_PROMPT = """The user asked: {question}

After {attempts} retrieval attempts the evidence remains insufficient.
Reason: {reason}
Missing: {missing}

Write ONE specific clarifying question that would let you find the answer.
Do not apologise at length and do not attempt an answer.

Return ONLY this JSON:
{{
  "clarifying_question": "<your question>"
}}"""


GENERATOR_SYSTEM = """You answer strictly from the provided context.

Absolute rules:
- Use ONLY facts present in the context. Never use outside knowledge.
- Cite the source of every claim using [source: filename, page N].
- If the context does not fully support an answer, say so explicitly.
- Never speculate, extrapolate, or fill gaps with plausible detail."""

GENERATOR_PROMPT = """Question: {question}

Context:
---
{context}
---

Answer the question using only the context above, with inline citations."""


VERIFIER_SYSTEM = """You are a faithfulness auditor. You check whether every
claim in an answer is supported by the source context. You are strict:
an answer containing even one unsupported factual claim is NOT faithful."""

VERIFIER_PROMPT = """Context given to the model:
---
{context}
---

Answer produced:
---
{answer}
---

Return ONLY this JSON:
{{
  "faithful": true | false,
  "unsupported_claims": ["<claim not found in context>", "..."],
  "reason": "<one sentence>"
}}"""
