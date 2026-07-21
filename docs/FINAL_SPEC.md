# VeriRAG — Final Project Specification

**A Self-Correcting Document Intelligence Agent**

| | |
|---|---|
| **Hackathon** | OneInbox AI Internship Hackathon 2026 — Round 2 Build Sprint |
| **Track** | AI Engineer · Problem Statement 1 — Self-Correcting RAG Pipeline |
| **Builder** | Jayant Raj (Individual) |
| **Build deadline** | 24 July 2026 |
| **Offline finale** | 25 July 2026 · OJONE, Bengaluru |
| **Status** | Specification locked — implementation begins |

---

## Part I — The Problem Statement

### 1.1 As issued by OneInbox

> **Problem Statement 1 – Self-Correcting RAG Pipeline**
>
> Build a RAG system over a messy, unstructured document set (mixed PDFs, scanned images needing OCR, inconsistent formatting) that detects when its own retrieved context is insufficient or contradictory — and either re-queries, asks a clarifying question, or explicitly flags low-confidence answers instead of hallucinating. Include a basic evaluation harness (10–15 test questions) showing hallucination rate before versus after introducing the self-correction layer.

### 1.2 The underlying problem, stated plainly

Retrieval-Augmented Generation is the dominant pattern for grounding language models in private data — and it is still not trusted in production. The reason is not retrieval quality. It is that **a conventional RAG system has no concept of not knowing.**

Given a question, it retrieves its top-k chunks whether or not anything relevant exists, hands them to a model trained above all to be helpful, and receives back a fluent, confident, plausibly-cited answer. When the evidence was thin, contradictory, or absent, that answer is a hallucination — and it is indistinguishable in tone and format from a correct one. The system cannot flag it, because the system never evaluated whether it should have answered at all.

This compounds with real-world data. Production document sets are not clean text: they are scanned contracts, photographed pages, multi-column PDFs, and tables mangled by naive extraction. Bad ingestion silently degrades retrieval, and the pipeline has no signal that it is now operating on garbage.

The consequence is that in exactly the domains where RAG is most valuable — legal, medical, financial, compliance, customer support — it is least deployable. **A confident wrong answer is worse than no answer.** Teams cannot ship a system whose failure mode is invisible and whose error rate is unmeasured.

### 1.3 Requirements decomposed

| # | Requirement | Deliverable |
|---|---|---|
| **R1** | Ingest a messy document set | Native PDFs **and** scanned images requiring OCR, with inconsistent formatting |
| **R2** | Working RAG system | Retrieval over embedded chunks + grounded, cited generation |
| **R3** | Detect insufficient context | Numeric, inspectable confidence signal |
| **R4** | Detect contradictory context | Explicit conflict recognition, not silent tie-breaking |
| **R5** | Re-query | Autonomous query reformulation and retry |
| **R6** | Ask a clarifying question | Targeted question returned to the user on ambiguity |
| **R7** | Flag low confidence / abstain | Explicit refusal instead of fabrication |
| **R8** | Evaluation harness | 10–15 automated test questions |
| **R9** | Before/after hallucination rate | Quantified baseline-vs-agent comparison |

R3–R7 constitute the self-correction layer — the substance of the challenge. R9 is the proof that it works.

### 1.4 Judging criteria

Functionality & Correctness · Engineering Quality · System Design · Robustness under Failure · Evaluation Results · Performance & Scalability · Code Quality · Demo Quality · Documentation

### 1.5 Round 2 deliverables

GitHub repository (mandatory) · working deployable demo · project description · technical documentation · **evaluation results (hallucination evaluation)** · demo video · live demonstration and architecture defense before the OneInbox engineering team.

---

## Part II — The Project

### 2.1 What VeriRAG is

**VeriRAG is an autonomous document-intelligence agent that knows the limits of its own evidence.**

Where a conventional RAG pipeline is a straight line — retrieve, generate, return — VeriRAG is an agent operating a bounded decision loop. It treats answering as a privilege it must earn: before it responds, it grades the evidence it retrieved; after it responds, it verifies its own output against the sources it cited. When the evidence does not support an answer, it takes one of three honest actions instead of inventing one:

- **Re-query** — reformulate the search and try again (up to a bounded limit)
- **Clarify** — return a targeted question to the user when intent is ambiguous
- **Abstain** — explicitly decline with a low-confidence flag

The governing thesis: **a system that reliably says "I don't know" is more valuable than one that is usually right.** VeriRAG's contribution is making that property both architectural and *measurable*.

### 2.2 Why this is an agent, not a pipeline

The self-correction loop satisfies every criterion of agentic behavior, which is precisely why it is built as a stateful LangGraph state machine rather than a chain:

| Agentic property | How VeriRAG exhibits it |
|---|---|
| **Autonomous decision-making** | The grader evaluates evidence and selects the execution path without human input |
| **Action selection** | Re-query, clarify, generate, and abstain are distinct actions with distinct consequences |
| **Persistent state** | `confidence`, `attempts`, `documents`, and `reasoning` persist and evolve across iterations |
| **Self-correction** | The agent rewrites its own queries and audits its own generated output |
| **Bounded planning** | Multi-hop questions are decomposed into sub-queries during reformulation |
| **Explainability** | Every transition records *why* it was taken, exposed via tracing |

### 2.3 The four failure modes it solves

1. **Garbage ingestion** — scanned and multi-column documents silently lose content under naive extraction. → Layout-aware parsing plus OCR, with extraction method and confidence retained as chunk metadata.
2. **Blind retrieval** — vector search always returns *something*, regardless of relevance. → An explicit context-sufficiency grader gates whether retrieval was good enough to proceed.
3. **Compulsive answering** — models interpolate over weak or conflicting context. → A bounded correction loop plus post-generation faithfulness verification.
4. **Unmeasured quality** — teams cannot quantify their own hallucination rate. → A built-in harness producing a hard before/after comparison.

### 2.4 How it works, end to end

**Ingestion.** Documents are uploaded via the API and processed asynchronously by a worker, so heavy OCR never blocks request handling. Native PDFs are parsed with layout awareness to preserve tables and headings; scanned images and image-only PDFs are routed through Tesseract OCR. Text is cleaned, normalized, and semantically chunked, with metadata preserved throughout — source file, page number, extraction method, OCR confidence. Chunks are embedded locally and upserted into Qdrant.

**Query — the agent loop.**

1. **Retrieve** — embed the question, pull top-k chunks from Qdrant.
2. **Grade context** — an LLM-as-judge, cross-checked against raw retrieval similarity so it cannot be blindly overconfident, emits a confidence score (0.0–1.0) and a one-line justification assessing whether the evidence is *sufficient* and *internally consistent*.
3. **Decide:**
   - **confidence > 0.7** → generate
   - **0.4 ≤ confidence ≤ 0.7** → if `attempts < 2`, rewrite the query (expand, add synonyms, decompose multi-hop) and retrieve again; otherwise ask a clarifying question
   - **confidence < 0.4** → abstain with an explicit low-confidence flag
4. **Generate** — answer strictly from retrieved context, with inline citations.
5. **Verify faithfulness** — confirm every claim is grounded in the cited sources; ungrounded output is downgraded to an abstention rather than returned.

The loop is **bounded at two re-query attempts** to control latency and cost and to guarantee termination.

**Evaluation.** An identical set of 12 golden questions is run through two systems — a baseline RAG that always answers, and the full VeriRAG agent — and scored on hallucination rate, abstention correctness, answer relevancy, and context precision. The output is the project's headline artifact.

### 2.5 Differentiators

- **Two-stage anti-hallucination** — grading *before* generation catches bad retrieval; faithfulness verification *after* generation catches invention beyond good retrieval.
- **Explicit, bounded agentic control flow** — decision logic lives in an inspectable state machine with declared thresholds and an attempt ceiling, not buried inside a prompt.
- **Contradiction detection** — explicitly required by the statement, and widely skipped.
- **Closed-domain integrity** — deliberately no web-search fallback; the honesty guarantee is scoped to the provided corpus, and reaching outside it would reintroduce ungrounded answers and evade the challenge.
- **Calibrated, not evasive** — abstention *correctness* is reported alongside hallucination rate, proving the agent is not gaming the metric by refusing everything.
- **Zero-cost, provider-agnostic, offline-capable** — runs entirely on free-tier or local models behind a single switchable interface.
- **Reproducible** — one command starts the stack; one command regenerates the headline number.

### 2.6 Success criteria

| Metric | Target |
|---|---|
| Hallucination rate (VeriRAG) | < 10% on the golden set |
| Reduction vs. baseline RAG | ≥ 3× |
| Abstention correctness | 100% on known-unanswerable questions |
| Answerable questions still answered | ≥ 90% (proves no over-abstention) |
| OCR extraction success on scanned documents | > 90% |
| Median query latency (no re-query) | < 5 s |

### 2.7 Technology stack

| Layer | Choice |
|---|---|
| Agent orchestration | Python 3.11, **LangGraph**, LangChain |
| LLM — primary | **Google Gemini 2.0 Flash** (free tier) — generation, grading, verification |
| LLM — fallbacks | **Groq** Llama 3.3 70B (free tier) · **Ollama** Llama 3.1 8B (fully local/offline) |
| LLM abstraction | Single provider interface, switchable via `LLM_PROVIDER` env var |
| Embeddings | **fastembed** — BAAI/bge-small-en-v1.5 (local, zero cost) |
| Vector store | **Qdrant** |
| Document parsing | **unstructured.io**, **Tesseract OCR**, pdfplumber |
| Cache / state | **Redis** |
| Async processing | **Celery** (Redis broker) |
| API | **FastAPI** + Pydantic |
| Evaluation | **Ragas**, DeepEval, custom hallucination scorer |
| Observability | **LangSmith** tracing, structured logging |
| Frontend | **React** + Vite + Tailwind |
| Infrastructure | **Docker Compose** |

**Cost: ₹0.** Embeddings run locally; all LLM calls target free tiers; Ollama provides a fully offline path so the live demo cannot be broken by venue connectivity.

### 2.8 Scope boundaries

**In scope:** OCR-aware ingestion · vector retrieval · the self-correcting agent loop · faithfulness verification · evaluation harness · minimal web UI · tracing and observability · containerized one-command startup · documentation and demo video.

**Out of scope:** authentication and multi-tenancy · Kubernetes orchestration · deployed autoscaling (designed for, not provisioned) · model fine-tuning · any web-search or external-knowledge fallback.

### 2.9 Deliverables

1. Public GitHub repository — clean, typed, documented code
2. `docker-compose up` → fully working system
3. Web UI demonstrating all four agent responses (answer / re-query / clarify / abstain)
4. Evaluation harness with `results.json` and comparison chart
5. README — architecture, setup, results table, design decisions, honest limitations
6. 2–3 minute demo video
7. Live demonstration and architecture defense, 25 July

---

## Part III — The One-Line Summary

> **VeriRAG is an autonomous document-intelligence agent for messy, OCR-heavy corpora that grades its own retrieved evidence, re-queries or asks for clarification when that evidence is weak, refuses to answer when it is absent — and proves the value of that restraint with a measured before-and-after hallucination rate.**

---

*Specification locked. Implementation begins.*
