# VeriRAG

**A self-correcting document intelligence agent that knows the limits of its own evidence.**

Built for the **OneInbox AI Internship Hackathon 2026** — AI Engineer track,
Problem Statement 1: *Self-Correcting RAG Pipeline*.

---

## The problem

A conventional RAG pipeline has no concept of *not knowing*. It retrieves its
top-k chunks whether or not anything relevant exists, hands them to a model
trained above all to be helpful, and gets back a fluent, confidently-cited
answer. When the evidence was thin, contradictory, or absent, that answer is a
hallucination — and it is indistinguishable in tone from a correct one.

This is why RAG is least deployable in exactly the domains where it is most
valuable: legal, medical, financial, compliance, support. **A confident wrong
answer is worse than no answer.**

## What VeriRAG does differently

VeriRAG treats answering as a privilege it must earn. Before it responds it
grades the evidence it retrieved; after it responds it verifies its own output
against the sources it cited. When the evidence doesn't support an answer, it
takes one of three honest actions instead of inventing one:

- **Re-query** — reformulate the search and try again (bounded at 2 attempts)
- **Clarify** — ask the user a targeted question when intent is ambiguous
- **Abstain** — explicitly decline, with a low-confidence flag

---

## The agent loop

```
retrieve → grade_context
   confidence ≥ 0.7                → generate → verify_faithfulness → END
   0.4 ≤ confidence < 0.7
        attempts < 2               → rewrite_query → retrieve  (loop)
        attempts == 2              → clarify → END
   confidence < 0.4                → abstain → END

verify_faithfulness
   grounded     → return the answer
   not grounded → downgrade to abstain
```

**Two-stage anti-hallucination.** `grade_context` runs *before* generation and
catches bad retrieval. `verify_faithfulness` runs *after* generation and catches
the model inventing beyond good retrieval. Either alone leaves a hole.

**The grader is cross-checked.** An LLM judge alone can be confidently wrong, so
its score is capped by a ceiling derived from the raw vector similarity of the
best-matching chunk. This is what stops the grader rubber-stamping weak
retrieval.

**The loop is bounded.** Two re-query attempts maximum, plus a
`recursion_limit` safety net. Guarantees termination; caps latency and cost.

---

## Results

> Run `python eval/run_eval.py` to reproduce. Paste your measured numbers here.

| Metric | Baseline RAG | VeriRAG |
|---|---:|---:|
| Hallucination rate | _run eval_ | _run eval_ |
| Correct abstentions | _run eval_ | _run eval_ |
| Answerable questions answered | _run eval_ | _run eval_ |
| Content accuracy | _run eval_ | _run eval_ |

![comparison](eval/results/comparison.png)

**Both** hallucination rate and abstention correctness are reported. Reporting
only the first would let a system game the metric by refusing everything;
reporting both proves the agent is *calibrated*, not merely evasive.

---

## Architecture

```
Web UI ── FastAPI ──┬── /api/ingest ──► background worker
                    │                     loader (pdfplumber │ Tesseract OCR)
                    │                       → clean → chunk → fastembed
                    │                       → Qdrant
                    │
                    └── /api/query  ──► LangGraph agent (7 nodes)
                                          ↕ Qdrant · Redis · LLM
```

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Gemini 2.0 Flash · Groq Llama 3.3 · Ollama (all free / local) |
| Embeddings | fastembed — BAAI/bge-small-en-v1.5 (local) |
| Vector store | Qdrant |
| OCR / parsing | Tesseract, pdfplumber, pdf2image |
| Async | Celery + Redis |
| API | FastAPI + Pydantic |
| Infra | Docker Compose |

**Total API cost: ₹0.** Embeddings run locally; all LLM calls target free
tiers. Ollama provides a fully offline path so the live demo cannot be broken
by venue connectivity.

---

## Quickstart

```bash
# 1. Free Gemini key: https://aistudio.google.com/apikey
cp .env.example .env          # set GEMINI_API_KEY

# 2. Generate the messy demo corpus
make corpus

# 3. Start the stack
make up                       # qdrant + redis + api + worker

# 4. Ingest
make seed

# 5. Open http://localhost:8000

# 6. Reproduce the headline number
make eval
```

Tests need no key and no services:

```bash
make test
```

---

## The demo corpus is engineered

`scripts/generate_corpus.py` builds three documents with deliberately planted
properties, so every agent behaviour is demonstrable:

| Planted property | Proves |
|---|---|
| Clean native-text policy PDF | Normal answering |
| Deployment-freeze details **only inside a scanned image** | The OCR path genuinely works |
| Notice period: **60 days** in the handbook vs **30 days** in the addendum | Contradiction detection |
| Contractor leave left underspecified | Clarification behaviour |
| No salary / address / insurance data anywhere | Honest abstention |

Try these four in the UI, in order:

1. *"How many casual leave days do full-time employees get?"* → answers
2. *"When does the production deployment freeze apply?"* → answers **from the image**
3. *"What is the notice period when resigning?"* → flags the **contradiction**
4. *"What is the CEO's home address?"* → **abstains**

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Status, indexed chunk count, active thresholds |
| `POST /api/ingest` | Upload documents (processed off the request path) |
| `GET /api/status/{task_id}` | Ingestion progress |
| `POST /api/ingest/seed` | Ingest everything in `data/raw` |
| `POST /api/query` | Run the agent |
| `DELETE /api/index` | Reset the vector store |

`POST /api/query` returns the answer plus `status`, `confidence`, `attempts`,
`contradiction`, `sources`, and the full **decision trace** — every node the
agent executed and why.

---

## Design decisions

**No web-search fallback.** The honesty guarantee is scoped to the provided
corpus. Reaching outside it would reintroduce ungrounded answers and evade the
point of the challenge.

**Fail safe means abstain.** Missing confidence, zero documents retrieved,
Qdrant unreachable, verifier unsure — all resolve to abstention. The system
never answers by default.

**Routing logic is dependency-free.** `app/graph/routing.py` imports nothing but
config, so the most safety-critical decision in the system is pure and
trivially unit-testable.

**Structure-aware chunking.** Tables are kept atomic and paragraph boundaries
respected, because fragmented context reads as *insufficient* to the grader and
needlessly triggers the correction loop.

---

## Known limitations

- Very low-resolution scans still defeat OCR; those documents are marked
  `flagged_for_review` rather than silently indexed as noise.
- The self-correction loop increases tail latency — a re-query roughly doubles
  time-to-answer. This is the deliberate trade for not hallucinating.
- Grader thresholds are tuned on a 12-question golden set; a production
  deployment would want a substantially larger calibration set.
- Multi-hop questions spanning more than two documents remain the weakest case.

---

## Project layout

```
app/
  config.py            settings + agent thresholds
  graph/
    routing.py         the decision function (pure, tested)
    nodes.py           the seven agent nodes
    build.py           LangGraph wiring + run_agent()
    prompts.py         all LLM prompts
    state.py           AgentState
  llm/                 Gemini / Groq / Ollama behind one interface
  ingestion/           loader (OCR) → clean → chunk → service → celery
  retrieval/           fastembed + Qdrant
  api/                 routes + schemas
eval/
  run_eval.py          baseline vs agent — the headline experiment
  scorer.py            hallucination + abstention scoring
  baseline_rag.py      the control group
data/golden/qa.yaml    the 12 golden questions
scripts/               corpus generator
web/index.html         UI with confidence bars and decision trace
```

---

Built by **Jayant Raj** for the OneInbox AI Internship Hackathon 2026.
