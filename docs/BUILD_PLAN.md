# VeriRAG — Round 2 Build Plan

**Self-Correcting RAG Pipeline · OneInbox AI Internship Hackathon 2026**
Build Sprint: now → 24 July · Offline finale: 25 July, OJONE Bengaluru
Author: Jayant Raj

---

## 0. How you win this (read this first)

You are competing with ~200 people building the *same* system. Nobody wins on "I built a RAG chatbot" — everyone will have that. You win on the three things the problem statement and rubric secretly care about most:

1. **The self-correction layer actually works and is visible.** Most people will bolt on a weak "confidence check" that never triggers. Yours must *demonstrably* re-query, clarify, and abstain — live, on camera.
2. **You have real before/after numbers.** The PS says: "showing hallucination rate before versus after introducing the self-correction layer." This single chart is your headline. Almost everyone will hand-wave it. You will show a table: *baseline RAG = 40% hallucination → VeriRAG = 6%.* That wins the "Evaluation Results" criterion outright.
3. **You break it on purpose and it holds.** The rubric has "Robustness under Failure." Feed it a scanned garbage image, an unanswerable question, and a contradictory-sources question — and show it abstaining honestly instead of lying.

Everything below is engineered to make those three things true, and to be *achievable in 6 days by one person*. Scope is deliberately ruthless — build the differentiators first, polish later.

**The one-sentence pitch to memorize:** *"VeriRAG doesn't just retrieve and generate — it grades its own evidence, and when the documents don't support an answer, it re-queries, asks you a clarifying question, or tells you it doesn't know. I measured it: hallucinations dropped from X% to Y%."*

---

## 1. Scope discipline — MUST / SHOULD / WON'T

| Priority | Feature | Why |
|---|---|---|
| **MUST** | Ingestion of native PDF + scanned image (OCR) | Core PS requirement; proves "messy data" handling |
| **MUST** | Vector retrieval (Qdrant) + generation (GPT-4o or open model) | Baseline RAG |
| **MUST** | Context-sufficiency grader with numeric confidence + thresholds | The differentiator |
| **MUST** | Bounded self-correction loop: re-query → clarify → abstain | The differentiator |
| **MUST** | Faithfulness verifier (answer grounded in sources) | Anti-hallucination |
| **MUST** | Eval harness: 12 golden Qs, baseline vs. self-correcting, hallucination-rate table | The headline result |
| **SHOULD** | Simple web UI showing confidence + sources + status badge | Demo quality |
| **SHOULD** | LangSmith tracing to *show* why the loop branched | Explainability, huge in the defense |
| **SHOULD** | Docker-compose one-command startup | Engineering quality |
| **WON'T (for now)** | Auth/JWT, K8s, multi-tenant, streaming polish | Not graded here; time sink |
| **WON'T** | Web-search fallback | Violates closed-domain integrity (defend this deliberately) |

If you fall behind, cut SHOULDs, never MUSTs. A working MUST-only build with real eval numbers beats a half-working feature-rich one.

---

## 2. Tech stack (locked)

The org said "no restrictions." Stick close to your submitted architecture — it's defensible and you already understand it.

- **Orchestration:** Python 3.11, **LangGraph** (the self-correction state machine), LangChain for utilities.
- **LLM (100% free):** **Google Gemini 2.0 Flash** for generation, grading, and verification — generous free tier, no card needed, and you already know it from DocQA. Wrapped behind a provider interface so `LLM_PROVIDER=gemini|groq|ollama` swaps the backend with no code change.
  - **Backup:** **Groq** (Llama 3.3 70B) — free tier, very fast, good if you hit Gemini rate limits.
  - **Offline insurance:** **Ollama** (Llama 3.1 8B) running locally — zero network dependency, so a dead venue WiFi on 25 July can't kill your demo.
- **Embeddings:** `fastembed` (BAAI/bge-small-en-v1.5) — fast, local, free, no API cost on ingestion.
- **Vector store:** **Qdrant** (docker).
- **Ingestion:** `unstructured.io` for layout/table parsing, **Tesseract OCR** for scanned images.
- **Cache/state:** **Redis**.
- **Async:** **Celery + RabbitMQ** (or Redis broker to save a container — see note in Day 2).
- **API:** **FastAPI**.
- **Eval:** **Ragas** (faithfulness, answer relevancy, context precision) + a custom hallucination scorer.
- **Observability:** **LangSmith** (tracing) + optional Prometheus/Grafana.
- **Frontend:** minimal **React + Vite + Tailwind** (or even a single-page HTML if time is tight).

> **Zero-cost guarantee:** embeddings run locally via fastembed (no API), and all LLM calls hit Gemini's free tier. Total spend for the sprint: **₹0**. This also means you can re-run the full eval harness as many times as you like while tuning thresholds — a real advantage over teams rate-limited by a paid budget.

---

## 3. Repository structure

```
verirag/
├── docker-compose.yml          # qdrant, redis, rabbitmq, api, worker
├── .env.example                # OPENAI_API_KEY, LANGSMITH_API_KEY, etc.
├── README.md                   # architecture, setup, results table (graded!)
├── pyproject.toml
├── data/
│   ├── raw/                    # messy source docs (native + scanned)
│   └── golden/qa.yaml          # 12 golden test questions + expected answers
├── app/
│   ├── main.py                 # FastAPI: /ingest, /status, /query
│   ├── config.py
│   ├── ingestion/
│   │   ├── loader.py           # unstructured + tesseract OCR
│   │   ├── clean.py            # normalization, dedup
│   │   ├── chunk.py            # semantic chunking + metadata (source, page)
│   │   └── tasks.py            # celery tasks
│   ├── retrieval/
│   │   ├── store.py            # qdrant client, upsert, search
│   │   └── embed.py            # fastembed wrapper
│   ├── graph/                  # ⭐ the differentiator
│   │   ├── state.py            # GraphState TypedDict
│   │   ├── nodes.py            # retrieve, grade, rewrite, clarify, generate, verify
│   │   ├── build.py            # LangGraph wiring + conditional edges
│   │   └── prompts.py          # grader, rewriter, verifier prompts
│   └── api/schemas.py
├── eval/
│   ├── baseline_rag.py         # plain retrieve→generate (no self-correction)
│   ├── run_eval.py             # runs both pipelines over golden set
│   ├── scorer.py               # hallucination + abstention correctness
│   └── results/                # generated: results.json, chart.png
└── web/                        # minimal React UI
```

The `graph/` directory is where the marks are. Spend your best hours here.

---

## 4. The core: how the self-correction graph actually works

This is the part judges will interrogate. Build it as an explicit LangGraph state machine, not hidden inside a prompt.

### 4.1 State

```python
class GraphState(TypedDict):
    question: str
    original_question: str
    documents: list          # retrieved chunks
    confidence: float        # 0.0–1.0 from the grader
    attempts: int            # re-query counter (bounded)
    answer: str
    sources: list
    status: str              # "success" | "clarify" | "abstain"
    reasoning: str           # why the loop branched (for tracing/demo)
```

### 4.2 Nodes

- **retrieve** — embed question, search Qdrant top-k (k=6), attach chunks + metadata.
- **grade_context** — LLM-as-judge scores whether the retrieved chunks are *sufficient and non-contradictory* to answer. Returns a float 0–1 plus a one-line reason. (Cross-check with a cheap heuristic: mean retrieval score, to avoid the grader being over-confident.)
- **rewrite_query** — reformulate/expand the query (add synonyms, decompose multi-part questions), increment `attempts`.
- **clarify** — produce a targeted clarifying question back to the user ("Did you mean the 2023 or 2024 policy?").
- **generate** — answer strictly from retrieved context, with an explicit "use only the provided sources" instruction and inline citations.
- **verify_faithfulness** — post-generation check: is every claim grounded in the sources? If not, downgrade to abstain or re-generate once.

### 4.3 Control flow (conditional edges)

```
retrieve → grade_context
  ├─ confidence > 0.7            → generate → verify_faithfulness → END
  ├─ 0.4 ≤ confidence ≤ 0.7
  │     ├─ attempts < 2          → rewrite_query → retrieve   (loop)
  │     └─ attempts == 2         → clarify → END
  └─ confidence < 0.4            → abstain → END

verify_faithfulness:
  ├─ grounded                    → return answer (status=success)
  └─ not grounded                → abstain (status=abstain)
```

**Two design decisions to defend out loud:**
1. **Bounded loop (max 2 re-queries).** Prevents infinite spin and runaway cost. State the number and why.
2. **Two-stage anti-hallucination** (grade *before* generation + verify *after*). Grading catches bad retrieval; verification catches the model inventing beyond good retrieval. Belt and suspenders — this is the sentence that impresses engineers.

---

## 5. The eval harness — your headline result

This is worth more than any feature. Build it early (Day 4), not the night before.

### 5.1 Golden set (`data/golden/qa.yaml`)

12 questions, deliberately spanning four categories so you can prove each branch:

| # | Category | Purpose | Expected status |
|---|---|---|---|
| 1–5 | Answerable, clean | Normal accuracy | success |
| 6–8 | Answerable only in scanned/OCR doc | Proves messy-data handling | success |
| 9–10 | Not in the corpus | Proves honest abstention | abstain |
| 11 | Contradictory sources | Proves contradiction detection | clarify/abstain |
| 12 | Ambiguous/underspecified | Proves clarification | clarify |

### 5.2 Two pipelines, same questions

Run `baseline_rag.py` (plain retrieve→generate, always answers) and the full VeriRAG graph over the identical 12 questions.

### 5.3 Metrics

- **Hallucination rate** = fraction of answers containing a claim not supported by sources OR a confident answer to an unanswerable question. Score with Ragas `faithfulness` + a manual/LLM check on the abstain cases.
- **Abstention correctness** = did it abstain exactly when it should have?
- **Answer relevancy / context precision** (Ragas) for the answerable ones.

### 5.4 The money table (put this in README + final slide)

```
                    Baseline RAG    VeriRAG (self-correcting)
Hallucination rate      ~42%              ~6%
Correct abstentions      0/4              4/4
Answer relevancy        0.71             0.88
```

*(Your real numbers will differ — run it and report honestly. Even 42%→12% is a compelling story.)* Generate a bar chart (`chart.png`) from `results.json` and embed it in the README.

---

## 6. Day-by-day plan (18 → 24 July)

You have ~7 days. This assumes a few focused hours/day, more on weekends.

### Day 1 (Fri 18) — Skeleton + infra
- Init repo, `pyproject.toml`, `.env.example`, `docker-compose.yml` (Qdrant + Redis + RabbitMQ).
- FastAPI app boots; `/health` works. Qdrant reachable from Python.
- Commit. **Goal: `docker-compose up` brings everything online.**

### Day 2 (Sat 19) — Ingestion pipeline
- `loader.py`: unstructured for PDFs, Tesseract for images. Test on 1 native PDF + 1 scanned image.
- `chunk.py` with metadata (source, page). `embed.py` with fastembed. Upsert into Qdrant.
- Wire `/ingest` → Celery task → `/status`. Ingest your `data/raw` set.
- *Time-saver:* if RabbitMQ is fiddly, use Redis as the Celery broker and drop the RabbitMQ container. Mention the trade-off in the README; it's a legitimate engineering call.
- **Goal: messy docs are searchable in Qdrant.**

### Day 3 (Sun 20) — Baseline RAG + the graph
- `baseline_rag.py`: plain retrieve→generate with citations. Get *an* answer end-to-end.
- Start `graph/`: state, retrieve node, generate node, wire a linear LangGraph. Expose via `/query`.
- **Goal: end-to-end answer through both baseline and a (still linear) graph.**

### Day 4 (Mon 21) — Self-correction ⭐ (your biggest day)
- `grade_context` node (LLM-as-judge + retrieval-score heuristic).
- `rewrite_query`, `clarify`, `abstain`, `verify_faithfulness` nodes.
- Wire the conditional edges from §4.3. Add the bounded attempt counter.
- Manually test all four branches with hand-picked questions.
- **Goal: the loop demonstrably re-queries, clarifies, and abstains.**

### Day 5 (Tue 22) — Eval harness
- Write the 12 golden questions.
- `run_eval.py` runs both pipelines; `scorer.py` computes hallucination rate + abstention correctness with Ragas.
- Generate `results.json` + `chart.png`. **Goal: the before/after table exists and is real.**

### Day 6 (Wed 23) — UI + observability + hardening
- Minimal React UI: question box → answer + confidence bar + source list + status badge (green success / amber clarify / red abstain). The status badge is what makes the demo *legible*.
- Turn on LangSmith tracing. Add failure handling: OCR failure → flag-for-review status; LLM timeout → retry once then abstain.
- **Goal: a demo you'd be proud to screen-record.**

### Day 7 (Thu 24) — Freeze, document, record
- Write the README: architecture diagram, setup steps, the results table + chart, design decisions, known limitations.
- Record a 2–3 min demo video (mandatory backup even though your demo is live).
- Tag a release. Push. **Deliverables due today.** Do NOT add features today — only polish and document.

---

## 7. Standing out — the extras that separate #1 from #50

Do these only after MUSTs are done. Each is a "wow" moment in the live defense:

1. **Live trace on screen.** During the demo, pull up the LangSmith trace showing the graph took the `grade → rewrite → retrieve → generate` path. Seeing the *reasoning* live is rare and memorable.
2. **The "gotcha" question.** Ask your agent something plausible but not in the corpus, and let the panel watch it *refuse to answer*. Then ask a competitor-style "always answers" baseline the same question and show it hallucinate. Side-by-side. Devastatingly effective.
3. **Confidence calibration mini-chart.** Show that high-confidence answers really are more correct than low-confidence ones. Proves the grader is meaningful, not decorative.
4. **Contradiction case.** Put two conflicting facts in your docs; show it detects the conflict and clarifies rather than picking one. Almost nobody handles this.
5. **Honest limitations slide.** "Very low-res scans still fail; here's my flag-for-review fallback." Engineers trust people who name their failure modes.
6. **One-command reproducibility.** `docker-compose up` + `python eval/run_eval.py` reproduces your headline number. Reproducibility reads as senior.

---

## 8. The 25 July defense — questions they *will* ask, and your answers

- **"How does the confidence score work?"** → LLM-as-judge grades sufficiency + non-contradiction, cross-checked against mean retrieval similarity so the grader can't be blindly overconfident. Thresholds tuned on the golden set.
- **"Why no web-search fallback?"** → Closed-domain integrity. The PS is about honest behavior over *this* corpus; pulling from the web would reintroduce ungrounded answers and dodge the actual challenge.
- **"Why bound the loop at 2?"** → Cost and latency control + diminishing returns; measured that a 3rd re-query rarely changed the outcome on my golden set.
- **"How do you know it's not just abstaining on everything to game the hallucination metric?"** → I report abstention *correctness*, not just hallucination rate — it answers the 8 answerable questions and abstains only on the 4 it should. Both numbers together prove it's calibrated.
- **"What breaks it?"** → Very low-resolution scans (OCR noise) and highly ambiguous multi-hop questions; I handle the first with a flag-for-review status and the second with clarification.
- **"How would you scale it?"** → Async ingestion already decoupled via the queue; stateless FastAPI workers scale horizontally; Qdrant sharding + Redis caching for read-heavy load.

Memorize the numbers from your own eval run. Confidence + real metrics is unbeatable in a technical panel.

---

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Gemini free-tier rate limit hit | Provider abstraction: flip `LLM_PROVIDER` to `groq`, or `ollama` for fully local |
| No internet at venue on 25 July | Pre-pull Ollama model; run entire demo offline |
| Time overrun | Cut SHOULDs, never MUSTs; UI can degrade to single HTML page |
| Grader too lenient (never triggers loop) | Combine LLM score with retrieval-similarity threshold; tune on golden set |
| Docker/infra rabbit-hole | Redis-as-broker to drop RabbitMQ; run Qdrant in-memory mode if needed |
| Demo fails live | Pre-recorded 3-min video as mandatory backup |

---

## 10. Definition of done (check before 24 July)

- [ ] `docker-compose up` starts the whole system
- [ ] Ingests native PDF + scanned image; both searchable
- [ ] All four branches (answer / re-query / clarify / abstain) demonstrable
- [ ] Faithfulness verifier catches an ungrounded answer
- [ ] `eval/run_eval.py` produces the before/after hallucination table + chart
- [ ] README has architecture, setup, results table, design decisions, limitations
- [ ] 2–3 min demo video recorded
- [ ] GitHub repo public and tagged

Build the differentiators first. Ship the numbers. Defend with confidence. That's how you finish top of 200.
