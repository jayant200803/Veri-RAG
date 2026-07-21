# Architecture

## System view

```
┌──────────────┐
│   Web UI     │  web/index.html — confidence bars, status badges, decision trace
└──────┬───────┘
       │ HTTP
┌──────▼───────────────────────────────────────────────┐
│  FastAPI  (app/main.py, app/api/routes.py)           │
│  /health  /ingest  /status  /query  /index           │
└──────┬─────────────────────────┬─────────────────────┘
       │                         │
   INGEST PATH               QUERY PATH
       │                         │
┌──────▼───────────────┐  ┌──────▼──────────────────────┐
│ ingestion/service.py │  │ graph/build.py — THE AGENT  │
│                      │  │                             │
│ loader.py            │  │  retrieve                   │
│  ├ pdfplumber        │  │      ↓                      │
│  │  (text layer)     │  │  grade_context ──┐          │
│  └ Tesseract OCR     │  │      ↓           │          │
│    (scans/images)    │  │  [routing.py]    │          │
│         ↓            │  │   ├ generate     │          │
│ clean.py             │  │   │   ↓          │          │
│         ↓            │  │   │ verify       │          │
│ chunk.py             │  │   ├ rewrite ─────┘ (loop)   │
│  (tables atomic)     │  │   ├ clarify                 │
│         ↓            │  │   └ abstain                 │
│ fastembed (local)    │  │                             │
└─────────┬────────────┘  └──────┬──────────────────────┘
          │                      │
     ┌────▼──────────────────────▼────┐
     │  Qdrant (vectors) · Redis      │
     └────────────────────────────────┘
```

## The seven agent nodes

| Node | Responsibility | Failure behaviour |
|---|---|---|
| `retrieve` | Embed query, search Qdrant top-k | Returns `[]` — never raises |
| `grade_context` | Score sufficiency + contradiction (0–1) | No docs → confidence 0.0 |
| `rewrite_query` | Reformulate, decompose multi-hop | Falls back to original query |
| `generate` | Answer from context only, with citations | — |
| `verify_faithfulness` | Audit answer against sources | Ungrounded → downgrade to abstain |
| `clarify` | Ask one targeted question | Generic fallback question |
| `abstain` | Decline, explain why | — |

## Why the grader is cross-checked

`grade_context` does not trust the LLM judge alone. It computes:

```python
retrieval_ceiling = min(1.0, top_vector_score / 0.75)
confidence = min(llm_confidence, max(retrieval_ceiling, 0.15))
```

An LLM judge can be confidently wrong about weak evidence. Capping its score by
the raw retrieval similarity means the agent cannot be talked into answering
when vector search clearly found nothing relevant. This single line is the
difference between a decorative confidence score and a meaningful one.

## Scalability

- **Ingestion is decoupled.** OCR is the slowest operation in the system and
  never touches the request path — uploads return immediately with a `task_id`.
- **The API is stateless.** All state lives in Qdrant and Redis, so API
  containers scale horizontally behind a load balancer with no coordination.
- **Embeddings are local.** No external rate limit on the ingestion path, so
  throughput is bounded by CPU rather than by an API quota.
- **Redis caches** repeated queries, which matters because the correction loop
  can issue several LLM calls per question.
- **The loop is bounded**, so worst-case latency and cost per query are known
  quantities rather than open-ended.

## Robustness under failure

| Failure | Behaviour |
|---|---|
| Qdrant unreachable | `search()` returns `[]` → confidence 0.0 → abstain |
| OCR yields nothing | Document marked `flagged_for_review`, not indexed as noise |
| LLM returns malformed JSON | Tolerant extraction; on total failure → safe default |
| LLM times out | Retry with exponential backoff (tenacity), then abstain |
| Grader misses confidence | Fails safe to abstain — never answers by default |
| Verifier rejects the answer | Answer discarded, downgraded to abstain |
| Pathological loop | Attempt ceiling (2) + `recursion_limit=25` |
