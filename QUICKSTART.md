# QUICKSTART — read this first

## 1. Prerequisites

- Docker Desktop (running)
- A free LLM API key — Groq (https://console.groq.com/keys) or
  Gemini (https://aistudio.google.com/apikey). Both have a free tier, no card.

## 2. Configure

```bash
cp .env.example .env
```

Open `.env` and set your provider + key. To use Groq:

```ini
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
```

(Or `LLM_PROVIDER=gemini` with `GEMINI_API_KEY`, or `LLM_PROVIDER=ollama` for a
fully offline run.)

## 3. Run it

```bash
docker compose up --build -d          # start qdrant + redis + api + worker
curl -X POST http://localhost:8001/api/ingest/seed   # ingest the demo corpus
```

Open the app:

- Landing page: http://localhost:8001/
- Console:      http://localhost:8001/app

## 4. Reproduce the headline number

```bash
python eval/run_eval.py
```

This runs the 12 golden questions through both a plain baseline RAG and the
VeriRAG agent, and prints the before/after hallucination table. That table is
the single most important artefact in the project.

## 5. Verify without any setup

```bash
pytest -q       # unit tests, no API key or services needed
```

---

## Try these four questions in the console

1. *"How many casual leave days do full-time employees get?"* → **answers**
2. *"When does the production deployment freeze apply?"* → **answers from a scanned image (OCR)**
3. *"What is the notice period when resigning?"* → **flags the contradiction**
4. *"What is the CEO's home address?"* → **abstains**

Open the **decision trace** under any answer to see the agent's confidence,
reasoning, and the path it chose.
