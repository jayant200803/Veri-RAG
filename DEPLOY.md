# Deploying VeriRAG

Two supported targets: **local Docker** (the demo-day default) and a **free
public URL** on Hugging Face Spaces.

---

## A. Local Docker (recommended for the live demo)

```bash
cp .env.example .env            # set LLM_PROVIDER + key
docker compose up --build -d    # qdrant + redis + api + worker
curl -X POST http://localhost:8001/api/ingest/seed
```

- Landing page: http://localhost:8001/
- Console:      http://localhost:8001/app

Most reliable, zero cost, and works fully offline if `LLM_PROVIDER=ollama`.

---

## B. Public URL — Hugging Face Spaces (free, Docker)

The app collapses to a **single container** for the cloud: Qdrant runs embedded
in-process, ingestion runs in-process (no Redis/Celery), and the demo corpus is
auto-seeded on boot. All of this is toggled by environment variables — no code
changes needed.

### Steps

1. Create a free account at https://huggingface.co and verify your email.
2. **New → Space.**
   - Space name: `verirag`
   - SDK: **Docker** (blank template)
   - Hardware: **CPU basic (free)** — 16 GB RAM
3. In the Space, open **Files → edit `README.md`** and make sure the top has:
   ```yaml
   ---
   title: VeriRAG
   emoji: 🛡️
   colorFrom: purple
   colorTo: indigo
   sdk: docker
   app_port: 8000
   ---
   ```
4. **Settings → Variables and secrets** — add:

   | Name | Type | Value |
   |---|---|---|
   | `LLM_PROVIDER` | Variable | `groq` |
   | `GROQ_API_KEY` | **Secret** | your `gsk_...` key |
   | `QDRANT_URL` | Variable | *(leave empty)* -> embedded Qdrant |
   | `AUTO_SEED` | Variable | `true` |

5. Push this repo's code into the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/verirag
   git push space main
   ```
   (Authenticate with a Hugging Face access token when prompted.)
6. The Space builds the Dockerfile, boots, auto-seeds the corpus, and serves at
   `https://<your-username>-verirag.hf.space`.

### Notes

- First boot downloads the embedding model and runs OCR on the scanned memo, so
  allow a minute before the first query.
- Storage on Spaces is ephemeral: the corpus is re-seeded automatically on every
  restart (`AUTO_SEED=true`), so the demo is always populated.
- Uploaded documents persist only until the Space restarts — expected for a demo.
