.PHONY: help setup corpus up down seed eval test fmt logs

help:
	@echo "VeriRAG - make targets"
	@echo "  setup    install python deps locally"
	@echo "  corpus   generate the messy demo corpus into data/raw"
	@echo "  up       start the full stack (qdrant, redis, api, worker)"
	@echo "  seed     ingest everything in data/raw"
	@echo "  eval     run baseline vs agent and print the headline table"
	@echo "  test     run unit tests"
	@echo "  down     stop the stack"

setup:
	pip install -r requirements.txt

corpus:
	python scripts/generate_corpus.py

up:
	docker compose up --build -d
	@echo "API on http://localhost:8000  ·  UI on http://localhost:8000/"

down:
	docker compose down

seed:
	curl -s -X POST http://localhost:8000/api/ingest/seed | python -m json.tool

eval:
	python eval/run_eval.py

test:
	pytest -q

logs:
	docker compose logs -f api
