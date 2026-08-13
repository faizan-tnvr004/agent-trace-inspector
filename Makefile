.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV       := .venv
PY         := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
CORPUS_DIR := data/corpus
RESULTS    := evaluation/results

# `uv` is used when present because it installs an order of magnitude faster and
# can provision Python 3.11 itself. The venv fallback keeps the Makefile usable
# on a machine that only has a system Python 3.11.
UV := $(shell command -v uv 2>/dev/null || echo "$$HOME/.local/bin/uv")

.PHONY: help install schema corpus stamp test coverage eval reproduce up down \
        dev dev-backend dev-frontend frontend-build clean

help:
	@echo "Multi-Agent Execution Trace Inspector"
	@echo
	@echo "  make install         create the venv, install backend and frontend deps"
	@echo "  make test            run the full pytest suite"
	@echo "  make coverage        run tests with coverage on the extraction module"
	@echo "  make corpus          regenerate the trace corpus (needs GEMINI_API_KEY, costs quota)"
	@echo "  make stamp           fill rejection_outcome on the committed corpus"
	@echo "  make eval            run the primary evaluation (needs GEMINI_API_KEY)"
	@echo "  make reproduce       recompute every published number from the committed corpus"
	@echo "  make schema          regenerate schema/trace.schema.json from the models"
	@echo "  make up              start backend and frontend via docker compose"
	@echo "  make dev             start backend and frontend natively (no Docker needed)"
	@echo

install:
	@if [ -x "$(UV)" ]; then \
		echo "==> provisioning Python 3.11 with uv"; \
		"$(UV)" python install 3.11; \
		"$(UV)" venv --python 3.11 $(VENV); \
		"$(UV)" pip install --python $(PY) -e "backend[dev,extraction,harness]"; \
	else \
		echo "==> uv not found, falling back to python3.11 -m venv"; \
		python3.11 -m venv $(VENV); \
		$(PIP) install --upgrade pip; \
		$(PIP) install -e "backend[dev,extraction,harness]"; \
	fi
	@cd frontend && npm install
	@echo "==> done. Copy .env.example to .env before running make corpus or make eval."

schema:
	@cd backend && ../$(PY) -c "from app.models import Run; import json; print(json.dumps(Run.model_json_schema(), indent=2))" > ../schema/trace.schema.json
	@echo "==> wrote schema/trace.schema.json"

# Two passes with different fault rates. The reviewer pipeline repairs most
# injected faults, so a single rate would either starve the evaluation of failed
# runs or leave the reviewer workflow with no clean controls.
corpus:
	@test -f .env || { echo "no .env; copy .env.example and add GEMINI_API_KEY"; exit 1; }
	set -a; . ./.env; set +a; \
	$(PY) harness/generate_corpus.py --workflow rag_qa \
		--n 60 --fault-rate 0.75 --out $(CORPUS_DIR) --expect-total 120 && \
	$(PY) harness/generate_corpus.py --workflow reviewer_pipeline \
		--n 60 --fault-rate 0.5 --out $(CORPUS_DIR) --append --expect-total 120
	@$(MAKE) stamp

stamp:
	@$(PY) harness/stamp_rejections.py --corpus $(CORPUS_DIR)

test:
	@cd backend && ../$(PY) -m pytest -q

coverage:
	@cd backend && ../$(PY) -m pytest --cov=app/extraction --cov-report=term-missing -q

eval:
	@test -f .env || { echo "no .env; copy .env.example and add GEMINI_API_KEY"; exit 1; }
	set -a; . ./.env; set +a; \
	$(PY) evaluation/run_study.py --out $(RESULTS) --corpus $(CORPUS_DIR)

# Recomputes every published number from the committed corpus. Corpus generation
# is deliberately excluded: it needs API quota and is not bit-reproducible, which
# is why the corpus itself is committed.
reproduce:
	@echo "==> extraction and rejection statistics from the committed corpus"
	@$(PY) evaluation/reproduce.py --corpus $(CORPUS_DIR) --out $(RESULTS)
	@echo
	@echo "==> primary study (requires GEMINI_API_KEY; skipped if absent)"
	@if [ -f .env ] && grep -q '^GEMINI_API_KEY=.\+' .env; then \
		$(MAKE) eval; \
	else \
		echo "    skipped: no GEMINI_API_KEY in .env"; \
		echo "    committed results remain in $(RESULTS)/primary_study.json"; \
	fi

frontend-build:
	@cd frontend && npm run build

up:
	docker compose up --build

down:
	docker compose down

# Native path, for a machine without Docker. Backend and frontend are separate
# targets so each can be run in its own terminal.
dev-backend:
	set -a; [ -f .env ] && . ./.env; set +a; \
	cd backend && ../$(VENV)/bin/uvicorn app.main:app --reload --port 8000

dev-frontend:
	@cd frontend && npm run dev

dev:
	@echo "Run these in two terminals:"
	@echo "  make dev-backend    -> http://localhost:8000"
	@echo "  make dev-frontend   -> http://localhost:5173"

clean:
	rm -rf $(VENV) frontend/node_modules frontend/dist data/traces.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .pytest_cache -type d -prune -exec rm -rf {} +
