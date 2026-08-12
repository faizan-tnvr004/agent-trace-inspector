"""FastAPI application: router mounting, CORS, and corpus bootstrap.

On startup the committed corpus is loaded into SQLite if the database is empty.
That is what makes `docker compose up` from a clean clone produce a working
system with data in it (NFR-4), without a separate seeding step a reviewer has
to know about.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.runs import router as runs_router
from app.db import connect, count_runs, init_db, load_corpus_directory
from app.deps import corpus_dir, db_path

logger = logging.getLogger("trace_inspector")

API_VERSION = "1.0.0"

# The frontend dev server and its container equivalent. No authentication
# anywhere in this system by design, so this is not a security boundary.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://frontend:5173",
]


def bootstrap_corpus() -> int:
    """Load the committed corpus into SQLite if the database has no runs."""
    conn = connect(db_path())
    try:
        init_db(conn)
        existing = count_runs(conn)
        if existing:
            logger.info("database already holds %d runs; skipping load", existing)
            return existing
        loaded = load_corpus_directory(conn, corpus_dir())
        logger.info("loaded %d runs from %s", loaded, corpus_dir())
        return loaded
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_corpus()
    yield


app = FastAPI(
    title="Multi-Agent Execution Trace Inspector",
    version=API_VERSION,
    summary=(
        "Ingests multi-agent execution traces, extracts the steps that "
        "determined the outcome, and predicts where failed runs went wrong."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(runs_router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    conn = connect(db_path())
    try:
        init_db(conn)
        runs = count_runs(conn)
    finally:
        conn.close()
    return {"status": "ok", "api_version": API_VERSION, "runs": runs}
