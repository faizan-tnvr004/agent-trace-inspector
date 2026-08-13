"""FastAPI application: router mounting, CORS, and corpus bootstrap.

On startup the committed corpus is loaded into SQLite if the database is empty
or if the corpus files have changed since the database was built. That is what
makes `docker compose up` from a clean clone produce a working system with data
in it (NFR-4), without a separate seeding step a reviewer has to know about.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.runs import router as runs_router
from app.db import (
    CORPUS_FINGERPRINT_KEY,
    connect,
    corpus_fingerprint,
    count_runs,
    get_meta,
    init_db,
    load_corpus_directory,
)
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
    """Project the committed corpus into SQLite when it is new or has changed.

    Skipping purely because the database already holds runs was wrong, and
    silently so. The container keeps its database in a named volume that
    survives `docker compose down`, so after the corpus was regenerated the
    volume went on serving the superseded traces while the results in the
    repository described the new ones. Nothing failed; the data was just old.
    """
    conn = connect(db_path())
    try:
        init_db(conn)
        existing = count_runs(conn)
        current = corpus_fingerprint(corpus_dir())
        if existing and get_meta(conn, CORPUS_FINGERPRINT_KEY) == current:
            logger.info(
                "database matches the corpus at %s (%d runs); skipping load",
                corpus_dir(),
                existing,
            )
            return existing
        if existing:
            logger.info("corpus at %s has changed; reloading", corpus_dir())
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
