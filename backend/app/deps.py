"""Request-scoped dependencies.

A connection per request rather than one shared connection: `sqlite3` objects
are not safe to use across threads, and FastAPI serves sync endpoints from a
thread pool. SQLite handles concurrent readers without difficulty and the corpus
is hundreds of runs, so there is nothing to gain from pooling.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from app.db import connect, init_db

__all__ = ["corpus_dir", "db_path", "get_conn"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def db_path() -> Path:
    """Database location. Regenerable from the corpus, so it is gitignored."""
    configured = os.environ.get("TRACE_DB_PATH", "data/traces.db")
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def corpus_dir() -> Path:
    configured = os.environ.get("CORPUS_DIR", "data/corpus")
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def get_conn() -> Iterator[sqlite3.Connection]:
    conn = connect(db_path())
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()
