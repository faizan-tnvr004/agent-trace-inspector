"""SQLite persistence for runs and steps.

SQLite rather than a server database is a deliberate constraint: the corpus is
hundreds of runs, not millions, and the whole system must come up from a clean
clone with no external service. The database file is regenerable from the
committed corpus, so it is gitignored.

Nested values (``injected_fault``, ``evidence_refs``, ``error``) are stored as
JSON text columns. Normalising them into further tables would buy nothing here:
nothing queries inside them, and they always travel with their owning row.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.models import Run, RunSummary, Step

__all__ = [
    "CORPUS_FINGERPRINT_KEY",
    "connect",
    "corpus_fingerprint",
    "count_runs",
    "delete_run",
    "get_meta",
    "get_run",
    "init_db",
    "insert_run",
    "insert_runs",
    "list_runs",
    "load_corpus_directory",
    "open_db",
    "run_ids",
    "set_meta",
]

#: Meta key holding the fingerprint of the corpus the runs table was built from.
CORPUS_FINGERPRINT_KEY = "corpus_fingerprint"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    workflow_type   TEXT    NOT NULL,
    workflow_version TEXT   NOT NULL,
    task_input      TEXT    NOT NULL,
    final_output    TEXT    NOT NULL,
    success         INTEGER NOT NULL,
    ground_truth    TEXT,
    injected_fault  TEXT,
    started_at      TEXT    NOT NULL,
    completed_at    TEXT    NOT NULL,
    total_cost_usd  REAL    NOT NULL DEFAULT 0.0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    -- 'corpus' for runs projected from the committed corpus files, 'api' for
    -- runs ingested through POST /runs. A corpus reload replaces the former and
    -- must never touch the latter.
    source          TEXT    NOT NULL DEFAULT 'api'
);

-- Small key/value store. Currently one key, the fingerprint of the corpus the
-- runs table was projected from.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    step_id          TEXT PRIMARY KEY,
    run_id           TEXT    NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    parent_step_id   TEXT,
    seq              INTEGER NOT NULL,
    agent_id         TEXT    NOT NULL,
    agent_role       TEXT    NOT NULL,
    model            TEXT    NOT NULL,
    event_type       TEXT    NOT NULL,
    input            TEXT    NOT NULL,
    output           TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL,
    latency_ms       INTEGER NOT NULL,
    prompt_tokens    INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd         REAL    NOT NULL DEFAULT 0.0,
    evidence_refs    TEXT    NOT NULL DEFAULT '[]',
    error            TEXT,
    retry_of         TEXT,
    rejection_outcome TEXT,
    UNIQUE (run_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_steps_run       ON steps (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_runs_success    ON runs (success);
CREATE INDEX IF NOT EXISTS idx_runs_workflow   ON runs (workflow_type);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs (started_at);
"""

_STEP_COLUMNS = (
    "step_id",
    "run_id",
    "parent_step_id",
    "seq",
    "agent_id",
    "agent_role",
    "model",
    "event_type",
    "input",
    "output",
    "timestamp",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "evidence_refs",
    "error",
    "retry_of",
    "rejection_outcome",
)


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with foreign keys on and rows accessible by name.

    ``check_same_thread=False`` is required, not merely convenient. FastAPI
    resolves a synchronous generator dependency on one worker thread and then
    runs the endpoint body on another, so a connection opened in the dependency
    is used from a different thread than the one that created it. With the
    default setting SQLite raises ProgrammingError intermittently, depending on
    which threads the pool happens to assign.

    It is safe here because connections are per request and never shared between
    concurrent requests, so disabling the assertion removes a false positive
    rather than a real guard.
    """
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def open_db(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Connect, ensure the schema exists, and close afterwards."""
    conn = connect(db_path)
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    # `CREATE TABLE IF NOT EXISTS` does not add a column to a table that already
    # exists, and the container keeps its database in a named volume that
    # outlives the image. A database created before `source` existed would
    # otherwise fail every query that mentions it.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    if "source" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN source TEXT NOT NULL DEFAULT 'api'")
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )


def corpus_fingerprint(corpus_dir: str | Path) -> str:
    """A digest of the corpus files' contents.

    Content rather than file count or modification time. The regenerated corpus
    that prompted this had the same 120 files with the same names; only the bytes
    inside 60 of them changed, so anything coarser would have missed it. Hashing
    120 small files costs a few milliseconds at boot.
    """
    directory = Path(corpus_dir)
    digest = hashlib.sha256()
    if not directory.is_dir():
        return ""
    for path in sorted(directory.glob("run_*.json")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def insert_run(conn: sqlite3.Connection, run: Run, *, source: str = "api") -> None:
    """Insert or replace a run and all of its steps in one transaction.

    Replacing rather than failing on conflict keeps corpus loading idempotent:
    ``make corpus`` can be re-run without first dropping the database.
    """
    with conn:
        conn.execute("DELETE FROM steps WHERE run_id = ?", (run.run_id,))
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, workflow_type, workflow_version, task_input,
                final_output, success, ground_truth, injected_fault,
                started_at, completed_at, total_cost_usd, total_tokens, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.workflow_type,
                run.workflow_version,
                run.task_input,
                run.final_output,
                int(run.success),
                run.ground_truth,
                (
                    run.injected_fault.model_dump_json()
                    if run.injected_fault is not None
                    else None
                ),
                run.started_at.isoformat(),
                run.completed_at.isoformat(),
                run.total_cost_usd,
                run.total_tokens,
                source,
            ),
        )
        conn.executemany(
            f"INSERT INTO steps ({', '.join(_STEP_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(_STEP_COLUMNS))})",
            [_step_row(step) for step in run.steps],
        )


def insert_runs(conn: sqlite3.Connection, runs: Iterable[Run]) -> int:
    count = 0
    for run in runs:
        insert_run(conn, run)
        count += 1
    return count


def get_run(conn: sqlite3.Connection, run_id: str) -> Run | None:
    """Return the full run including steps, or ``None`` if unknown."""
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    step_rows = conn.execute(
        "SELECT * FROM steps WHERE run_id = ? ORDER BY seq", (run_id,)
    ).fetchall()
    return Run.model_validate(
        {**_run_fields(row), "steps": [_step_fields(r) for r in step_rows]}
    )


def list_runs(
    conn: sqlite3.Connection,
    *,
    success: bool | None = None,
    workflow_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[RunSummary], int]:
    """Return a page of summaries and the total matching the filters.

    The total ignores ``limit`` and ``offset`` so the caller can paginate.
    """
    where: list[str] = []
    params: list[Any] = []
    if success is not None:
        where.append("r.success = ?")
        params.append(int(success))
    if workflow_type is not None:
        where.append("r.workflow_type = ?")
        params.append(workflow_type)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM runs r {clause}", params
    ).fetchone()["n"]

    rows = conn.execute(
        f"""
        SELECT r.*, (
            SELECT COUNT(*) FROM steps s WHERE s.run_id = r.run_id
        ) AS step_count
        FROM runs r
        {clause}
        ORDER BY r.started_at, r.run_id
        LIMIT ? OFFSET ?
        """,
        [*params, limit, offset],
    ).fetchall()

    items = [
        RunSummary(
            run_id=row["run_id"],
            workflow_type=row["workflow_type"],
            workflow_version=row["workflow_version"],
            task_input=row["task_input"],
            success=bool(row["success"]),
            has_injected_fault=row["injected_fault"] is not None,
            fault_type=(
                json.loads(row["injected_fault"])["fault_type"]
                if row["injected_fault"] is not None
                else None
            ),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            total_cost_usd=row["total_cost_usd"],
            total_tokens=row["total_tokens"],
            step_count=row["step_count"],
        )
        for row in rows
    ]
    return items, total


def load_corpus_directory(conn: sqlite3.Connection, corpus_dir: str | Path) -> int:
    """Ingest every ``run_*.json`` in a directory. Returns the number loaded.

    The corpus files are the source of truth and the database is a queryable
    projection of them, so this is safe to re-run: `insert_run` replaces.

    Only ``run_*.json`` is matched, never ``*.json``. Sibling files such as the
    generation summary are not traces and would fail validation.
    """
    directory = Path(corpus_dir)
    if not directory.is_dir():
        return 0

    loaded = 0
    seen: list[str] = []
    for path in sorted(directory.glob("run_*.json")):
        run = Run.model_validate(json.loads(path.read_text()))
        insert_run(conn, run, source="corpus")
        seen.append(run.run_id)
        loaded += 1

    # Regenerating the corpus mints fresh run ids, so the previous projection
    # does not overlap the new one and replacing by id leaves every old run
    # behind. Without this, reloading a regenerated corpus doubles the run count
    # and serves both versions side by side. Scoped to `source = 'corpus'` so a
    # run ingested through POST /runs is never collateral.
    placeholders = ", ".join("?" * len(seen))
    with conn:
        conn.execute(
            f"DELETE FROM steps WHERE run_id IN ("  # noqa: S608 - ids are bound
            f"SELECT run_id FROM runs WHERE source = 'corpus'"
            f"{f' AND run_id NOT IN ({placeholders})' if seen else ''})",
            seen,
        )
        conn.execute(
            "DELETE FROM runs WHERE source = 'corpus'"
            + (f" AND run_id NOT IN ({placeholders})" if seen else ""),
            seen,
        )

    set_meta(conn, CORPUS_FINGERPRINT_KEY, corpus_fingerprint(directory))
    return loaded


def count_runs(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]


def run_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT run_id FROM runs ORDER BY started_at, run_id")
    return [r["run_id"] for r in rows]


def delete_run(conn: sqlite3.Connection, run_id: str) -> bool:
    with conn:
        cur = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM steps WHERE run_id = ?", (run_id,))
    return cur.rowcount > 0


# --------------------------------------------------------------------------
# Row <-> model conversion
# --------------------------------------------------------------------------


def _step_row(step: Step) -> tuple[Any, ...]:
    return (
        step.step_id,
        step.run_id,
        step.parent_step_id,
        step.seq,
        step.agent_id,
        step.agent_role,
        step.model,
        step.event_type,
        step.input,
        step.output,
        step.timestamp.isoformat(),
        step.latency_ms,
        step.prompt_tokens,
        step.completion_tokens,
        step.cost_usd,
        json.dumps(step.evidence_refs),
        step.error.model_dump_json() if step.error is not None else None,
        step.retry_of,
        step.rejection_outcome,
    )


def _run_fields(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "workflow_type": row["workflow_type"],
        "workflow_version": row["workflow_version"],
        "task_input": row["task_input"],
        "final_output": row["final_output"],
        "success": bool(row["success"]),
        "ground_truth": row["ground_truth"],
        "injected_fault": (
            json.loads(row["injected_fault"])
            if row["injected_fault"] is not None
            else None
        ),
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "total_cost_usd": row["total_cost_usd"],
        "total_tokens": row["total_tokens"],
    }


def _step_fields(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "step_id": row["step_id"],
        "run_id": row["run_id"],
        "parent_step_id": row["parent_step_id"],
        "seq": row["seq"],
        "agent_id": row["agent_id"],
        "agent_role": row["agent_role"],
        "model": row["model"],
        "event_type": row["event_type"],
        "input": row["input"],
        "output": row["output"],
        "timestamp": row["timestamp"],
        "latency_ms": row["latency_ms"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "cost_usd": row["cost_usd"],
        "evidence_refs": json.loads(row["evidence_refs"]),
        "error": json.loads(row["error"]) if row["error"] is not None else None,
        "retry_of": row["retry_of"],
        "rejection_outcome": row["rejection_outcome"],
    }
