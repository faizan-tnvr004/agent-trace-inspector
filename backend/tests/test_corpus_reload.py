"""Tests for keeping the database in step with the corpus files.

The corpus files are the source of truth and the database is a projection of
them. The container keeps that database in a named volume which outlives
`docker compose down`, so "load it once and never again" quietly served the
superseded traces for as long as the volume lived. These tests pin the two
properties that make a reload safe: it notices a content change, and it does not
take runs that arrived through the API with it.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.db import (
    CORPUS_FINGERPRINT_KEY,
    connect,
    corpus_fingerprint,
    count_runs,
    get_meta,
    init_db,
    insert_run,
    load_corpus_directory,
    run_ids,
)
from app.models import Run
from tests.conftest import make_run, make_step


def _write_corpus(directory: Path, run_ids_wanted: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, run_id in enumerate(run_ids_wanted, start=1):
        run = make_run(
            steps=[make_step(i, run_id=run_id) for i in range(3)],
            run_id=run_id,
        )
        (directory / f"run_{index:04d}.json").write_text(json.dumps(run))


def _db(tmp_path: Path):
    conn = connect(tmp_path / "traces.db")
    init_db(conn)
    return conn


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_a_file_changes(tmp_path: Path) -> None:
    """The case that motivated this: the same 120 filenames, 60 different
    bodies. A count or a filename list would not have noticed."""
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, ["run-a", "run-b"])
    before = corpus_fingerprint(corpus)

    path = corpus / "run_0001.json"
    run = json.loads(path.read_text())
    run["final_output"] = "a different answer"
    path.write_text(json.dumps(run))

    assert corpus_fingerprint(corpus) != before


def test_fingerprint_is_stable_when_nothing_changes(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, ["run-a", "run-b"])
    assert corpus_fingerprint(corpus) == corpus_fingerprint(corpus)


def test_fingerprint_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert corpus_fingerprint(tmp_path / "nope") == ""


def test_loading_records_the_fingerprint(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, ["run-a"])
    conn = _db(tmp_path)
    load_corpus_directory(conn, corpus)
    assert get_meta(conn, CORPUS_FINGERPRINT_KEY) == corpus_fingerprint(corpus)


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------


def test_a_regenerated_corpus_replaces_rather_than_accumulates(
    tmp_path: Path,
) -> None:
    """Regenerating the corpus mints fresh run ids, so replacing by id leaves
    every old run behind. Without the sweep this database would hold four runs
    from two different corpora and serve both."""
    corpus = tmp_path / "corpus"
    conn = _db(tmp_path)

    _write_corpus(corpus, ["old-a", "old-b"])
    load_corpus_directory(conn, corpus)
    assert count_runs(conn) == 2

    _write_corpus(corpus, ["new-a", "new-b"])
    load_corpus_directory(conn, corpus)

    assert count_runs(conn) == 2
    assert sorted(run_ids(conn)) == ["new-a", "new-b"]


def test_reloading_the_same_corpus_is_idempotent(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    conn = _db(tmp_path)
    _write_corpus(corpus, ["run-a", "run-b"])
    load_corpus_directory(conn, corpus)
    load_corpus_directory(conn, corpus)
    assert count_runs(conn) == 2


def test_a_reload_does_not_delete_runs_ingested_through_the_api(
    tmp_path: Path,
) -> None:
    """A run that arrived through POST /runs is somebody's data, not a
    projection of a file, and a corpus change is no reason to drop it."""
    corpus = tmp_path / "corpus"
    conn = _db(tmp_path)

    _write_corpus(corpus, ["old-a"])
    load_corpus_directory(conn, corpus)

    posted = Run.model_validate(
        make_run(steps=[make_step(i, run_id="posted") for i in range(3)], run_id="posted")
    )
    insert_run(conn, posted)

    _write_corpus(corpus, ["new-a"])
    load_corpus_directory(conn, corpus)

    assert sorted(run_ids(conn)) == ["new-a", "posted"]


def test_an_emptied_corpus_clears_only_the_projection(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    conn = _db(tmp_path)
    _write_corpus(corpus, ["old-a", "old-b"])
    load_corpus_directory(conn, corpus)

    posted = Run.model_validate(
        make_run(steps=[make_step(i, run_id="posted") for i in range(3)], run_id="posted")
    )
    insert_run(conn, posted)

    for path in corpus.glob("run_*.json"):
        path.unlink()
    load_corpus_directory(conn, corpus)

    assert run_ids(conn) == ["posted"]


def test_reload_removes_the_orphaned_steps_too(tmp_path: Path) -> None:
    """A left-behind step row would still satisfy a run count check while
    corrupting any query that reads steps directly."""
    corpus = tmp_path / "corpus"
    conn = _db(tmp_path)
    _write_corpus(corpus, ["old-a"])
    load_corpus_directory(conn, corpus)

    _write_corpus(corpus, ["new-a"])
    load_corpus_directory(conn, corpus)

    rows = conn.execute("SELECT DISTINCT run_id FROM steps").fetchall()
    assert [r["run_id"] for r in rows] == ["new-a"]


# ---------------------------------------------------------------------------
# Startup decision
# ---------------------------------------------------------------------------


def _bootstrap(tmp_path: Path, monkeypatch) -> int:
    monkeypatch.setenv("TRACE_DB_PATH", str(tmp_path / "traces.db"))
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    from app.main import bootstrap_corpus

    return bootstrap_corpus()


def test_startup_loads_an_empty_database(tmp_path: Path, monkeypatch) -> None:
    _write_corpus(tmp_path / "corpus", ["run-a", "run-b"])
    assert _bootstrap(tmp_path, monkeypatch) == 2


def test_startup_skips_when_the_corpus_is_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    _write_corpus(tmp_path / "corpus", ["run-a", "run-b"])
    _bootstrap(tmp_path, monkeypatch)
    assert _bootstrap(tmp_path, monkeypatch) == 2


def test_startup_reloads_a_changed_corpus(tmp_path: Path, monkeypatch) -> None:
    """The regression: a database that survives a restart must not go on
    serving traces the corpus no longer contains."""
    corpus = tmp_path / "corpus"
    _write_corpus(corpus, ["old-a", "old-b"])
    _bootstrap(tmp_path, monkeypatch)

    _write_corpus(corpus, ["new-a", "new-b"])
    _bootstrap(tmp_path, monkeypatch)

    conn = connect(tmp_path / "traces.db")
    assert sorted(run_ids(conn)) == ["new-a", "new-b"]


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_a_database_predating_the_source_column_is_migrated(tmp_path: Path) -> None:
    """The container's volume outlives the image, so an existing database will
    not have the column that `CREATE TABLE IF NOT EXISTS` would have added."""
    path = tmp_path / "traces.db"
    conn = connect(path)
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY, workflow_type TEXT NOT NULL,
            workflow_version TEXT NOT NULL, task_input TEXT NOT NULL,
            final_output TEXT NOT NULL, success INTEGER NOT NULL,
            ground_truth TEXT, injected_fault TEXT, started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL, total_cost_usd REAL NOT NULL DEFAULT 0.0,
            total_tokens INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()

    init_db(conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    assert "source" in columns

    corpus = tmp_path / "corpus"
    _write_corpus(corpus, ["run-a"])
    assert load_corpus_directory(conn, corpus) == 1
