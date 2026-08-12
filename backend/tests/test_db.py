"""Storage round-trip tests.

The database is regenerated from the committed corpus rather than being the
source of truth, so what matters here is that a run survives the round trip
byte-for-byte in its model form, and that the filters backing ``GET /runs``
behave.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.db import delete_run, get_run, insert_run, insert_runs, list_runs, run_ids
from app.models import Run
from tests.conftest import BASE_TIME, make_run, make_step


def _run(run_id: str, **overrides) -> Run:
    steps = [make_step(i, run_id=run_id) for i in range(3)]
    return Run.model_validate(
        make_run(steps=steps, run_id=run_id, **overrides)
    )


def test_run_round_trips_unchanged(db_conn) -> None:
    original = _run("run-rt")
    insert_run(db_conn, original)
    assert get_run(db_conn, "run-rt") == original


def test_get_run_returns_none_for_unknown_id(db_conn) -> None:
    assert get_run(db_conn, "no-such-run") is None


def test_nested_fields_survive_round_trip(db_conn) -> None:
    steps = [
        make_step(0, run_id="run-nested", evidence_refs=["doc-1", "doc-2"]),
        make_step(
            1,
            run_id="run-nested",
            event_type="error",
            error={"error_type": "TimeoutError", "message": "tool call timed out"},
        ),
        make_step(
            2,
            run_id="run-nested",
            event_type="critique",
            rejection_outcome="damage",
        ),
    ]
    original = Run.model_validate(
        make_run(
            steps=steps,
            run_id="run-nested",
            injected_fault={
                "fault_type": "injected_contradiction",
                "target_step_seq": 1,
                "description": "inserted a contradicting fact into context",
            },
        )
    )
    insert_run(db_conn, original)
    restored = get_run(db_conn, "run-nested")

    assert restored is not None
    assert restored.steps[0].evidence_refs == ["doc-1", "doc-2"]
    assert restored.steps[1].error is not None
    assert restored.steps[1].error.error_type == "TimeoutError"
    assert restored.steps[2].rejection_outcome == "damage"
    assert restored.injected_fault is not None
    assert restored.injected_fault.fault_type == "injected_contradiction"


def test_reinserting_a_run_replaces_it_without_duplicating_steps(db_conn) -> None:
    """`make corpus` must be re-runnable without dropping the database first."""
    insert_run(db_conn, _run("run-idem"))
    insert_run(db_conn, _run("run-idem", final_output="revised answer"))

    restored = get_run(db_conn, "run-idem")
    assert restored is not None
    assert restored.final_output == "revised answer"
    assert len(restored.steps) == 3
    assert run_ids(db_conn) == ["run-idem"]


def test_list_runs_filters_by_success(db_conn) -> None:
    insert_runs(
        db_conn,
        [
            _run("run-a", success=True),
            _run("run-b", success=False),
            _run("run-c", success=False),
        ],
    )
    failed, total = list_runs(db_conn, success=False)
    assert total == 2
    assert {r.run_id for r in failed} == {"run-b", "run-c"}

    passed, total = list_runs(db_conn, success=True)
    assert total == 1
    assert passed[0].run_id == "run-a"


def test_list_runs_filters_by_workflow_type(db_conn) -> None:
    insert_runs(
        db_conn,
        [
            _run("run-a", workflow_type="reviewer_pipeline"),
            _run("run-b", workflow_type="rag_qa"),
        ],
    )
    items, total = list_runs(db_conn, workflow_type="rag_qa")
    assert total == 1
    assert items[0].run_id == "run-b"


def test_list_runs_filters_compose(db_conn) -> None:
    insert_runs(
        db_conn,
        [
            _run("run-a", workflow_type="rag_qa", success=True),
            _run("run-b", workflow_type="rag_qa", success=False),
            _run("run-c", workflow_type="reviewer_pipeline", success=False),
        ],
    )
    items, total = list_runs(db_conn, workflow_type="rag_qa", success=False)
    assert total == 1
    assert items[0].run_id == "run-b"


def test_list_runs_paginates_with_a_stable_order(db_conn) -> None:
    insert_runs(
        db_conn,
        [
            Run.model_validate(
                make_run(
                    steps=[make_step(0, run_id=f"run-{i}")],
                    run_id=f"run-{i}",
                    started_at=(BASE_TIME + timedelta(minutes=i)).isoformat(),
                )
            )
            for i in range(5)
        ],
    )
    first, total = list_runs(db_conn, limit=2, offset=0)
    second, _ = list_runs(db_conn, limit=2, offset=2)

    assert total == 5
    assert [r.run_id for r in first] == ["run-0", "run-1"]
    assert [r.run_id for r in second] == ["run-2", "run-3"]


def test_list_runs_summary_reports_fault_without_loading_steps(db_conn) -> None:
    insert_run(
        db_conn,
        Run.model_validate(
            make_run(
                steps=[make_step(i, run_id="run-f") for i in range(4)],
                run_id="run-f",
                injected_fault={
                    "fault_type": "dropped_retrieval",
                    "target_step_seq": 3,
                    "description": "removed the chunk containing the answer",
                },
            )
        ),
    )
    items, _ = list_runs(db_conn)
    assert items[0].has_injected_fault is True
    assert items[0].fault_type == "dropped_retrieval"
    assert items[0].step_count == 4


def test_delete_run_removes_run_and_steps(db_conn) -> None:
    insert_run(db_conn, _run("run-del"))
    assert delete_run(db_conn, "run-del") is True
    assert get_run(db_conn, "run-del") is None
    assert delete_run(db_conn, "run-del") is False
    orphans = db_conn.execute(
        "SELECT COUNT(*) AS n FROM steps WHERE run_id = ?", ("run-del",)
    ).fetchone()["n"]
    assert orphans == 0


def test_duplicate_seq_within_a_run_is_rejected_by_the_database(db_conn) -> None:
    """Defence in depth: the model already forbids this, but the UNIQUE
    constraint means a direct write cannot corrupt the store either."""
    import sqlite3

    insert_run(db_conn, _run("run-uniq"))
    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            "INSERT INTO steps (step_id, run_id, seq, agent_id, agent_role, "
            "model, event_type, input, output, timestamp, latency_ms) "
            "VALUES ('x', 'run-uniq', 0, 'a', 'executor', 'm', 'plan', 'i', "
            "'o', '2026-01-01T12:00:00+00:00', 1)"
        )
