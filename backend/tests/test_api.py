"""Endpoint tests for the backend API.

The 422 tests are the ones that matter most. FR-6 requires a rejected trace to
name the offending field, and the six trace invariants are enforced by
model-level validators, which Pydantic reports with an empty ``loc``. Without
the recovery step in the router those responses would say "Value error" and
nothing else.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.db import connect, init_db, insert_run
from app.models import Run
from tests.conftest import make_run, make_step, sid

FACT = "The deep basin at Grethe Bay reaches a maximum depth of 312 metres."


def _run(run_id: str, **overrides) -> Run:
    steps = overrides.pop(
        "steps",
        [
            make_step(0, run_id=run_id, event_type="plan", output="a plan"),
            make_step(
                1,
                run_id=run_id,
                event_type="retrieval",
                output=FACT,
                evidence_refs=["doc-08#c2"],
                agent_id="retriever-1",
                agent_role="retriever",
            ),
            make_step(
                2,
                run_id=run_id,
                event_type="reasoning",
                output=FACT,
                agent_id="answerer-1",
            ),
            make_step(
                3, run_id=run_id, event_type="final", output=FACT, agent_id="answerer-1"
            ),
        ],
    )
    overrides.setdefault("final_output", FACT)
    return Run.model_validate(make_run(steps=steps, run_id=run_id, **overrides))


@pytest.fixture
def db_file(tmp_path, monkeypatch):
    """A file-backed database, not in-memory.

    TestClient serves sync endpoints from a thread pool, and a `sqlite3`
    connection may only be used on the thread that created it. Sharing one
    in-memory connection therefore fails with a ProgrammingError. Using a file
    lets each request open its own connection, which is also what happens in
    production.
    """
    path = tmp_path / "test.db"
    monkeypatch.setenv("TRACE_DB_PATH", str(path))
    # Point the corpus bootstrap at a directory that does not exist, so the
    # fixtures below are the only data in play.
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "no-corpus-here"))
    connection = connect(path)
    init_db(connection)
    connection.close()
    return path


@pytest.fixture
def conn(db_file) -> Iterator[sqlite3.Connection]:
    """A connection used only by the tests to insert fixtures."""
    connection = connect(db_file)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def client(db_file) -> Iterator[TestClient]:
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def test_list_runs_returns_items_and_total(client, conn) -> None:
    insert_run(conn, _run("run-a"))
    insert_run(conn, _run("run-b"))

    body = client.get("/runs").json()
    assert body["total"] == 2
    assert {item["run_id"] for item in body["items"]} == {"run-a", "run-b"}
    assert body["items"][0]["step_count"] == 4


def test_list_runs_filters_on_success(client, conn) -> None:
    insert_run(conn, _run("run-ok", success=True))
    insert_run(conn, _run("run-bad", success=False))

    failed = client.get("/runs", params={"success": "false"}).json()
    assert failed["total"] == 1
    assert failed["items"][0]["run_id"] == "run-bad"


def test_list_runs_filters_on_workflow_type(client, conn) -> None:
    insert_run(conn, _run("run-rag", workflow_type="rag_qa"))
    insert_run(conn, _run("run-rev", workflow_type="reviewer_pipeline"))

    body = client.get("/runs", params={"workflow_type": "rag_qa"}).json()
    assert body["total"] == 1
    assert body["items"][0]["run_id"] == "run-rag"


def test_list_runs_paginates(client, conn) -> None:
    for i in range(5):
        insert_run(conn, _run(f"run-{i}"))

    page = client.get("/runs", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 5
    assert len(page["items"]) == 2
    assert page["offset"] == 2


def test_list_runs_rejects_an_out_of_range_limit(client) -> None:
    assert client.get("/runs", params={"limit": 0}).status_code == 422
    assert client.get("/runs", params={"offset": -1}).status_code == 422


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def test_get_run_returns_the_full_trace(client, conn) -> None:
    insert_run(conn, _run("run-full"))
    body = client.get("/runs/run-full").json()
    assert body["run_id"] == "run-full"
    assert len(body["steps"]) == 4
    assert [s["seq"] for s in body["steps"]] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "path",
    ["", "/critical", "/attribution", "/provenance", "/cost", "/export"],
)
def test_unknown_run_id_is_404_on_every_endpoint(client, path: str) -> None:
    response = client.get(f"/runs/nonexistent{path}")
    assert response.status_code == 404
    assert "nonexistent" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Extraction endpoints
# ---------------------------------------------------------------------------


def test_critical_returns_at_most_k_steps_with_components(client, conn) -> None:
    insert_run(conn, _run("run-crit"))
    body = client.get("/runs/run-crit/critical", params={"k": 2}).json()

    assert body["k"] == 2
    assert len(body["steps"]) == 2
    step = body["steps"][0]
    for field in ("evidence_survival", "branch", "error", "weights", "reasons"):
        assert field in step, f"{field} missing: score must not be opaque"


def test_attribution_reports_ground_truth_when_a_fault_was_injected(
    client, conn
) -> None:
    insert_run(
        conn,
        _run(
            "run-attr",
            success=False,
            steps=[
                make_step(0, run_id="run-attr", event_type="plan", output="a plan"),
                make_step(
                    1,
                    run_id="run-attr",
                    event_type="retrieval",
                    output="",
                    evidence_refs=[],
                ),
                make_step(2, run_id="run-attr", event_type="final", output="not found"),
            ],
            injected_fault={
                "fault_type": "dropped_retrieval",
                "target_step_seq": 1,
                "description": "removed the chunk containing the answer",
            },
        ),
    )
    body = client.get("/runs/run-attr/attribution").json()
    assert body["actual_fault_step_seq"] == 1
    assert body["predicted_step_seq"] == 1
    assert body["reason"]
    assert body["rule_weights"]


def test_provenance_counts_unsupported_claims(client, conn) -> None:
    insert_run(
        conn,
        _run(
            "run-prov",
            steps=[
                make_step(
                    0,
                    run_id="run-prov",
                    event_type="retrieval",
                    output=FACT,
                    evidence_refs=["doc-08#c2"],
                ),
                make_step(1, run_id="run-prov", event_type="final", output=FACT),
            ],
            final_output=(
                f"{FACT} The station also recorded the first magnitude nine "
                "earthquake ever seen in the region."
            ),
        ),
    )
    body = client.get("/runs/run-prov/provenance").json()
    assert body["total"] == 2
    assert body["unsupported"] == 1
    assert any(c["supported"] is False for c in body["claims"])


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_reconciles_with_the_run_total(client, conn) -> None:
    """FR-28: totals must reconcile with run.total_cost_usd."""
    insert_run(conn, _run("run-cost"))
    body = client.get("/runs/run-cost/cost").json()

    assert body["reconciles"] is True
    assert body["total"]["cost_usd"] == pytest.approx(
        body["run_total_cost_usd"], abs=1e-6
    )


def test_cost_breaks_down_by_agent_and_event_type(client, conn) -> None:
    insert_run(conn, _run("run-cost2"))
    body = client.get("/runs/run-cost2/cost").json()

    assert set(body["by_agent"]) == {"executor-1", "retriever-1", "answerer-1"}
    assert set(body["by_event_type"]) == {"plan", "retrieval", "reasoning", "final"}
    assert sum(g["steps"] for g in body["by_agent"].values()) == 4
    assert sum(g["steps"] for g in body["by_event_type"].values()) == 4


def test_cost_states_that_the_basis_is_notional(client, conn) -> None:
    """Generation ran on a free tier. A number that looks like money spent, but
    is not, has to say so."""
    insert_run(conn, _run("run-cost3"))
    assert "notional" in client.get("/runs/run-cost3/cost").json()["cost_basis"]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_is_self_contained(client, conn) -> None:
    insert_run(conn, _run("run-exp"))
    body = client.get("/runs/run-exp/export").json()

    for key in (
        "export_format_version",
        "exported_at",
        "run",
        "critical_steps",
        "attribution",
        "provenance",
        "rejection_outcomes",
        "cost",
        "ground_truth",
    ):
        assert key in body, f"audit bundle is missing {key}"
    assert body["run"]["run_id"] == "run-exp"
    assert len(body["run"]["steps"]) == 4


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def test_post_accepts_a_valid_trace(client) -> None:
    payload = make_run(
        steps=[make_step(0, run_id="uploaded"), make_step(1, run_id="uploaded")],
        run_id="uploaded",
    )
    response = client.post("/runs", json=payload)
    assert response.status_code == 201
    assert response.json() == {"run_id": "uploaded", "steps": 2}
    assert client.get("/runs/uploaded").status_code == 200


def test_post_rejects_a_non_contiguous_seq_naming_the_field(client) -> None:
    payload = make_run(
        steps=[make_step(0, run_id="bad"), make_step(2, run_id="bad")], run_id="bad"
    )
    response = client.post("/runs", json=payload)
    assert response.status_code == 422
    assert ["seq"] in [error["loc"] for error in response.json()["detail"]]


def test_post_rejects_a_cost_mismatch_naming_the_field(client) -> None:
    payload = make_run(
        steps=[make_step(0, run_id="bad2", cost_usd=0.001)],
        run_id="bad2",
        reconcile_cost=False,
        total_cost_usd=9.99,
    )
    response = client.post("/runs", json=payload)
    assert response.status_code == 422
    assert ["total_cost_usd"] in [e["loc"] for e in response.json()["detail"]]


def test_post_rejects_a_bad_fault_target_naming_the_nested_field(client) -> None:
    payload = make_run(
        steps=[make_step(0, run_id="bad3")],
        run_id="bad3",
        injected_fault={
            "fault_type": "dropped_retrieval",
            "target_step_seq": 99,
            "description": "targets a step that does not exist",
        },
    )
    response = client.post("/runs", json=payload)
    assert response.status_code == 422
    assert ["injected_fault", "target_step_seq"] in [
        e["loc"] for e in response.json()["detail"]
    ]


def test_post_rejects_an_unknown_event_type_naming_the_field(client) -> None:
    payload = make_run(
        steps=[make_step(0, run_id="bad4", event_type="hallucination")], run_id="bad4"
    )
    response = client.post("/runs", json=payload)
    assert response.status_code == 422
    locs = [e["loc"] for e in response.json()["detail"]]
    assert any("event_type" in loc for loc in locs)


def test_post_rejects_a_rejection_outcome_on_a_non_critique(client) -> None:
    payload = make_run(
        steps=[
            make_step(
                0, run_id="bad5", event_type="revision", rejection_outcome="repair"
            )
        ],
        run_id="bad5",
    )
    response = client.post("/runs", json=payload)
    assert response.status_code == 422
    locs = [e["loc"] for e in response.json()["detail"]]
    assert any("rejection_outcome" in loc for loc in locs)


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


def test_health_reports_ok(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "runs" in body
