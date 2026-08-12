"""Shared trace builders.

Tests construct traces as plain dicts rather than model instances so that a test
can corrupt a single field and then assert that validation rejects it. Building
a ``Step`` object first would trip the validators before the corruption could be
applied.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
RUN_ID = "run-0001"


def sid(seq: int, run_id: str = RUN_ID) -> str:
    """The step_id the builder assigns at ``seq``.

    Real step ids are uuid4 and globally unique, and the store enforces that
    with a PRIMARY KEY, so fixture ids are namespaced by run to match.
    """
    return f"{run_id}-step-{seq:04d}"


def make_step(seq: int, run_id: str = RUN_ID, **overrides: Any) -> dict[str, Any]:
    """A valid step at ``seq``. Any field can be overridden or corrupted."""
    step: dict[str, Any] = {
        "step_id": sid(seq, run_id),
        "run_id": run_id,
        "parent_step_id": None,
        "seq": seq,
        "agent_id": "executor-1",
        "agent_role": "executor",
        "model": "gemini-2.0-flash",
        "event_type": "reasoning",
        "input": f"input for step {seq}",
        "output": f"output for step {seq}",
        "timestamp": (BASE_TIME + timedelta(seconds=seq)).isoformat(),
        "latency_ms": 100 + seq,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_usd": 0.001,
        "evidence_refs": [],
        "error": None,
        "retry_of": None,
        "rejection_outcome": None,
    }
    step.update(overrides)
    return step


def make_run(
    steps: list[dict[str, Any]] | None = None,
    *,
    reconcile_cost: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """A valid run. ``reconcile_cost`` recomputes totals from the steps."""
    if steps is None:
        steps = [make_step(i) for i in range(3)]

    run: dict[str, Any] = {
        "run_id": RUN_ID,
        "workflow_type": "reviewer_pipeline",
        "workflow_version": "1.0.0",
        "task_input": "What is 2 + 2?",
        "final_output": "The answer is 4.",
        "success": True,
        "ground_truth": "4",
        "injected_fault": None,
        "started_at": BASE_TIME.isoformat(),
        "completed_at": (BASE_TIME + timedelta(seconds=30)).isoformat(),
        "total_cost_usd": 0.0,
        "total_tokens": 0,
        "steps": steps,
    }
    if reconcile_cost:
        run["total_cost_usd"] = round(sum(s["cost_usd"] for s in steps), 12)
        run["total_tokens"] = sum(
            s["prompt_tokens"] + s["completion_tokens"] for s in steps
        )
    run.update(overrides)
    return run


@pytest.fixture
def valid_run() -> dict[str, Any]:
    return make_run()


@pytest.fixture
def db_conn():
    """An in-memory database with the schema applied."""
    from app.db import connect, init_db

    conn = connect(":memory:")
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()
