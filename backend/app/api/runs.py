"""All ``/runs`` endpoints.

The backend is deliberately thin. It stores traces and serves extractions
computed by `app.extraction`; it holds no analysis logic of its own. That
separation is what lets the extraction engine be unit-tested independently of
the UI (NFR-9) and keeps every number the frontend shows reproducible from a
script.

Error contract:

* ``404`` for an unknown ``run_id``
* ``422`` naming the offending field for a trace that violates the schema

The 422 case needs care. Pydantic reports failures from model-level validators
with an empty ``loc``, so the six trace invariants would otherwise produce a
422 that says only "Value error". `offending_field` recovers the field name from
the message, and the handler below puts it in ``loc`` where a client expects it.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import ValidationError

from app.db import get_run, insert_run, list_runs
from app.deps import get_conn
from app.extraction.attribution import predict_failure_origin
from app.extraction.claims import analyse_claims
from app.extraction.rejection import classify_all_rejections
from app.extraction.scoring import rank_critical
from app.models import Run, offending_field

router = APIRouter(prefix="/runs", tags=["runs"])

Conn = Annotated[sqlite3.Connection, Depends(get_conn)]

# Schema contract version reported in the audit export, so an exported bundle
# says which shape of trace it was produced from.
EXPORT_FORMAT_VERSION = "1.0.0"


def _require_run(conn: sqlite3.Connection, run_id: str) -> Run:
    run = get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return run


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    """Turn a ValidationError into a 422 body that always names a field.

    A field name recovered from the message is appended to ``loc`` rather than
    only substituted when ``loc`` is empty. Run-level validators report an empty
    ``loc``, but a step-level validator inside the steps list reports
    ``("steps", 3)`` and names the field only in the message, so substitution
    alone would leave the caller knowing which step failed but not which field.
    """
    detail: list[dict[str, Any]] = []
    for error in exc.errors():
        loc = [str(part) for part in error["loc"]]
        recovered = offending_field(error["msg"])
        if recovered:
            parts = recovered.split(".")
            if loc[-len(parts) :] != parts:
                loc = [*loc, *parts]
        detail.append(
            {
                "loc": loc or ["__root__"],
                "msg": error["msg"],
                "type": error["type"],
            }
        )
    return detail


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


@router.get("")
def list_runs_endpoint(
    conn: Conn,
    success: bool | None = Query(None, description="filter on run outcome"),
    workflow_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    items, total = list_runs(
        conn,
        success=success,
        workflow_type=workflow_type,
        limit=limit,
        offset=offset,
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.post("", status_code=201)
def create_run(conn: Conn, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Accept an uploaded trace, validate it, store it.

    Validation is done explicitly rather than by declaring `Run` as the body
    type, so the 422 body can be given the offending field name.
    """
    try:
        run = Run.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=_validation_detail(exc)
        ) from exc

    insert_run(conn, run)
    return {"run_id": run.run_id, "steps": len(run.steps)}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


@router.get("/{run_id}")
def get_run_endpoint(conn: Conn, run_id: str) -> Run:
    return _require_run(conn, run_id)


@router.get("/{run_id}/critical")
def get_critical(
    conn: Conn, run_id: str, k: int = Query(5, ge=0, le=200)
) -> dict[str, Any]:
    run = _require_run(conn, run_id)
    return {"steps": rank_critical(run, k=k), "k": k, "step_count": len(run.steps)}


@router.get("/{run_id}/attribution")
def get_attribution(conn: Conn, run_id: str) -> Any:
    return predict_failure_origin(_require_run(conn, run_id))


@router.get("/{run_id}/provenance")
def get_provenance(conn: Conn, run_id: str) -> dict[str, Any]:
    run = _require_run(conn, run_id)
    claims = analyse_claims(run)
    unsupported = sum(1 for c in claims if not c.supported)
    return {
        "claims": claims,
        "total": len(claims),
        "unsupported": unsupported,
        "final_output": run.final_output,
    }


@router.get("/{run_id}/cost")
def get_cost(conn: Conn, run_id: str) -> dict[str, Any]:
    """Cost, tokens and latency aggregated per agent and per event type.

    Totals are summed from the same steps as the breakdowns, so the view can
    reconcile against ``run.total_cost_usd`` rather than trusting it.
    """
    run = _require_run(conn, run_id)

    def bucket() -> dict[str, Any]:
        return {
            "cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
            "steps": 0,
        }

    by_agent: dict[str, dict[str, Any]] = {}
    by_event_type: dict[str, dict[str, Any]] = {}
    total = bucket()

    for step in run.steps:
        for target in (
            by_agent.setdefault(step.agent_id, bucket()),
            by_event_type.setdefault(step.event_type, bucket()),
            total,
        ):
            target["cost_usd"] += step.cost_usd
            target["prompt_tokens"] += step.prompt_tokens
            target["completion_tokens"] += step.completion_tokens
            target["total_tokens"] += step.prompt_tokens + step.completion_tokens
            target["latency_ms"] += step.latency_ms
            target["steps"] += 1

    for group in (*by_agent.values(), *by_event_type.values(), total):
        group["cost_usd"] = round(group["cost_usd"], 8)

    return {
        "by_agent": by_agent,
        "by_event_type": by_event_type,
        "total": total,
        "run_total_cost_usd": run.total_cost_usd,
        "reconciles": abs(total["cost_usd"] - run.total_cost_usd) < 1e-6,
        # Generation ran on a free tier, so no money was spent. Costs are
        # computed from published list prices; see the README.
        "cost_basis": "notional, from published list prices",
    }


@router.get("/{run_id}/export")
def export_run(conn: Conn, run_id: str) -> dict[str, Any]:
    """A self-contained audit bundle.

    Everything an auditor needs in one document: the trace, what the extraction
    engine concluded, the weights it used, and the ground truth where it exists.
    No further requests are required to check the reasoning.
    """
    run = _require_run(conn, run_id)
    attribution = predict_failure_origin(run)
    claims = analyse_claims(run)

    return {
        "export_format_version": EXPORT_FORMAT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "critical_steps": rank_critical(run, k=len(run.steps) or 1),
        "attribution": attribution,
        "provenance": {
            "claims": claims,
            "total": len(claims),
            "unsupported": sum(1 for c in claims if not c.supported),
        },
        "rejection_outcomes": classify_all_rejections(run),
        "cost": get_cost(conn, run_id),
        "ground_truth": {
            "expected_answer": run.ground_truth,
            "injected_fault": run.injected_fault,
        },
    }
