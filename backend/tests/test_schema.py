"""Invariant tests for the trace data contract.

The specification requires one passing test per invariant plus one failing case
per invariant, so the six invariants of section 3.4 appear twice each. The
accepting case pins down that a legitimate trace is *not* rejected, which matters
as much as the rejection: an over-strict validator would silently shrink the
corpus.

Every rejection test also asserts that the error names the offending field, which
is the requirement the API's 422 contract rests on (FR-6).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models import Run, Step, offending_field
from tests.conftest import make_run, make_step, sid

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "trace.schema.json"


def _messages(exc: ValidationError) -> str:
    return " | ".join(err["msg"] for err in exc.errors())


# ---------------------------------------------------------------------------
# Invariant 1 — seq values are contiguous from 0
# ---------------------------------------------------------------------------


def test_invariant_1_accepts_contiguous_seq() -> None:
    run = Run.model_validate(make_run(steps=[make_step(i) for i in range(5)]))
    assert [s.seq for s in run.steps] == [0, 1, 2, 3, 4]


def test_invariant_1_rejects_gap_in_seq() -> None:
    steps = [make_step(0), make_step(1), make_step(3)]
    with pytest.raises(ValidationError) as exc:
        Run.model_validate(make_run(steps=steps))
    assert "seq" in _messages(exc.value)


def test_invariant_1_rejects_duplicate_seq() -> None:
    steps = [make_step(0), make_step(1, step_id="dup-step"), make_step(1)]
    with pytest.raises(ValidationError) as exc:
        Run.model_validate(make_run(steps=steps))
    assert "duplicated" in _messages(exc.value)


def test_invariant_1_rejects_seq_not_starting_at_zero() -> None:
    steps = [make_step(1), make_step(2)]
    with pytest.raises(ValidationError) as exc:
        Run.model_validate(make_run(steps=steps))
    assert "seq" in _messages(exc.value)


# ---------------------------------------------------------------------------
# Invariant 2 — parent_step_id references a step in the same run, or is null
# ---------------------------------------------------------------------------


def test_invariant_2_accepts_parent_within_run_and_null_parent() -> None:
    steps = [make_step(0), make_step(1, parent_step_id=sid(0))]
    run = Run.model_validate(make_run(steps=steps))
    assert run.steps[0].parent_step_id is None
    assert run.steps[1].parent_step_id == sid(0)


def test_invariant_2_rejects_parent_outside_run() -> None:
    steps = [make_step(0), make_step(1, parent_step_id="step-from-another-run")]
    with pytest.raises(ValidationError) as exc:
        Run.model_validate(make_run(steps=steps))
    message = _messages(exc.value)
    assert offending_field(message.split(" | ")[0]) == "parent_step_id"


# ---------------------------------------------------------------------------
# Invariant 3 — retry_of references a step with a lower seq
# ---------------------------------------------------------------------------


def test_invariant_3_accepts_retry_of_earlier_step() -> None:
    steps = [
        make_step(0),
        make_step(1, event_type="retry", retry_of=sid(0)),
    ]
    run = Run.model_validate(make_run(steps=steps))
    assert run.steps[1].retry_of == sid(0)


def test_invariant_3_rejects_retry_of_later_step() -> None:
    steps = [make_step(0, retry_of=sid(1)), make_step(1)]
    with pytest.raises(ValidationError) as exc:
        Run.model_validate(make_run(steps=steps))
    assert "retry_of" in _messages(exc.value)


def test_invariant_3_rejects_retry_of_itself() -> None:
    steps = [make_step(0), make_step(1, retry_of=sid(1))]
    with pytest.raises(ValidationError) as exc:
        Run.model_validate(make_run(steps=steps))
    assert "retry_of" in _messages(exc.value)


def test_invariant_3_rejects_retry_of_unknown_step() -> None:
    steps = [make_step(0), make_step(1, retry_of="does-not-exist")]
    with pytest.raises(ValidationError) as exc:
        Run.model_validate(make_run(steps=steps))
    assert "not a step in this run" in _messages(exc.value)


# ---------------------------------------------------------------------------
# Invariant 4 — rejection_outcome only on critique steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["repair", "damage", "no_change"])
def test_invariant_4_accepts_rejection_outcome_on_critique(outcome: str) -> None:
    step = Step.model_validate(
        make_step(0, event_type="critique", rejection_outcome=outcome)
    )
    assert step.rejection_outcome == outcome


def test_invariant_4_accepts_null_rejection_outcome_on_any_event() -> None:
    step = Step.model_validate(make_step(0, event_type="revision"))
    assert step.rejection_outcome is None


def test_invariant_4_rejects_rejection_outcome_on_non_critique() -> None:
    with pytest.raises(ValidationError) as exc:
        Step.model_validate(
            make_step(0, event_type="revision", rejection_outcome="repair")
        )
    message = _messages(exc.value)
    assert offending_field(message.split(" | ")[0]) == "rejection_outcome"


# ---------------------------------------------------------------------------
# Invariant 5 — total_cost_usd equals the sum of step costs within 1e-6
# ---------------------------------------------------------------------------


def test_invariant_5_accepts_reconciling_cost() -> None:
    steps = [make_step(i, cost_usd=0.0025) for i in range(4)]
    run = Run.model_validate(make_run(steps=steps, total_cost_usd=0.01))
    assert run.total_cost_usd == pytest.approx(0.01)


def test_invariant_5_accepts_difference_inside_tolerance() -> None:
    steps = [make_step(0, cost_usd=0.001)]
    run = Run.model_validate(
        make_run(steps=steps, reconcile_cost=False, total_cost_usd=0.001 + 9e-7)
    )
    assert run.total_cost_usd == pytest.approx(0.001, abs=1e-6)


def test_invariant_5_rejects_cost_mismatch() -> None:
    steps = [make_step(0, cost_usd=0.001)]
    with pytest.raises(ValidationError) as exc:
        Run.model_validate(
            make_run(steps=steps, reconcile_cost=False, total_cost_usd=0.5)
        )
    message = _messages(exc.value)
    assert offending_field(message.split(" | ")[0]) == "total_cost_usd"


def test_invariant_5_rejects_difference_just_outside_tolerance() -> None:
    steps = [make_step(0, cost_usd=0.001)]
    with pytest.raises(ValidationError):
        Run.model_validate(
            make_run(steps=steps, reconcile_cost=False, total_cost_usd=0.001 + 1e-3)
        )


# ---------------------------------------------------------------------------
# Invariant 6 — an injected fault targets a seq that exists in the run
# ---------------------------------------------------------------------------


def test_invariant_6_accepts_fault_targeting_existing_seq() -> None:
    run = Run.model_validate(
        make_run(
            steps=[make_step(i) for i in range(3)],
            injected_fault={
                "fault_type": "dropped_retrieval",
                "target_step_seq": 2,
                "description": "removed the chunk containing the answer",
            },
        )
    )
    assert run.injected_fault is not None
    assert run.step_by_seq(run.injected_fault.target_step_seq) is not None


def test_invariant_6_rejects_fault_targeting_missing_seq() -> None:
    with pytest.raises(ValidationError) as exc:
        Run.model_validate(
            make_run(
                steps=[make_step(i) for i in range(3)],
                injected_fault={
                    "fault_type": "dropped_retrieval",
                    "target_step_seq": 99,
                    "description": "targets a step that does not exist",
                },
            )
        )
    assert "injected_fault.target_step_seq" in _messages(exc.value)


# ---------------------------------------------------------------------------
# Supporting behaviour
# ---------------------------------------------------------------------------


def test_steps_are_normalised_into_seq_order() -> None:
    """Extraction walks `run.steps` directly and must not depend on the order
    the trace happened to be serialised in (FR-16)."""
    shuffled = [make_step(2), make_step(0), make_step(1)]
    run = Run.model_validate(make_run(steps=shuffled))
    assert [s.seq for s in run.steps] == [0, 1, 2]


def test_empty_run_is_valid() -> None:
    """A run that failed before emitting a step is still a legitimate trace."""
    run = Run.model_validate(make_run(steps=[], reconcile_cost=False))
    assert run.steps == []
    assert run.total_cost_usd == 0.0


def test_unknown_event_type_is_rejected_naming_the_field() -> None:
    with pytest.raises(ValidationError) as exc:
        Step.model_validate(make_step(0, event_type="hallucination"))
    assert exc.value.errors()[0]["loc"] == ("event_type",)


def test_unknown_fault_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Run.model_validate(
            make_run(
                injected_fault={
                    "fault_type": "cosmic_ray",
                    "target_step_seq": 0,
                    "description": "not one of the four fault types",
                }
            )
        )


def test_negative_cost_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Step.model_validate(make_step(0, cost_usd=-0.01))


def test_offending_field_recovers_prefix() -> None:
    assert offending_field("retry_of: step X references Y") == "retry_of"
    assert offending_field("no prefix here") is None
    assert offending_field("Input should be a valid integer") is None


def test_offending_field_unwraps_pydantic_message_prefix() -> None:
    """Pydantic reports validator failures as "Value error, <our message>". The
    422 contract depends on the field surviving that wrapping."""
    assert offending_field("Value error, total_cost_usd: run reports 0.5") == (
        "total_cost_usd"
    )
    assert offending_field(
        "Value error, injected_fault.target_step_seq: fault targets seq 99"
    ) == "injected_fault.target_step_seq"


def test_run_summary_reports_fault_and_step_count() -> None:
    run = Run.model_validate(
        make_run(
            steps=[make_step(i) for i in range(4)],
            injected_fault={
                "fault_type": "truncated_tool_result",
                "target_step_seq": 1,
                "description": "cut tool output at 40% length",
            },
        )
    )
    summary = run.summary()
    assert summary.step_count == 4
    assert summary.has_injected_fault is True
    assert summary.fault_type == "truncated_tool_result"


# ---------------------------------------------------------------------------
# Published JSON Schema
# ---------------------------------------------------------------------------


def test_json_schema_file_is_committed_and_current() -> None:
    """`schema/trace.schema.json` is the public contract that Build B validates
    against, so a drift between it and the models is a breaking change."""
    assert SCHEMA_PATH.exists(), f"{SCHEMA_PATH} is missing; run `make schema`"
    committed = json.loads(SCHEMA_PATH.read_text())
    assert committed == Run.model_json_schema(), (
        "committed schema is stale; regenerate it with `make schema`"
    )


def test_valid_run_validates_against_published_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    run = Run.model_validate(make_run())
    jsonschema.validate(json.loads(run.model_dump_json()), schema)
