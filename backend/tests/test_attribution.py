"""Tests for failure attribution.

`is_correct()` is tested carefully because every accuracy number in the primary
study is a sum over it. A version that returned True when ground truth was
absent would inflate reported accuracy without failing anything.
"""

from __future__ import annotations

from app.extraction.attribution import (
    EXPECTED_EVIDENCE_REFS,
    RULE_WEIGHTS,
    predict_failure_origin,
)
from app.models import Run
from tests.conftest import make_run, make_step, sid

FACT = "The deep basin at Grethe Bay reaches a maximum depth of 312 metres."


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predicts_the_step_that_raised_an_error() -> None:
    run = make_run(
        steps=[
            make_step(0, event_type="plan", output="a plan"),
            make_step(
                1,
                event_type="tool_result",
                output="",
                error={"error_type": "TimeoutError", "message": "tool timed out"},
            ),
            make_step(2, event_type="final", output="unknown"),
        ],
        success=False,
        final_output="unknown",
    )
    result = predict_failure_origin(run)
    assert result.predicted_step_seq == 1
    assert "TimeoutError" in result.reason


def test_predicts_an_empty_retrieval() -> None:
    """Every dropped_retrieval fault must be reachable by this signal."""
    run = make_run(
        steps=[
            make_step(0, event_type="plan", output="a plan"),
            make_step(1, event_type="retrieval", output="", evidence_refs=[]),
            make_step(2, event_type="final", output="not found"),
        ],
        success=False,
        final_output="not found",
    )
    assert predict_failure_origin(run).predicted_step_seq == 1


def test_scores_a_thin_retrieval() -> None:
    run = make_run(
        steps=[
            make_step(
                0,
                event_type="retrieval",
                output=FACT,
                evidence_refs=["doc-08#c0"],
            ),
            make_step(1, event_type="final", output=FACT),
        ],
        success=False,
        final_output=FACT,
    )
    result = predict_failure_origin(run)
    seqs = [c.seq for c in result.candidates]
    assert 0 in seqs
    # Tracks the retriever's TOP_K, not an independent guess: a healthy retrieval
    # returns 4 chunks and dropped_retrieval removes one.
    assert EXPECTED_EVIDENCE_REFS == 4


def test_a_full_retrieval_is_not_scored_as_thin() -> None:
    run = make_run(
        steps=[
            make_step(
                0,
                event_type="retrieval",
                output=FACT,
                evidence_refs=["a", "b", "c", "d"],
            ),
            make_step(1, event_type="final", output=FACT),
        ],
        success=False,
        final_output=FACT,
    )
    thin = [
        c
        for c in predict_failure_origin(run).candidates
        if any("fewer than" in r for r in c.reasons)
    ]
    assert thin == []


def test_candidates_are_ordered_by_descending_score() -> None:
    run = make_run(
        steps=[
            make_step(0, event_type="retrieval", output="", evidence_refs=[]),
            make_step(1, event_type="critique", output="this is wrong"),
            make_step(2, event_type="final", output="an answer entirely unlike it"),
        ],
        success=False,
        final_output="an answer entirely unlike it",
    )
    scores = [c.score for c in predict_failure_origin(run).candidates]
    assert scores == sorted(scores, reverse=True)


def test_ties_resolve_to_the_earlier_step() -> None:
    """Attribution asks for the origin, so the earlier of two equally
    suspicious steps is the better answer."""
    run = make_run(
        steps=[
            make_step(0, event_type="retrieval", output="", evidence_refs=[]),
            make_step(1, event_type="retrieval", output="", evidence_refs=[]),
            make_step(2, event_type="final", output="no answer"),
        ],
        success=False,
        final_output="no answer",
    )
    result = predict_failure_origin(run)
    assert result.predicted_step_seq == 0


def test_reports_when_nothing_localises_the_failure() -> None:
    """A wrong answer with no recorded signal is an honest 'do not know', not a
    guess at step 0."""
    run = make_run(
        steps=[
            make_step(0, event_type="reasoning", output=FACT),
            make_step(1, event_type="final", output=FACT),
        ],
        success=False,
        final_output=FACT,
    )
    result = predict_failure_origin(run)
    assert result.predicted_step_id is None
    assert "does not localise" in result.reason


def test_handles_a_run_with_no_steps() -> None:
    result = predict_failure_origin(make_run(steps=[], reconcile_cost=False))
    assert result.predicted_step_id is None
    assert "no steps" in result.reason


def test_exposes_the_rule_weights() -> None:
    """NFR-7: no opaque single number."""
    result = predict_failure_origin(make_run())
    assert result.rule_weights == RULE_WEIGHTS


# ---------------------------------------------------------------------------
# Ground truth comparison
# ---------------------------------------------------------------------------


def _faulted_run(target_seq: int) -> dict:
    return make_run(
        steps=[
            make_step(0, event_type="plan", output="a plan"),
            make_step(1, event_type="retrieval", output="", evidence_refs=[]),
            make_step(2, event_type="final", output="not found"),
        ],
        success=False,
        final_output="not found",
        injected_fault={
            "fault_type": "dropped_retrieval",
            "target_step_seq": target_seq,
            "description": "removed the chunk containing the answer",
        },
    )


def test_is_correct_when_the_prediction_matches_ground_truth() -> None:
    result = predict_failure_origin(_faulted_run(1))
    assert result.actual_fault_step_seq == 1
    assert result.actual_fault_type == "dropped_retrieval"
    assert result.is_correct() is True


def test_is_not_correct_when_the_prediction_misses() -> None:
    result = predict_failure_origin(_faulted_run(0))
    assert result.predicted_step_seq == 1
    assert result.is_correct() is False


def test_is_not_correct_without_ground_truth() -> None:
    """Summing is_correct() over a population must never count an unknowable
    case as a hit."""
    run = make_run(
        steps=[
            make_step(0, event_type="retrieval", output="", evidence_refs=[]),
            make_step(1, event_type="final", output="x"),
        ],
        success=False,
        final_output="x",
    )
    result = predict_failure_origin(run)
    assert result.actual_fault_step_seq is None
    assert result.is_correct() is False


def test_is_not_correct_when_no_prediction_was_made() -> None:
    run = make_run(
        steps=[make_step(0, event_type="reasoning", output=FACT)],
        success=False,
        final_output=FACT,
        injected_fault={
            "fault_type": "injected_contradiction",
            "target_step_seq": 0,
            "description": "inserted a contradicting fact",
        },
    )
    result = predict_failure_origin(run)
    assert result.predicted_step_seq is None
    assert result.is_correct() is False


# ---------------------------------------------------------------------------
# Determinism and input forms
# ---------------------------------------------------------------------------


def test_attribution_is_deterministic() -> None:
    run = _faulted_run(1)
    assert predict_failure_origin(run) == predict_failure_origin(run)


def test_accepts_both_a_run_object_and_a_dict() -> None:
    """The Phase 3 acceptance check passes dicts straight from json.load."""
    run_dict = _faulted_run(1)
    assert predict_failure_origin(run_dict) == predict_failure_origin(
        Run.model_validate(run_dict)
    )


def test_a_retry_that_changed_nothing_is_not_credited_as_divergent() -> None:
    run = make_run(
        steps=[
            make_step(0, event_type="reasoning", output=FACT),
            make_step(1, event_type="retry", output=FACT, retry_of=sid(0)),
            make_step(2, event_type="final", output=FACT),
        ],
        success=False,
        final_output=FACT,
    )
    divergent = [
        c
        for c in predict_failure_origin(run).candidates
        if any("diverged" in r for r in c.reasons)
    ]
    assert divergent == []
