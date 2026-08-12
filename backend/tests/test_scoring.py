"""Tests for critical-step scoring.

Determinism is tested explicitly rather than assumed. It is a stated
requirement (FR-16, NFR-6), and it is the kind of property that breaks silently:
an unstable sort or an unrounded float would still produce plausible rankings
while making two identical inputs disagree.
"""

from __future__ import annotations

import pytest

from app.extraction.scoring import (
    SHORT_TOOL_RESULT_CHARS,
    WEIGHTS,
    critical_score,
    error_signal,
    evidence_survival,
    is_branch_point,
    rank_critical,
)
from app.models import Run, Step
from tests.conftest import make_run, make_step, sid


def _step(**overrides) -> Step:
    return Step.model_validate(make_step(overrides.pop("seq", 0), **overrides))


# ---------------------------------------------------------------------------
# Evidence survival
# ---------------------------------------------------------------------------


def test_evidence_survival_is_one_for_identical_text() -> None:
    step = _step(output="The answer is 96 loaves.")
    assert evidence_survival(step, "The answer is 96 loaves.") == 1.0


def test_evidence_survival_is_high_for_a_paraphrase() -> None:
    step = _step(output="The answer is 96 loaves.")
    assert evidence_survival(step, "There are 96 loaves in total.") > 0.6


def test_evidence_survival_is_low_for_unrelated_text() -> None:
    step = _step(output="The answer is 96 loaves.")
    assert evidence_survival(step, "Seismometers are buried three metres down.") < 0.3


def test_evidence_survival_is_bounded() -> None:
    step = _step(output="anything at all")
    for final in ["anything at all", "", "utterly different subject matter"]:
        assert 0.0 <= evidence_survival(step, final) <= 1.0


def test_evidence_survival_is_zero_for_empty_output() -> None:
    """An empty output cannot have survived into the final answer."""
    assert evidence_survival(_step(output=""), "the answer is 96") == 0.0


# ---------------------------------------------------------------------------
# Branch points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event_type", ["retry", "critique", "revision"])
def test_branch_point_flags_the_declared_event_types(event_type: str) -> None:
    run = Run.model_validate(make_run(steps=[make_step(0, event_type=event_type)]))
    assert is_branch_point(run.steps[0], run) is True


def test_branch_point_flags_a_step_that_retries_another() -> None:
    run = Run.model_validate(
        make_run(steps=[make_step(0), make_step(1, retry_of=sid(0))])
    )
    assert is_branch_point(run.steps[1], run) is True


def test_branch_point_flags_a_changed_plan() -> None:
    run = Run.model_validate(
        make_run(
            steps=[
                make_step(0, event_type="plan", output="use the calculator"),
                make_step(1, event_type="plan", output="solve it by hand instead"),
            ]
        )
    )
    assert is_branch_point(run.steps[1], run) is True


def test_branch_point_ignores_an_unchanged_plan() -> None:
    run = Run.model_validate(
        make_run(
            steps=[
                make_step(0, event_type="plan", output="use the calculator"),
                make_step(1, event_type="plan", output="use the calculator"),
            ]
        )
    )
    assert is_branch_point(run.steps[1], run) is False


def test_first_plan_is_not_a_branch_point() -> None:
    """With nothing to differ from, a first plan is not a divergence."""
    run = Run.model_validate(
        make_run(steps=[make_step(0, event_type="plan", output="a plan")])
    )
    assert is_branch_point(run.steps[0], run) is False


def test_ordinary_reasoning_is_not_a_branch_point() -> None:
    run = Run.model_validate(make_run(steps=[make_step(0, event_type="reasoning")]))
    assert is_branch_point(run.steps[0], run) is False


# ---------------------------------------------------------------------------
# Error signal
# ---------------------------------------------------------------------------


def test_error_signal_is_one_when_the_step_raised() -> None:
    step = _step(error={"error_type": "TimeoutError", "message": "timed out"})
    assert error_signal(step) == 1.0


def test_error_signal_is_one_for_a_retrieval_with_no_references() -> None:
    """This is the signal that must catch every dropped_retrieval fault
    (FR-11)."""
    assert error_signal(_step(event_type="retrieval", evidence_refs=[])) == 1.0


def test_error_signal_is_zero_for_a_retrieval_that_returned_references() -> None:
    step = _step(event_type="retrieval", evidence_refs=["doc-1#c0"])
    assert error_signal(step) == 0.0


def test_error_signal_is_half_for_a_short_tool_result() -> None:
    step = _step(event_type="tool_result", output="x" * (SHORT_TOOL_RESULT_CHARS - 1))
    assert error_signal(step) == 0.5


def test_error_signal_is_zero_for_a_full_length_tool_result() -> None:
    step = _step(event_type="tool_result", output="x" * SHORT_TOOL_RESULT_CHARS)
    assert error_signal(step) == 0.0


def test_error_signal_is_zero_for_an_ordinary_step() -> None:
    assert error_signal(_step(event_type="reasoning", output="fine")) == 0.0


# ---------------------------------------------------------------------------
# Critical score
# ---------------------------------------------------------------------------


def test_critical_score_returns_components_never_a_bare_float() -> None:
    """NFR-7: every score shown must decompose into its contributing signals."""
    run = Run.model_validate(make_run(steps=[make_step(0, event_type="critique")]))
    breakdown = critical_score(run.steps[0], run)

    assert not isinstance(breakdown, float)
    assert breakdown.weights == WEIGHTS
    assert breakdown.branch == 1.0
    assert breakdown.reasons, "a scored step must explain itself"


def test_critical_score_equals_the_weighted_sum_of_its_parts() -> None:
    run = Run.model_validate(
        make_run(
            steps=[
                make_step(
                    0,
                    event_type="critique",
                    output="The reasoning is wrong.",
                    error={"error_type": "ValueError", "message": "bad"},
                )
            ],
            final_output="The reasoning is wrong.",
        )
    )
    b = critical_score(run.steps[0], run)
    expected = (
        WEIGHTS["evidence_survival"] * b.evidence_survival
        + WEIGHTS["branch"] * b.branch
        + WEIGHTS["error"] * b.error
    )
    assert b.critical_score == pytest.approx(expected)


def test_critical_score_is_bounded_by_the_sum_of_weights() -> None:
    run = Run.model_validate(
        make_run(
            steps=[
                make_step(
                    0,
                    event_type="critique",
                    output="identical",
                    error={"error_type": "E", "message": "m"},
                )
            ],
            final_output="identical",
        )
    )
    assert critical_score(run.steps[0], run).critical_score <= sum(WEIGHTS.values())


def test_critical_score_accepts_a_run_dict() -> None:
    """Corpus files are read with json.load, so dicts must work everywhere."""
    run_dict = make_run(steps=[make_step(0, event_type="critique")])
    run = Run.model_validate(run_dict)
    assert critical_score(run.steps[0], run_dict) == critical_score(run.steps[0], run)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_rank_critical_returns_at_most_k_steps() -> None:
    run = make_run(steps=[make_step(i) for i in range(9)])
    assert len(rank_critical(run, k=3)) == 3
    assert len(rank_critical(run, k=50)) == 9
    assert rank_critical(run, k=0) == []


def test_rank_critical_orders_by_descending_score() -> None:
    run = make_run(
        steps=[
            make_step(0, event_type="reasoning", output="irrelevant filler"),
            make_step(1, event_type="critique", output="the answer is wrong"),
            make_step(2, event_type="retrieval", evidence_refs=[]),
        ],
        final_output="the answer is wrong",
    )
    scores = [s.critical_score for s in rank_critical(run)]
    assert scores == sorted(scores, reverse=True)


def test_rank_critical_breaks_ties_on_seq() -> None:
    """Equal scores must return in a fixed order or extraction is not
    deterministic."""
    run = make_run(
        steps=[make_step(i, event_type="reasoning", output="same text") for i in range(4)]
    )
    ranked = rank_critical(run, k=4)
    tied = [s.seq for s in ranked if s.critical_score == ranked[0].critical_score]
    assert tied == sorted(tied)


def test_rank_critical_is_deterministic() -> None:
    """The Phase 3 acceptance check asserts exactly this."""
    run = make_run(
        steps=[
            make_step(0, event_type="plan", output="plan a"),
            make_step(1, event_type="critique", output="wrong"),
            make_step(2, event_type="retrieval", evidence_refs=[]),
            make_step(3, event_type="final", output="the answer is 96"),
        ],
        final_output="the answer is 96",
    )
    assert rank_critical(run) == rank_critical(run)


def test_rank_critical_on_an_empty_run() -> None:
    assert rank_critical(make_run(steps=[], reconcile_cost=False)) == []
