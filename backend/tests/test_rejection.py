"""Tests for the rejection-outcome taxonomy (FR-15).

The three outcomes are the substance of the author's prior study, so each is
tested with the trace shape that produces it. `no_change` gets the most
attention because it is the outcome that finding turned on: review that happens
but changes nothing.
"""

from __future__ import annotations

from app.extraction.rejection import (
    classify_all_rejections,
    classify_rejection,
    is_rejection,
    rejection_summary,
)
from app.models import Run, Step
from tests.conftest import make_run, make_step

WRONG = "Working through it, the speed is 90 km/h so the distance is 567 km."
RIGHT = "Speed is 240/3 = 80 km/h, so over 7 hours the distance is 560 km."


def _pipeline(before: str, after: str | None, *, truth: str = "560") -> Run:
    steps = [
        make_step(0, event_type="reasoning", output=before),
        make_step(1, event_type="critique", output="This looks wrong. VERDICT: REJECT"),
    ]
    if after is not None:
        steps.append(make_step(2, event_type="revision", output=after))
    steps.append(
        make_step(len(steps), event_type="final", output=after if after else before)
    )
    return Run.model_validate(
        make_run(
            steps=steps,
            ground_truth=truth,
            final_output=after if after else before,
        )
    )


def _critique(run: Run):
    return next(s for s in run.steps if s.event_type == "critique")


# ---------------------------------------------------------------------------
# The three outcomes
# ---------------------------------------------------------------------------


def test_repair_when_a_wrong_answer_becomes_right() -> None:
    run = _pipeline(WRONG, RIGHT)
    assert classify_rejection(_critique(run), run) == "repair"


def test_damage_when_a_right_answer_becomes_wrong() -> None:
    run = _pipeline(RIGHT, WRONG)
    assert classify_rejection(_critique(run), run) == "damage"


def test_no_change_when_there_is_no_revision() -> None:
    """The executor ignored the critique outright."""
    run = _pipeline(RIGHT, None)
    assert classify_rejection(_critique(run), run) == "no_change"


def test_no_change_when_the_revision_restates_the_same_answer() -> None:
    """Rewording is not a change of outcome. This is the case that made the
    prior study's reviewer look safe while being inert."""
    run = _pipeline(RIGHT, RIGHT)
    assert classify_rejection(_critique(run), run) == "no_change"


def test_no_change_when_a_wrong_answer_stays_wrong() -> None:
    run = _pipeline(WRONG, "The distance must be 567 km after rechecking it.")
    assert classify_rejection(_critique(run), run) == "no_change"


def test_no_change_when_a_right_answer_is_rephrased_but_still_right() -> None:
    run = _pipeline(RIGHT, "The train covers 560 km in seven hours at 80 km/h.")
    assert classify_rejection(_critique(run), run) == "no_change"


# ---------------------------------------------------------------------------
# Ground truth requirement
# ---------------------------------------------------------------------------


def test_returns_none_without_ground_truth() -> None:
    """Repair and damage are not distinguishable without a known answer, so the
    classifier declines rather than guessing."""
    run = Run.model_validate(
        make_run(
            steps=[
                make_step(0, event_type="reasoning", output=WRONG),
                make_step(1, event_type="critique", output="wrong. REJECT"),
                make_step(2, event_type="revision", output=RIGHT),
            ],
            ground_truth=None,
            final_output=RIGHT,
        )
    )
    assert classify_rejection(_critique(run), run) is None


# ---------------------------------------------------------------------------
# Multi-round review
# ---------------------------------------------------------------------------


def test_each_critique_is_paired_with_its_own_revision() -> None:
    """In a two-round review the first critique must not be credited with the
    second round's revision."""
    run = Run.model_validate(
        make_run(
            steps=[
                make_step(0, event_type="reasoning", output=WRONG),
                make_step(1, event_type="critique", output="wrong. REJECT"),
                make_step(2, event_type="revision", output=WRONG),
                make_step(3, event_type="critique", output="still wrong. REJECT"),
                make_step(4, event_type="revision", output=RIGHT),
                make_step(5, event_type="final", output=RIGHT),
            ],
            ground_truth="560",
            final_output=RIGHT,
        )
    )
    critiques = [s for s in run.steps if s.event_type == "critique"]
    assert classify_rejection(critiques[0], run) == "no_change"
    assert classify_rejection(critiques[1], run) == "repair"


def test_classify_all_rejections_covers_every_critique() -> None:
    run = _pipeline(WRONG, RIGHT)
    outcomes = classify_all_rejections(run)
    assert len(outcomes) == 1
    assert list(outcomes.values()) == ["repair"]


def test_classify_all_rejections_is_empty_without_critiques() -> None:
    run = Run.model_validate(
        make_run(steps=[make_step(0, event_type="retrieval", evidence_refs=["a"])])
    )
    assert classify_all_rejections(run) == {}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_rejection_summary_counts_across_runs() -> None:
    summary = rejection_summary(
        [_pipeline(WRONG, RIGHT), _pipeline(RIGHT, WRONG), _pipeline(RIGHT, None)]
    )
    counts = summary["all_critiques"]["counts"]
    assert counts["repair"] == 1
    assert counts["damage"] == 1
    assert counts["no_change"] == 1
    assert counts["unclassified"] == 0
    assert summary["all_critiques"]["total"] == 3


def test_rejection_summary_separates_rejections_from_acceptances() -> None:
    """An accepting critique is not a rejection. Blending the two makes the
    no-change rate a statistic about how often reviewers approve, not about how
    often criticism was ignored, which is what the taxonomy is for."""
    accepting = Run.model_validate(
        make_run(
            steps=[
                make_step(0, event_type="reasoning", output=RIGHT),
                make_step(
                    1, event_type="critique", output="Looks correct. VERDICT: ACCEPT"
                ),
                make_step(2, event_type="final", output=RIGHT),
            ],
            ground_truth="560",
            final_output=RIGHT,
        )
    )
    summary = rejection_summary([_pipeline(WRONG, RIGHT), accepting])

    # Both critiques are classified, but only one of them rejected anything.
    assert summary["all_critiques"]["total"] == 2
    assert summary["rejections_only"]["total"] == 1
    assert summary["rejections_only"]["counts"]["repair"] == 1
    assert summary["rejections_only"]["rates"]["repair"] == 1.0


def test_is_rejection_reads_the_verdict_line_not_the_tone() -> None:
    rejecting = Step.model_validate(
        make_step(
            0,
            event_type="critique",
            output="This is dreadful and sloppy. VERDICT: ACCEPT",
        )
    )
    accepting = Step.model_validate(
        make_step(
            0, event_type="critique", output="Fine work. VERDICT: REJECT"
        )
    )
    # Verdict wins over sentiment in both directions.
    assert is_rejection(rejecting) is False
    assert is_rejection(accepting) is True


def test_classification_is_deterministic() -> None:
    run = _pipeline(WRONG, RIGHT)
    critique = _critique(run)
    assert classify_rejection(critique, run) == classify_rejection(critique, run)
