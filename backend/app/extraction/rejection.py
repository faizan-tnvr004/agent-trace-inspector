"""Rejection outcome: what a critique actually caused downstream.

This is the taxonomy from the author's prior study, carried into the tool as a
per-critique classification:

* **repair** — the answer was wrong before the critique and right after
* **damage** — the answer was right before and wrong after
* **no_change** — there was no revision, or the revision changed nothing

The distinction matters because of what the prior work found: a reviewer that
detected errors at the highest rate of any condition produced no accuracy gain,
and its apparent safety turned out to be an artifact of the executor ignoring
bad critiques rather than of good reviewing. A high `no_change` rate is that
phenomenon made visible: review that is performed but not consequential.

Classification requires ground truth. Without a known correct answer there is no
way to distinguish a repair from damage, so `classify_rejection` returns None
rather than guessing.
"""

from __future__ import annotations

from typing import Any, Literal

from app.extraction.embeddings import cosine_similarity
from app.extraction.scoring import as_run
from app.models import Run, Step

__all__ = [
    "UNCHANGED_SIMILARITY",
    "classify_rejection",
    "classify_all_rejections",
    "is_rejection",
    "rejection_summary",
]

# Revisions at or above this similarity to the pre-critique answer are treated
# as having changed nothing of substance. Rewording is not a change of outcome.
UNCHANGED_SIMILARITY = 0.98

RejectionOutcome = Literal["repair", "damage", "no_change"]

# Event types whose output represents "the answer as it currently stands".
_ANSWER_EVENTS = frozenset({"reasoning", "revision"})


def is_rejection(critique_step: Step) -> bool:
    """Whether this critique actually rejected the answer it reviewed.

    The specification defines `no_change` to cover a critique with no following
    revision, which is correct as written but lumps two different things
    together: a reviewer that *approved* the answer, and a reviewer that
    objected and was ignored. Only the second is a rejection whose outcome the
    taxonomy was built to classify.

    Reporting the two together makes the no-change rate look enormous simply
    because most critiques are acceptances. Callers that want the population the
    prior study actually coded should filter on this.

    Read from the reviewer's own verdict line, not inferred from tone.
    """
    text = critique_step.output.upper()
    if "VERDICT:" in text:
        _, _, after = text.partition("VERDICT:")
        return "REJECT" in after.split("\n", 1)[0]
    return "REJECT" in text


def _answer_before(run: Run, critique: Step) -> Step | None:
    """The step holding the answer the critique was reviewing."""
    prior = [
        s for s in run.steps if s.seq < critique.seq and s.event_type in _ANSWER_EVENTS
    ]
    return prior[-1] if prior else None


def _revision_after(run: Run, critique: Step) -> Step | None:
    """The revision this critique triggered, if any.

    Only revisions before the *next* critique count, so in a multi-round review
    each critique is paired with its own revision rather than with a later one.
    """
    next_critique_seq = min(
        (
            s.seq
            for s in run.steps
            if s.event_type == "critique" and s.seq > critique.seq
        ),
        default=None,
    )
    for step in run.steps:
        if step.seq <= critique.seq or step.event_type != "revision":
            continue
        if next_critique_seq is not None and step.seq > next_critique_seq:
            break
        return step
    return None


def classify_rejection(
    critique_step: Step, run: Run | dict[str, Any]
) -> RejectionOutcome | None:
    """Classify a critique's downstream effect.

    Returns None when ``run.ground_truth`` is absent, since repair and damage
    are not distinguishable without it.
    """
    resolved = as_run(run)
    if not resolved.ground_truth:
        return None

    # Imported here rather than at module scope: the harness imports grading,
    # and importing it the other way round at module level would make the
    # extraction package depend on the harness package.
    from harness.grading import answer_matches

    revision = _revision_after(resolved, critique_step)
    if revision is None:
        return "no_change"

    before = _answer_before(resolved, critique_step)
    before_text = before.output if before is not None else ""

    if (
        before_text
        and cosine_similarity(before_text, revision.output) >= UNCHANGED_SIMILARITY
    ):
        return "no_change"

    was_right = answer_matches(resolved.ground_truth, before_text)
    now_right = answer_matches(resolved.ground_truth, revision.output)

    if not was_right and now_right:
        return "repair"
    if was_right and not now_right:
        return "damage"
    return "no_change"


def classify_all_rejections(
    run: Run | dict[str, Any],
) -> dict[str, RejectionOutcome | None]:
    """Outcome per critique step, keyed by step id."""
    resolved = as_run(run)
    return {
        step.step_id: classify_rejection(step, resolved)
        for step in resolved.steps
        if step.event_type == "critique"
    }


def rejection_summary(runs: list[Run | dict[str, Any]]) -> dict[str, Any]:
    """Aggregate outcome counts across runs.

    Reports two populations, because they answer different questions:

    * ``all_critiques`` — every critique, using the specification's definition
    * ``rejections_only`` — critiques that actually returned a REJECT verdict,
      which is the population the prior study coded and the only one where a
      no-change rate means "the executor ignored the criticism"
    """
    template = {"repair": 0, "damage": 0, "no_change": 0, "unclassified": 0}
    overall = dict(template)
    rejections = dict(template)

    for run in runs:
        resolved = as_run(run)
        outcomes = classify_all_rejections(resolved)
        for step in resolved.steps:
            if step.event_type != "critique":
                continue
            key = outcomes.get(step.step_id) or "unclassified"
            overall[key] += 1
            if is_rejection(step):
                rejections[key] += 1

    def rates(counts: dict[str, int]) -> dict[str, float]:
        total = sum(counts.values())
        return {
            outcome: round(counts[outcome] / total, 4) if total else 0.0
            for outcome in ("repair", "damage", "no_change")
        }

    return {
        "all_critiques": {
            "counts": overall,
            "total": sum(overall.values()),
            "rates": rates(overall),
        },
        "rejections_only": {
            "counts": rejections,
            "total": sum(rejections.values()),
            "rates": rates(rejections),
        },
    }
