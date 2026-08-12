"""Failure attribution: predicting where a failed run first went wrong.

Error origin is distinct from error manifestation. A run produces a wrong final
answer at the last step, but the cause may be a retrieval three steps earlier
that returned nothing. This module walks a trace in ``seq`` order and scores
each step as a candidate origin.

The scoring rules and their weights come from the specification and were fixed
before the primary study was run. They have not been adjusted since. Tuning them
against the evaluation result would be fitting to the test set, and the whole
point of the injected-fault corpus is that the answer is known in advance.

Where ground truth exists, `AttributionResult.is_correct()` compares the
prediction against `injected_fault.target_step_seq`. There is no minimum
accuracy target. If attribution performs near chance that is a finding to
report, not a defect to hide.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.extraction.claims import analyse_claims
from app.extraction.embeddings import PRECISION, cosine_similarity
from app.extraction.scoring import as_run, error_signal, is_branch_point
from app.models import Run, Step

__all__ = [
    "RULE_WEIGHTS",
    "AttributionCandidate",
    "AttributionResult",
    "predict_failure_origin",
]

# Fixed before Phase 6. Do not tune against evaluation results.
RULE_WEIGHTS: dict[str, float] = {
    "error_signal": 3.0,
    "first_unsupported_claim": 2.0,
    "divergent_branch": 1.5,
    "thin_retrieval": 1.0,
}

# A retrieval returning fewer than this many references is treated as thin.
#
# This is the retriever's configured `TOP_K` (see `harness/workflows/rag_qa.py`),
# not an independent guess. "Fewer than expected" only means anything relative to
# how many chunks the retriever was asked for: a healthy retrieval returns 4, and
# `dropped_retrieval` removes one, leaving 3. An earlier value of 3 here made the
# rule unfireable for exactly the fault it exists to catch, which was found while
# diagnosing why attribution scored 0/33. The rule *weights* are unchanged.
EXPECTED_EVIDENCE_REFS = 4

# Below this similarity, the steps downstream of a branch are taken to have
# diverged from it.
DIVERGENCE_SIMILARITY = 0.5


class AttributionCandidate(BaseModel):
    """One step considered as the origin of the failure."""

    step_id: str
    seq: int
    agent_id: str
    agent_role: str
    event_type: str
    score: float
    reasons: list[str] = Field(default_factory=list)


class AttributionResult(BaseModel):
    """A prediction, its alternates, and the ground truth where known."""

    run_id: str
    success: bool
    predicted_step_id: str | None = None
    predicted_step_seq: int | None = None
    reason: str = ""
    candidates: list[AttributionCandidate] = Field(default_factory=list)
    actual_fault_step_seq: int | None = None
    actual_fault_type: str | None = None
    # Weights are surfaced so a reader can see the rules that produced the
    # ranking, rather than being handed an opaque number (NFR-7).
    rule_weights: dict[str, float] = Field(default_factory=lambda: dict(RULE_WEIGHTS))

    def is_correct(self) -> bool:
        """True when the prediction matches the injected fault's target step.

        Returns False when there is no ground truth to compare against, so that
        summing `is_correct()` over a population never silently counts an
        unknowable case as a hit.
        """
        if self.actual_fault_step_seq is None or self.predicted_step_seq is None:
            return False
        return self.predicted_step_seq == self.actual_fault_step_seq


def _downstream_diverges(step: Step, run: Run) -> tuple[bool, float]:
    """Whether the steps after ``step`` depart from what it produced.

    A branch only matters if the run actually changed course at it. Comparing
    the branch's own output against the run's final output distinguishes a
    critique that was acted on from one that was ignored, which is exactly the
    distinction the rejection taxonomy turns on.
    """
    later = [s for s in run.steps if s.seq > step.seq]
    if not later:
        return False, 1.0
    similarity = cosine_similarity(step.output, run.final_output)
    return similarity < DIVERGENCE_SIMILARITY, similarity


def predict_failure_origin(run: Run | dict[str, Any]) -> AttributionResult:
    """Predict which step introduced the error in a failed run."""
    resolved = as_run(run)

    result = AttributionResult(
        run_id=resolved.run_id,
        success=resolved.success,
        actual_fault_step_seq=(
            resolved.injected_fault.target_step_seq
            if resolved.injected_fault is not None
            else None
        ),
        actual_fault_type=(
            resolved.injected_fault.fault_type
            if resolved.injected_fault is not None
            else None
        ),
    )

    if not resolved.steps:
        result.reason = "The run recorded no steps, so there is nothing to attribute."
        return result

    # Which steps carry an unsupported claim. Computed once: claim analysis
    # embeds every claim against every candidate step and is the expensive part.
    unsupported_texts = [
        claim.text for claim in analyse_claims(resolved) if not claim.supported
    ]

    candidates: list[AttributionCandidate] = []
    first_unsupported_awarded = False

    for step in resolved.steps:
        score = 0.0
        reasons: list[str] = []

        error = error_signal(step)
        if error > 0:
            score += RULE_WEIGHTS["error_signal"] * error
            if step.error is not None:
                reasons.append(f"raised {step.error.error_type}: {step.error.message}")
            elif step.event_type == "retrieval":
                reasons.append("retrieval returned no evidence references")
            else:
                reasons.append(
                    f"tool result is only {len(step.output)} characters, which "
                    "suggests a truncated or failed call"
                )

        # Only the *first* step carrying an unsupported claim is credited: the
        # rule is about where an ungrounded assertion entered the run, and every
        # later step repeating it is a symptom, not the origin.
        if not first_unsupported_awarded and unsupported_texts and step.output:
            if any(
                cosine_similarity(text, step.output) >= 0.6
                for text in unsupported_texts
            ):
                score += RULE_WEIGHTS["first_unsupported_claim"]
                reasons.append(
                    "first step whose output contains a claim the trace does "
                    "not ground"
                )
                first_unsupported_awarded = True

        if is_branch_point(step, resolved):
            diverges, similarity = _downstream_diverges(step, resolved)
            if diverges:
                score += RULE_WEIGHTS["divergent_branch"]
                reasons.append(
                    f"{step.event_type} step and the run diverged from it "
                    f"afterwards (similarity to final answer {similarity:.2f})"
                )

        if step.event_type == "retrieval" and 0 < len(step.evidence_refs) < (
            EXPECTED_EVIDENCE_REFS
        ):
            score += RULE_WEIGHTS["thin_retrieval"]
            reasons.append(
                f"retrieval returned {len(step.evidence_refs)} references, "
                f"fewer than the {EXPECTED_EVIDENCE_REFS} expected"
            )

        if score > 0:
            candidates.append(
                AttributionCandidate(
                    step_id=step.step_id,
                    seq=step.seq,
                    agent_id=step.agent_id,
                    agent_role=step.agent_role,
                    event_type=step.event_type,
                    score=round(score, PRECISION),
                    reasons=reasons,
                )
            )

    # Highest score first; earliest step wins a tie, because attribution is
    # asking for the *origin* and the earlier of two equally suspicious steps is
    # the better answer. The tiebreak also makes the ordering total, which
    # determinism requires.
    candidates.sort(key=lambda c: (-c.score, c.seq))
    result.candidates = candidates

    if not candidates:
        result.reason = (
            "No step matched any failure-origin rule. The run produced a wrong "
            "answer without any recorded error, ungrounded claim, divergent "
            "branch or thin retrieval, so the trace does not localise the cause."
        )
        return result

    best = candidates[0]
    result.predicted_step_id = best.step_id
    result.predicted_step_seq = best.seq
    result.reason = (
        f"Step {best.seq} ({best.event_type}, {best.agent_role}) scored "
        f"{best.score:g}, the highest of {len(candidates)} candidate step(s). "
        + "; ".join(best.reasons)
        + "."
    )
    return result
