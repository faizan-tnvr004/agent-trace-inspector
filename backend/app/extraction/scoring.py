"""Critical-step scoring: which steps in a trace determined the outcome.

Most steps in a trace are inert. This module ranks them by influence on the
final output, combining three signals:

* **evidence survival** — how much of the step's output persists into the final
  answer, as embedding cosine similarity
* **branch** — whether the step is a point where the run could have gone
  differently: a retry, a critique, a revision, or a changed plan
* **error** — whether the step carries an exception, an empty retrieval, or a
  suspiciously short tool result

Two constraints shape the implementation.

*No LLM calls.* Embeddings only. The same trace must produce byte-identical
output on every run (FR-16), which an LLM in the scoring path would destroy.

*No opaque numbers.* `critical_score` returns a `ScoreBreakdown` carrying every
component and the weights used, never a bare float (NFR-7). A reviewer must be
able to see why a step was ranked where it was.

The weights below were fixed before the primary study was run and have not been
changed since. Tuning them against the evaluation result would be fitting to
the test set.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.extraction.embeddings import PRECISION, cosine_similarity
from app.models import Run, Step

__all__ = [
    "WEIGHTS",
    "ScoreBreakdown",
    "critical_score",
    "error_signal",
    "evidence_survival",
    "is_branch_point",
    "rank_critical",
]

# Fixed before Phase 6. Do not tune against evaluation results.
WEIGHTS: dict[str, float] = {
    "evidence_survival": 0.4,
    "branch": 0.3,
    "error": 0.3,
}

# A tool result shorter than this is treated as a partial failure. Matches the
# specification's threshold.
SHORT_TOOL_RESULT_CHARS = 20

_BRANCH_EVENT_TYPES = frozenset({"retry", "critique", "revision"})


class ScoreBreakdown(BaseModel):
    """A step's criticality with every contributing signal exposed."""

    step_id: str
    seq: int
    agent_id: str
    agent_role: str
    event_type: str
    evidence_survival: float
    branch: float
    error: float
    critical_score: float
    weights: dict[str, float] = Field(default_factory=lambda: dict(WEIGHTS))
    reasons: list[str] = Field(default_factory=list)


def as_run(run: Run | dict[str, Any]) -> Run:
    """Accept either a `Run` or the dict form loaded from a corpus file.

    The corpus is committed as JSON and scripts read it with `json.load`, so
    every public entry point in the extraction engine takes both.
    """
    return run if isinstance(run, Run) else Run.model_validate(run)


def evidence_survival(step: Step, final_output: str) -> float:
    """How much of ``step.output`` persists into the final output, in [0, 1]."""
    return cosine_similarity(step.output, final_output)


def is_branch_point(step: Step, run: Run | dict[str, Any]) -> bool:
    """True where the run could have proceeded differently.

    A changed plan counts: comparing against the *previous* plan step rather
    than against all of them means a plan that oscillates between two options
    is flagged each time it changes, which is the behaviour of interest.
    """
    if step.event_type in _BRANCH_EVENT_TYPES:
        return True
    if step.retry_of is not None:
        return True
    if step.event_type == "plan":
        previous = _previous_plan(as_run(run), step)
        if previous is not None and previous.output.strip() != step.output.strip():
            return True
    return False


def _previous_plan(run: Run, step: Step) -> Step | None:
    candidates = [
        s for s in run.steps if s.event_type == "plan" and s.seq < step.seq
    ]
    return candidates[-1] if candidates else None


def error_signal(step: Step) -> float:
    """Strength of evidence that this step went wrong, in [0, 1]."""
    if step.error is not None:
        return 1.0
    if step.event_type == "retrieval" and not step.evidence_refs:
        return 1.0
    if (
        step.event_type == "tool_result"
        and len(step.output) < SHORT_TOOL_RESULT_CHARS
    ):
        return 0.5
    return 0.0


def _reasons(step: Step, survival: float, branch: float, error: float) -> list[str]:
    """Human-readable justification for each non-zero signal."""
    out: list[str] = []
    if survival >= 0.5:
        out.append(
            f"{survival:.0%} of this step's output survives into the final "
            "answer"
        )
    elif survival > 0.0:
        out.append(f"weak overlap with the final answer ({survival:.0%})")

    if branch > 0.0:
        if step.retry_of is not None:
            out.append("retries an earlier step")
        elif step.event_type in _BRANCH_EVENT_TYPES:
            out.append(f"{step.event_type} step: the run could have diverged here")
        else:
            out.append("plan changed from the previous plan step")

    if error >= 1.0:
        if step.error is not None:
            out.append(f"raised {step.error.error_type}")
        else:
            out.append("retrieval returned no evidence references")
    elif error > 0.0:
        out.append(
            f"tool result is only {len(step.output)} characters, below the "
            f"{SHORT_TOOL_RESULT_CHARS}-character threshold"
        )
    return out


def critical_score(step: Step, run: Run | dict[str, Any]) -> ScoreBreakdown:
    """Weighted criticality with per-signal components.

    Never returns a bare float: the breakdown is the contract.
    """
    resolved = as_run(run)
    survival = evidence_survival(step, resolved.final_output)
    branch = 1.0 if is_branch_point(step, resolved) else 0.0
    error = error_signal(step)

    total = (
        WEIGHTS["evidence_survival"] * survival
        + WEIGHTS["branch"] * branch
        + WEIGHTS["error"] * error
    )

    return ScoreBreakdown(
        step_id=step.step_id,
        seq=step.seq,
        agent_id=step.agent_id,
        agent_role=step.agent_role,
        event_type=step.event_type,
        evidence_survival=survival,
        branch=branch,
        error=error,
        critical_score=round(total, PRECISION),
        weights=dict(WEIGHTS),
        reasons=_reasons(step, survival, branch, error),
    )


def rank_critical(
    run: Run | dict[str, Any], k: int = 5
) -> list[ScoreBreakdown]:
    """The ``k`` most critical steps, highest first.

    Ties break on ``seq`` ascending so the ordering is total and repeatable;
    without that, equal scores could come back in either order and extraction
    would not be deterministic.
    """
    resolved = as_run(run)
    scored = [critical_score(step, resolved) for step in resolved.steps]
    scored.sort(key=lambda s: (-s.critical_score, s.seq))
    return scored[: max(0, k)]
