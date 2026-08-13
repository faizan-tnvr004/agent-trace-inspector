"""The answer key must name where the error was introduced.

A fault acting on retrieved context is re-applied on every retrieval round, or a
later round re-retrieves from the full corpus and silently undoes it. Each
application yields its own `InjectedFault`, and the one that gets recorded
becomes ground truth for the whole evaluation.

Recording the last application made the answer key name the final retrieval
while the error originated at the first. Scored against it the raw-log judge got
1 of 24; scored against the first retrieval it got 23 of 24. Nothing failed, and
no test caught it, because on a single-round workflow first and last are the
same step. These tests fix the invariant so a future workflow with more rounds
cannot reintroduce it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from app.models import InjectedFault, Run
from harness.faults import applicable_faults, earliest
from harness.llm import StubClient
from harness.workflows.rag_qa import load_corpus, run_rag_qa

CHUNKS, QUESTIONS = load_corpus()
CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


def _fault(seq: int) -> InjectedFault:
    return InjectedFault(
        fault_type="dropped_retrieval",
        target_step_seq=seq,
        description=f"applied at {seq}",
    )


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


def test_the_first_application_is_kept() -> None:
    assert earliest(_fault(3), _fault(9)).target_step_seq == 3


def test_an_earlier_application_replaces_a_later_one() -> None:
    """Order of arrival must not decide the answer key."""
    assert earliest(_fault(9), _fault(3)).target_step_seq == 3


def test_the_first_application_is_kept_when_there_is_nothing_yet() -> None:
    assert earliest(None, _fault(7)).target_step_seq == 7


def test_repeated_application_at_the_same_seq_is_stable() -> None:
    assert earliest(_fault(4), _fault(4)).target_step_seq == 4


# ---------------------------------------------------------------------------
# The invariant, over freshly generated runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fault_type", applicable_faults("rag_qa"))
@pytest.mark.parametrize("question_index", [0, 7, 14])
def test_recorded_seq_is_the_earliest_faulted_step(
    fault_type: str, question_index: int
) -> None:
    """For a context fault, the earliest application is the first retrieval.

    The fault is applied immediately after each retrieval is computed and before
    it is recorded, so the recorded seq must equal the first retrieval's seq. Any
    later value means a subsequent application overwrote the answer key.
    """
    run = run_rag_qa(
        QUESTIONS[question_index],
        StubClient(),
        CHUNKS,
        fault_type=fault_type,
        rng=random.Random(1),
    )
    assert run.injected_fault is not None
    recorded = run.injected_fault.target_step_seq

    if fault_type in {"dropped_retrieval", "injected_contradiction"}:
        retrievals = [s.seq for s in run.steps if s.event_type == "retrieval"]
        assert recorded == retrievals[0], (
            f"{fault_type} recorded seq {recorded}, but the first faulted "
            f"retrieval is {retrievals[0]}"
        )
    # The other two faults are applied once, at their own step, so there is no
    # earliest-versus-latest question for them. The invariant below still holds.
    assert recorded <= max(s.seq for s in run.steps)


@pytest.mark.parametrize("fault_type", applicable_faults("rag_qa"))
def test_the_recorded_step_precedes_the_final_step(fault_type: str) -> None:
    """A cause cannot be recorded at the step that merely reports the result."""
    run = run_rag_qa(
        QUESTIONS[0], StubClient(), CHUNKS, fault_type=fault_type,
        rng=random.Random(1),
    )
    assert run.injected_fault is not None
    final_seq = max(s.seq for s in run.steps if s.event_type == "final")
    assert run.injected_fault.target_step_seq < final_seq


# ---------------------------------------------------------------------------
# The invariant, over the committed corpus
# ---------------------------------------------------------------------------


def _committed_runs() -> list[Run]:
    return [
        Run.model_validate(json.loads(path.read_text()))
        for path in sorted(CORPUS.glob("run_*.json"))
    ]


@pytest.mark.skipif(not CORPUS.is_dir(), reason="corpus not present")
def test_committed_context_faults_point_at_the_first_retrieval() -> None:
    """Guards the published answer key, not just the generator.

    A corpus committed before the fix would pass every generator test above and
    still carry mislabelled runs, so the shipped files are checked directly.
    """
    offenders = []
    for run in _committed_runs():
        fault = run.injected_fault
        if fault is None or run.workflow_type != "rag_qa":
            continue
        if fault.fault_type not in {"dropped_retrieval", "injected_contradiction"}:
            continue
        retrievals = [s.seq for s in run.steps if s.event_type == "retrieval"]
        if retrievals and fault.target_step_seq != retrievals[0]:
            offenders.append(
                f"{run.run_id[:8]} {fault.fault_type}: "
                f"recorded {fault.target_step_seq}, first retrieval {retrievals[0]}"
            )
    assert not offenders, "mislabelled runs in the committed corpus:\n" + "\n".join(
        offenders
    )


@pytest.mark.skipif(not CORPUS.is_dir(), reason="corpus not present")
def test_every_committed_fault_targets_a_step_that_exists() -> None:
    for run in _committed_runs():
        if run.injected_fault is None:
            continue
        seqs = {s.seq for s in run.steps}
        assert run.injected_fault.target_step_seq in seqs, run.run_id
