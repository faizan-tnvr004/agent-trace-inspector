"""Tests for the deepened rag_qa workflow (version 2.0.0).

The workflow was deepened for one reason: at 4 to 8 steps a run, a top-5
extraction kept the whole trace, so the primary study compared two
serialisations of identical content rather than a pruned trace against a full
one. These tests pin the properties that make the deeper trace worth having,
and the fault mechanics that have to keep working across the new step layout.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from harness.faults import applicable_faults
from harness.llm import StubClient
from harness.workflows.rag_qa import (
    WORKFLOW_VERSION,
    derive_value,
    extract_measurements,
    load_corpus,
    run_rag_qa,
    split_sentences,
    verify_claim,
    _round_count,
    _round_queries,
)

CHUNKS, QUESTIONS = load_corpus()


def _run(index: int = 0, fault_type: str | None = None):
    return run_rag_qa(
        QUESTIONS[index],
        StubClient(),
        CHUNKS,
        fault_type=fault_type,
        rng=random.Random(1),
    )


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------


def test_the_workflow_version_records_the_change() -> None:
    """A corpus mixing 1.0.0 and 2.0.0 traces must be distinguishable, since
    the two have very different step counts."""
    assert WORKFLOW_VERSION == "2.0.0"


def test_runs_are_deep_enough_for_extraction_to_prune() -> None:
    """The point of the rewrite. Under a top-5 extraction a trace of 5 steps or
    fewer is kept whole, and the study's two conditions become identical."""
    for index in range(0, len(QUESTIONS), 5):
        assert len(_run(index).steps) > 10


def test_trace_length_varies_across_questions() -> None:
    """A fixed length would let a judge learn the faulted step's position
    rather than read the trace."""
    lengths = {len(_run(i).steps) for i in range(len(QUESTIONS))}
    assert len(lengths) > 1


def test_every_phase_of_the_pipeline_is_recorded() -> None:
    counts = Counter(s.event_type for s in _run().steps)
    for event_type in ("plan", "decision", "retrieval", "tool_call", "tool_result",
                       "reasoning", "critique", "final"):
        assert counts[event_type] >= 1, event_type


def test_there_is_more_than_one_retrieval_round() -> None:
    counts = Counter(s.event_type for s in _run().steps)
    assert counts["retrieval"] >= 2
    assert counts["decision"] >= 2


def test_rounds_are_two_or_three() -> None:
    assert {_round_count(q["question"]) for q in QUESTIONS} <= {2, 3}


def test_each_round_issues_a_different_query() -> None:
    queries = _round_queries(QUESTIONS[0]["question"])
    assert len(queries) == 3
    assert len(set(queries)) > 1


# ---------------------------------------------------------------------------
# Faults still land
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fault_type", applicable_faults("rag_qa"))
def test_every_applicable_fault_is_recorded_against_a_real_step(
    fault_type: str,
) -> None:
    run = _run(fault_type=fault_type)
    assert run.injected_fault is not None
    assert run.injected_fault.fault_type == fault_type
    seqs = {s.seq for s in run.steps}
    assert run.injected_fault.target_step_seq in seqs


def test_forced_false_rejection_is_now_applicable_to_rag() -> None:
    """It became applicable when the workflow gained a verification stage.
    Before 2.0.0 there was no reviewer to force."""
    assert "forced_false_rejection" in applicable_faults("rag_qa")


def test_forced_false_rejection_produces_a_rejecting_critique() -> None:
    run = _run(fault_type="forced_false_rejection")
    critiques = [s for s in run.steps if s.event_type == "critique"]
    assert any("REJECT" in s.output for s in critiques)


def test_dropped_retrieval_survives_every_later_round() -> None:
    """Rounds accumulate evidence and re-retrieve from the whole corpus, so a
    fault applied once would be undone by the next round and the run would
    carry ground truth for a fault that changed nothing."""
    run = _run(fault_type="dropped_retrieval")
    answer_chunk = QUESTIONS[0].get("answer_chunk_id")
    for step in run.steps:
        if step.event_type == "retrieval":
            assert answer_chunk not in step.evidence_refs


def test_truncated_tool_result_cuts_the_tool_step_not_the_context() -> None:
    """The fault is named for the tool result, and now there is a real one."""
    run = _run(fault_type="truncated_tool_result")
    assert run.injected_fault is not None
    assert "tool_result" in run.injected_fault.description


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_extraction_puts_the_answer_candidate_last() -> None:
    """Truncation keeps the first 40%. A digest with the useful value at the
    front would survive the fault and make it inert."""
    context = "[doc-01#c0] The mast is 12 metres tall and was built in 1998."
    result = extract_measurements(context, "1998")
    lines = [ln for ln in result.splitlines() if "candidate" in ln]
    assert "1998" in lines[-1]


def test_extraction_reports_when_there_is_nothing_to_find() -> None:
    assert "no numeric candidates" in extract_measurements("[doc-01#c0] text", "x")


def test_derive_turns_a_year_into_an_elapsed_duration() -> None:
    assert "year(s) before 2026" in derive_value("Since what year...?", ["1962"])


def test_derive_converts_a_length() -> None:
    assert "feet" in derive_value("What depth in metres?", ["100"])


def test_derive_handles_a_non_numeric_value() -> None:
    assert "not numeric" in derive_value("q", ["abc"])


def test_derive_handles_no_values() -> None:
    assert "no value" in derive_value("q", [])


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_a_grounded_claim_is_accepted() -> None:
    chunks = [{"chunk_id": "doc-01#c0",
               "text": "Vellmar Station was established in 1962 on the coast."}]
    supported, note = verify_claim("Vellmar Station was established in 1962.", chunks)
    assert supported
    assert "doc-01#c0" in note


def test_a_claim_asserting_an_unretrieved_number_is_rejected() -> None:
    chunks = [{"chunk_id": "doc-01#c0",
               "text": "Vellmar Station was established in 1962 on the coast."}]
    supported, note = verify_claim("Vellmar Station was established in 1988.", chunks)
    assert not supported
    assert "1988" in note


def test_verification_reports_its_own_weakness() -> None:
    """The check is lexical, and the step output has to say so rather than
    presenting itself as a judgement of truth."""
    chunks = [{"chunk_id": "doc-01#c0", "text": "unrelated text entirely"}]
    _, note = verify_claim("Some quite different assertion about stations.", chunks)
    assert "overlap" in note


def test_sentence_splitting_drops_fragments() -> None:
    assert split_sentences("Yes. A full sentence about a station here.") == [
        "A full sentence about a station here."
    ]
