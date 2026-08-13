"""Tests for the workflow harness: grading, fault injection, tracing, retrieval.

The corpus is the input to every downstream measurement, so the properties
tested here are the ones whose silent failure would invalidate results rather
than crash anything:

* a fault must actually damage what its ground truth says it damaged
* the tracer must emit a schema-valid run even when a workflow crashes
* retrieval must be deterministic, or extraction cannot be
* the committed task data must have exactly one source for each answer
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from harness.faults import (
    ALL_FAULT_TYPES,
    TRUNCATION_RATIO,
    applicable_faults,
    apply_fault,
)
from app.grading import answer_matches, normalise
from harness.llm import LLMResponse, StubClient, notional_cost_usd
from harness.tracer import TraceRecorder
from harness.workflows.rag_qa import load_corpus, retrieve

TASKS = Path(__file__).resolve().parents[2] / "harness" / "tasks"


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expected,produced",
    [
        ("96", "ANSWER: 96"),
        ("96", "the answer is 96 loaves"),
        ("96", "96"),
        ("96", "So we get 96."),
        ("24.5", "ANSWER: 24.5"),
        ("2420", "ANSWER: 2,420"),
        ("1240", "the station stands at 1,240 metres"),
    ],
)
def test_answer_matches_accepts_correct_presentations(
    expected: str, produced: str
) -> None:
    assert answer_matches(expected, produced) is True


@pytest.mark.parametrize(
    "expected,produced",
    [
        ("96", "960"),
        ("96", "1.96"),
        ("96", "96.5"),
        ("4", "1974"),
        ("4", "42"),
        ("18", "18.5"),
        ("96", ""),
        ("96", "the answer is 69"),
    ],
)
def test_answer_matches_rejects_near_misses(expected: str, produced: str) -> None:
    """A number inside a longer number is not the answer. Without this the
    corpus would record wrong runs as successes and there would be nothing
    left to attribute."""
    assert answer_matches(expected, produced) is False


def test_normalise_strips_separators_and_currency() -> None:
    assert normalise("£2,420") == "2420"
    assert normalise("  1,180  ") == "1180"


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


def _rag_state() -> dict:
    return {
        "ground_truth": "312",
        "answer_chunk_id": "doc-08#c2",
        "chunks": [
            {"chunk_id": "doc-08#c0", "doc_id": "doc-08", "text": "Opened in 1985."},
            {
                "chunk_id": "doc-08#c2",
                "doc_id": "doc-08",
                "text": "The deep basin reaches a maximum depth of 312 metres.",
            },
        ],
    }


def test_every_declared_fault_type_has_an_implementation() -> None:
    for fault_type in ALL_FAULT_TYPES:
        state, record = apply_fault(fault_type, _rag_state(), 1)
        assert record.fault_type == fault_type
        assert record.target_step_seq == 1
        assert record.description.strip()


def test_dropped_retrieval_removes_only_the_answer_chunk() -> None:
    state, record = apply_fault("dropped_retrieval", _rag_state(), 1)
    ids = [c["chunk_id"] for c in state["chunks"]]
    assert "doc-08#c2" not in ids
    assert "doc-08#c0" in ids
    assert "doc-08#c2" in record.description


def test_dropped_retrieval_leaves_plausible_context_behind() -> None:
    """The retrieval must still look successful, otherwise the fault is
    trivially visible and attribution is not being tested."""
    state, _ = apply_fault("dropped_retrieval", _rag_state(), 1)
    assert len(state["chunks"]) > 0


def test_truncated_tool_result_cuts_to_the_declared_ratio() -> None:
    state = {"tool_result": "x" * 100, "ground_truth": "1"}
    out, record = apply_fault("truncated_tool_result", state, 2)
    assert len(out["tool_result"]) == int(100 * TRUNCATION_RATIO)
    assert "40%" in record.description


def test_truncated_tool_result_falls_back_to_context_text() -> None:
    state = {"context_text": "y" * 50, "ground_truth": "1"}
    out, _ = apply_fault("truncated_tool_result", state, 2)
    assert len(out["context_text"]) == int(50 * TRUNCATION_RATIO)


def test_truncated_tool_result_never_empties_the_output() -> None:
    state = {"tool_result": "ab", "ground_truth": "1"}
    out, _ = apply_fault("truncated_tool_result", state, 1)
    assert len(out["tool_result"]) >= 1


def test_forced_false_rejection_sets_the_flag_the_reviewer_reads() -> None:
    state, record = apply_fault("forced_false_rejection", {"rounds": 0}, 4)
    assert state["force_reject"] is True
    assert "REJECT" in record.description


def test_injected_contradiction_conflicts_with_the_known_answer() -> None:
    state, record = apply_fault("injected_contradiction", _rag_state(), 1)
    injected = state["chunks"][0]
    # Shaped like a real corpus chunk id. A self-labelling id would appear in the
    # retrieval step's evidence_refs and let the evaluation judge read the answer
    # off the trace instead of reasoning about it.
    assert re.fullmatch(r"doc-\d+#c\d+", injected["chunk_id"])
    assert "inject" not in injected["chunk_id"].lower()
    assert "312" in injected["text"]
    # It must propose a different value, not merely mention the right one.
    assert "319" in injected["text"]
    assert "312" in record.description


def test_injected_contradiction_is_placed_first_in_context() -> None:
    state, _ = apply_fault("injected_contradiction", _rag_state(), 1)
    assert state["chunks"][0]["text"].startswith("Correction notice")
    assert len(state["chunks"]) == 3


def test_faults_do_not_mutate_the_caller_state() -> None:
    original = _rag_state()
    apply_fault("dropped_retrieval", original, 1)
    assert len(original["chunks"]) == 2


def test_unknown_fault_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fault_type"):
        apply_fault("cosmic_ray", {}, 0)  # type: ignore[arg-type]


def test_faults_are_only_offered_to_workflows_that_can_carry_them() -> None:
    """`dropped_retrieval` needs a retrieval step and `forced_false_rejection`
    needs a reviewer. Offering either to the wrong workflow would record ground
    truth for a defect that was never introduced."""
    assert "dropped_retrieval" not in applicable_faults("reviewer_pipeline")
    assert "forced_false_rejection" not in applicable_faults("rag_qa")
    assert "dropped_retrieval" in applicable_faults("rag_qa")
    assert "forced_false_rejection" in applicable_faults("reviewer_pipeline")


def test_union_of_applicable_faults_covers_all_four_types() -> None:
    covered = set(applicable_faults("rag_qa")) | set(
        applicable_faults("reviewer_pipeline")
    )
    assert covered == set(ALL_FAULT_TYPES)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


def test_tracer_emits_a_valid_run_with_reconciled_cost() -> None:
    with TraceRecorder("rag_qa", "1.0.0", "q?", ground_truth="4") as tr:
        tr.record(
            agent_id="a",
            agent_role="retriever",
            event_type="retrieval",
            input="q?",
            output="ctx",
            cost_usd=0.001,
            prompt_tokens=10,
            completion_tokens=2,
        )
        tr.record(
            agent_id="a",
            agent_role="executor",
            event_type="final",
            input="ctx",
            output="4",
            cost_usd=0.002,
            prompt_tokens=5,
            completion_tokens=1,
        )
        tr.set_result(final_output="4", success=True)

    run = tr.run
    assert run.total_cost_usd == pytest.approx(0.003)
    assert run.total_tokens == 18
    assert [s.seq for s in run.steps] == [0, 1]
    assert run.success is True


def test_tracer_chains_parent_step_ids_by_default() -> None:
    with TraceRecorder("rag_qa", "1.0.0", "q?") as tr:
        tr.record(
            agent_id="a", agent_role="r", event_type="plan", input="", output=""
        )
        tr.record(
            agent_id="a", agent_role="r", event_type="final", input="", output=""
        )
        tr.set_result(final_output="", success=False)

    run = tr.run
    assert run.steps[0].parent_step_id is None
    assert run.steps[1].parent_step_id == run.steps[0].step_id


def test_tracer_records_a_crash_as_a_failed_run_and_reraises() -> None:
    """A workflow that raises must still yield a valid trace. Discarding
    crashed runs would bias the corpus towards runs that completed."""
    recorder = TraceRecorder("rag_qa", "1.0.0", "q?")
    with pytest.raises(ZeroDivisionError):
        with recorder as tr:
            tr.record(
                agent_id="a",
                agent_role="retriever",
                event_type="retrieval",
                input="q?",
                output="ctx",
            )
            raise ZeroDivisionError("boom")

    run = recorder.run
    assert run.success is False
    assert run.steps[-1].event_type == "error"
    assert run.steps[-1].error is not None
    assert run.steps[-1].error.error_type == "ZeroDivisionError"


def test_tracer_run_is_unavailable_before_the_context_exits() -> None:
    recorder = TraceRecorder("rag_qa", "1.0.0", "q?")
    with pytest.raises(RuntimeError):
        _ = recorder.run


def test_tracer_ids_are_reproducible_when_seeded() -> None:
    import uuid

    source = uuid.UUID(int=7)

    def build() -> list[str]:
        with TraceRecorder(
            "rag_qa", "1.0.0", "q?", run_id="fixed", id_source=source
        ) as tr:
            tr.record(
                agent_id="a", agent_role="r", event_type="plan", input="", output=""
            )
            tr.record(
                agent_id="a", agent_role="r", event_type="final", input="", output=""
            )
            tr.set_result(final_output="", success=True)
        return [s.step_id for s in tr.run.steps]

    assert build() == build()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_retrieval_is_deterministic() -> None:
    chunks, questions = load_corpus()
    q = questions[0]["question"]
    assert [c["chunk_id"] for c, _ in retrieve(q, chunks)] == [
        c["chunk_id"] for c, _ in retrieve(q, chunks)
    ]


def test_retrieval_scores_are_bounded() -> None:
    chunks, questions = load_corpus()
    for question in questions[:5]:
        for _, score in retrieve(question["question"], chunks):
            assert 0.0 <= score <= 1.0


def test_retrieval_finds_the_answer_chunk_for_most_questions() -> None:
    """Recall is a property of the corpus, so it is pinned. If retrieval
    degraded, faulted runs would fail for the wrong reason."""
    chunks, questions = load_corpus()
    hits = sum(
        1
        for q in questions
        if q["answer_chunk_id"] in [c["chunk_id"] for c, _ in retrieve(q["question"], chunks)]
    )
    assert hits >= 28, f"recall@4 dropped to {hits}/{len(questions)}"


def test_retrieval_handles_a_query_with_no_usable_tokens() -> None:
    """Tokens of three characters or fewer are discarded, so a query made only
    of them leaves nothing to score against."""
    chunks, _ = load_corpus()
    assert retrieve("a of an is", chunks) == []


# ---------------------------------------------------------------------------
# Committed task data
# ---------------------------------------------------------------------------


def test_math_tasks_meet_the_declared_size_and_are_unique() -> None:
    tasks = json.loads((TASKS / "math_tasks.json").read_text())
    assert len(tasks) >= 40
    assert len({t["id"] for t in tasks}) == len(tasks)
    for task in tasks:
        assert task["question"].strip()
        assert str(task["answer"]).strip()


def test_qa_corpus_meets_the_declared_size() -> None:
    documents = json.loads((TASKS / "qa_corpus" / "documents.json").read_text())
    questions = json.loads((TASKS / "qa_corpus" / "questions.json").read_text())
    assert len(documents) >= 30
    assert len(questions) >= 30
    assert len({d["doc_id"] for d in documents}) == len(documents)


def test_each_qa_answer_appears_in_exactly_one_chunk_of_its_document() -> None:
    """This is what makes `dropped_retrieval` a real fault. If the answer also
    sat in a sibling chunk, removing the target chunk would leave the run able
    to succeed, and the recorded ground truth would be a lie."""
    import re

    documents = json.loads((TASKS / "qa_corpus" / "documents.json").read_text())
    questions = json.loads((TASKS / "qa_corpus" / "questions.json").read_text())
    chunks = {c["chunk_id"]: c["text"] for d in documents for c in d["chunks"]}

    for q in questions:
        answer = q["answer"].replace(",", "")
        # Must not match inside a longer number; a trailing full stop is fine.
        pattern = re.compile(
            rf"(?<![\d.]){re.escape(answer)}(?!\d)(?!\.\d)"
        )
        own = q["answer_chunk_id"]
        assert own in chunks, f"{q['id']} references missing chunk {own}"
        assert pattern.search(chunks[own].replace(",", "")), (
            f"{q['id']}: answer {answer!r} is not in its own chunk {own}"
        )
        siblings = [
            cid
            for cid, text in chunks.items()
            if cid != own
            and cid.split("#")[0] == q["doc_id"]
            and pattern.search(text.replace(",", ""))
        ]
        assert not siblings, (
            f"{q['id']}: answer {answer!r} also appears in {siblings}, so "
            "dropping the target chunk would not remove the answer"
        )


# ---------------------------------------------------------------------------
# Cost accounting and the stub client
# ---------------------------------------------------------------------------


def test_notional_cost_uses_published_rates() -> None:
    # gemini-3.5-flash-lite: $0.10/MTok in, $0.40/MTok out
    cost = notional_cost_usd("gemini-3.5-flash-lite", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.50)


def test_notional_cost_falls_back_for_an_unknown_model() -> None:
    assert notional_cost_usd("some-future-model", 1_000_000, 0) > 0


def test_llm_response_exposes_cost_from_usage() -> None:
    response = LLMResponse(
        text="x",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        latency_ms=1,
        model="gemini-3.5-flash-lite",
    )
    assert response.cost_usd == pytest.approx(0.10)


def test_stub_client_is_deterministic_for_the_same_prompt() -> None:
    client = StubClient()
    a = client.complete("what is 2+2?", hint={"role": "executor", "expected": "4"})
    b = client.complete("what is 2+2?", hint={"role": "executor", "expected": "4"})
    assert a.text == b.text


def test_stub_client_hint_does_not_come_from_the_prompt() -> None:
    """Ground truth must never travel in a prompt: with a real client that
    would leak the answer into the model call and invalidate the corpus."""
    client = StubClient()
    without = client.complete("Question: what is the depth?")
    assert "EXPECTED" not in without.text
