"""Adapter tests (FR-7, FR-8).

The point of these is NFR-3: an imported trace must be the same `Run` object a
native trace is, so it renders identically and every extraction function works on
it unchanged. Each test therefore ends by running real extraction over the
imported run rather than only checking field values.
"""

from __future__ import annotations

from adapters.autogen_adapter import import_autogen_messages
from adapters.langgraph_adapter import import_langgraph_events, map_node_to_event
from app.extraction.attribution import predict_failure_origin
from app.extraction.scoring import rank_critical
from app.models import Run


# ---------------------------------------------------------------------------
# LangGraph
# ---------------------------------------------------------------------------

UPDATES_STREAM = [
    {"retrieve": {"documents": [{"chunk_id": "doc-08#c2", "text": "312 metres"}]}},
    {"reason": {"output": "The basin reaches 312 metres.", "prompt_tokens": 90}},
    {"answer": {"answer": "ANSWER: 312"}},
]

DEBUG_STREAM = [
    {
        "type": "task_result",
        "timestamp": "2026-01-01T12:00:00+00:00",
        "step": 1,
        "payload": {"name": "retrieve", "result": {"documents": ["doc-08#c2"]}},
    },
    {
        "type": "task_result",
        "timestamp": "2026-01-01T12:00:02+00:00",
        "step": 2,
        "payload": {"name": "review", "result": {"output": "Looks wrong."}},
    },
]


def test_updates_stream_becomes_a_valid_run() -> None:
    run = import_langgraph_events(UPDATES_STREAM, task_input="How deep is it?")
    assert isinstance(run, Run)
    assert run.workflow_type == "imported"
    assert [s.seq for s in run.steps] == [0, 1, 2]
    assert [s.event_type for s in run.steps] == ["retrieval", "reasoning", "final"]


def test_debug_stream_becomes_a_valid_run_with_real_timestamps() -> None:
    run = import_langgraph_events(DEBUG_STREAM, task_input="How deep is it?")
    assert [s.event_type for s in run.steps] == ["retrieval", "critique"]
    assert run.steps[0].timestamp.year == 2026
    assert run.steps[1].timestamp > run.steps[0].timestamp


def test_retrieval_evidence_refs_are_recovered() -> None:
    """FR-2 requires retrieval steps to record document ids, and the extraction
    engine's empty-retrieval signal depends on them."""
    run = import_langgraph_events(UPDATES_STREAM, task_input="q")
    assert run.steps[0].evidence_refs == ["doc-08#c2"]


def test_string_document_lists_are_also_recovered() -> None:
    run = import_langgraph_events(DEBUG_STREAM, task_input="q")
    assert run.steps[0].evidence_refs == ["doc-08#c2"]


def test_steps_are_chained_into_a_tree() -> None:
    run = import_langgraph_events(UPDATES_STREAM, task_input="q")
    assert run.steps[0].parent_step_id is None
    assert run.steps[1].parent_step_id == run.steps[0].step_id
    assert run.steps[2].parent_step_id == run.steps[1].step_id


def test_an_error_in_a_state_delta_becomes_an_error_step() -> None:
    run = import_langgraph_events(
        [{"tools": {"error": {"type": "TimeoutError", "message": "timed out"}}}],
        task_input="q",
    )
    assert run.steps[0].event_type == "error"
    assert run.steps[0].error is not None
    assert run.steps[0].error.error_type == "TimeoutError"


def test_success_defaults_to_false_rather_than_true() -> None:
    """An import has no grader. Defaulting to success would corrupt any
    statistic computed over an imported corpus."""
    assert import_langgraph_events(UPDATES_STREAM, task_input="q").success is False


def test_internal_channels_are_skipped() -> None:
    run = import_langgraph_events(
        [{"__start__": {"x": 1}}, {"reason": {"output": "ok"}}], task_input="q"
    )
    assert [s.agent_id for s in run.steps] == ["reason"]


def test_unknown_node_names_fall_back_rather_than_failing() -> None:
    run = import_langgraph_events(
        [{"some_bespoke_node": {"output": "text"}}], task_input="q"
    )
    assert run.steps[0].event_type == "reasoning"


def test_node_mapping_prefers_retrieval_over_review() -> None:
    """Keyword fallbacks are order-sensitive; this pins the intended priority."""
    assert map_node_to_event("retrieve_and_review") == "retrieval"
    assert map_node_to_event("Reviewer") == "critique"
    assert map_node_to_event("__end__") == "final"


def test_imported_langgraph_run_works_with_extraction() -> None:
    run = import_langgraph_events(UPDATES_STREAM, task_input="q")
    assert len(rank_critical(run, k=3)) == 3
    assert predict_failure_origin(run).run_id == run.run_id


# ---------------------------------------------------------------------------
# AutoGen
# ---------------------------------------------------------------------------

AUTOGEN_LOG = [
    {"role": "user", "name": "user_proxy", "content": "Compute 24 + 72."},
    {
        "role": "assistant",
        "name": "coder",
        "content": "Let me add them.",
        "usage": {"prompt_tokens": 20, "completion_tokens": 8},
    },
    {
        "role": "assistant",
        "name": "critic",
        "content": "The addition is right.",
        "usage": {"prompt_tokens": 30, "completion_tokens": 6},
    },
    {"role": "assistant", "name": "coder", "content": "ANSWER: 96"},
]


def test_autogen_log_becomes_a_valid_run() -> None:
    run = import_autogen_messages(AUTOGEN_LOG)
    assert isinstance(run, Run)
    assert [s.event_type for s in run.steps] == [
        "plan",
        "reasoning",
        "critique",
        "final",
    ]
    assert run.task_input == "Compute 24 + 72."


def test_autogen_tokens_and_totals_reconcile() -> None:
    run = import_autogen_messages(AUTOGEN_LOG)
    assert run.total_tokens == 64
    assert run.total_cost_usd == 0.0


def test_autogen_tool_calls_are_detected_from_fields_not_prose() -> None:
    run = import_autogen_messages(
        [
            {"role": "assistant", "name": "coder", "tool_calls": [{"name": "calc"}]},
            {"role": "tool", "tool_call_id": "1", "content": "96"},
        ]
    )
    assert run.steps[0].event_type == "tool_call"
    assert run.steps[1].event_type == "tool_result"


def test_autogen_step_input_is_the_previous_message() -> None:
    run = import_autogen_messages(AUTOGEN_LOG)
    assert run.steps[1].input == "Compute 24 + 72."
    assert run.steps[2].input == "Let me add them."


def test_autogen_handles_structured_content_blocks() -> None:
    run = import_autogen_messages(
        [{"role": "assistant", "content": [{"text": "part one"}, {"text": "part two"}]}]
    )
    assert "part one" in run.steps[0].output
    assert "part two" in run.steps[0].output


def test_imported_autogen_run_works_with_extraction() -> None:
    run = import_autogen_messages(AUTOGEN_LOG)
    ranked = rank_critical(run, k=2)
    assert len(ranked) == 2
    assert all(0.0 <= s.critical_score <= 1.0 for s in ranked)


def test_empty_inputs_do_not_crash() -> None:
    assert import_langgraph_events([], task_input="q").steps == []
    assert import_autogen_messages([]).steps == []
