"""Tests for the study's result-integrity guards.

Neither guard affects a study that runs start to finish in one sitting. Both
exist because the study does not always get to: free-tier daily quotas are small
enough that a population is judged across several sessions, and the corpus itself
was regenerated part-way through the project. Each guard closes a way for those
facts to corrupt a published number without anything failing.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.run_study import load_checkpoint, resolve_output_path


def _row(run_id: str, condition: str, *, predicted: int) -> dict:
    return {
        "condition": condition,
        "run_id": run_id,
        "workflow_type": "rag_qa",
        "fault_type": "dropped_retrieval",
        "actual_seq": 1,
        "predicted_seq": predicted,
        "correct": predicted == 1,
        "prompt_tokens": 100,
        "latency_ms": 10,
        "raw_response": str(predicted),
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def test_a_repeated_trial_is_counted_once(tmp_path: Path) -> None:
    """Trials are appended, so the same run can appear twice. Counting both
    would inflate n and every accuracy derived from it."""
    checkpoint = tmp_path / "trials.jsonl"
    _write(
        checkpoint,
        [
            _row("run-a", "raw_log", predicted=9),
            _row("run-a", "extracted", predicted=9),
            _row("run-a", "raw_log", predicted=1),
            _row("run-a", "extracted", predicted=1),
        ],
    )
    trials = load_checkpoint(checkpoint)
    assert len(trials["raw_log"]) == 1
    assert len(trials["extracted"]) == 1


def test_the_newest_trial_wins(tmp_path: Path) -> None:
    """The older row may have been judged against a corpus that no longer
    exists, so the last write is the only one that describes the current one."""
    checkpoint = tmp_path / "trials.jsonl"
    _write(
        checkpoint,
        [
            _row("run-a", "raw_log", predicted=9),
            _row("run-a", "raw_log", predicted=1),
        ],
    )
    assert load_checkpoint(checkpoint)["raw_log"][0]["predicted_seq"] == 1


def test_distinct_runs_are_all_kept(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trials.jsonl"
    _write(
        checkpoint,
        [
            _row("run-a", "raw_log", predicted=1),
            _row("run-b", "raw_log", predicted=1),
            _row("run-c", "raw_log", predicted=1),
        ],
    )
    assert len(load_checkpoint(checkpoint)["raw_log"]) == 3


def test_blank_lines_are_tolerated(tmp_path: Path) -> None:
    """A run killed mid-write can leave a trailing newline."""
    checkpoint = tmp_path / "trials.jsonl"
    checkpoint.write_text(json.dumps(_row("run-a", "raw_log", predicted=1)) + "\n\n")
    assert len(load_checkpoint(checkpoint)["raw_log"]) == 1


def test_both_conditions_are_present_even_when_empty(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trials.jsonl"
    checkpoint.write_text("")
    assert load_checkpoint(checkpoint) == {"raw_log": [], "extracted": []}


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------


def _study(*, complete: bool, n: int = 31) -> str:
    return json.dumps({"n": n, "population_complete": complete})


def test_a_complete_run_writes_to_the_published_path(tmp_path: Path) -> None:
    out = tmp_path / "primary_study.json"
    out.write_text(_study(complete=True))
    path, displaced = resolve_output_path(out, complete=True)
    assert path == out
    assert displaced == {}


def test_a_partial_run_does_not_overwrite_a_complete_one(tmp_path: Path) -> None:
    """The case this exists for: a quota wall cut a rerun short at 11 of 31 and
    the partial figures replaced a finished study of the whole population."""
    out = tmp_path / "primary_study.json"
    out.write_text(_study(complete=True, n=31))
    path, displaced = resolve_output_path(out, complete=False)
    assert path == tmp_path / "primary_study_partial.json"
    assert displaced["n"] == 31


def test_a_partial_run_may_replace_a_partial_one(tmp_path: Path) -> None:
    """Resuming onto a bigger partial population is progress, not a regression."""
    out = tmp_path / "primary_study.json"
    out.write_text(_study(complete=False, n=11))
    path, _ = resolve_output_path(out, complete=False)
    assert path == out


def test_a_partial_run_writes_normally_when_there_is_nothing_to_lose(
    tmp_path: Path,
) -> None:
    out = tmp_path / "primary_study.json"
    path, displaced = resolve_output_path(out, complete=False)
    assert path == out
    assert displaced == {}


def test_unreadable_existing_results_do_not_block_the_write(tmp_path: Path) -> None:
    """A truncated or hand-edited file is not evidence of a complete study, and
    must not wedge the study into never writing again."""
    out = tmp_path / "primary_study.json"
    out.write_text("{not json")
    path, displaced = resolve_output_path(out, complete=False)
    assert path == out
    assert displaced == {}
