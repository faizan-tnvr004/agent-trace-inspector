"""Primary evaluation: do extracted critical-trace summaries beat raw logs?

Hypothesis: extracted summaries let a reader identify the cause of a failed run
more accurately, and with less reading, than the raw log.

* **Population** — every failed run carrying a known injected fault
* **Condition A (`raw_log`)** — the full trace serialised as JSON
* **Condition B (`extracted`)** — the top 5 critical steps plus the final output
* **Task** — name the step that introduced the error
* **Score** — against `injected_fault.target_step_seq`
* **Secondary measures** — input tokens presented, judge latency

The judge prompt and model are identical in both conditions; only the trace
content differs. Extraction weights were fixed before this script was first run
and have not been changed since.

A null result is a valid outcome. If extraction does not beat the raw log, that
is the finding, and section 10.4 of the SRS anticipates it.

Usage::

    python evaluation/run_study.py --out evaluation/results/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "backend", REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.extraction.scoring import rank_critical  # noqa: E402
from app.models import Run  # noqa: E402
from evaluation.judge import JUDGE_PROMPT, JUDGE_SYSTEM, ask_judge  # noqa: E402
from harness.llm import LLMUnavailable, build_client  # noqa: E402

CONDITIONS = ("raw_log", "extracted")
CRITICAL_K = 5
# The judge behind the published table. Free-tier daily caps differ by an order
# of magnitude between models (500/day on the lite tiers against 20/day on
# gemini-3.5-flash), so the default is a model whose cap fits the whole
# population in one sitting.
DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-flash-lite")


# ---------------------------------------------------------------------------
# Condition serialisation
# ---------------------------------------------------------------------------


#: Fields removed from a trace before the judge sees it, in *both* conditions.
#: `injected_fault` carries `target_step_seq`, which is the answer the judge is
#: being asked for, and `ground_truth` is the expected answer. Serialising a whole
#: `Run` would hand the judge the answer key and condition A would score near
#: 100% for a reason that has nothing to do with reading the trace.
BLINDED_RUN_FIELDS = ("injected_fault", "ground_truth")

#: The committed corpus was generated before `harness/faults.py` stopped naming
#: the injected chunk "injected-contradiction". That id appears in the retrieval
#: step's `evidence_refs` and in its rendered output, so a judge could locate the
#: faulted step by reading the label rather than by reasoning about the trace.
#: It is rewritten to a neutral, corpus-shaped id in **both** conditions with the
#: same substitution, so the blinding cannot advantage either one. Corpora
#: generated after that fix need no rewriting; this is retained so the committed
#: corpus stays usable without regeneration.
_GIVEAWAY_CHUNK_ID = "injected-contradiction"
_NEUTRAL_CHUNK_ID = "doc-47#c0"


def blind_text(text: str) -> str:
    """Remove fault labels a judge could read instead of reasoning."""
    return text.replace(_GIVEAWAY_CHUNK_ID, _NEUTRAL_CHUNK_ID)


def blind(run: Run) -> dict[str, Any]:
    """The trace as the judge sees it, with ground truth removed."""
    payload = json.loads(run.model_dump_json())
    for field in BLINDED_RUN_FIELDS:
        payload.pop(field, None)
    return payload


def serialise_raw_log(run: Run) -> str:
    """Condition A: the whole trace as JSON, which is what a developer reads."""
    return blind_text(json.dumps(blind(run), indent=2))


def serialise_extracted(run: Run) -> str:
    """Condition B: the top-5 critical steps plus the final output.

    Rendered as readable text rather than JSON. That is a deliberate part of the
    condition being tested: the claim under test is that a short, structured
    summary beats a long raw log, and the summary's presentation is part of the
    treatment. It is noted as a confound in the README, since format and length
    change together.
    """
    ranked = rank_critical(run, k=CRITICAL_K)
    by_seq = {step.seq: step for step in run.steps}

    lines = [
        f"task: {run.task_input}",
        f"workflow: {run.workflow_type} v{run.workflow_version}",
        f"total steps in run: {len(run.steps)}",
        "",
        f"top {len(ranked)} steps by influence on the outcome:",
    ]
    for score in ranked:
        step = by_seq[score.seq]
        lines.append("")
        lines.append(
            f"  seq {score.seq} | {score.event_type} | agent {score.agent_id} "
            f"({score.agent_role}) | critical score {score.critical_score:.3f}"
        )
        lines.append(
            f"    signals: evidence_survival={score.evidence_survival:.3f} "
            f"branch={score.branch:.1f} error={score.error:.1f}"
        )
        if score.reasons:
            lines.append(f"    notes: {'; '.join(score.reasons)}")
        if step.evidence_refs:
            lines.append(f"    evidence_refs: {', '.join(step.evidence_refs)}")
        if step.error is not None:
            lines.append(
                f"    error: {step.error.error_type}: {step.error.message}"
            )
        lines.append(f"    input: {step.input}")
        lines.append(f"    output: {step.output}")

    lines.extend(["", f"final output: {run.final_output}"])
    return blind_text("\n".join(lines))


SERIALISERS = {
    "raw_log": serialise_raw_log,
    "extracted": serialise_extracted,
}


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


#: Any occurrence of this in a serialised trace means the fault is still
#: identifiable by name rather than by reasoning. Substring blinding removes the
#: exact chunk id, but an agent that *read* that id can paraphrase it into its
#: own prose — one run's final answer says "corrected by the injection notice",
#: which no literal substitution catches. A judge shown that text can locate the
#: faulted step without inspecting the trace at all.
_FAULT_RESIDUE = re.compile(r"inject", re.IGNORECASE)


def is_compromised(run: Run) -> bool:
    """True if either condition would still name the fault to the judge.

    Checked against the *blinded* serialisations, so this catches only what
    survives blinding. The test is on the trace text and is independent of what
    the judge answered, and it excludes a run from both conditions together, so
    it cannot favour either one.
    """
    return bool(
        _FAULT_RESIDUE.search(serialise_raw_log(run))
        or _FAULT_RESIDUE.search(serialise_extracted(run))
    )


def load_checkpoint(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Checkpointed trials, one row per run and condition.

    Rows are keyed rather than appended. A checkpoint accumulated across several
    sessions can hold more than one row for the same run and condition, and only
    the newest was judged against the current corpus. Appending blindly would
    count a run twice and inflate n, which is a silent corruption of every
    accuracy figure downstream.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[(row["run_id"], row["condition"])] = row

    trials: dict[str, list[dict[str, Any]]] = {c: [] for c in CONDITIONS}
    for row in latest.values():
        if row["condition"] in trials:
            trials[row["condition"]].append(row)
    return trials


def resolve_output_path(out_path: Path, *, complete: bool) -> tuple[Path, dict[str, Any]]:
    """Where to write this study, and what it would displace.

    An incomplete run must never overwrite a complete one. The published table is
    read from this file, and a run cut short by a daily quota is
    indistinguishable from a finished study once written: same fields, smaller n.
    It is diverted to a ``_partial`` sibling instead.

    Returns ``(path, displaced)``, where ``displaced`` is the existing complete
    study when the write was diverted and empty otherwise.
    """
    if complete or not out_path.exists():
        return out_path, {}
    try:
        existing = json.loads(out_path.read_text())
    except (OSError, json.JSONDecodeError):
        return out_path, {}
    if not existing.get("population_complete"):
        return out_path, {}
    return out_path.with_name(out_path.stem + "_partial.json"), existing


def load_population(
    corpus_dir: Path, *, exclude_compromised: bool = True
) -> tuple[list[Run], list[Run]]:
    """Failed runs with a known injected fault, in a stable order.

    Returns ``(population, excluded)``. Runs whose trace still names the injected
    fault after blinding are excluded by default: leaving them in measures how
    well a judge can read a label, not how well it can read a trace.
    """
    eligible = []
    for path in sorted(corpus_dir.glob("run_*.json")):
        run = Run.model_validate(json.loads(path.read_text()))
        if not run.success and run.injected_fault is not None:
            eligible.append(run)

    if not exclude_compromised:
        return eligible, []

    excluded = [run for run in eligible if is_compromised(run)]
    excluded_ids = {run.run_id for run in excluded}
    return [r for r in eligible if r.run_id not in excluded_ids], excluded


def summarise(trials: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [t for t in trials if t["correct"]]
    unparseable = [t for t in trials if t["predicted_seq"] is None]
    return {
        "n": len(trials),
        "correct": len(correct),
        "accuracy": round(len(correct) / len(trials), 4) if trials else 0.0,
        "mean_input_tokens": (
            round(statistics.mean(t["prompt_tokens"] for t in trials), 1)
            if trials
            else 0.0
        ),
        "median_input_tokens": (
            round(statistics.median(t["prompt_tokens"] for t in trials), 1)
            if trials
            else 0.0
        ),
        "mean_latency_ms": (
            round(statistics.mean(t["latency_ms"] for t in trials), 1)
            if trials
            else 0.0
        ),
        "unparseable_responses": len(unparseable),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rpm-limit", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the population size (for a quick check, not for reporting)",
    )
    parser.add_argument(
        "--stub-llm",
        action="store_true",
        help="smoke-test the pipeline without a key; results are meaningless",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="ignore the checkpoint and judge every run again",
    )
    parser.add_argument(
        "--include-compromised",
        action="store_true",
        help="keep runs whose trace still names the injected fault after "
        "blinding (reported for comparison only; not a valid primary result)",
    )
    args = parser.parse_args(argv)

    population, excluded = load_population(
        args.corpus, exclude_compromised=not args.include_compromised
    )
    if args.limit:
        population = population[: args.limit]
    if excluded:
        print(
            f"Excluded {len(excluded)} run(s) whose trace still names the "
            "injected fault after blinding; they would measure label-reading, "
            "not trace-reading."
        )

    if not population:
        print(
            f"no failed, fault-injected runs found in {args.corpus}",
            file=sys.stderr,
        )
        return 1

    try:
        client = build_client(
            args.model, stub=args.stub_llm, rpm_limit=args.rpm_limit
        )
    except LLMUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Population: {len(population)} failed runs with a known injected "
        f"fault.\nJudge: {'stub' if args.stub_llm else args.model}, identical "
        f"prompt in both conditions.\n"
    )

    # Trials are checkpointed after every run and reloaded on start. Free-tier
    # daily quotas are small enough that a full population may not fit in one
    # day, and losing a completed half of the study to a quota wall would mean
    # re-spending quota that had already been paid.
    # The checkpoint name carries the judge model. The study requires one judge
    # across the whole population, and resuming a run started under a different
    # model would silently blend two judges into one accuracy figure.
    judge_id = "stub" if args.stub_llm else args.model
    checkpoint = args.out / f"primary_study_trials_{judge_id}.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)

    trials: dict[str, list[dict[str, Any]]] = {c: [] for c in CONDITIONS}
    done_run_ids: set[str] = set()
    if args.restart and checkpoint.exists():
        # Trials are appended, so ignoring the checkpoint is not enough: the old
        # rows would stay in the file and a later resume would load them
        # alongside the new ones. Those rows may have been judged against a
        # different corpus, which is exactly the contamination the checkpoint is
        # meant to avoid. --restart therefore discards them.
        checkpoint.unlink()
    if checkpoint.exists():
        trials = load_checkpoint(checkpoint)
        population_ids = {r.run_id for r in population}
        complete = {
            row["run_id"]
            for row in trials[CONDITIONS[0]]
            if row["run_id"] in {r["run_id"] for r in trials[CONDITIONS[1]]}
            # A checkpointed trial for a run no longer in the population must
            # not be counted; otherwise excluding a run would leave its result
            # in the summary.
            and row["run_id"] in population_ids
        }
        done_run_ids = complete
        # Drop any half-finished run so it is retried as a pair, keeping the two
        # conditions on exactly the same population.
        for condition in CONDITIONS:
            trials[condition] = [
                row for row in trials[condition] if row["run_id"] in done_run_ids
            ]
        if done_run_ids:
            print(f"Resuming: {len(done_run_ids)} run(s) already judged.\n")

    started = time.monotonic()
    quota_exhausted = False

    with checkpoint.open("a") as handle:
        for index, run in enumerate(population, start=1):
            assert run.injected_fault is not None
            if run.run_id in done_run_ids:
                continue
            actual = run.injected_fault.target_step_seq

            pending: list[dict[str, Any]] = []
            try:
                for condition in CONDITIONS:
                    trace_text = SERIALISERS[condition](run)
                    verdict = ask_judge(client, trace_text)
                    pending.append(
                        {
                            "condition": condition,
                            "run_id": run.run_id,
                            "workflow_type": run.workflow_type,
                            "fault_type": run.injected_fault.fault_type,
                            "actual_seq": actual,
                            "predicted_seq": verdict.predicted_seq,
                            "correct": verdict.predicted_seq == actual,
                            "prompt_tokens": verdict.prompt_tokens,
                            "latency_ms": verdict.latency_ms,
                            "raw_response": verdict.raw_response[:120],
                        }
                    )
            except LLMUnavailable as exc:
                # Stop cleanly and report the partial population rather than
                # dying with nothing written.
                print(f"\n  judge unavailable, stopping early: {exc}", file=sys.stderr)
                quota_exhausted = True
                break

            # Both conditions succeeded, so the pair is committed together.
            for row in pending:
                trials[row["condition"]].append(row)
                handle.write(json.dumps(row) + "\n")
            handle.flush()

            summary_bits = [
                f"{row['condition']}: {row['predicted_seq']} "
                f"{'HIT ' if row['correct'] else 'miss'} ({row['prompt_tokens']}tok)"
                for row in pending
            ]
            print(
                f"  [{index:>3}/{len(population)}] "
                f"{run.injected_fault.fault_type:<24} actual={actual}  "
                + "  |  ".join(summary_bits)
            )

    judged = len(trials[CONDITIONS[0]])
    if judged == 0:
        print("no trials completed; nothing to report", file=sys.stderr)
        return 2

    # Per-fault-type breakdown, and the both-conditions-failed count the SRS
    # asks for in section 10.3.
    by_fault_type: dict[str, dict[str, Any]] = defaultdict(dict)
    for condition in CONDITIONS:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trial in trials[condition]:
            grouped[trial["fault_type"]].append(trial)
        for fault_type, rows in grouped.items():
            by_fault_type[fault_type][condition] = {
                "n": len(rows),
                "correct": sum(1 for r in rows if r["correct"]),
                "accuracy": round(
                    sum(1 for r in rows if r["correct"]) / len(rows), 4
                ),
            }

    raw_by_run = {t["run_id"]: t["correct"] for t in trials["raw_log"]}
    extracted_by_run = {t["run_id"]: t["correct"] for t in trials["extracted"]}
    both_failed = sum(
        1
        for run_id in raw_by_run
        if not raw_by_run[run_id] and not extracted_by_run.get(run_id)
    )
    both_correct = sum(
        1
        for run_id in raw_by_run
        if raw_by_run[run_id] and extracted_by_run.get(run_id)
    )

    conditions_summary = {c: summarise(trials[c]) for c in CONDITIONS}
    token_reduction = None
    raw_tokens = conditions_summary["raw_log"]["mean_input_tokens"]
    if raw_tokens:
        token_reduction = round(
            1 - conditions_summary["extracted"]["mean_input_tokens"] / raw_tokens,
            4,
        )

    result: dict[str, Any] = {
        "study": "primary: raw log versus extracted critical-trace summary",
        # What was actually judged, which may be less than the eligible
        # population if a free-tier daily quota was reached.
        "n": judged,
        "eligible_population": len(population),
        "excluded_compromised": [
            {
                "run_id": r.run_id,
                "fault_type": r.injected_fault.fault_type if r.injected_fault else None,
                "reason": "trace still names the injected fault after blinding",
            }
            for r in excluded
        ],
        "population_complete": judged >= len(population),
        "stopped_early_on_quota": quota_exhausted,
        "judge_model": "stub" if args.stub_llm else args.model,
        "judge_prompt_identical_across_conditions": True,
        "judge_prompt": JUDGE_PROMPT,
        "judge_system": JUDGE_SYSTEM,
        "critical_k": CRITICAL_K,
        "conditions": conditions_summary,
        "by_fault_type": dict(by_fault_type),
        "agreement": {
            "both_correct": both_correct,
            "both_failed": both_failed,
            "raw_only": sum(
                1
                for r in raw_by_run
                if raw_by_run[r] and not extracted_by_run.get(r)
            ),
            "extracted_only": sum(
                1
                for r in raw_by_run
                if not raw_by_run[r] and extracted_by_run.get(r)
            ),
        },
        "mean_input_token_reduction": token_reduction,
        "wall_clock_seconds": round(time.monotonic() - started, 1),
        "is_stub": args.stub_llm,
        "trials": trials,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / (
        "primary_study_stub.json" if args.stub_llm else "primary_study.json"
    )

    intended = out_path
    out_path, displaced = resolve_output_path(
        out_path, complete=result["population_complete"]
    )
    if displaced:
        print(
            f"\n  incomplete run: {judged}/{len(population)} judged under "
            f"{result['judge_model']}. Not overwriting {intended.name}, which "
            f"holds a complete study of {displaced.get('n')} run(s) under "
            f"{displaced.get('judge_model')}.\n  Writing to {out_path.name} "
            f"instead. Rerun without --restart to resume onto the full "
            f"population.",
            file=sys.stderr,
        )

    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 68)
    print(f"{'condition':<14}{'n':>5}{'correct':>9}{'accuracy':>11}"
          f"{'mean tokens':>14}{'mean ms':>10}")
    print("-" * 68)
    for condition in CONDITIONS:
        s = conditions_summary[condition]
        print(
            f"{condition:<14}{s['n']:>5}{s['correct']:>9}"
            f"{s['accuracy'] * 100:>10.1f}%{s['mean_input_tokens']:>14.0f}"
            f"{s['mean_latency_ms']:>10.0f}"
        )
    print("=" * 68)
    if token_reduction is not None:
        print(f"input token reduction: {token_reduction * 100:.1f}%")
    print(
        f"both correct: {both_correct}   both failed: {both_failed}   "
        f"raw only: {result['agreement']['raw_only']}   "
        f"extracted only: {result['agreement']['extracted_only']}"
    )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
