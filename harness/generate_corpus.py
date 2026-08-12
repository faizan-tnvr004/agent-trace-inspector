"""CLI that generates the trace corpus.

Fault assignment is round-robin over the fault types each workflow supports,
not random. The corpus has to satisfy hard minimums (every fault type present
at least five times), and sampling randomly would meet them only most of the
time. A corpus that silently comes up short on one fault type would weaken the
per-fault-type breakdown in the primary study without anything failing.

Usage::

    python harness/generate_corpus.py --workflow both --n 60 --fault-rate 0.5 \\
        --out data/corpus/

Add ``--stub-llm`` to exercise the whole pipeline with no API key and no
quota. Stub runs are written with a ``+stub`` workflow_version suffix and are
never committed as corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.models import Run  # noqa: E402
from harness.faults import ALL_FAULT_TYPES, applicable_faults  # noqa: E402
from harness.llm import LLMUnavailable, build_client  # noqa: E402
from harness.workflows import rag_qa, reviewer_pipeline  # noqa: E402

WORKFLOWS = ("reviewer_pipeline", "rag_qa")
DEFAULT_MODEL = os.environ.get("CORPUS_MODEL", "gemini-2.0-flash")


def plan_faults(
    workflow: str, n: int, fault_rate: float, rng: random.Random
) -> list[str | None]:
    """Decide, per run, which fault (if any) to inject.

    Faulted positions are spread evenly and the fault types cycle, so the
    counts are exact rather than approximate.
    """
    types = applicable_faults(workflow)
    n_faulted = int(round(n * fault_rate))
    if not types:
        return [None] * n

    positions = sorted(rng.sample(range(n), min(n_faulted, n)))
    plan: list[str | None] = [None] * n
    for i, pos in enumerate(positions):
        plan[pos] = types[i % len(types)]
    return plan


def _tasks_for(workflow: str) -> list[dict[str, Any]]:
    if workflow == "reviewer_pipeline":
        return json.loads(
            (REPO_ROOT / "harness" / "tasks" / "math_tasks.json").read_text()
        )
    _, questions = rag_qa.load_corpus()
    return questions


def generate(
    workflow: str,
    n: int,
    fault_rate: float,
    client: Any,
    rng: random.Random,
    *,
    stub: bool,
) -> list[Run]:
    tasks = _tasks_for(workflow)
    plan = plan_faults(workflow, n, fault_rate, rng)
    corpus_chunks = rag_qa.load_corpus()[0] if workflow == "rag_qa" else []

    runs: list[Run] = []
    for i in range(n):
        task = tasks[i % len(tasks)]
        fault_type = plan[i]
        started = time.monotonic()
        try:
            if workflow == "reviewer_pipeline":
                run = reviewer_pipeline.run_reviewer_pipeline(
                    task, client, fault_type=fault_type, rng=rng
                )
            else:
                run = rag_qa.run_rag_qa(
                    task, client, corpus_chunks, fault_type=fault_type, rng=rng
                )
        except LLMUnavailable:
            raise
        except Exception as exc:  # a crashed workflow still yields a trace
            print(f"  run {i + 1}/{n} raised {type(exc).__name__}: {exc}")
            continue

        if stub:
            # Mark stub traces so they can never be mistaken for real runs.
            run = run.model_copy(
                update={"workflow_version": f"{run.workflow_version}+stub"}
            )
        runs.append(run)

        elapsed = time.monotonic() - started
        flag = fault_type or "-"
        status = "ok " if run.success else "FAIL"
        print(
            f"  [{workflow}] run {i + 1:>3}/{n}  {status}  "
            f"steps={len(run.steps):<2} fault={flag:<24} {elapsed:5.1f}s"
        )
    return runs


def summarise(runs: list[Run]) -> dict[str, Any]:
    faults = Counter(
        r.injected_fault.fault_type for r in runs if r.injected_fault is not None
    )
    by_workflow = Counter(r.workflow_type for r in runs)
    failed_and_faulted = [
        r for r in runs if not r.success and r.injected_fault is not None
    ]
    return {
        "runs": len(runs),
        "by_workflow": dict(by_workflow),
        "successful": sum(1 for r in runs if r.success),
        "failed": sum(1 for r in runs if not r.success),
        "faulted": sum(1 for r in runs if r.injected_fault is not None),
        "fault_types": dict(faults),
        "failed_and_faulted": len(failed_and_faulted),
        "total_notional_cost_usd": round(sum(r.total_cost_usd for r in runs), 6),
        "total_tokens": sum(r.total_tokens for r in runs),
        "total_steps": sum(len(r.steps) for r in runs),
    }


def check_targets(summary: dict[str, Any], expected_total: int) -> list[str]:
    """Report shortfalls rather than silently accepting a thin corpus."""
    problems = []
    if summary["runs"] < expected_total:
        problems.append(
            f"generated {summary['runs']} runs, expected {expected_total}"
        )
    missing = [f for f in ALL_FAULT_TYPES if summary["fault_types"].get(f, 0) < 5]
    if missing:
        problems.append(
            "fault types under the minimum of 5: "
            + ", ".join(f"{f}={summary['fault_types'].get(f, 0)}" for f in missing)
        )
    if summary["faulted"] * 2 < summary["runs"]:
        problems.append(
            f"only {summary['faulted']}/{summary['runs']} runs carry a fault, "
            "target is at least half"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow", choices=[*WORKFLOWS, "both"], default="both"
    )
    parser.add_argument("--n", type=int, default=60, help="total runs to generate")
    parser.add_argument("--fault-rate", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("data/corpus"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--stub-llm",
        action="store_true",
        help="use deterministic stub responses; no API key or quota needed. "
        "Output is for smoke-testing only and is never committed as corpus.",
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=None,
        help="requests per minute cap (default: LLM_RPM_LIMIT or 10)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="add to an existing corpus directory instead of replacing it, so "
        "each workflow can be generated with its own fault rate",
    )
    parser.add_argument(
        "--expect-total",
        type=int,
        default=None,
        help="total corpus size to check targets against (default: --n). Use "
        "when building a corpus across several appending passes.",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    try:
        client = build_client(
            args.model, stub=args.stub_llm, rpm_limit=args.rpm_limit
        )
    except LLMUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    workflows = list(WORKFLOWS) if args.workflow == "both" else [args.workflow]
    per_workflow = args.n // len(workflows)

    print(
        f"Generating {args.n} runs across {len(workflows)} workflow(s) using "
        f"{'stub responses' if args.stub_llm else args.model}"
    )

    runs: list[Run] = []
    for workflow in workflows:
        runs.extend(
            generate(
                workflow,
                per_workflow,
                args.fault_rate,
                client,
                rng,
                stub=args.stub_llm,
            )
        )

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    start = 1
    if args.append:
        existing_names = sorted(out_dir.glob("run_*.json"))
        start = len(existing_names) + 1
    else:
        for existing in sorted(out_dir.glob("run_*.json")):
            existing.unlink()

    for offset, run in enumerate(runs):
        path = out_dir / f"run_{start + offset:04d}.json"
        path.write_text(json.dumps(json.loads(run.model_dump_json()), indent=2))

    # Summarise everything in the directory rather than just this invocation's
    # runs. The two workflows are generated in separate passes with different
    # fault rates, because their faults have very different potency, and the
    # corpus targets apply to the whole corpus.
    all_runs = [
        Run.model_validate(json.loads(p.read_text()))
        for p in sorted(out_dir.glob("run_*.json"))
    ]
    summary = summarise(all_runs)
    # Written beside the corpus directory rather than inside it. Downstream
    # consumers glob `data/corpus/*.json` and expect every match to be a run,
    # so a summary file living in there would be parsed as a trace and fail.
    summary_path = out_dir.parent / f"{out_dir.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nCorpus summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    problems = check_targets(summary, args.expect_total or args.n)
    if problems:
        print("\nTargets not met:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"\nWrote {len(runs)} runs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
