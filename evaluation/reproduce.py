"""Recompute every published number that does not need an API call.

`make reproduce` has to regenerate the README's figures from committed code
(NFR-5). Most of them do not involve a model at all: the extraction engine is
deterministic and embedding-only, so corpus statistics, fault potency, the
heuristic attribution accuracy and the rejection-outcome distribution are all
recoverable from the committed corpus alone. Only the primary study needs the
judge, and that is invoked separately.

Corpus *generation* is deliberately not reproduced here. It needs API quota and
model output is not bit-reproducible, which is precisely why the corpus is
committed rather than regenerated.

Usage::

    python evaluation/reproduce.py --corpus data/corpus --out evaluation/results
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "backend", REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.extraction.attribution import predict_failure_origin  # noqa: E402
from app.extraction.claims import analyse_claims  # noqa: E402
from app.extraction.rejection import rejection_summary  # noqa: E402
from app.models import Run  # noqa: E402


def load_corpus(corpus_dir: Path) -> list[Run]:
    paths = sorted(corpus_dir.glob("run_*.json"))
    if not paths:
        raise SystemExit(f"no run_*.json found in {corpus_dir}")
    return [Run.model_validate(json.loads(p.read_text())) for p in paths]


def corpus_stats(runs: list[Run]) -> dict[str, Any]:
    faulted = [r for r in runs if r.injected_fault is not None]
    return {
        "runs": len(runs),
        "by_workflow": dict(Counter(r.workflow_type for r in runs)),
        "successful": sum(1 for r in runs if r.success),
        "failed": sum(1 for r in runs if not r.success),
        "faulted": len(faulted),
        "faulted_share": round(len(faulted) / len(runs), 4) if runs else 0.0,
        "fault_types": dict(
            Counter(r.injected_fault.fault_type for r in faulted)  # type: ignore[union-attr]
        ),
        "failed_and_faulted": sum(1 for r in faulted if not r.success),
        "steps_total": sum(len(r.steps) for r in runs),
        "steps_mean": round(statistics.mean(len(r.steps) for r in runs), 2),
        "steps_min": min(len(r.steps) for r in runs),
        "steps_max": max(len(r.steps) for r in runs),
        "total_notional_cost_usd": round(sum(r.total_cost_usd for r in runs), 6),
        "total_tokens": sum(r.total_tokens for r in runs),
    }


def fault_potency(runs: list[Run]) -> dict[str, Any]:
    """How often each fault type actually changed the outcome.

    A fault that never causes a failure produces no failed run to attribute, so
    this table explains the shape of the evaluation population rather than being
    an incidental statistic.
    """
    grouped: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        if run.injected_fault is not None:
            grouped[(run.workflow_type, run.injected_fault.fault_type)].append(run)

    out: dict[str, Any] = {}
    for (workflow, fault_type), group in sorted(grouped.items()):
        failed = sum(1 for r in group if not r.success)
        out.setdefault(workflow, {})[fault_type] = {
            "injected": len(group),
            "caused_failure": failed,
            "potency": round(failed / len(group), 4) if group else 0.0,
        }
    return out


def attribution_accuracy(runs: list[Run]) -> dict[str, Any]:
    """Accuracy of the heuristic attribution engine, with no LLM involved.

    This is the tool's own attribution, distinct from the judge's accuracy in the
    primary study. Reported separately because they answer different questions:
    this one asks whether the heuristic localises the fault, the study asks
    whether a reader given the extraction does.
    """
    population = [r for r in runs if not r.success and r.injected_fault is not None]
    if not population:
        return {"n": 0}

    by_fault: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "no_prediction": 0}
    )
    hits = 0
    no_prediction = 0

    for run in population:
        result = predict_failure_origin(run)
        fault_type = run.injected_fault.fault_type  # type: ignore[union-attr]
        by_fault[fault_type]["n"] += 1
        if result.predicted_step_seq is None:
            no_prediction += 1
            by_fault[fault_type]["no_prediction"] += 1
        elif result.is_correct():
            hits += 1
            by_fault[fault_type]["correct"] += 1

    return {
        "n": len(population),
        "correct": hits,
        "accuracy": round(hits / len(population), 4),
        "no_prediction": no_prediction,
        "by_fault_type": {
            fault_type: {
                **counts,
                "accuracy": round(counts["correct"] / counts["n"], 4)
                if counts["n"]
                else 0.0,
            }
            for fault_type, counts in sorted(by_fault.items())
        },
    }


def rejection_stats(runs: list[Run]) -> dict[str, Any]:
    """Rejection-outcome distribution, the taxonomy from the prior study.

    Split into all critiques and rejections only. Most critiques in this corpus
    accept the answer, and an acceptance has no rejection outcome to speak of, so
    a single blended no-change rate would be dominated by approvals and would not
    mean what the prior study's no-change rate meant.
    """
    summary = rejection_summary(list(runs))
    runs_with_critiques = sum(
        1 for run in runs if any(s.event_type == "critique" for s in run.steps)
    )
    return {
        "runs_with_critiques": runs_with_critiques,
        **summary,
    }


def provenance_stats(runs: list[Run]) -> dict[str, Any]:
    """How much of each final answer the trace actually grounds."""
    totals, unsupported, runs_with_unsupported = 0, 0, 0
    for run in runs:
        claims = analyse_claims(run)
        totals += len(claims)
        missing = sum(1 for c in claims if not c.supported)
        unsupported += missing
        if missing:
            runs_with_unsupported += 1
    return {
        "claims_total": totals,
        "claims_unsupported": unsupported,
        "unsupported_share": round(unsupported / totals, 4) if totals else 0.0,
        "runs_with_at_least_one_unsupported_claim": runs_with_unsupported,
        "runs": len(runs),
    }


def determinism_check(runs: list[Run]) -> dict[str, Any]:
    """Verify FR-16 over the real corpus, not only over fixtures."""
    sample = runs[: min(10, len(runs))]
    stable = all(
        predict_failure_origin(r) == predict_failure_origin(r) for r in sample
    )
    return {"runs_checked": len(sample), "deterministic": stable}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/results"))
    args = parser.parse_args(argv)

    started = time.monotonic()
    runs = load_corpus(args.corpus)

    result = {
        "corpus": corpus_stats(runs),
        "fault_potency": fault_potency(runs),
        "heuristic_attribution": attribution_accuracy(runs),
        "rejection_outcomes": rejection_stats(runs),
        "provenance": provenance_stats(runs),
        "determinism": determinism_check(runs),
        "wall_clock_seconds": round(time.monotonic() - started, 1),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "corpus_and_extraction.json"
    out_path.write_text(json.dumps(result, indent=2))

    c = result["corpus"]
    a = result["heuristic_attribution"]
    r = result["rejection_outcomes"]
    p = result["provenance"]

    print(f"Corpus: {c['runs']} runs, {c['by_workflow']}")
    print(
        f"  {c['failed']} failed, {c['faulted']} faulted "
        f"({c['faulted_share'] * 100:.0f}%), "
        f"{c['failed_and_faulted']} failed and faulted"
    )
    print(f"  fault types: {c['fault_types']}")
    print(
        f"  steps: {c['steps_total']} total, mean {c['steps_mean']}, "
        f"range {c['steps_min']}-{c['steps_max']}"
    )
    print(f"  notional cost: ${c['total_notional_cost_usd']:.6f}")

    print("\nFault potency (share of injected faults that changed the outcome):")
    for workflow, faults in result["fault_potency"].items():
        print(f"  {workflow}")
        for fault_type, stats in faults.items():
            print(
                f"    {fault_type:<24} {stats['caused_failure']:>3}/"
                f"{stats['injected']:<3} = {stats['potency'] * 100:>5.1f}%"
            )

    print(f"\nHeuristic attribution: {a.get('correct', 0)}/{a.get('n', 0)}"
          f" = {a.get('accuracy', 0) * 100:.1f}%"
          f" ({a.get('no_prediction', 0)} with no prediction)")
    for fault_type, stats in a.get("by_fault_type", {}).items():
        print(
            f"  {fault_type:<24} {stats['correct']:>3}/{stats['n']:<3} = "
            f"{stats['accuracy'] * 100:>5.1f}%"
        )

    print(f"\nRejection outcomes ({r['runs_with_critiques']} runs have critiques)")
    for population in ("all_critiques", "rejections_only"):
        block = r[population]
        print(f"  {population} (n={block['total']}): {block['counts']}")
        for outcome, rate in block["rates"].items():
            print(f"    {outcome:<10} {rate * 100:>5.1f}%")

    print(
        f"\nProvenance: {p['claims_unsupported']}/{p['claims_total']} claims "
        f"unsupported ({p['unsupported_share'] * 100:.1f}%), affecting "
        f"{p['runs_with_at_least_one_unsupported_claim']}/{p['runs']} runs"
    )
    print(
        f"\nDeterminism: {'OK' if result['determinism']['deterministic'] else 'FAILED'}"
        f" over {result['determinism']['runs_checked']} runs"
    )
    print(f"\nWrote {out_path} in {result['wall_clock_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
