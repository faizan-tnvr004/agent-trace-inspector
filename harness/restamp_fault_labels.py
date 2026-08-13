"""Correct `injected_fault.target_step_seq` on an already-generated corpus.

Why this exists rather than a regeneration
------------------------------------------

A fault acting on retrieved context is re-applied on every retrieval round, and
the generator used to record the *last* application. The error originates at the
first. The generator is fixed, but regenerating the corpus would spend quota and
mint fresh run ids for traces whose steps are already correct: only one integer
per run is wrong.

The corrected value is recoverable from the trace itself. The fault is applied
immediately after each retrieval is computed and before that retrieval is
recorded, so the earliest application is by construction the run's first
`retrieval` step. Nothing about the steps changes, so the judge's answers stay
valid and only the scoring against them moves.

Scope is deliberately narrow. Only `dropped_retrieval` and
`injected_contradiction` on `rag_qa` are re-applied per round;
`truncated_tool_result` and `forced_false_rejection` are applied once at their
own step and are left untouched.

Usage::

    python harness/restamp_fault_labels.py --corpus data/corpus --check
    python harness/restamp_fault_labels.py --corpus data/corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.models import Run  # noqa: E402

#: Faults that are re-applied on every retrieval round, and so can carry a
#: target_step_seq later than the step that introduced the error.
PER_ROUND_FAULTS = frozenset({"dropped_retrieval", "injected_contradiction"})


def corrected_seq(run: Run) -> int | None:
    """The seq the answer key should carry, or None if there is nothing to fix."""
    fault = run.injected_fault
    if fault is None or fault.fault_type not in PER_ROUND_FAULTS:
        return None
    retrievals = [s.seq for s in run.steps if s.event_type == "retrieval"]
    if not retrievals:
        return None
    return retrievals[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change and exit non-zero if anything would",
    )
    args = parser.parse_args(argv)

    paths = sorted(args.corpus.glob("run_*.json"))
    if not paths:
        print(f"no runs found in {args.corpus}", file=sys.stderr)
        return 2

    changed = []
    for path in paths:
        payload = json.loads(path.read_text())
        run = Run.model_validate(payload)
        target = corrected_seq(run)
        if target is None:
            continue
        assert run.injected_fault is not None
        current = run.injected_fault.target_step_seq
        if current == target:
            continue

        changed.append((path.name, run.injected_fault.fault_type, current, target))
        if not args.check:
            payload["injected_fault"]["target_step_seq"] = target
            # Re-validate before writing: a corrected seq still has to satisfy
            # the invariant that the fault targets a step that exists.
            Run.model_validate(payload)
            path.write_text(json.dumps(payload, indent=2))

    verb = "would correct" if args.check else "corrected"
    print(f"{verb} {len(changed)} of {len(paths)} run(s)")
    for name, fault_type, current, target in changed[:10]:
        print(f"  {name}  {fault_type}: {current} -> {target}")
    if len(changed) > 10:
        print(f"  ... and {len(changed) - 10} more")

    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
