"""Fill in `rejection_outcome` on the critique steps of a generated corpus.

Not in the specification's file tree. It exists because of an ordering problem:
FR-15 requires the classification to be *recorded* in `Step.rejection_outcome`,
but the classifier lives in `app.extraction.rejection`, which is Phase 3, while
the corpus is produced in Phase 2. Rather than duplicating the taxonomy inside
the workflow, the corpus is generated first and stamped afterwards with the
single authoritative implementation.

Idempotent: re-running it recomputes the same values from the same traces.

Usage::

    python harness/stamp_rejections.py --corpus data/corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "backend", REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.extraction.rejection import classify_all_rejections  # noqa: E402
from app.models import Run  # noqa: E402


def stamp_file(path: Path) -> tuple[int, Counter[str]]:
    """Stamp one corpus file. Returns (critiques stamped, outcome counts)."""
    payload = json.loads(path.read_text())
    run = Run.model_validate(payload)

    outcomes = classify_all_rejections(run)
    if not outcomes:
        return 0, Counter()

    counts: Counter[str] = Counter()
    stamped = 0
    for step in payload["steps"]:
        outcome = outcomes.get(step["step_id"])
        if step["event_type"] != "critique" or outcome is None:
            continue
        step["rejection_outcome"] = outcome
        counts[outcome] += 1
        stamped += 1

    # Re-validate before writing: invariant 4 forbids a rejection_outcome on any
    # step that is not a critique, so a stamping bug must fail here rather than
    # write an invalid trace into the corpus.
    Run.model_validate(payload)
    path.write_text(json.dumps(payload, indent=2))
    return stamped, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    args = parser.parse_args(argv)

    files = sorted(args.corpus.glob("run_*.json"))
    if not files:
        print(f"no run_*.json found in {args.corpus}", file=sys.stderr)
        return 1

    total = 0
    counts: Counter[str] = Counter()
    runs_with_critiques = 0
    for path in files:
        stamped, file_counts = stamp_file(path)
        if stamped:
            runs_with_critiques += 1
        total += stamped
        counts.update(file_counts)

    print(f"stamped {total} critique step(s) across {runs_with_critiques} run(s)")
    for outcome in ("repair", "damage", "no_change"):
        share = counts[outcome] / total * 100 if total else 0.0
        print(f"  {outcome:<10} {counts[outcome]:>4}  ({share:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
