"""Label review comments with their downstream outcome, and report the results.

The taxonomy is the same one used for critiques inside synthetic agent traces:

* **repair** — the comment produced a genuine fix
* **damage** — the change made in response made things worse
* **no_change** — the comment was not acted on, without consequence

This step is manual and cannot be automated: the repair/damage/no-change
judgement *is* the taxonomy. FR-32 is explicit about that, and an automated
labeller would be measuring a proxy for the thing under study rather than the
thing itself.

Labels are written back after every decision, so the session is resumable and an
interrupted run loses nothing.

Usage::

    python pr_corpus/label.py --data pr_corpus/data/comments.jsonl
    python pr_corpus/label.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUTCOMES = {
    "r": "repair",
    "d": "damage",
    "n": "no_change",
}

LABELLING_RULE = """Labelling rule, applied consistently to every comment:

  repair     the comment identified a real problem AND a following commit
             changed the code it pointed at in a way that addresses it
  damage     a following commit changed the code in response and made it
             worse, or the comment itself was wrong and was acted on
  no_change  no following commit touched what the comment pointed at, or the
             change was cosmetic, or the comment was not a criticism

Judge only from the comment, the diff hunk it is attached to, and whether
commits followed. Do not judge from the commenter's tone or seniority."""


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run pr_corpus/collect.py first.")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write atomically: a crash mid-write must not truncate the corpus."""
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    temp.replace(path)


def label_interactively(path: Path, rows: list[dict[str, Any]]) -> None:
    pending = [i for i, row in enumerate(rows) if not row.get("outcome")]
    done = len(rows) - len(pending)

    print(LABELLING_RULE)
    print(f"\n{done} of {len(rows)} already labelled. {len(pending)} to go.")
    print("Keys: [r]epair  [d]amage  [n]o_change  [s]kip  [q]uit and save\n")

    for position, index in enumerate(pending, start=1):
        row = rows[index]
        print("=" * 78)
        print(
            f"[{position}/{len(pending)}]  {row['repo']} #{row['pr_number']}  "
            f"PR author: {row['pr_author_type']} "
            f"(via {row['detection_method']})  "
            f"merged: {row['pr_merged']}"
        )
        print(f"commenter: {row['comment_author']}")
        print(f"file: {row.get('comment_path') or '(whole PR)'}")
        print(
            f"commits after this comment: "
            f"{row.get('commits_after_comment_count', 0)}"
        )
        print(f"url: {row.get('html_url')}")
        if row.get("comment_diff_hunk"):
            print("\n--- diff hunk ---")
            print(row["comment_diff_hunk"][:1200])
        print("\n--- comment ---")
        print(row["comment_body"][:1500])
        print()

        while True:
            choice = input("outcome [r/d/n/s/q] > ").strip().lower()
            if choice == "q":
                save(path, rows)
                print(f"\nSaved. {sum(1 for r in rows if r.get('outcome'))} labelled.")
                return
            if choice == "s":
                break
            if choice in OUTCOMES:
                row["outcome"] = OUTCOMES[choice]
                row["labelled_at"] = datetime.now(timezone.utc).isoformat()
                # Written after every decision so the session is resumable.
                save(path, rows)
                break
            print("  expected one of r, d, n, s, q")

    save(path, rows)
    print(f"\nDone. {sum(1 for r in rows if r.get('outcome'))} labelled.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reviewer behaviour toward agent- versus human-authored pull requests.

    Restricted to repositories containing both authorship types. Comparing a
    repository that only has agent pull requests against a different repository
    that only has human ones would confound authorship with project norms, which
    FR-33 forbids.
    """
    by_repo_types: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_repo_types[row["repo"]].add(row["pr_author_type"])
    within = {
        repo for repo, types in by_repo_types.items() if {"agent", "human"} <= types
    }

    eligible = [r for r in rows if r["repo"] in within]
    labelled = [r for r in eligible if r.get("outcome")]

    out: dict[str, Any] = {
        "comments_total": len(rows),
        "comments_in_repos_with_both_authorships": len(eligible),
        "comments_labelled": len(labelled),
        "repositories_total": len(by_repo_types),
        "repositories_with_both_authorships": len(within),
        "detection_methods": dict(Counter(r["detection_method"] for r in rows)),
        "by_authorship": {},
    }

    for author_type in ("agent", "human"):
        subset = [r for r in labelled if r["pr_author_type"] == author_type]
        prs = {(r["repo"], r["pr_number"]) for r in eligible if r["pr_author_type"] == author_type}
        criticised_prs = {
            (r["repo"], r["pr_number"])
            for r in subset
            if r["outcome"] in ("repair", "damage")
        }
        merged_after_criticism = {
            (r["repo"], r["pr_number"])
            for r in subset
            if r["outcome"] in ("repair", "damage") and r["pr_merged"]
        }
        counts = Counter(r["outcome"] for r in subset)
        n = len(subset)

        out["by_authorship"][author_type] = {
            "pull_requests": len(prs),
            "comments_labelled": n,
            "comment_rate_per_pr": round(len(subset) / len(prs), 2) if prs else 0.0,
            "outcomes": dict(counts),
            "repair_rate": round(counts["repair"] / n, 4) if n else 0.0,
            "damage_rate": round(counts["damage"] / n, 4) if n else 0.0,
            "no_change_rate": round(counts["no_change"] / n, 4) if n else 0.0,
            "criticised_prs": len(criticised_prs),
            "merge_rate_after_criticism": (
                round(len(merged_after_criticism) / len(criticised_prs), 4)
                if criticised_prs
                else None
            ),
        }

    return out


def print_report(summary: dict[str, Any]) -> None:
    print("Reviewer leniency toward agent-authored code")
    print(f"  comments collected:            {summary['comments_total']}")
    print(
        f"  in repos with both authorships: "
        f"{summary['comments_in_repos_with_both_authorships']}"
    )
    print(f"  labelled:                      {summary['comments_labelled']}")
    print(
        f"  repositories (both authorships): "
        f"{summary['repositories_with_both_authorships']}"
        f" of {summary['repositories_total']}"
    )
    print(f"  detection methods:             {summary['detection_methods']}")
    print()
    header = (
        f"{'authorship':<12}{'PRs':>6}{'comments':>10}{'per PR':>9}"
        f"{'repair':>9}{'damage':>9}{'no_change':>11}{'merged after':>14}"
    )
    print(header)
    print("-" * len(header))
    for author_type in ("agent", "human"):
        row = summary["by_authorship"].get(author_type)
        if not row:
            continue
        merged = row["merge_rate_after_criticism"]
        print(
            f"{author_type:<12}{row['pull_requests']:>6}"
            f"{row['comments_labelled']:>10}{row['comment_rate_per_pr']:>9.2f}"
            f"{row['repair_rate'] * 100:>8.1f}%{row['damage_rate'] * 100:>8.1f}%"
            f"{row['no_change_rate'] * 100:>10.1f}%"
            f"{'n/a' if merged is None else f'{merged * 100:.1f}%':>14}"
        )
    if summary["comments_labelled"] < 100:
        print(
            f"\nOnly {summary['comments_labelled']} comments are labelled. "
            "State this n as a limitation; do not claim significance.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("pr_corpus/data/comments.jsonl")
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the leniency comparison instead of labelling",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pr_corpus/data/leniency_summary.json"),
    )
    args = parser.parse_args(argv)

    rows = load(args.data)

    if args.report:
        summary = report(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print_report(summary)
        print(f"\nWrote {args.out}")
        return 0

    label_interactively(args.data, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
