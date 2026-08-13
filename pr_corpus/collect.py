"""Collect review comments from public pull requests.

This extension applies the rejection-outcome taxonomy used inside synthetic
agent traces to human reviewers acting on real pull requests, testing whether it
generalises beyond the synthetic setting.

**All data is archival and public.** No participants are recruited, no private
repositories are read, and no personal data beyond the public author login on a
public pull request is stored.

**Agent authorship is established by bot account or commit trailer only.**
Never by writing style. Inferring authorship from prose would be unreliable and
would invalidate the comparison, so every row records `detection_method`, and a
pull request that cannot be classified by one of the two mechanical signals is
recorded as `human` with `detection_method: "no_agent_signal"` rather than
guessed at.

The comparison is *within repository*: agent- and human-authored pull requests
are collected from the same repositories so that reviewer identity and project
norms are held constant. Cross-repository comparison is explicitly not performed.

Usage::

    python pr_corpus/collect.py --repos config/repos.txt --out pr_corpus/data/
    python pr_corpus/collect.py --discover --out pr_corpus/data/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"

# Accounts that publish pull requests written by an LLM coding agent. Matching
# is exact on the lowercased login, so a human account cannot match by
# resemblance.
AGENT_ACCOUNTS = {
    "claude[bot]",
    "claude-code[bot]",
    "devin-ai-integration[bot]",
    "copilot-swe-agent[bot]",
    "copilot[bot]",
    "cursoragent",
    "cursor[bot]",
    "sweep-ai[bot]",
    "codegen-sh[bot]",
    "google-labs-jules[bot]",
    "jules[bot]",
    "openai-codex[bot]",
    "codex[bot]",
    "sourcery-ai[bot]",
    "coderabbitai[bot]",
    "ellipsis-dev[bot]",
    "graphite-app[bot]",
    "gemini-code-assist[bot]",
}

# Commit-message trailers that identify an agent as an author. Lowercased
# substring match against the commit message.
AGENT_TRAILERS = (
    "co-authored-by: claude",
    "generated with [claude code]",
    "co-authored-by: devin",
    "co-authored-by: copilot",
    "co-authored-by: cursor",
    "co-authored-by: openai",
    "co-authored-by: gemini",
    "generated with claude code",
)

DETECTION_BOT_ACCOUNT = "bot_account"
DETECTION_COMMIT_TRAILER = "commit_trailer"
DETECTION_NONE = "no_agent_signal"


class GitHub:
    """Minimal REST client with rate-limit awareness.

    Uses the `gh` CLI token when `GITHUB_TOKEN` is unset, so a machine that has
    already authenticated `gh` needs no extra configuration.
    """

    def __init__(self, token: str | None = None, *, pause: float = 0.0) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN") or _gh_cli_token()
        if not self.token:
            raise SystemExit(
                "No GitHub token. Set GITHUB_TOKEN or run: gh auth login"
            )
        self.pause = pause
        self.requests_made = 0

    def get(self, path: str, **params: Any) -> Any:
        url = f"{API_ROOT}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "agent-trace-inspector-pr-corpus",
            },
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    self.requests_made += 1
                    if self.pause:
                        time.sleep(self.pause)
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 429):
                    # Secondary rate limit. Back off rather than hammering.
                    wait = min(2**attempt * 10, 120)
                    print(f"    rate limited, waiting {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                if exc.code == 404:
                    return None
                if exc.code == 422:
                    # Search rejects a query naming an account that does not
                    # exist. That is expected while probing a list of candidate
                    # agent accounts, so it is not fatal.
                    return None
                raise
        raise SystemExit(f"gave up on {path} after repeated rate limiting")

    def paginate(self, path: str, *, limit: int, **params: Any) -> list[Any]:
        out: list[Any] = []
        page = 1
        while len(out) < limit:
            batch = self.get(path, per_page=100, page=page, **params)
            if not batch:
                break
            out.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return out[:limit]


def _gh_cli_token() -> str | None:
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=15
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


# ---------------------------------------------------------------------------
# Authorship detection
# ---------------------------------------------------------------------------


def classify_author(
    pr: dict[str, Any], commits: list[dict[str, Any]]
) -> tuple[str, str, str]:
    """Return (author_type, detection_method, evidence).

    Two mechanical signals only. Order matters: a bot account is the stronger
    signal, so it is checked first and reported in preference to a trailer.
    """
    user = pr.get("user") or {}
    login = (user.get("login") or "").lower()

    if login in AGENT_ACCOUNTS or user.get("type") == "Bot":
        return "agent", DETECTION_BOT_ACCOUNT, login

    for commit in commits:
        message = ((commit.get("commit") or {}).get("message") or "").lower()
        for trailer in AGENT_TRAILERS:
            if trailer in message:
                return "agent", DETECTION_COMMIT_TRAILER, trailer

    return "human", DETECTION_NONE, login


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect_repo(
    api: GitHub, repo: str, *, per_repo: int, comment_cap: int
) -> list[dict[str, Any]]:
    """Collect review comments for recent closed pull requests in one repo."""
    owner_name = repo.strip()
    if not owner_name or owner_name.startswith("#"):
        return []

    print(f"  {owner_name}")
    prs = api.paginate(
        f"/repos/{owner_name}/pulls",
        limit=per_repo,
        state="closed",
        sort="updated",
        direction="desc",
    )
    if not prs:
        print("    no pull requests returned (repo missing or empty)")
        return []

    rows: list[dict[str, Any]] = []
    seen_authors: Counter[str] = Counter()

    for pr in prs:
        number = pr["number"]
        commits = api.paginate(
            f"/repos/{owner_name}/pulls/{number}/commits", limit=50
        ) or []
        author_type, detection_method, evidence = classify_author(pr, commits)

        # Review comments on the diff, plus review summaries. Both are review
        # comments in the sense FR-31 means; issue comments on the PR thread are
        # excluded because they are frequently not review.
        review_comments = (
            api.paginate(
                f"/repos/{owner_name}/pulls/{number}/comments", limit=comment_cap
            )
            or []
        )
        reviews = (
            api.paginate(
                f"/repos/{owner_name}/pulls/{number}/reviews", limit=comment_cap
            )
            or []
        )

        pr_author = ((pr.get("user") or {}).get("login") or "")
        merged = pr.get("merged_at") is not None
        commit_shas = [c.get("sha") for c in commits]

        def base_row(
            comment: dict[str, Any], kind: str, body: str
        ) -> dict[str, Any] | None:
            commenter = ((comment.get("user") or {}).get("login") or "")
            if not body.strip():
                return None
            # A comment by the PR author on their own PR is not review.
            if commenter and commenter == pr_author:
                return None
            created_at = comment.get("created_at") or comment.get("submitted_at")
            # Commits pushed after the comment are the evidence a label is
            # judged against, so they are recorded rather than the diff itself.
            following = [
                c.get("sha")
                for c in commits
                if ((c.get("commit") or {}).get("committer") or {}).get("date", "")
                > (created_at or "")
            ]
            return {
                "repo": owner_name,
                "pr_number": number,
                "pr_title": pr.get("title") or "",
                "pr_author": pr_author,
                "pr_author_type": author_type,
                "detection_method": detection_method,
                "detection_evidence": evidence,
                "pr_merged": merged,
                "pr_commit_count": len(commit_shas),
                "comment_id": comment.get("id"),
                "comment_kind": kind,
                "comment_author": commenter,
                "comment_author_is_bot": (
                    ((comment.get("user") or {}).get("type") == "Bot")
                ),
                "comment_created_at": created_at,
                "comment_body": body.strip()[:4000],
                "comment_path": comment.get("path"),
                "comment_diff_hunk": (comment.get("diff_hunk") or "")[:2000],
                "commits_after_comment": following,
                "commits_after_comment_count": len(following),
                "html_url": comment.get("html_url") or pr.get("html_url"),
                # Filled in by label.py.
                "outcome": None,
                "labelled_at": None,
            }

        for comment in review_comments:
            row = base_row(comment, "review_comment", comment.get("body") or "")
            if row:
                rows.append(row)
                seen_authors[author_type] += 1

        for review in reviews:
            # Only reviews that actually said something. An APPROVED review with
            # an empty body carries no criticism to classify.
            row = base_row(review, "review_summary", review.get("body") or "")
            if row:
                row["review_state"] = review.get("state")
                rows.append(row)
                seen_authors[author_type] += 1

    print(
        f"    {len(rows)} comments  "
        f"(agent-authored PRs: {seen_authors['agent']}, "
        f"human-authored: {seen_authors['human']})"
    )
    return rows


def discover_repos(api: GitHub, *, want: int) -> list[str]:
    """Find repositories that visibly receive agent-authored pull requests."""
    found: Counter[str] = Counter()
    for account in sorted(AGENT_ACCOUNTS):
        # Search rejects `author:name[bot]`. A GitHub App's pull requests are
        # found with `author:app/<slug>`, so both forms are tried and whichever
        # the account actually is will match.
        queries = [f"is:pr author:{account}"]
        if account.endswith("[bot]"):
            queries = [f"is:pr author:app/{account[: -len('[bot]')]}"]

        before = sum(found.values())
        for query in queries:
            result = api.get(
                "/search/issues", q=query, per_page=100, sort="updated", order="desc"
            )
            for item in (result or {}).get("items", []):
                url = item.get("repository_url") or ""
                if "/repos/" in url:
                    found[url.split("/repos/", 1)[1]] += 1

        gained = sum(found.values()) - before
        if gained:
            print(f"  {account}: {gained} agent PRs, {len(found)} repos so far")
        if len(found) >= want * 3:
            break
    return [repo for repo, _ in found.most_common(want)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", type=Path, default=Path("config/repos.txt"))
    parser.add_argument("--out", type=Path, default=Path("pr_corpus/data"))
    parser.add_argument("--per-repo", type=int, default=40)
    parser.add_argument("--comment-cap", type=int, default=60)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument(
        "--discover",
        action="store_true",
        help="search for repositories receiving agent PRs and write --repos",
    )
    parser.add_argument("--discover-count", type=int, default=20)
    parser.add_argument("--pause", type=float, default=0.0)
    args = parser.parse_args(argv)

    api = GitHub(pause=args.pause)

    if args.discover:
        print("Discovering repositories with agent-authored pull requests")
        repos = discover_repos(api, want=args.discover_count)
        args.repos.parent.mkdir(parents=True, exist_ok=True)
        args.repos.write_text(
            "# Repositories that visibly receive agent-authored pull requests.\n"
            "# Discovered by searching for pull requests authored by known\n"
            "# agent bot accounts. Authorship is never inferred from style.\n"
            + "\n".join(repos)
            + "\n"
        )
        print(f"\nWrote {len(repos)} repositories to {args.repos}")
        return 0

    if not args.repos.exists():
        print(
            f"{args.repos} not found. Run with --discover first, or write the "
            "file by hand (one owner/name per line).",
            file=sys.stderr,
        )
        return 1

    repos = [
        line.strip()
        for line in args.repos.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    print(f"Collecting from {len(repos)} repositories")

    rows: list[dict[str, Any]] = []
    for repo in repos:
        rows.extend(
            collect_repo(
                api, repo, per_repo=args.per_repo, comment_cap=args.comment_cap
            )
        )
        if len(rows) >= args.target:
            print(f"\nReached the target of {args.target} comments")
            break

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "comments.jsonl"
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    by_type = Counter(r["pr_author_type"] for r in rows)
    by_detection = Counter(r["detection_method"] for r in rows)
    repos_seen = {r["repo"] for r in rows}
    repos_with_both = {
        repo
        for repo in repos_seen
        if {r["pr_author_type"] for r in rows if r["repo"] == repo} == {"agent", "human"}
    }

    print(f"\nWrote {len(rows)} comments to {out_path}")
    print(f"  repositories:            {len(repos_seen)}")
    print(f"  with both authorships:   {len(repos_with_both)}")
    print(f"  by PR authorship:        {dict(by_type)}")
    print(f"  by detection method:     {dict(by_detection)}")
    print(f"  API requests made:       {api.requests_made}")

    if len(rows) < 150:
        print(
            f"\nBelow the stated minimum of 150 comments ({len(rows)}). "
            "Add repositories to the list or raise --per-repo; do not report "
            "this as a complete corpus.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
