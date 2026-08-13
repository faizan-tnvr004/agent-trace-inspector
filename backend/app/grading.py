"""Deciding whether a run produced the correct answer.

Not in the specification's file tree; separated out because the workflows, the
corpus generator and the rejection classifier all need the same rule, and a
grading rule that drifted between them would make `success` incomparable across
the corpus.

It lives under `app/` rather than `harness/` because of import direction. The
harness already imports `app.models`, so `harness` depends on `app`; putting
grading in `harness` and importing it from `app.extraction.rejection` made the
dependency circular in practice. `harness` is not an installed package, so that
import failed whenever the process ran with `backend/` as its working directory,
which is how both uvicorn and the container run: `GET /runs/{id}/export`
returned 500 on every run that had a critique step. Tests missed it because
pytest puts the repository root on the path, and the endpoint check missed it
because the sampled run had no critiques.

Grading is deliberately lenient about presentation and strict about value. The
model is asked for a final answer in prose, so "the answer is 96" and "96
loaves" must both count, while "960" and "9.6" must not. The check therefore
looks for the expected value as a complete number, not as a substring: "96"
does not match inside "960" or "1.96".
"""

from __future__ import annotations

import re

__all__ = ["answer_matches", "normalise"]


def normalise(text: str) -> str:
    """Strip thousands separators, currency and casing before comparison.

    Separators and currency symbols are deleted rather than replaced with a
    space: replacing them would turn "2,420" into "2 420", which then fails to
    match "2420" and would mark a correct answer as wrong. Whitespace is
    collapsed separately.
    """
    cleaned = re.sub(r"[,£$€]", "", str(text))
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def answer_matches(expected: str, produced: str) -> bool:
    """True if ``produced`` states ``expected`` as a value.

    Numeric answers are matched on a number boundary so that a longer or more
    precise number is not counted as correct. Non-numeric answers fall back to
    a containment test.
    """
    expected_n = normalise(expected)
    produced_n = normalise(produced)
    if not expected_n or not produced_n:
        return False

    if re.fullmatch(r"-?\d+(?:\.\d+)?", expected_n):
        # Reject a match that sits inside a longer number: 96 must not match
        # 960, 1.96 or 96.5, but may be followed by a full stop or a word.
        pattern = rf"(?<![\d.]){re.escape(expected_n)}(?!\d)(?!\.\d)"
        return re.search(pattern, produced_n) is not None

    return expected_n in produced_n
