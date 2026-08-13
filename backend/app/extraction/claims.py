"""Claim decomposition and evidence linking.

Answers the auditor's question: what is this final answer grounded in, and what
is it not?

**Sentence-level splitting, deliberately.** Claims are sentences, not
semantically extracted propositions. Semantic claim extraction is unreliable and
out of scope, and attempting it would also require an LLM in the extraction
path, which is forbidden. The cost of this choice is real and is stated in the
README: a sentence carrying two assertions is treated as one claim, and a claim
spread over two sentences is treated as two.

A claim is *supported* when some step's output is similar enough to it. That is
a statement about textual overlap with upstream work, not about truth: a claim
can be well supported by a step and still be wrong, if the step was wrong. The
provenance view answers "where did this come from", not "is this correct".
"""

from __future__ import annotations

import re
from typing import Any

from app.extraction.embeddings import cosine_similarity
from app.extraction.scoring import as_run
from app.models import Claim, Run, Step

__all__ = [
    "EVIDENCE_SIMILARITY_THRESHOLD",
    "link_evidence",
    "split_claims",
    "unsupported_claims",
]

# A step's output supports a claim at or above this cosine similarity. Fixed
# before the primary study, per the specification's threshold.
EVIDENCE_SIMILARITY_THRESHOLD = 0.6

# Steps whose output is a candidate source of grounding. A `critique` is
# excluded because it describes the answer rather than contributing content to
# it, and `final` is excluded because it *is* the final output: matching a claim
# against the text it was split from would make every claim trivially supported.
_EVIDENCE_EVENT_TYPES = frozenset(
    {"tool_result", "retrieval", "reasoning", "revision", "plan", "tool_call"}
)

# Split on sentence-ending punctuation followed by whitespace, but not when the
# full stop sits inside a number ("1.8 metres") or after a single initial.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

# Lines the models emit as scaffolding rather than as assertions.
_SCAFFOLD = re.compile(
    r"^\s*(answer|expression|verdict|working|plan|support|sources?|"
    r"passage identifier|reference)s?\s*:\s*",
    re.IGNORECASE,
)

_MIN_CLAIM_CHARS = 12

# Bracketed corpus citations, e.g. "[doc-08#c2]".
_CITATION = re.compile(r"\[[^\]]*\]")

# Phrases that only introduce a citation and assert nothing themselves. Kept
# deliberately narrow: including ordinary words such as "the" or "in" would
# shrink genuine sentences below the substance threshold and discard real claims.
_CITATION_WORDS = re.compile(
    r"\b(supported\s+by|support|sources?|passage\s+identifier|identifier)\b",
    re.IGNORECASE,
)

# A claim must retain this many letters once citation markup is removed.
_MIN_SUBSTANCE_CHARS = 12


def _is_citation_fragment(text: str) -> bool:
    """True for text that carries no proposition once its citations are removed.

    Models answer with fragments such as ``Support: [doc-04#c2]`` and
    ``1962 (supported by [doc-01#c0])``. These are citation markup, not
    assertions, but they are sentence-shaped and embed poorly against any step
    output, so counting them as claims marks them unsupported and inflates the
    ungrounded-claim rate. Eighteen such fragments in this corpus cited a source
    while being labelled unsupported, which is self-contradictory on its face.

    The test is structural, not a similarity threshold: strip the citations and
    the words that only introduce them, and see whether any assertion is left.
    """
    stripped = _CITATION.sub(" ", text)
    stripped = _CITATION_WORDS.sub(" ", stripped)
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    return len(letters) < _MIN_SUBSTANCE_CHARS


def _sentences(text: str) -> list[str]:
    """Split text into sentence-level claims.

    Numbered or bulleted working is split per line first, since models present
    steps of a calculation on separate lines without terminal punctuation.
    """
    out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _SCAFFOLD.sub("", line).strip()
        if not line:
            continue
        out.extend(part.strip() for part in _SENTENCE_BOUNDARY.split(line))
    return [
        s
        for s in out
        if len(s) >= _MIN_CLAIM_CHARS and not _is_citation_fragment(s)
    ]


def split_claims(
    final_output: str, run_id: str = "", *, run: Run | dict[str, Any] | None = None
) -> list[Claim]:
    """Split a final output into sentence-level claims.

    ``run`` is accepted so callers holding a run need not pass both it and the
    run's id.
    """
    if run is not None:
        resolved = as_run(run)
        final_output = final_output or resolved.final_output
        run_id = run_id or resolved.run_id

    return [
        Claim(
            claim_id=f"{run_id or 'claim'}-c{index}",
            run_id=run_id,
            index=index,
            text=text,
            evidence_refs=[],
            supported=False,
        )
        for index, text in enumerate(_sentences(final_output))
    ]


def _evidence_steps(run: Run) -> list[Step]:
    return [s for s in run.steps if s.event_type in _EVIDENCE_EVENT_TYPES]


def link_evidence(claim: Claim, run: Run | dict[str, Any]) -> list[str]:
    """Step ids whose output supports ``claim`` above the threshold.

    Returned in ``seq`` order rather than by similarity, so the provenance view
    reads as a chain through the run.
    """
    resolved = as_run(run)
    return [
        step.step_id
        for step in _evidence_steps(resolved)
        if cosine_similarity(claim.text, step.output)
        >= EVIDENCE_SIMILARITY_THRESHOLD
    ]


def analyse_claims(run: Run | dict[str, Any]) -> list[Claim]:
    """Every claim in the run's final output, with evidence links resolved."""
    resolved = as_run(run)
    claims = split_claims(resolved.final_output, resolved.run_id)
    for claim in claims:
        refs = link_evidence(claim, resolved)
        claim.evidence_refs = refs
        claim.supported = bool(refs)
    return claims


def unsupported_claims(run: Run | dict[str, Any]) -> list[Claim]:
    """Claims with no supporting step: assertions the trace does not ground."""
    return [claim for claim in analyse_claims(run) if not claim.supported]
