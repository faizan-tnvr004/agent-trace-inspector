"""Tests for claim decomposition and evidence linking."""

from __future__ import annotations

import pytest

from app.extraction.claims import (
    EVIDENCE_SIMILARITY_THRESHOLD,
    analyse_claims,
    link_evidence,
    split_claims,
    unsupported_claims,
)
from app.models import Run
from tests.conftest import make_run, make_step

DEPTH_FACT = "The deep basin at Grethe Bay reaches a maximum depth of 312 metres."


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_split_claims_splits_on_sentence_boundaries() -> None:
    claims = split_claims(
        "The basin is deep. It reaches 312 metres. That is the answer here."
    )
    assert len(claims) == 3
    assert claims[0].text.startswith("The basin is deep")


def test_split_claims_does_not_split_inside_a_decimal() -> None:
    """"1.8 metres" is one claim, not two. Splitting on every full stop would
    shatter numeric answers into fragments."""
    claims = split_claims("The telescope has a diameter of 1.8 metres exactly.")
    assert len(claims) == 1


def test_split_claims_indexes_claims_in_order() -> None:
    claims = split_claims("First assertion here. Second assertion here.")
    assert [c.index for c in claims] == [0, 1]


def test_split_claims_strips_scaffolding_prefixes() -> None:
    """`ANSWER:` and friends are formatting the harness asked for, not
    assertions the run is making."""
    claims = split_claims("ANSWER: the depth is 312 metres below the surface")
    assert len(claims) == 1
    assert not claims[0].text.lower().startswith("answer:")


def test_split_claims_drops_trivial_fragments() -> None:
    claims = split_claims("96\nok\nThe total number of loaves sold today is 96.")
    assert len(claims) == 1


def test_split_claims_on_empty_output() -> None:
    assert split_claims("") == []


@pytest.mark.parametrize(
    "fragment",
    [
        "Support: [doc-04#c2]",
        "1962 (supported by [doc-01#c0])",
        "4,600 ([doc-06#c2])",
        "10 (supported by [doc-07#c1])",
        "Passage identifier: [doc-12#c2]",
    ],
)
def test_citation_fragments_are_not_claims(fragment: str) -> None:
    """Citation markup is not an assertion.

    Models answer with sentence-shaped fragments that only cite a source. They
    embed poorly against any step output, so counting them as claims marked them
    unsupported and inflated the ungrounded-claim rate from 25.5% to 30.9%.
    Eighteen such fragments in the corpus cited a source while being labelled
    unsupported, which is self-contradictory on its face.
    """
    assert split_claims(fragment) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "The basin is deep.",
        "It reaches 312 metres.",
        "The station was established in 1962 by the Trust.",
        "The context does not contain the answer.",
    ],
)
def test_real_sentences_survive_the_citation_filter(sentence: str) -> None:
    """The filter must not eat genuine short assertions. An earlier version
    stripped ordinary words such as "the" and "in" before measuring substance,
    which discarded real claims."""
    assert len(split_claims(sentence)) == 1


def test_split_claims_splits_multiline_working() -> None:
    text = (
        "Morning sales were 24 loaves in total.\n"
        "Afternoon sales were three times that figure.\n"
        "ANSWER: 96"
    )
    assert len(split_claims(text)) == 2


def test_split_claims_carries_the_run_id_into_claim_ids() -> None:
    claims = split_claims("A sufficiently long assertion here.", "run-xyz")
    assert claims[0].run_id == "run-xyz"
    assert claims[0].claim_id.startswith("run-xyz")


def test_split_claims_can_take_a_run_instead_of_both_arguments() -> None:
    run = Run.model_validate(
        make_run(steps=[make_step(0)], final_output=DEPTH_FACT, run_id="run-r")
    )
    claims = split_claims("", run=run)
    assert claims and claims[0].run_id == "run-r"


# ---------------------------------------------------------------------------
# Evidence linking
# ---------------------------------------------------------------------------


def _grounded_run() -> Run:
    return Run.model_validate(
        make_run(
            steps=[
                make_step(
                    0,
                    event_type="retrieval",
                    output=DEPTH_FACT,
                    evidence_refs=["doc-08#c2"],
                ),
                make_step(1, event_type="reasoning", output=DEPTH_FACT),
                make_step(2, event_type="final", output=DEPTH_FACT),
            ],
            final_output=DEPTH_FACT,
        )
    )


def test_link_evidence_finds_the_supporting_steps() -> None:
    run = _grounded_run()
    claim = split_claims(run.final_output, run.run_id)[0]
    refs = link_evidence(claim, run)
    assert run.steps[0].step_id in refs
    assert run.steps[1].step_id in refs


def test_link_evidence_excludes_the_final_step() -> None:
    """The final step *is* the final output. Counting it as evidence would make
    every claim trivially supported and the provenance view meaningless."""
    run = _grounded_run()
    claim = split_claims(run.final_output, run.run_id)[0]
    assert run.steps[2].step_id not in link_evidence(claim, run)


def test_link_evidence_returns_refs_in_seq_order() -> None:
    run = _grounded_run()
    claim = split_claims(run.final_output, run.run_id)[0]
    refs = link_evidence(claim, run)
    seqs = [run.step_by_id(r).seq for r in refs]  # type: ignore[union-attr]
    assert seqs == sorted(seqs)


def test_link_evidence_returns_nothing_for_an_ungrounded_claim() -> None:
    run = Run.model_validate(
        make_run(
            steps=[
                make_step(
                    0,
                    event_type="retrieval",
                    output="Tarnholt maintains 240 survey stakes.",
                    evidence_refs=["doc-02#c2"],
                )
            ],
            final_output="Corvenna Aurora Camp runs five all-sky cameras.",
        )
    )
    claim = split_claims(run.final_output, run.run_id)[0]
    assert link_evidence(claim, run) == []


def test_threshold_is_the_documented_value() -> None:
    assert EVIDENCE_SIMILARITY_THRESHOLD == 0.6


# ---------------------------------------------------------------------------
# Support labelling
# ---------------------------------------------------------------------------


def test_analyse_claims_labels_support() -> None:
    run = _grounded_run()
    claims = analyse_claims(run)
    assert claims[0].supported is True
    assert claims[0].evidence_refs


def test_unsupported_claims_reports_ungrounded_assertions() -> None:
    run = Run.model_validate(
        make_run(
            steps=[
                make_step(
                    0,
                    event_type="retrieval",
                    output=DEPTH_FACT,
                    evidence_refs=["doc-08#c2"],
                )
            ],
            final_output=(
                f"{DEPTH_FACT} The station was also the first to record a "
                "magnitude nine earthquake in the region."
            ),
        )
    )
    unsupported = unsupported_claims(run)
    assert len(unsupported) == 1
    assert "earthquake" in unsupported[0].text
    assert unsupported[0].supported is False


def test_unsupported_claims_is_empty_when_everything_is_grounded() -> None:
    assert unsupported_claims(_grounded_run()) == []


def test_claim_analysis_is_deterministic() -> None:
    run = _grounded_run()
    assert analyse_claims(run) == analyse_claims(run)


def test_analyse_claims_accepts_a_run_dict() -> None:
    run_dict = make_run(
        steps=[
            make_step(0, event_type="reasoning", output=DEPTH_FACT),
        ],
        final_output=DEPTH_FACT,
    )
    assert analyse_claims(run_dict) == analyse_claims(Run.model_validate(run_dict))
