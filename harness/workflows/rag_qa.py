"""Workflow B: retrieval-augmented question answering over a controlled corpus.

Graph: ``gather -> tools -> reason -> verify -> answer``.

The document set is synthetic and describes fictional research stations. That
is a methodological requirement, not set dressing. If the documents were real
public text, the model could answer from parametric knowledge, and dropping the
retrieved chunk containing the answer would not cause a failure. The fault
would then be recorded as ground truth for a run that failed elsewhere, or
succeeded anyway, and the attribution evaluation would be measuring noise.
Fictional facts force the answer to come from retrieval or not at all.

Retrieval is lexical (IDF-weighted token overlap) rather than embedding-based.
It is deterministic, needs no model load during corpus generation, and is
adequate over 90 chunks. Embeddings are confined to the extraction engine,
where the specification requires them.

Trace depth
-----------

An earlier version of this workflow emitted 4 to 8 steps per run. That made the
primary study untestable: a top-5 extraction kept the whole trace on most runs,
so the two study conditions differed by formatting rather than by content. This
version emits roughly 20 to 30 steps by doing more of the work the pipeline was
always implicitly doing, in separate recorded steps:

===================  =========================================================
phase                steps
===================  =========================================================
opening plan         one ``plan``
retrieval rounds     per round: ``plan``, ``decision`` (query reformulation),
                     ``retrieval``; plus ``retry`` + ``retrieval`` when the
                     round comes back thin. Two or three rounds per run.
tool use             ``tool_call`` + ``tool_result`` for evidence extraction,
                     then again for the derived arithmetic
draft                one ``reasoning``
verification         one ``critique`` per claim in the draft
revision             one ``revision`` when any claim is rejected
close                one ``final``
===================  =========================================================

Only ``reasoning`` and ``revision`` call the model. Every other step is a real
deterministic computation that was previously folded into its neighbour and not
recorded. Nothing here is padding: each step consumes its predecessor's output
and its removal would change what the next step sees. Most of them are
nonetheless inert with respect to the outcome, which is the premise the
extraction engine exists to exploit.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.models import InjectedFault
from harness.faults import apply_fault
from app.grading import answer_matches
from harness.llm import LLMClient
from harness.tracer import TraceRecorder

__all__ = [
    "WORKFLOW_VERSION",
    "load_corpus",
    "retrieve",
    "run_rag_qa",
    "split_sentences",
]

WORKFLOW_VERSION = "2.0.0"

CORPUS_DIR = Path(__file__).resolve().parents[1] / "tasks" / "qa_corpus"
TOP_K = 4

# Below this best-match score the round reformulates and retrieves again. Set
# from the measured score distribution over the 30 committed questions so that
# roughly 40% of rounds take the retry branch: trace length and the position of
# the faulted step then vary across the corpus instead of being fixed, which
# would let the evaluation judge exploit position rather than content.
# Fixed before the primary study and not revisited afterwards.
WEAK_RETRIEVAL_SCORE = 0.45

# Faults that act on retrieved context, and so must be re-applied on every round
# and every attempt within a round.
_RAG_FAULTS = frozenset(
    {"dropped_retrieval", "injected_contradiction", "truncated_tool_result"}
)

RETRIEVER_ID = "retriever-1"
ANSWERER_ID = "answerer-1"
PLANNER_ID = "planner-1"
EXTRACTOR_ID = "extractor-1"
CALCULATOR_ID = "calculator-1"
VERIFIER_ID = "verifier-1"

# The answerer is asked to show its supporting facts as separate sentences.
# That is not verbosity for its own sake: the verification stage checks one
# claim at a time, and a single-sentence answer gives it exactly one claim to
# check. Citing each fact separately is also ordinary practice for a RAG
# pipeline that has to be auditable.
_SYSTEM_ANSWERER = (
    "You answer questions using only the supplied context. If the context "
    "does not contain the answer, say so plainly rather than guessing.\n"
    "Structure every reply as:\n"
    "  one sentence per supporting fact you relied on, each naming the "
    "passage identifier it came from;\n"
    "  one sentence stating what those facts together imply;\n"
    "  a final line of the form 'ANSWER: <value>'."
)

_SYSTEM_REVISER = (
    "You revise an answer in the light of a verifier's objections, using only "
    "the supplied context. If the context still does not support an answer, "
    "say so plainly rather than inventing one. State the final answer on its "
    "own line in the form 'ANSWER: <value>'."
)


class RagState(TypedDict, total=False):
    question: str
    ground_truth: str
    chunks: list[dict[str, Any]]
    answer_chunk_id: str
    context_text: str
    tool_result: str
    derived_note: str
    reasoning: str
    draft: str
    critiques: list[dict[str, Any]]
    final_output: str
    stated_answer: str
    force_reject: bool


def load_corpus(
    corpus_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (chunks, questions) from the committed QA corpus."""
    base = corpus_dir or CORPUS_DIR
    documents = json.loads((base / "documents.json").read_text())
    questions = json.loads((base / "questions.json").read_text())
    chunks = [
        {"chunk_id": c["chunk_id"], "doc_id": d["doc_id"], "text": c["text"]}
        for d in documents
        for c in d["chunks"]
    ]
    return chunks, questions


_TOKEN = re.compile(r"[a-z0-9.]+")


def _tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace(",", ""))


def retrieve(
    query: str, chunks: list[dict[str, Any]], k: int = TOP_K
) -> list[tuple[dict[str, Any], float]]:
    """IDF-weighted token overlap, normalised by query length.

    Deterministic: ties break on chunk_id so the same query always returns the
    same ordering.
    """
    n_docs = len(chunks) or 1
    doc_freq: Counter[str] = Counter()
    tokenised = []
    for chunk in chunks:
        tokens = set(_tokenise(chunk["text"]))
        tokenised.append(tokens)
        doc_freq.update(tokens)

    query_tokens = [t for t in set(_tokenise(query)) if len(t) > 2]
    if not query_tokens:
        return []

    max_score = sum(math.log(1 + n_docs / (1 + doc_freq.get(t, 0))) for t in query_tokens)
    scored: list[tuple[dict[str, Any], float]] = []
    for chunk, tokens in zip(chunks, tokenised):
        score = sum(
            math.log(1 + n_docs / (1 + doc_freq.get(t, 0)))
            for t in query_tokens
            if t in tokens
        )
        scored.append((chunk, score / max_score if max_score else 0.0))

    scored.sort(key=lambda pair: (-pair[1], pair[0]["chunk_id"]))
    return scored[:k]


def _format_context(chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def _extract_answer(text: str) -> str:
    match = re.search(r"ANSWER:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".")
    return text.strip().splitlines()[-1][:120] if text.strip() else ""


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Sentence-level split, local to the harness.

    Deliberately not imported from `app.extraction.claims`: that module pulls in
    the embedding stack, and corpus generation must not load a 90MB model to cut
    a string on full stops.
    """
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    return [p for p in parts if len(p) > 15]


# ---------------------------------------------------------------------------
# Query reformulation
# ---------------------------------------------------------------------------

_MEASUREMENT_WORDS = (
    "depth", "diameter", "height", "length", "width", "thickness", "frequency",
    "year", "number", "count", "rooms", "buoys", "quadrats", "cameras",
    "metres", "megahertz", "capacity", "area",
)


def _round_queries(question: str) -> list[str]:
    """The query each retrieval round issues.

    Three deterministic reformulations of the same information need, in
    decreasing breadth. Rounds accumulate evidence rather than replacing it, so
    a later round can add the chunk an earlier one missed.
    """
    content = [t for t in _tokenise(question) if len(t) > 3]
    proper = re.findall(r"\b[A-Z][a-z]+\b", question)
    measure = [w for w in _MEASUREMENT_WORDS if w in question.lower()]

    queries = [question]
    queries.append(" ".join(content) if content else question)
    focus = " ".join(proper + measure)
    queries.append(focus if focus.strip() else question)
    return queries


def _round_count(question: str) -> int:
    """Two or three rounds, deterministic per question.

    Varying the count varies trace length across the corpus, so neither the
    extraction engine nor the study judge can rely on the faulted step sitting
    at a fixed position.
    """
    return 2 + (sum(ord(c) for c in question) % 2)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def extract_measurements(context: str, answer_hint: str) -> str:
    """Pull candidate values out of retrieved context.

    A real extraction over the retrieved text, and load-bearing: the reasoning
    step reads this digest alongside the raw context. The candidate that matches
    the question's expected shape is placed **last**, so `truncated_tool_result`
    (which keeps the first 40%) actually destroys it. A tool result whose useful
    content sits at the front would survive truncation and the fault would be
    recorded as ground truth for a run it did not affect.
    """
    lines = []
    for chunk_line in context.split("\n\n"):
        ref = re.match(r"\[([^\]]+)\]", chunk_line)
        # Scan the passage text only. Chunk ids such as [doc-01#c0] are full of
        # digits, and scanning the whole line harvests "01" and "0" as candidate
        # measurements, feeding the answerer numbers that appear nowhere in the
        # corpus text.
        body = chunk_line[ref.end() :] if ref else chunk_line
        for value in _NUMBER.findall(body):
            lines.append(f"  candidate {value} from [{ref.group(1) if ref else '?'}]")

    # Stable order, with any candidate equal to the expected shape moved last.
    tail = [ln for ln in lines if answer_hint and answer_hint in ln]
    head = [ln for ln in lines if ln not in tail]
    body = "\n".join(head + tail) or "  no numeric candidates found"
    return f"extract_measurements: {len(lines)} candidate(s)\n{body}"


def derive_value(question: str, values: list[str]) -> str:
    """One arithmetic step over an extracted value.

    Genuine arithmetic, not decoration: a year is turned into an elapsed
    duration, a metric length into feet. The result is reported to the answerer
    as supporting detail. It is deliberately *not* the answer itself, so a
    truncation of this second tool result degrades the trace without
    guaranteeing failure.
    """
    if not values:
        return "derive: no value to work from"
    raw = values[0].replace(",", "")
    try:
        number = float(raw)
    except ValueError:
        return f"derive: {values[0]!r} is not numeric"

    if re.search(r"\byear\b|\bsince\b", question, flags=re.IGNORECASE) and (
        1800 < number < 2100
    ):
        return f"derive: {int(number)} is {2026 - int(number)} year(s) before 2026"
    feet = number * 3.28084
    return f"derive: {raw} metres is {feet:.1f} feet"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_claim(claim: str, chunks: list[dict[str, Any]]) -> tuple[bool, str]:
    """Whether any retrieved chunk lexically supports this claim.

    Deterministic on purpose. A model-based verifier would put an LLM call in
    every claim of every run, which the free tier cannot fund, and would make
    the critique steps non-reproducible. The check is content-word overlap plus
    agreement on any number the claim asserts, which is weak but honest: it is
    reported as a lexical check in the step output, not as a judgement of truth.
    """
    claim_tokens = {t for t in _tokenise(claim) if len(t) > 3}
    if not claim_tokens:
        return True, "no content words to check"

    numbers = set(_NUMBER.findall(claim))
    best_ref, best_overlap = "", 0.0
    for chunk in chunks:
        chunk_tokens = {t for t in _tokenise(chunk["text"]) if len(t) > 3}
        overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
        if overlap > best_overlap:
            best_overlap, best_ref = overlap, chunk["chunk_id"]

    numbers_ok = not numbers or any(
        n in chunk["text"] for n in numbers for chunk in chunks
    )
    supported = best_overlap >= 0.34 and numbers_ok

    if supported:
        return True, f"supported by [{best_ref}] (overlap {best_overlap:.2f})"
    if not numbers_ok:
        return False, f"asserts {sorted(numbers)}, which no retrieved chunk states"
    return False, f"best overlap {best_overlap:.2f} against [{best_ref or 'nothing'}]"


def run_rag_qa(
    task: dict[str, str],
    client: LLMClient,
    corpus_chunks: list[dict[str, Any]],
    *,
    fault_type: str | None = None,
    run_id: str | None = None,
    rng: random.Random | None = None,
) -> Any:
    """Execute one run and return the recorded `Run`."""
    rng = rng or random.Random()
    question = task["question"]
    truth = str(task["answer"])

    with TraceRecorder(
        "rag_qa",
        WORKFLOW_VERSION,
        question,
        ground_truth=truth,
        run_id=run_id,
    ) as tracer:
        fault: InjectedFault | None = None
        state: RagState = {
            "question": question,
            "ground_truth": truth,
            "answer_chunk_id": task.get("answer_chunk_id", ""),
            "chunks": [],
            "critiques": [],
        }

        def _apply_rag_fault(s: RagState) -> RagState:
            """Re-apply the context fault to the state as it now stands.

            Called after every retrieval, not once at the start. Rounds
            accumulate chunks, so a later round re-retrieves from the full
            corpus and would silently restore what an earlier fault removed:
            the run would then carry ground truth for a fault with no effect.
            Overwriting `fault` each time also leaves `target_step_seq` naming
            the retrieval that actually feeds the answer.
            """
            nonlocal fault
            if fault_type not in _RAG_FAULTS or fault_type == "truncated_tool_result":
                return s
            updated, fault = apply_fault(fault_type, dict(s), tracer.next_seq)
            updated["context_text"] = _format_context(updated.get("chunks", []))
            return updated  # type: ignore[return-value]

        def gather_node(s: RagState) -> RagState:
            s = dict(s)  # type: ignore[assignment]

            tracer.record(
                agent_id=PLANNER_ID,
                agent_role="planner",
                event_type="plan",
                input=s["question"],
                output=(
                    "Plan: gather evidence over several retrieval rounds, "
                    "extract the candidate values, verify each claim against "
                    "the retrieved passages, then answer."
                ),
            )

            queries = _round_queries(s["question"])
            rounds = _round_count(s["question"])
            seen: dict[str, dict[str, Any]] = {}

            for index in range(rounds):
                query = queries[index % len(queries)]

                tracer.record(
                    agent_id=PLANNER_ID,
                    agent_role="planner",
                    event_type="plan",
                    input=s["question"],
                    output=(
                        f"Round {index + 1} of {rounds}: look for the passage "
                        f"naming the station and the quantity asked for."
                    ),
                )
                tracer.record(
                    agent_id=RETRIEVER_ID,
                    agent_role="retriever",
                    event_type="decision",
                    input=s["question"],
                    output=f"Query for round {index + 1}: {query}",
                )

                for attempt in range(2):
                    hits = retrieve(query, corpus_chunks)
                    best = hits[0][1] if hits else 0.0
                    for chunk, _ in hits:
                        seen[chunk["chunk_id"]] = chunk

                    s["chunks"] = list(seen.values())
                    s["context_text"] = _format_context(s["chunks"])
                    s = _apply_rag_fault(s)
                    seen = {c["chunk_id"]: c for c in s.get("chunks", [])}

                    tracer.record(
                        agent_id=RETRIEVER_ID,
                        agent_role="retriever",
                        event_type="retrieval",
                        input=query,
                        output=s.get("context_text", ""),
                        evidence_refs=[c["chunk_id"] for c in s.get("chunks", [])],
                    )

                    if best >= WEAK_RETRIEVAL_SCORE or attempt == 1:
                        break

                    tracer.record(
                        agent_id=RETRIEVER_ID,
                        agent_role="retriever",
                        event_type="retry",
                        input=query,
                        output=(
                            f"Best match scored {best:.2f}, below the "
                            f"{WEAK_RETRIEVAL_SCORE} threshold. Widening the "
                            "query to the station name alone."
                        ),
                        retry_of=tracer.steps[-1].step_id,
                    )
                    query = " ".join(
                        t for t in _tokenise(s["question"]) if len(t) > 3
                    )
            return s  # type: ignore[return-value]

        def tools_node(s: RagState) -> RagState:
            nonlocal fault
            s = dict(s)  # type: ignore[assignment]
            context = s.get("context_text", "")

            tracer.record(
                agent_id=EXTRACTOR_ID,
                agent_role="tool",
                event_type="tool_call",
                input=context,
                output="extract_measurements(context)",
                evidence_refs=[c["chunk_id"] for c in s.get("chunks", [])],
            )

            s["tool_result"] = extract_measurements(context, str(s["ground_truth"]))
            if fault_type == "truncated_tool_result":
                s, fault = apply_fault(  # type: ignore[assignment]
                    fault_type, dict(s), tracer.next_seq
                )

            tracer.record(
                agent_id=EXTRACTOR_ID,
                agent_role="tool",
                event_type="tool_result",
                input="extract_measurements(context)",
                output=s.get("tool_result", ""),
            )

            values = _NUMBER.findall(s.get("tool_result", ""))
            tracer.record(
                agent_id=CALCULATOR_ID,
                agent_role="tool",
                event_type="tool_call",
                input=s.get("tool_result", ""),
                output=f"derive({values[0] if values else 'none'})",
            )
            s["derived_note"] = derive_value(s["question"], values)
            tracer.record(
                agent_id=CALCULATOR_ID,
                agent_role="tool",
                event_type="tool_result",
                input=f"derive({values[0] if values else 'none'})",
                output=s["derived_note"],
            )
            return s  # type: ignore[return-value]

        def reason_node(s: RagState) -> RagState:
            s = dict(s)  # type: ignore[assignment]
            response = client.complete(
                "Using only the context below, work out the answer. State "
                "which passage identifier supports it, or say that the "
                "context does not contain the answer.\n\n"
                f"Context:\n{s.get('context_text', '')}\n\n"
                f"Extracted candidates:\n{s.get('tool_result', '')}\n\n"
                f"Derived: {s.get('derived_note', '')}\n\n"
                f"Question: {s['question']}",
                system=_SYSTEM_ANSWERER,
                hint={"role": "executor", "expected": s["ground_truth"]},
            )
            tracer.record_llm(
                agent_id=ANSWERER_ID,
                agent_role="executor",
                event_type="reasoning",
                input=s.get("context_text", ""),
                response=response,
                evidence_refs=[c["chunk_id"] for c in s.get("chunks", [])],
            )
            s["draft"] = response.text
            s["reasoning"] = response.text
            return s  # type: ignore[return-value]

        def verify_node(s: RagState) -> RagState:
            nonlocal fault
            s = dict(s)  # type: ignore[assignment]
            claims = split_sentences(s.get("draft", "")) or [s.get("draft", "")]
            chunks = s.get("chunks", [])

            if fault_type == "forced_false_rejection":
                s, fault = apply_fault(  # type: ignore[assignment]
                    fault_type, dict(s), tracer.next_seq
                )

            forced = bool(s.get("force_reject"))
            critiques: list[dict[str, Any]] = []

            for index, claim in enumerate(claims):
                supported, note = verify_claim(claim, chunks)
                # The forced fault flips the first claim only. Rejecting every
                # claim would be an obviously broken verifier rather than the
                # plausible false rejection the fault is meant to model.
                if forced and index == 0:
                    supported, note = False, (
                        "claim does not follow from the retrieved passages"
                    )
                verdict = "ACCEPT" if supported else "REJECT"
                critiques.append({"claim": claim, "verdict": verdict, "note": note})
                tracer.record(
                    agent_id=VERIFIER_ID,
                    agent_role="reviewer",
                    event_type="critique",
                    input=claim,
                    output=f"VERDICT: {verdict}\nLexical check: {note}",
                    evidence_refs=[c["chunk_id"] for c in chunks],
                )

            s["critiques"] = critiques
            return s  # type: ignore[return-value]

        def revise_node(s: RagState) -> RagState:
            s = dict(s)  # type: ignore[assignment]
            rejected = [c for c in s.get("critiques", []) if c["verdict"] == "REJECT"]
            if not rejected:
                s["final_output"] = s.get("draft", "")
                return s  # type: ignore[return-value]

            objections = "\n".join(
                f"- {c['claim']}  ({c['note']})" for c in rejected
            )
            response = client.complete(
                "A verifier rejected part of your answer. Revise it using only "
                "the context below.\n\n"
                f"Context:\n{s.get('context_text', '')}\n\n"
                f"Your answer:\n{s.get('draft', '')}\n\n"
                f"Objections:\n{objections}",
                system=_SYSTEM_REVISER,
                hint={"role": "executor", "expected": s["ground_truth"]},
            )
            tracer.record_llm(
                agent_id=ANSWERER_ID,
                agent_role="executor",
                event_type="revision",
                input=objections,
                response=response,
                evidence_refs=[c["chunk_id"] for c in s.get("chunks", [])],
            )
            s["final_output"] = response.text
            return s  # type: ignore[return-value]

        def answer_node(s: RagState) -> RagState:
            s = dict(s)  # type: ignore[assignment]
            # The full closing prose, not the bare value: see the equivalent
            # comment in reviewer_pipeline. Grading uses the stated answer.
            prose = s.get("final_output") or s.get("draft", "")
            tracer.record(
                agent_id=ANSWERER_ID,
                agent_role="executor",
                event_type="final",
                input=s.get("context_text", ""),
                output=prose,
                evidence_refs=[c["chunk_id"] for c in s.get("chunks", [])],
            )
            s["final_output"] = prose
            s["stated_answer"] = _extract_answer(prose)
            return s  # type: ignore[return-value]

        builder: StateGraph = StateGraph(RagState)
        builder.add_node("gather", gather_node)
        builder.add_node("tools", tools_node)
        builder.add_node("reason", reason_node)
        builder.add_node("verify", verify_node)
        builder.add_node("revise", revise_node)
        builder.add_node("answer", answer_node)
        builder.add_edge(START, "gather")
        builder.add_edge("gather", "tools")
        builder.add_edge("tools", "reason")
        builder.add_edge("reason", "verify")
        builder.add_edge("verify", "revise")
        builder.add_edge("revise", "answer")
        builder.add_edge("answer", END)

        result = builder.compile().invoke(state)

        tracer.set_injected_fault(fault)
        tracer.set_result(
            final_output=result.get("final_output", ""),
            success=answer_matches(truth, result.get("stated_answer", "")),
        )

    return tracer.run
