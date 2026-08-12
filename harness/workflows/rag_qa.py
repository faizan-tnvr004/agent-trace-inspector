"""Workflow B: retrieval-augmented question answering over a controlled corpus.

Graph: ``retrieve -> reason -> answer``.

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

Steps emitted per run:

===  =================  ==============================================
seq  event_type         fault that targets it
===  =================  ==============================================
0    plan               -
1    retrieval          dropped_retrieval, injected_contradiction,
                        truncated_tool_result
2    reasoning          - (or `retry` + a second `retrieval`, then
                          `reasoning`, when the first retrieval is weak)
n    final              -
===  =================  ==============================================

When the first retrieval scores poorly the agent reformulates and retrieves
again, so the faulted step is not always at the same position.
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
from harness.grading import answer_matches
from harness.llm import LLMClient
from harness.tracer import TraceRecorder

__all__ = [
    "WORKFLOW_VERSION",
    "load_corpus",
    "retrieve",
    "run_rag_qa",
]

WORKFLOW_VERSION = "1.0.0"

CORPUS_DIR = Path(__file__).resolve().parents[1] / "tasks" / "qa_corpus"
TOP_K = 4

# Below this best-match score the agent reformulates and retrieves again. Set
# from the measured score distribution over the 30 committed questions so that
# roughly 40% of runs take the retry branch: trace length and the position of
# the faulted step then vary across the corpus instead of being fixed, which
# would let the evaluation judge exploit position rather than content.
# Fixed before the primary study and not revisited afterwards.
WEAK_RETRIEVAL_SCORE = 0.45

# Faults that act on retrieval, and so must be re-applied on every attempt.
_RAG_FAULTS = frozenset(
    {"dropped_retrieval", "injected_contradiction", "truncated_tool_result"}
)

RETRIEVER_ID = "retriever-1"
ANSWERER_ID = "answerer-1"

_SYSTEM_ANSWERER = (
    "You answer questions using only the supplied context. If the context "
    "does not contain the answer, say so plainly rather than guessing. State "
    "the final answer on its own line in the form 'ANSWER: <value>'."
)


class RagState(TypedDict, total=False):
    question: str
    ground_truth: str
    chunks: list[dict[str, Any]]
    answer_chunk_id: str
    context_text: str
    reasoning: str
    final_output: str
    stated_answer: str


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
        }

        def retrieve_node(s: RagState) -> RagState:
            nonlocal fault
            s = dict(s)  # type: ignore[assignment]

            tracer.record(
                agent_id=RETRIEVER_ID,
                agent_role="planner",
                event_type="plan",
                input=s["question"],
                output=(
                    "Plan: search the station corpus for the passage naming "
                    "this station and this quantity, then answer from it."
                ),
            )

            query = s["question"]
            for attempt in range(2):
                hits = retrieve(query, corpus_chunks)
                best = hits[0][1] if hits else 0.0
                s["chunks"] = [chunk for chunk, _ in hits]
                s["context_text"] = _format_context(s["chunks"])

                # Re-applied on every attempt, not just the first. A retry
                # re-runs retrieval from the full corpus, so applying the fault
                # once would let the second attempt silently restore what the
                # fault removed: the run would carry ground truth for a fault
                # that had no effect. Overwriting `fault` each time also leaves
                # `target_step_seq` naming the retrieval that actually feeds
                # the reasoning step.
                if fault_type in _RAG_FAULTS:
                    s, fault = apply_fault(  # type: ignore[assignment]
                        fault_type, dict(s), tracer.next_seq
                    )
                    if fault_type != "truncated_tool_result":
                        # These faults change the chunk set, so the rendered
                        # context has to be rebuilt from it.
                        s["context_text"] = _format_context(s["chunks"])

                tracer.record(
                    agent_id=RETRIEVER_ID,
                    agent_role="retriever",
                    event_type="retrieval",
                    input=query,
                    output=s.get("context_text", ""),
                    evidence_refs=[c["chunk_id"] for c in s["chunks"]],
                )

                if best >= WEAK_RETRIEVAL_SCORE or attempt == 1:
                    break

                # Weak match: reformulate and try once more. This is a genuine
                # branch point, and it moves the position of the faulted step.
                tracer.record(
                    agent_id=RETRIEVER_ID,
                    agent_role="retriever",
                    event_type="retry",
                    input=query,
                    output=(
                        f"Best match scored {best:.2f}, below the "
                        f"{WEAK_RETRIEVAL_SCORE} threshold. Reformulating the "
                        "query around the station name."
                    ),
                    retry_of=tracer.steps[-1].step_id,
                )
                query = " ".join(
                    t for t in _tokenise(s["question"]) if len(t) > 3
                )
            return s  # type: ignore[return-value]

        def reason_node(s: RagState) -> RagState:
            s = dict(s)  # type: ignore[assignment]
            response = client.complete(
                "Using only the context below, work out the answer. State "
                "which passage identifier supports it, or say that the "
                "context does not contain the answer.\n\n"
                f"Context:\n{s.get('context_text', '')}\n\n"
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
            s["reasoning"] = response.text
            return s  # type: ignore[return-value]

        def answer_node(s: RagState) -> RagState:
            s = dict(s)  # type: ignore[assignment]
            # The full closing prose, not the bare value: see the equivalent
            # comment in reviewer_pipeline. Grading uses the stated answer.
            prose = s.get("reasoning", "")
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
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("reason", reason_node)
        builder.add_node("answer", answer_node)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "reason")
        builder.add_edge("reason", "answer")
        builder.add_edge("answer", END)

        result = builder.compile().invoke(state)

        tracer.set_injected_fault(fault)
        tracer.set_result(
            final_output=result.get("final_output", ""),
            success=answer_matches(truth, result.get("stated_answer", "")),
        )

    return tracer.run
