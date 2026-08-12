"""Workflow A: an executor reviewed by a second agent, over mathematics tasks.

Graph: ``execute -> review -> (revise | accept) -> final``.

LangGraph is used even though plain function calls would do the same work. The
workflow being an explicit state machine is itself a requirement: branch points
are what the extraction engine scores, and a graph makes the branch structure a
property of the program rather than something inferred from prose.

This pipeline reproduces the setting of the author's prior study, in which a
reviewer falsely rejected 35% of the executor's own correct answers. The
`forced_false_rejection` fault makes that failure mode reproducible on demand,
so the rejection taxonomy has cases to classify.

Steps emitted per run:

===  =================  ==============================================
seq  event_type         fault that targets it
===  =================  ==============================================
0    plan               injected_contradiction
1    tool_call          -
2    tool_result        truncated_tool_result
3    reasoning          -
4    critique           forced_false_rejection
5    revision           - (only when the reviewer rejects)
6    final              -
===  =================  ==============================================

A second review round occurs when the first revision is itself rejected, so
trace length and the position of later steps vary across runs.
"""

from __future__ import annotations

import ast
import operator
import random
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.models import InjectedFault
from harness.faults import apply_fault
from harness.grading import answer_matches
from harness.llm import LLMClient
from harness.tracer import TraceRecorder

__all__ = ["WORKFLOW_VERSION", "calculate", "run_reviewer_pipeline"]

WORKFLOW_VERSION = "1.0.0"
MAX_REVIEW_ROUNDS = 2

EXECUTOR_ID = "executor-1"
REVIEWER_ID = "reviewer-1"

# The executor operates under a strict tool policy, and that is a
# methodological requirement rather than a stylistic choice.
#
# An earlier version let the executor solve the arithmetic itself. Measured over
# 21 runs, every context fault then had zero effect: the model answered these
# tasks correctly from the question alone, so truncating the calculator output
# or contradicting the reference notes changed nothing. Those runs carried
# ground truth for faults that demonstrably did not cause the outcome, which is
# exactly the contamination `faults.py` warns about.
#
# Making the tool result and the notes authoritative restores the causal link
# the fault definitions assume. The trade-off is stated in the README: a
# workflow whose agents ignore their context is immune to context faults, which
# is itself a finding about where injected-fault studies apply.
_SYSTEM_EXECUTOR = (
    "You are a mathematics assistant operating under a strict tool policy. "
    "Take every numeric result from the calculator output you are given; do "
    "not perform the arithmetic yourself. Treat the reference notes as "
    "authoritative: if they state that a value has been corrected, use the "
    "corrected value. If the calculator output is missing or looks truncated, "
    "say so explicitly and then give the best answer you can justify from what "
    "you were given. Show brief working, then state the final answer on its "
    "own line in the form 'ANSWER: <value>'. Give the value as a plain number "
    "with no units."
)
_SYSTEM_REVIEWER = (
    "You review another agent's mathematical working. Reply with a short "
    "critique, then a verdict on its own line in the form 'VERDICT: ACCEPT' "
    "or 'VERDICT: REJECT'. Reject only if the working or the final value is "
    "actually wrong."
)


class ReviewState(TypedDict, total=False):
    question: str
    ground_truth: str
    context_text: str
    tool_call: str
    tool_result: str
    draft: str
    critique: str
    verdict: str
    rounds: int
    final_output: str
    stated_answer: str
    force_reject: bool


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression.

    A real tool with a real failure mode, which is what makes
    `truncated_tool_result` a meaningful fault rather than a cosmetic one.
    Parsed through `ast` and restricted to arithmetic, so a malformed or
    hostile expression raises instead of executing.
    """

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression element: {ast.dump(node)}")

    cleaned = expression.strip().rstrip("=").strip()
    if not cleaned:
        raise ValueError("empty expression")
    value = _eval(ast.parse(cleaned, mode="eval"))
    rendered = str(int(value)) if float(value).is_integer() else f"{value:g}"

    # The result line comes last, and that ordering is deliberate.
    #
    # `truncated_tool_result` keeps the first 40% of this string. An earlier
    # version led with "Result: <value>", so truncation preserved the very thing
    # it was supposed to destroy and the fault had no effect on any of 21
    # measured runs. Verbose preamble followed by the payload is also how real
    # tools behave, so putting the result at the end is realistic as well as
    # necessary.
    return (
        f"calculator v1.2 (exact rational arithmetic)\n"
        f"input expression: {cleaned}\n"
        f"parse status: OK\n"
        f"evaluation mode: exact, no rounding applied\n"
        f"note: the calculator evaluates the expression exactly as written. It "
        f"does not check that the expression models the question. Verify the "
        f"expression before relying on the value below.\n"
        f"Result: {rendered}"
    )


def _extract_answer(text: str) -> str:
    match = re.search(r"ANSWER:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else text.strip()[:80]


def _extract_expression(text: str) -> str:
    match = re.search(r"EXPRESSION:\s*([^\n]+)", text, flags=re.IGNORECASE)
    candidate = match.group(1) if match else text
    allowed = re.findall(r"[-+*/().\d\s%]+", candidate.replace(",", ""))
    for chunk in sorted(allowed, key=len, reverse=True):
        if any(ch.isdigit() for ch in chunk):
            return chunk.strip()
    return ""


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def run_reviewer_pipeline(
    task: dict[str, str],
    client: LLMClient,
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
        "reviewer_pipeline",
        WORKFLOW_VERSION,
        question,
        ground_truth=truth,
        run_id=run_id,
    ) as tracer:
        fault: InjectedFault | None = None
        state: ReviewState = {
            "question": question,
            "ground_truth": truth,
            "context_text": (
                "Reference notes: work in the units given in the question and "
                "give the final value as a plain number."
            ),
            "rounds": 0,
            "force_reject": False,
        }

        # `injected_contradiction` contaminates the framing the planner reads,
        # so it is introduced at seq 0 before any step is recorded.
        if fault_type == "injected_contradiction":
            merged, fault = apply_fault(
                "injected_contradiction", dict(state), tracer.next_seq
            )
            state["context_text"] = merged["context_text"]

        def execute(s: ReviewState) -> ReviewState:
            tracer.record(
                agent_id=EXECUTOR_ID,
                agent_role="executor",
                event_type="plan",
                input=s["context_text"],
                output=(
                    "Plan: restate the quantities, form a single arithmetic "
                    "expression, evaluate it with the calculator, then state "
                    "the answer."
                ),
            )

            call = client.complete(
                "Write a single arithmetic expression that answers the "
                "question. Reply with one line only, in the form "
                "'EXPRESSION: <expression>'.\n\n"
                f"Notes: {s['context_text']}\n"
                f"Question: {s['question']}",
                system=_SYSTEM_EXECUTOR,
                hint={"role": "executor", "expected": s["ground_truth"]},
            )
            expression = _extract_expression(call.text)
            tracer.record_llm(
                agent_id=EXECUTOR_ID,
                agent_role="executor",
                event_type="tool_call",
                input=s["question"],
                response=call,
            )

            try:
                result = calculate(expression)
                tool_error = None
            except Exception as exc:
                result = ""
                tool_error = exc

            s = dict(s)  # type: ignore[assignment]
            s["tool_call"] = expression
            s["tool_result"] = result

            nonlocal fault
            if fault_type == "truncated_tool_result" and result:
                s, fault = apply_fault(  # type: ignore[assignment]
                    "truncated_tool_result", dict(s), tracer.next_seq
                )

            from app.models import ErrorInfo

            tracer.record(
                agent_id="calculator",
                agent_role="tool",
                event_type="tool_result",
                input=expression,
                output=s.get("tool_result", ""),
                error=(
                    ErrorInfo(
                        error_type=type(tool_error).__name__, message=str(tool_error)
                    )
                    if tool_error is not None
                    else None
                ),
            )

            draft_response = client.complete(
                "Answer the question. The calculator output below is the "
                "authoritative source for the numeric result, and the "
                "reference notes are authoritative for which values to use.\n\n"
                f"Reference notes: {s['context_text']}\n"
                f"Calculator output: {s.get('tool_result') or '(unavailable)'}\n"
                f"Question: {s['question']}",
                system=_SYSTEM_EXECUTOR,
                hint={"role": "executor", "expected": s["ground_truth"]},
            )
            tracer.record_llm(
                agent_id=EXECUTOR_ID,
                agent_role="executor",
                event_type="reasoning",
                input=s.get("tool_result", ""),
                response=draft_response,
            )
            s["draft"] = draft_response.text
            return s  # type: ignore[return-value]

        def review(s: ReviewState) -> ReviewState:
            nonlocal fault
            s = dict(s)  # type: ignore[assignment]

            if fault_type == "forced_false_rejection" and s["rounds"] == 0:
                s, fault = apply_fault(  # type: ignore[assignment]
                    "forced_false_rejection", dict(s), tracer.next_seq
                )

            response = client.complete(
                "Review the following working and give a verdict.\n\n"
                f"Question: {s['question']}\n"
                f"Working: {s['draft']}",
                system=_SYSTEM_REVIEWER,
                hint={"role": "reviewer"},
            )

            forced = bool(s.get("force_reject")) and s["rounds"] == 0
            if forced:
                verdict = "REJECT"
                critique_text = (
                    f"{response.text}\n\nVERDICT: REJECT"
                    if "VERDICT:" not in response.text.upper()
                    else re.sub(
                        r"VERDICT:\s*ACCEPT",
                        "VERDICT: REJECT",
                        response.text,
                        flags=re.IGNORECASE,
                    )
                )
            else:
                verdict = (
                    "REJECT" if "REJECT" in response.text.upper() else "ACCEPT"
                )
                critique_text = response.text

            tracer.record(
                agent_id=REVIEWER_ID,
                agent_role="reviewer",
                event_type="critique",
                input=s["draft"],
                output=critique_text,
                model=response.model,
                latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost_usd=response.cost_usd,
            )
            s["critique"] = critique_text
            s["verdict"] = verdict
            s["rounds"] = s["rounds"] + 1
            return s  # type: ignore[return-value]

        def revise(s: ReviewState) -> ReviewState:
            s = dict(s)  # type: ignore[assignment]
            response = client.complete(
                "A reviewer rejected your answer. Produce a corrected "
                "answer, or restate the original if the criticism is "
                "mistaken.\n\n"
                f"Question: {s['question']}\n"
                f"Your answer: {s['draft']}\n"
                f"Reviewer: {s['critique']}",
                system=_SYSTEM_EXECUTOR,
                hint={"role": "executor", "expected": s["ground_truth"]},
            )
            tracer.record_llm(
                agent_id=EXECUTOR_ID,
                agent_role="executor",
                event_type="revision",
                input=s["critique"],
                response=response,
            )
            s["draft"] = response.text
            return s  # type: ignore[return-value]

        def finalise(s: ReviewState) -> ReviewState:
            s = dict(s)  # type: ignore[assignment]
            # `final_output` carries the executor's full closing prose, not the
            # bare value. Provenance splits the final output into claims, and a
            # single number splits into exactly one claim, which would make the
            # provenance view and the unsupported-claim count vacuous (FR-12,
            # FR-26). Grading still uses the stated value, below, so a wrong
            # answer cannot pass by mentioning the right number mid-working.
            tracer.record(
                agent_id=EXECUTOR_ID,
                agent_role="executor",
                event_type="final",
                input=s["critique"] if s.get("critique") else s["question"],
                output=s["draft"],
            )
            s["final_output"] = s["draft"]
            s["stated_answer"] = _extract_answer(s["draft"])
            return s  # type: ignore[return-value]

        def route(s: ReviewState) -> str:
            if s.get("verdict") == "REJECT" and s["rounds"] < MAX_REVIEW_ROUNDS:
                return "revise"
            return "finalise"

        builder: StateGraph = StateGraph(ReviewState)
        builder.add_node("execute", execute)
        builder.add_node("review", review)
        builder.add_node("revise", revise)
        builder.add_node("finalise", finalise)
        builder.add_edge(START, "execute")
        builder.add_edge("execute", "review")
        builder.add_conditional_edges(
            "review", route, {"revise": "revise", "finalise": "finalise"}
        )
        builder.add_edge("revise", "review")
        builder.add_edge("finalise", END)

        result = builder.compile().invoke(state)

        tracer.set_injected_fault(fault)
        tracer.set_result(
            final_output=result.get("final_output", ""),
            success=answer_matches(truth, result.get("stated_answer", "")),
        )

    return tracer.run
