"""LLM judge for the primary study.

The judge is asked one question: which step introduced the error? It sees a
serialised trace and answers with a single ``seq`` number.

**The prompt and the model are identical across both conditions.** That is the
whole design of the study: the only difference between condition A and condition
B is the trace content placed in the ``{trace}`` slot. Anything else that
differed, a hint about length, a different instruction, a different temperature,
would confound the comparison and there would be nothing to conclude.

The judge is not told which condition it is in, how long the trace is, whether a
fault was injected, or what the correct answer is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from harness.llm import LLMClient

__all__ = ["JUDGE_PROMPT", "JUDGE_SYSTEM", "JudgeVerdict", "ask_judge"]

JUDGE_SYSTEM = (
    "You analyse execution traces from multi-agent language model systems. "
    "You identify the single step at which a run first went wrong. You answer "
    "with a number and nothing else."
)

# One template, both conditions. Only `{trace}` changes.
JUDGE_PROMPT = """The trace below is from a multi-agent run that produced an incorrect final answer.

Each step has a `seq` number. Exactly one step first introduced the error that led to the wrong answer. Note that the step where the error becomes visible is often not the step that introduced it: an earlier step may have supplied missing, truncated or contradictory information.

Which step introduced the error?

Answer with a single seq number and nothing else. No explanation, no punctuation, no other text.

TRACE:
{trace}
"""


@dataclass(frozen=True)
class JudgeVerdict:
    predicted_seq: int | None
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model: str


def _parse_seq(text: str) -> int | None:
    """Recover a seq number from the judge's reply.

    The prompt asks for a bare number, but a model sometimes answers "Step 3" or
    "seq 3." anyway. Taking the first integer is deliberate: a reply of
    "3 because the retrieval..." means the judge chose 3, and discarding it as
    unparseable would understate accuracy in both conditions.
    """
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    try:
        return int(match.group())
    except ValueError:  # pragma: no cover - regex guarantees an int
        return None


def ask_judge(client: LLMClient, trace_text: str) -> JudgeVerdict:
    """Put one serialised trace to the judge."""
    response = client.complete(
        JUDGE_PROMPT.format(trace=trace_text), system=JUDGE_SYSTEM
    )
    return JudgeVerdict(
        predicted_seq=_parse_seq(response.text),
        raw_response=response.text,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
        model=response.model,
    )
