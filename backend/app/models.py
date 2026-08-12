"""Pydantic data contract for multi-agent execution traces.

This module is the central artifact of the system. Every other component depends
on it, so the six invariants from the specification are enforced *here* rather
than at the storage or API layer: a malformed trace cannot then enter the system
through any route, whether that is corpus generation, an import adapter, or the
``POST /runs`` endpoint.

Invariants enforced (specification section 3.4):

1. ``seq`` values within a run are contiguous from 0
2. ``parent_step_id`` references a step in the same run, or is null
3. ``retry_of`` references a step with a lower ``seq``
4. ``rejection_outcome`` is non-null only on ``event_type == "critique"``
5. ``run.total_cost_usd`` equals the sum of ``step.cost_usd`` within 1e-6
6. If ``injected_fault`` is set, its ``target_step_seq`` exists in the run

Error message convention
------------------------
Invariant failures raise ``ValueError`` whose message begins ``"<field>: "``.
Pydantic reports model-level validator failures with an empty ``loc``, which
would leave the API unable to satisfy the requirement that a rejected trace name
the offending field (FR-6). The prefix carries that name instead, and
``offending_field`` recovers it.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "COST_TOLERANCE_USD",
    "AgentRole",
    "Claim",
    "ErrorInfo",
    "EventType",
    "FaultType",
    "InjectedFault",
    "RejectionOutcome",
    "Run",
    "RunSummary",
    "Step",
    "WorkflowType",
    "offending_field",
]

# Tolerance for invariant 5. Steps carry per-call costs that are summed in
# floating point, so exact equality is not achievable.
COST_TOLERANCE_USD = 1e-6

WorkflowType = Literal["reviewer_pipeline", "rag_qa", "imported"]

# Deliberately small vocabulary. A larger set would be more expressive but less
# portable across frameworks; a smaller one would collapse distinctions the
# extraction engine depends on. `critique` and `revision` are separate because
# the rejection taxonomy turns on a critique and the revision it triggers having
# different outcomes.
EventType = Literal[
    "plan",
    "tool_call",
    "tool_result",
    "retrieval",
    "reasoning",
    "critique",
    "revision",
    "decision",
    "error",
    "retry",
    "final",
]

FaultType = Literal[
    "dropped_retrieval",
    "truncated_tool_result",
    "forced_false_rejection",
    "injected_contradiction",
]

RejectionOutcome = Literal["repair", "damage", "no_change"]

# Not a closed enum. The specification gives these as examples and the schema is
# required to stay framework-agnostic (NFR-3), so `agent_role` is a free string
# and this tuple exists only for documentation and fixture construction.
AgentRole = Literal["executor", "reviewer", "retriever", "planner", "judge"]


# Pydantic wraps exceptions raised inside a validator, prefixing the message it
# reports. The prefix has to come off before the field name can be recovered.
_PYDANTIC_WRAPPERS = ("Value error, ", "Assertion failed, ")


def offending_field(message: str) -> str | None:
    """Recover the field name from an invariant failure message.

    Accepts either a bare message or one as Pydantic reports it in
    ``ValidationError.errors()[i]["msg"]``.

    Returns ``None`` when the message does not carry the ``"<field>: "`` prefix,
    which is the case for ordinary Pydantic type errors, where ``loc`` already
    names the field.
    """
    for wrapper in _PYDANTIC_WRAPPERS:
        if message.startswith(wrapper):
            message = message[len(wrapper) :]
            break
    head, sep, _ = message.partition(": ")
    if not sep or not head or " " in head:
        return None
    return head


class ErrorInfo(BaseModel):
    """An exception recorded against a step.

    The specification names this field but does not define its shape beyond
    "exception type and message" (SRS 7.2).
    """

    error_type: str
    message: str


class InjectedFault(BaseModel):
    """A deliberate defect introduced to create known ground truth.

    ``target_step_seq`` is the ground truth that failure attribution is scored
    against in the primary evaluation.
    """

    fault_type: FaultType
    target_step_seq: int
    description: str


class Step(BaseModel):
    """One recorded event within a trace."""

    step_id: str
    run_id: str
    parent_step_id: str | None = None
    seq: int = Field(ge=0)
    agent_id: str
    agent_role: str
    model: str
    event_type: EventType
    input: str
    output: str
    timestamp: datetime
    latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    # step_ids of upstream steps, or document chunk ids for retrieval steps.
    evidence_refs: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None
    retry_of: str | None = None
    rejection_outcome: RejectionOutcome | None = None

    @model_validator(mode="after")
    def _rejection_outcome_only_on_critique(self) -> Step:
        """Invariant 4.

        A rejection outcome classifies what a critique caused downstream. It is
        meaningless on any other event type, and allowing it elsewhere would let
        the rejection taxonomy be silently applied to steps it was never defined
        for.
        """
        if self.rejection_outcome is not None and self.event_type != "critique":
            raise ValueError(
                "rejection_outcome: may only be set on a step with "
                f'event_type "critique", but step {self.step_id} has event_type '
                f'"{self.event_type}"'
            )
        return self


class Run(BaseModel):
    """The complete recorded sequence of a single agent run."""

    run_id: str
    workflow_type: WorkflowType
    workflow_version: str
    task_input: str
    final_output: str
    success: bool
    ground_truth: str | None = None
    injected_fault: InjectedFault | None = None
    started_at: datetime
    completed_at: datetime
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    steps: list[Step] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_invariants(self) -> Run:
        # Steps are normalised into seq order so that every downstream consumer
        # can walk `run.steps` directly. Extraction must be deterministic
        # (FR-16), and relying on incoming list order would make it depend on
        # how the trace happened to be serialised.
        self.steps.sort(key=lambda s: s.seq)

        self._check_seq_contiguous()
        self._check_parent_refs()
        self._check_retry_refs()
        self._check_cost_reconciles()
        self._check_fault_target_exists()
        return self

    # -- invariant 1 ---------------------------------------------------------
    def _check_seq_contiguous(self) -> None:
        seqs = [s.seq for s in self.steps]
        expected = list(range(len(seqs)))
        if seqs != expected:
            duplicates = sorted({v for v in seqs if seqs.count(v) > 1})
            detail = f"expected contiguous 0..{len(seqs) - 1}, got {seqs}"
            if duplicates:
                detail += f" (duplicated: {duplicates})"
            raise ValueError(f"seq: {detail}")

    # -- invariant 2 ---------------------------------------------------------
    def _check_parent_refs(self) -> None:
        known = {s.step_id for s in self.steps}
        for step in self.steps:
            if step.parent_step_id is not None and step.parent_step_id not in known:
                raise ValueError(
                    f"parent_step_id: step {step.step_id} (seq {step.seq}) "
                    f"references {step.parent_step_id!r}, which is not a step "
                    "in this run"
                )

    # -- invariant 3 ---------------------------------------------------------
    def _check_retry_refs(self) -> None:
        seq_by_id = {s.step_id: s.seq for s in self.steps}
        for step in self.steps:
            if step.retry_of is None:
                continue
            target_seq = seq_by_id.get(step.retry_of)
            if target_seq is None:
                raise ValueError(
                    f"retry_of: step {step.step_id} (seq {step.seq}) references "
                    f"{step.retry_of!r}, which is not a step in this run"
                )
            if target_seq >= step.seq:
                raise ValueError(
                    f"retry_of: step {step.step_id} (seq {step.seq}) retries a "
                    f"step at seq {target_seq}; a retry must follow the step it "
                    "retries"
                )

    # -- invariant 5 ---------------------------------------------------------
    def _check_cost_reconciles(self) -> None:
        step_total = math.fsum(s.cost_usd for s in self.steps)
        if not math.isclose(
            self.total_cost_usd, step_total, rel_tol=0.0, abs_tol=COST_TOLERANCE_USD
        ):
            raise ValueError(
                f"total_cost_usd: run reports {self.total_cost_usd!r} but its "
                f"steps sum to {step_total!r}, a difference of "
                f"{abs(self.total_cost_usd - step_total):.3e} which exceeds the "
                f"tolerance of {COST_TOLERANCE_USD:.0e}"
            )

    # -- invariant 6 ---------------------------------------------------------
    def _check_fault_target_exists(self) -> None:
        if self.injected_fault is None:
            return
        seqs = {s.seq for s in self.steps}
        if self.injected_fault.target_step_seq not in seqs:
            raise ValueError(
                "injected_fault.target_step_seq: fault targets seq "
                f"{self.injected_fault.target_step_seq}, which does not exist "
                f"in this run (present: {sorted(seqs)})"
            )

    # -- convenience ---------------------------------------------------------
    def step_by_seq(self, seq: int) -> Step | None:
        for step in self.steps:
            if step.seq == seq:
                return step
        return None

    def step_by_id(self, step_id: str) -> Step | None:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def summary(self) -> RunSummary:
        return RunSummary(
            run_id=self.run_id,
            workflow_type=self.workflow_type,
            workflow_version=self.workflow_version,
            task_input=self.task_input,
            success=self.success,
            has_injected_fault=self.injected_fault is not None,
            fault_type=(
                self.injected_fault.fault_type if self.injected_fault else None
            ),
            started_at=self.started_at,
            completed_at=self.completed_at,
            total_cost_usd=self.total_cost_usd,
            total_tokens=self.total_tokens,
            step_count=len(self.steps),
        )


class RunSummary(BaseModel):
    """Row shape for ``GET /runs``. Excludes step bodies, which dominate size."""

    run_id: str
    workflow_type: WorkflowType
    workflow_version: str
    task_input: str
    success: bool
    has_injected_fault: bool
    fault_type: FaultType | None = None
    started_at: datetime
    completed_at: datetime
    total_cost_usd: float
    total_tokens: int
    step_count: int


class Claim(BaseModel):
    """One assertion extracted from a run's final output.

    Claims are produced by sentence-level splitting, not semantic extraction.
    That choice is deliberate and is documented in the README: semantic claim
    extraction is unreliable and out of scope.
    """

    claim_id: str
    run_id: str
    index: int = Field(ge=0)
    text: str
    # step_ids whose output supports this claim above the similarity threshold.
    evidence_refs: list[str] = Field(default_factory=list)
    supported: bool = False
