"""Records workflow execution as a schema-conformant `Run`.

The recorder is the only thing in the harness that knows about the trace
schema. Workflows call `record_*` and never construct a `Step` themselves, so
there is exactly one place where a change to the data contract has to land.

Emitting a valid trace is enforced rather than hoped for: `__exit__` builds a
`Run` through the Pydantic models, so a workflow that produces a malformed
trace fails at generation time instead of writing a broken file into the
corpus.

A workflow that raises part-way through still produces a valid failed trace.
That is deliberate: a crashed run is a legitimate object of study, and
discarding it would bias the corpus towards runs that completed.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Any, Literal

from app.models import ErrorInfo, EventType, InjectedFault, Run, Step

from harness.llm import LLMResponse

__all__ = ["TraceRecorder"]

# Sentinel: chain this step to the one before it. Explicit `None` means the step
# is a root, which is not the same thing.
AUTO_PARENT = "auto"


class TraceRecorder:
    """Context manager that records steps and emits a valid `Run`.

    Usage::

        with TraceRecorder("rag_qa", "1.0.0", question, ground_truth="4") as tr:
            tr.record(agent_id="retriever-1", agent_role="retriever",
                      event_type="retrieval", input=question, output=chunks)
            tr.set_result(final_output=answer, success=True)
        run = tr.run
    """

    def __init__(
        self,
        workflow_type: str,
        workflow_version: str,
        task_input: str,
        *,
        ground_truth: str | None = None,
        run_id: str | None = None,
        started_at: datetime | None = None,
        id_source: uuid.UUID | None = None,
    ) -> None:
        self.workflow_type = workflow_type
        self.workflow_version = workflow_version
        self.task_input = task_input
        self.ground_truth = ground_truth

        self.run_id = run_id or str(uuid.uuid4())
        self.started_at = started_at or datetime.now(timezone.utc)
        self._clock = self.started_at

        self._steps: list[Step] = []
        self._final_output: str = ""
        self._success: bool = False
        self._injected_fault: InjectedFault | None = None
        self._run: Run | None = None
        self._id_source = id_source

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> TraceRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        if exc is not None:
            # Record the crash as a step so the trace explains itself, then
            # finalise as a failed run. The exception is not suppressed.
            self.record(
                agent_id="harness",
                agent_role="harness",
                event_type="error",
                input=self.task_input,
                output="",
                error=ErrorInfo(error_type=type(exc).__name__, message=str(exc)),
            )
            self._success = False
            if not self._final_output:
                self._final_output = ""
        self._run = self._build()
        return False

    @property
    def run(self) -> Run:
        if self._run is None:
            raise RuntimeError(
                "TraceRecorder.run is only available after the context exits"
            )
        return self._run

    @property
    def steps(self) -> list[Step]:
        return list(self._steps)

    @property
    def next_seq(self) -> int:
        return len(self._steps)

    # -- recording -----------------------------------------------------------

    def record(
        self,
        *,
        agent_id: str,
        agent_role: str,
        event_type: EventType,
        input: str,
        output: str,
        model: str = "none",
        latency_ms: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        evidence_refs: list[str] | None = None,
        error: ErrorInfo | None = None,
        retry_of: str | None = None,
        rejection_outcome: str | None = None,
        parent_step_id: str | None = AUTO_PARENT,
    ) -> Step:
        """Append a step and return it."""
        if parent_step_id == AUTO_PARENT:
            parent_step_id = self._steps[-1].step_id if self._steps else None

        self._clock += timedelta(milliseconds=max(latency_ms, 1))
        step = Step(
            step_id=self._new_id(),
            run_id=self.run_id,
            parent_step_id=parent_step_id,
            seq=len(self._steps),
            agent_id=agent_id,
            agent_role=agent_role,
            model=model,
            event_type=event_type,
            input=input,
            output=output,
            timestamp=self._clock,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            evidence_refs=list(evidence_refs or []),
            error=error,
            retry_of=retry_of,
            rejection_outcome=rejection_outcome,
        )
        self._steps.append(step)
        return step

    def record_llm(
        self,
        *,
        agent_id: str,
        agent_role: str,
        event_type: EventType,
        input: str,
        response: LLMResponse,
        **kwargs: Any,
    ) -> Step:
        """Record a step from a model response, carrying its usage across."""
        return self.record(
            agent_id=agent_id,
            agent_role=agent_role,
            event_type=event_type,
            input=input,
            output=response.text,
            model=response.model,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=response.cost_usd,
            **kwargs,
        )

    # -- result --------------------------------------------------------------

    def set_result(self, *, final_output: str, success: bool) -> None:
        self._final_output = final_output
        self._success = success

    def set_injected_fault(self, fault: InjectedFault | None) -> None:
        self._injected_fault = fault

    # -- internals -----------------------------------------------------------

    def _new_id(self) -> str:
        if self._id_source is None:
            return str(uuid.uuid4())
        # Derived ids keep a seeded corpus run reproducible end to end.
        return str(uuid.uuid5(self._id_source, str(len(self._steps))))

    def _build(self) -> Run:
        total_cost = math.fsum(s.cost_usd for s in self._steps)
        total_tokens = sum(s.prompt_tokens + s.completion_tokens for s in self._steps)
        completed_at = self._clock if self._steps else self.started_at

        return Run(
            run_id=self.run_id,
            workflow_type=self.workflow_type,  # type: ignore[arg-type]
            workflow_version=self.workflow_version,
            task_input=self.task_input,
            final_output=self._final_output,
            success=self._success,
            ground_truth=self.ground_truth,
            injected_fault=self._injected_fault,
            started_at=self.started_at,
            completed_at=completed_at,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            steps=self._steps,
        )
