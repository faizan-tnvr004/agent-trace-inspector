"""Import LangGraph stream events as a `Run` (FR-7).

Demonstrates that the trace schema is not tied to one framework (NFR-3): a
LangGraph export renders in the inspector identically to a natively emitted
trace, because it becomes the same `Run` object.

Two LangGraph stream shapes are supported, since which one a user has depends on
how they called `stream()`:

* ``stream_mode="updates"`` — a list of ``{node_name: state_delta}`` dicts
* ``stream_mode="debug"`` — a list of ``{"type", "timestamp", "step",
  "payload"}`` dicts, which carry node timing

Node names are mapped onto the event-type vocabulary by `NODE_EVENT_MAP`, falling
back to a keyword match and then to ``reasoning``. The fallback is deliberate:
an unrecognised node should import as a plausible step rather than fail the whole
trace, and `imported` is a first-class `workflow_type` for exactly this reason.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.models import ErrorInfo, EventType, Run, Step

__all__ = ["NODE_EVENT_MAP", "import_langgraph_events", "map_node_to_event"]

# Exact node-name matches, checked before the keyword fallback.
NODE_EVENT_MAP: dict[str, EventType] = {
    "plan": "plan",
    "planner": "plan",
    "retrieve": "retrieval",
    "retriever": "retrieval",
    "search": "retrieval",
    "tools": "tool_call",
    "tool": "tool_call",
    "action": "tool_call",
    "tool_node": "tool_call",
    "observation": "tool_result",
    "reason": "reasoning",
    "agent": "reasoning",
    "execute": "reasoning",
    "executor": "reasoning",
    "generate": "reasoning",
    "review": "critique",
    "reviewer": "critique",
    "critic": "critique",
    "reflect": "critique",
    "revise": "revision",
    "rewrite": "revision",
    "decide": "decision",
    "route": "decision",
    "router": "decision",
    "retry": "retry",
    "answer": "final",
    "final": "final",
    "respond": "final",
    "__end__": "final",
}

# Substring fallbacks, in priority order. Ordering matters: "retrieve" must be
# tested before "review" would match a node called "retrieve_and_review".
_KEYWORD_FALLBACKS: tuple[tuple[str, EventType], ...] = (
    ("retriev", "retrieval"),
    ("search", "retrieval"),
    ("critiq", "critique"),
    ("review", "critique"),
    ("revis", "revision"),
    ("plan", "plan"),
    ("tool", "tool_call"),
    ("error", "error"),
    ("retry", "retry"),
    ("final", "final"),
    ("answer", "final"),
    ("decid", "decision"),
)


def map_node_to_event(node_name: str) -> EventType:
    lowered = node_name.strip().lower()
    if lowered in NODE_EVENT_MAP:
        return NODE_EVENT_MAP[lowered]
    for needle, event_type in _KEYWORD_FALLBACKS:
        if needle in lowered:
            return event_type
    return "reasoning"


def _stringify(value: Any) -> str:
    """Render a state value as text without losing structure."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        # LangChain message dicts are the common case worth reading nicely.
        if "content" in value:
            return _stringify(value["content"])
        return "\n".join(f"{k}: {_stringify(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(item) for item in value)
    content = getattr(value, "content", None)
    return _stringify(content) if content is not None else str(value)


def _extract_evidence_refs(delta: dict[str, Any]) -> list[str]:
    """Pull document ids out of a retrieval state delta."""
    refs: list[str] = []
    for key in ("documents", "docs", "chunks", "context", "sources"):
        items = delta.get(key)
        if not isinstance(items, (list, tuple)):
            continue
        for item in items:
            if isinstance(item, dict):
                for id_key in ("chunk_id", "doc_id", "id", "source"):
                    if item.get(id_key):
                        refs.append(str(item[id_key]))
                        break
            elif isinstance(item, str):
                refs.append(item)
    return refs


def _normalise_events(events: list[Any]) -> list[tuple[str, dict[str, Any], Any]]:
    """Flatten either stream shape into (node_name, state_delta, timestamp)."""
    out: list[tuple[str, dict[str, Any], Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue

        # debug shape
        if "payload" in event and "type" in event:
            if event.get("type") not in (None, "task_result", "task"):
                continue
            payload = event.get("payload") or {}
            node = payload.get("name") or payload.get("id") or "unknown"
            delta = payload.get("result") or payload.get("input") or {}
            if isinstance(delta, list):
                delta = {"result": delta}
            if not isinstance(delta, dict):
                delta = {"result": delta}
            out.append((str(node), delta, event.get("timestamp")))
            continue

        # updates shape: one key per node that ran
        for node, delta in event.items():
            if node.startswith("__") and node != "__end__":
                continue
            if not isinstance(delta, dict):
                delta = {"result": delta}
            out.append((str(node), delta, None))
    return out


def import_langgraph_events(
    events: list[Any],
    *,
    task_input: str,
    run_id: str | None = None,
    workflow_version: str = "langgraph-import",
    ground_truth: str | None = None,
    success: bool | None = None,
    final_output: str | None = None,
    model: str = "unknown",
    default_latency_ms: int = 0,
) -> Run:
    """Convert LangGraph stream events into a `Run`.

    ``success`` defaults to False when not supplied. An import has no grader, and
    silently marking unknown runs as successful would corrupt any statistic
    computed over an imported corpus.
    """
    normalised = _normalise_events(events)
    started_at = datetime.now(timezone.utc)
    resolved_run_id = run_id or str(uuid.uuid4())

    steps: list[Step] = []
    clock = started_at
    previous_id: str | None = None

    for seq, (node, delta, timestamp) in enumerate(normalised):
        event_type = map_node_to_event(node)

        parsed_time: datetime | None = None
        if isinstance(timestamp, str):
            try:
                parsed_time = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                )
            except ValueError:
                parsed_time = None
        if parsed_time is None:
            clock += timedelta(milliseconds=max(default_latency_ms, 1))
            parsed_time = clock
        else:
            clock = parsed_time

        error = delta.get("error")
        error_info = None
        if error:
            error_info = ErrorInfo(
                error_type=(
                    error.get("type", "Error")
                    if isinstance(error, dict)
                    else type(error).__name__
                ),
                message=_stringify(
                    error.get("message") if isinstance(error, dict) else error
                ),
            )

        step = Step(
            step_id=str(uuid.uuid4()),
            run_id=resolved_run_id,
            parent_step_id=previous_id,
            seq=seq,
            agent_id=node,
            agent_role=node,
            model=str(delta.get("model") or model),
            event_type="error" if error_info is not None else event_type,
            input=_stringify(delta.get("input") or delta.get("question") or ""),
            output=_stringify(
                delta.get("output")
                or delta.get("answer")
                or delta.get("messages")
                or delta.get("result")
                or {k: v for k, v in delta.items() if k not in ("error", "model")}
            ),
            timestamp=parsed_time,
            latency_ms=int(delta.get("latency_ms") or default_latency_ms),
            prompt_tokens=int(delta.get("prompt_tokens") or 0),
            completion_tokens=int(delta.get("completion_tokens") or 0),
            cost_usd=float(delta.get("cost_usd") or 0.0),
            evidence_refs=_extract_evidence_refs(delta),
            error=error_info,
        )
        steps.append(step)
        previous_id = step.step_id

    resolved_final = final_output
    if resolved_final is None:
        resolved_final = steps[-1].output if steps else ""

    total_cost = sum(s.cost_usd for s in steps)
    return Run(
        run_id=resolved_run_id,
        workflow_type="imported",
        workflow_version=workflow_version,
        task_input=task_input,
        final_output=resolved_final,
        success=bool(success),
        ground_truth=ground_truth,
        injected_fault=None,
        started_at=started_at,
        completed_at=clock,
        total_cost_usd=total_cost,
        total_tokens=sum(s.prompt_tokens + s.completion_tokens for s in steps),
        steps=steps,
    )
