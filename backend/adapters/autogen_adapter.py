"""Import AutoGen message logs as a `Run` (FR-8).

Lowest priority in the specification's cut order, and it is the thinnest of the
adapters. It handles the common AutoGen export: a flat list of chat messages,
each with a sender, a recipient and content.

AutoGen conversations carry less structure than a LangGraph graph does. There are
no node boundaries, so event types are inferred from the speaker's configured
role and from explicit tool-call fields, never from the prose. That limit is
worth stating plainly: an AutoGen import produces a usable timeline and cost
breakdown, but its branch-point detection is weaker than a native trace's,
because a critique in AutoGen is just another message unless the agent was named
as a critic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.models import EventType, Run, Step

__all__ = ["ROLE_EVENT_MAP", "import_autogen_messages"]

ROLE_EVENT_MAP: dict[str, EventType] = {
    "user": "plan",
    "user_proxy": "plan",
    "planner": "plan",
    "assistant": "reasoning",
    "coder": "reasoning",
    "executor": "reasoning",
    "critic": "critique",
    "reviewer": "critique",
    "retriever": "retrieval",
    "tool": "tool_result",
    "function": "tool_result",
}


def _event_for(message: dict[str, Any], index: int, total: int) -> EventType:
    if message.get("tool_calls") or message.get("function_call"):
        return "tool_call"
    if message.get("role") in ("tool", "function") or message.get("tool_call_id"):
        return "tool_result"
    if index == total - 1:
        return "final"

    name = str(message.get("name") or message.get("role") or "").lower()
    for key, event_type in ROLE_EVENT_MAP.items():
        if key in name:
            return event_type
    return "reasoning"


def _content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    if content is None:
        for key in ("tool_calls", "function_call"):
            if message.get(key):
                return str(message[key])
        return ""
    return str(content)


def import_autogen_messages(
    messages: list[dict[str, Any]],
    *,
    task_input: str | None = None,
    run_id: str | None = None,
    workflow_version: str = "autogen-import",
    ground_truth: str | None = None,
    success: bool | None = None,
    model: str = "unknown",
    default_latency_ms: int = 0,
) -> Run:
    """Convert a flat AutoGen message log into a `Run`.

    ``success`` defaults to False for the same reason as in the LangGraph
    adapter: an import has no grader, and defaulting to success would corrupt any
    statistic over an imported corpus.
    """
    resolved_run_id = run_id or str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    clock = started_at

    steps: list[Step] = []
    previous_id: str | None = None
    total = len(messages)

    for seq, message in enumerate(messages):
        clock += timedelta(milliseconds=max(default_latency_ms, 1))
        sender = str(message.get("name") or message.get("role") or "agent")
        usage = message.get("usage") or {}

        step = Step(
            step_id=str(uuid.uuid4()),
            run_id=resolved_run_id,
            parent_step_id=previous_id,
            seq=seq,
            agent_id=sender,
            agent_role=str(message.get("role") or sender),
            model=str(message.get("model") or model),
            event_type=_event_for(message, seq, total),
            input=str(messages[seq - 1].get("content") or "") if seq else (
                task_input or ""
            ),
            output=_content(message),
            timestamp=clock,
            latency_ms=int(message.get("latency_ms") or default_latency_ms),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=float(message.get("cost_usd") or usage.get("cost") or 0.0),
        )
        steps.append(step)
        previous_id = step.step_id

    first_content = _content(messages[0]) if messages else ""
    return Run(
        run_id=resolved_run_id,
        workflow_type="imported",
        workflow_version=workflow_version,
        task_input=task_input or first_content,
        final_output=steps[-1].output if steps else "",
        success=bool(success),
        ground_truth=ground_truth,
        injected_fault=None,
        started_at=started_at,
        completed_at=clock,
        total_cost_usd=sum(s.cost_usd for s in steps),
        total_tokens=sum(s.prompt_tokens + s.completion_tokens for s in steps),
        steps=steps,
    )
