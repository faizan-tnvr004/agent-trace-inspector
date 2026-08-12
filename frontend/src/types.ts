/**
 * Mirrors the backend Pydantic models in `backend/app/models.py` and the
 * extraction result types.
 *
 * These are hand-written rather than generated. The published JSON Schema
 * (`schema/trace.schema.json`) covers `Run` and its nested types, but not the
 * extraction results (`ScoreBreakdown`, `AttributionResult`) or the endpoint
 * envelopes, so a generator would cover roughly half the surface and leave the
 * rest to drift silently. The union types below are kept identical to the
 * `Literal` types on the Python side; a mismatch there is a compile error at
 * the point of use rather than a runtime surprise.
 */

export type WorkflowType = 'reviewer_pipeline' | 'rag_qa' | 'imported'

export type EventType =
  | 'plan'
  | 'tool_call'
  | 'tool_result'
  | 'retrieval'
  | 'reasoning'
  | 'critique'
  | 'revision'
  | 'decision'
  | 'error'
  | 'retry'
  | 'final'

export type FaultType =
  | 'dropped_retrieval'
  | 'truncated_tool_result'
  | 'forced_false_rejection'
  | 'injected_contradiction'

export type RejectionOutcome = 'repair' | 'damage' | 'no_change'

export interface ErrorInfo {
  error_type: string
  message: string
}

export interface InjectedFault {
  fault_type: FaultType
  target_step_seq: number
  description: string
}

export interface Step {
  step_id: string
  run_id: string
  parent_step_id: string | null
  seq: number
  agent_id: string
  agent_role: string
  model: string
  event_type: EventType
  input: string
  output: string
  timestamp: string
  latency_ms: number
  prompt_tokens: number
  completion_tokens: number
  cost_usd: number
  evidence_refs: string[]
  error: ErrorInfo | null
  retry_of: string | null
  rejection_outcome: RejectionOutcome | null
}

export interface Run {
  run_id: string
  workflow_type: WorkflowType
  workflow_version: string
  task_input: string
  final_output: string
  success: boolean
  ground_truth: string | null
  injected_fault: InjectedFault | null
  started_at: string
  completed_at: string
  total_cost_usd: number
  total_tokens: number
  steps: Step[]
}

export interface RunSummary {
  run_id: string
  workflow_type: WorkflowType
  workflow_version: string
  task_input: string
  success: boolean
  has_injected_fault: boolean
  fault_type: FaultType | null
  started_at: string
  completed_at: string
  total_cost_usd: number
  total_tokens: number
  step_count: number
}

export interface RunListResponse {
  items: RunSummary[]
  total: number
  limit: number
  offset: number
}

export interface ScoreBreakdown {
  step_id: string
  seq: number
  agent_id: string
  agent_role: string
  event_type: EventType
  evidence_survival: number
  branch: number
  error: number
  critical_score: number
  weights: Record<string, number>
  reasons: string[]
}

export interface CriticalResponse {
  steps: ScoreBreakdown[]
  k: number
  step_count: number
}

export interface Claim {
  claim_id: string
  run_id: string
  index: number
  text: string
  evidence_refs: string[]
  supported: boolean
}

export interface ProvenanceResponse {
  claims: Claim[]
  total: number
  unsupported: number
  final_output: string
}

export interface AttributionCandidate {
  step_id: string
  seq: number
  agent_id: string
  agent_role: string
  event_type: EventType
  score: number
  reasons: string[]
}

export interface AttributionResult {
  run_id: string
  success: boolean
  predicted_step_id: string | null
  predicted_step_seq: number | null
  reason: string
  candidates: AttributionCandidate[]
  actual_fault_step_seq: number | null
  actual_fault_type: FaultType | null
  rule_weights: Record<string, number>
}

export interface CostBucket {
  cost_usd: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  latency_ms: number
  steps: number
}

export interface CostResponse {
  by_agent: Record<string, CostBucket>
  by_event_type: Record<string, CostBucket>
  total: CostBucket
  run_total_cost_usd: number
  reconciles: boolean
  cost_basis: string
}
