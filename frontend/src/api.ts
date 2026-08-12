/**
 * Typed fetch wrappers for the backend.
 *
 * The frontend holds no analysis logic: every number it shows comes from an
 * endpoint, so the same figures are reproducible from a script without the UI.
 */

import type {
  AttributionResult,
  CostResponse,
  CriticalResponse,
  ProvenanceResponse,
  Run,
  RunListResponse,
} from './types'

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  'http://localhost:8000'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function get<T>(path: string): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`)
  } catch {
    throw new ApiError(
      `Could not reach the backend at ${API_BASE}. Is it running?`,
      0,
    )
  }

  if (!response.ok) {
    // 404 carries a useful detail string; surface it rather than a bare code.
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* response had no JSON body; keep the status line */
    }
    throw new ApiError(detail, response.status)
  }

  return (await response.json()) as T
}

export interface ListRunsParams {
  success?: boolean
  workflowType?: string
  limit?: number
  offset?: number
}

export function listRuns(params: ListRunsParams = {}): Promise<RunListResponse> {
  const query = new URLSearchParams()
  if (params.success !== undefined) query.set('success', String(params.success))
  if (params.workflowType) query.set('workflow_type', params.workflowType)
  query.set('limit', String(params.limit ?? 200))
  query.set('offset', String(params.offset ?? 0))
  return get<RunListResponse>(`/runs?${query.toString()}`)
}

export const getRun = (runId: string) => get<Run>(`/runs/${runId}`)

export const getCritical = (runId: string, k = 5) =>
  get<CriticalResponse>(`/runs/${runId}/critical?k=${k}`)

export const getAttribution = (runId: string) =>
  get<AttributionResult>(`/runs/${runId}/attribution`)

export const getProvenance = (runId: string) =>
  get<ProvenanceResponse>(`/runs/${runId}/provenance`)

export const getCost = (runId: string) =>
  get<CostResponse>(`/runs/${runId}/cost`)

export const exportUrl = (runId: string) => `${API_BASE}/runs/${runId}/export`
