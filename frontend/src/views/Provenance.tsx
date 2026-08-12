/**
 * Provenance view (FR-26).
 *
 * The final output split into claims, each expandable to the steps that support
 * it. Unsupported claims render in the danger colour with a warning icon and are
 * counted in the header, so "how much of this answer is grounded" is answerable
 * at a glance.
 *
 * "Supported" means some upstream step's output is textually similar to the
 * claim. It is not a claim of truth: a claim can be well supported by a step and
 * still be wrong if that step was wrong. The header says so, because the
 * distinction is easy to misread.
 */

import { useEffect, useState } from 'react'
import { getProvenance } from '../api'
import type { Claim, ProvenanceResponse, Run, Step } from '../types'
import {
  EmptyState,
  EventBadge,
  Panel,
  WarningIcon,
  agentColour,
  truncate,
} from '../components/ui'

function ClaimRow({
  claim,
  stepsById,
}: {
  claim: Claim
  stepsById: Map<string, Step>
}) {
  const [open, setOpen] = useState(false)
  const supporting = claim.evidence_refs
    .map((ref) => stepsById.get(ref))
    .filter((step): step is Step => step !== undefined)

  return (
    <li
      className={`rounded border ${
        claim.supported
          ? 'border-line bg-white'
          : 'border-danger/40 bg-dangerBg'
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-start gap-2 px-3 py-2 text-left"
      >
        {claim.supported ? (
          <span
            className="mt-0.5 shrink-0 font-mono text-[11px] text-success"
            aria-label="supported"
          >
            ✓
          </span>
        ) : (
          <WarningIcon className="mt-0.5 text-danger" />
        )}
        <span
          className={`min-w-0 flex-1 text-sm ${
            claim.supported ? 'text-gray-800' : 'font-medium text-danger'
          }`}
        >
          {claim.text}
        </span>
        <span className="shrink-0 font-mono text-[11px] text-muted">
          {claim.supported ? `${supporting.length} source(s)` : 'no source'}
        </span>
        <span className="shrink-0 text-xs text-muted">{open ? '▾' : '▸'}</span>
      </button>

      {open && (
        <div className="space-y-2 border-t border-line/70 px-3 py-2">
          {supporting.length === 0 ? (
            <p className="text-xs text-danger">
              No step in this trace produced text matching this claim. The run
              asserted it without recorded grounding.
            </p>
          ) : (
            supporting.map((step) => {
              const colour = agentColour(step.agent_id)
              return (
                <div
                  key={step.step_id}
                  className="rounded border border-line bg-white p-2"
                >
                  <div className="mb-1 flex items-center gap-2">
                    <span
                      className={`h-4 w-1 rounded ${colour.bar}`}
                      aria-hidden
                    />
                    <span className="font-mono text-[11px] text-muted">
                      seq {step.seq}
                    </span>
                    <EventBadge eventType={step.event_type} />
                    <span className={`rounded px-1.5 text-[11px] ${colour.chip}`}>
                      {step.agent_role}
                    </span>
                    {step.evidence_refs.length > 0 && (
                      <span className="font-mono text-[11px] text-muted">
                        {step.evidence_refs.join(', ')}
                      </span>
                    )}
                  </div>
                  <p className="font-mono text-[11px] leading-relaxed text-gray-700">
                    {truncate(step.output, 400)}
                  </p>
                </div>
              )
            })
          )}
        </div>
      )}
    </li>
  )
}

export default function Provenance({ run }: { run: Run }) {
  const [data, setData] = useState<ProvenanceResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setData(null)
    setError(null)
    getProvenance(run.run_id)
      .then((response) => {
        if (!cancelled) setData(response)
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message)
      })
    return () => {
      cancelled = true
    }
  }, [run.run_id])

  if (error) return <EmptyState>Could not load provenance: {error}</EmptyState>
  if (!data) return <EmptyState>Loading provenance…</EmptyState>

  const stepsById = new Map(run.steps.map((step) => [step.step_id, step]))

  return (
    <Panel
      title={
        data.total === 0
          ? 'No claims extracted'
          : `${data.unsupported} of ${data.total} claims unsupported`
      }
      right={
        data.unsupported > 0 ? (
          <span className="flex items-center gap-1 rounded bg-dangerBg px-2 py-0.5 text-[11px] font-medium text-danger">
            <WarningIcon />
            ungrounded assertions present
          </span>
        ) : (
          <span className="rounded bg-successBg px-2 py-0.5 text-[11px] font-medium text-success">
            every claim traced to a step
          </span>
        )
      }
    >
      {data.total === 0 ? (
        <EmptyState>
          The final output produced no sentence-level claims long enough to
          analyse.
        </EmptyState>
      ) : (
        <>
          <p className="mb-3 text-[11px] leading-relaxed text-muted">
            Claims are sentences, split from the final output. Supported means
            some upstream step produced closely matching text, which shows where
            a claim came from, not whether it is true.
          </p>
          <ul className="space-y-1.5">
            {data.claims.map((claim) => (
              <ClaimRow
                key={claim.claim_id}
                claim={claim}
                stepsById={stepsById}
              />
            ))}
          </ul>
        </>
      )}
    </Panel>
  )
}
