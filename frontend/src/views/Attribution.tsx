/**
 * Attribution view (FR-27).
 *
 * Only meaningful for failed runs, so successful ones get an explicit empty
 * state rather than a misleading ranking. Where an injected fault gives ground
 * truth, a banner shows predicted against actual in green or red, naming both
 * steps, so whether the tool got it right is answerable in one screen.
 */

import { useEffect, useState } from 'react'
import { getAttribution } from '../api'
import type { AttributionResult, Run, Step } from '../types'
import {
  EmptyState,
  EventBadge,
  Panel,
  WarningIcon,
  agentColour,
  truncate,
} from '../components/ui'

function describe(step: Step | undefined, seq: number | null): string {
  if (seq === null) return 'no step'
  if (!step) return `step ${seq}`
  return `step ${seq} (${step.event_type}, ${step.agent_role})`
}

function Verdict({
  result,
  stepsBySeq,
}: {
  result: AttributionResult
  stepsBySeq: Map<number, Step>
}) {
  if (result.actual_fault_step_seq === null) {
    return (
      <div className="rounded border border-line bg-gray-50 px-4 py-3 text-sm">
        <p className="font-medium">No ground truth for this run</p>
        <p className="mt-1 text-xs text-muted">
          This run carries no injected fault, so the prediction below cannot be
          scored. It is a hypothesis, not a verified result.
        </p>
      </div>
    )
  }

  const correct =
    result.predicted_step_seq !== null &&
    result.predicted_step_seq === result.actual_fault_step_seq

  return (
    <div
      className={`rounded border px-4 py-3 ${
        correct
          ? 'border-success/40 bg-successBg'
          : 'border-danger/40 bg-dangerBg'
      }`}
    >
      <p
        className={`flex items-center gap-2 text-sm font-semibold ${
          correct ? 'text-success' : 'text-danger'
        }`}
      >
        {correct ? '✓' : <WarningIcon />}
        {correct ? 'Prediction correct' : 'Prediction incorrect'}
      </p>
      <dl className="mt-2 grid gap-x-8 gap-y-1 text-xs sm:grid-cols-2">
        <div>
          <dt className="uppercase tracking-wide text-muted">Predicted</dt>
          <dd className="font-mono">
            {describe(
              result.predicted_step_seq === null
                ? undefined
                : stepsBySeq.get(result.predicted_step_seq),
              result.predicted_step_seq,
            )}
          </dd>
        </div>
        <div>
          <dt className="uppercase tracking-wide text-muted">
            Actual (injected fault)
          </dt>
          <dd className="font-mono">
            {describe(
              stepsBySeq.get(result.actual_fault_step_seq),
              result.actual_fault_step_seq,
            )}
            {result.actual_fault_type && ` — ${result.actual_fault_type}`}
          </dd>
        </div>
      </dl>
    </div>
  )
}

export default function Attribution({ run }: { run: Run }) {
  const [result, setResult] = useState<AttributionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setResult(null)
    setError(null)
    getAttribution(run.run_id)
      .then((response) => {
        if (!cancelled) setResult(response)
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message)
      })
    return () => {
      cancelled = true
    }
  }, [run.run_id])

  if (run.success) {
    return (
      <EmptyState>
        This run succeeded, so there is no failure to attribute. Attribution
        applies to failed runs.
        {run.injected_fault && (
          <>
            <br />
            <span className="mt-2 inline-block text-xs">
              It does carry an injected{' '}
              <span className="font-mono">{run.injected_fault.fault_type}</span>{' '}
              fault at step {run.injected_fault.target_step_seq}, which did not
              change the outcome.
            </span>
          </>
        )}
      </EmptyState>
    )
  }

  if (error) return <EmptyState>Could not load attribution: {error}</EmptyState>
  if (!result) return <EmptyState>Loading attribution…</EmptyState>

  const stepsBySeq = new Map(run.steps.map((step) => [step.seq, step]))

  return (
    <div className="space-y-3">
      <Verdict result={result} stepsBySeq={stepsBySeq} />

      <Panel
        title="Reasoning"
        right={
          <span className="font-mono text-[11px] text-muted">
            {result.candidates.length} candidate step(s)
          </span>
        }
      >
        <p className="text-sm leading-relaxed">{result.reason}</p>
        <p className="mt-2 font-mono text-[11px] text-muted">
          rule weights:{' '}
          {Object.entries(result.rule_weights)
            .map(([rule, weight]) => `${rule}=${weight}`)
            .join('  ')}
        </p>
      </Panel>

      <Panel title="Ranked candidates">
        {result.candidates.length === 0 ? (
          <EmptyState>
            No step matched any failure-origin rule, so the trace does not
            localise the cause. That is reported rather than guessed at.
          </EmptyState>
        ) : (
          <ol className="space-y-2">
            {result.candidates.map((candidate, index) => {
              const colour = agentColour(candidate.agent_id)
              const isPredicted = candidate.step_id === result.predicted_step_id
              const isActual =
                result.actual_fault_step_seq === candidate.seq
              return (
                <li
                  key={candidate.step_id}
                  className={`rounded border px-3 py-2 ${
                    isPredicted
                      ? 'border-ink bg-gray-50'
                      : 'border-line bg-white'
                  }`}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-[11px] text-muted">
                      #{index + 1}
                    </span>
                    <span className={`h-4 w-1 rounded ${colour.bar}`} aria-hidden />
                    <span className="font-mono text-[11px] text-muted">
                      seq {candidate.seq}
                    </span>
                    <EventBadge eventType={candidate.event_type} />
                    <span className={`rounded px-1.5 text-[11px] ${colour.chip}`}>
                      {candidate.agent_role}
                    </span>
                    <span className="font-mono text-xs font-semibold">
                      {candidate.score.toFixed(2)}
                    </span>
                    {isPredicted && (
                      <span className="rounded bg-ink px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-white">
                        predicted
                      </span>
                    )}
                    {isActual && (
                      <span className="rounded bg-success px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-white">
                        actual fault
                      </span>
                    )}
                  </div>
                  <ul className="mt-1 list-inside list-disc text-xs text-gray-700">
                    {candidate.reasons.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                  {stepsBySeq.get(candidate.seq) && (
                    <p className="mt-1 font-mono text-[11px] text-muted">
                      {truncate(stepsBySeq.get(candidate.seq)!.output, 200) ||
                        '(empty output)'}
                    </p>
                  )}
                </li>
              )
            })}
          </ol>
        )}
      </Panel>
    </div>
  )
}
