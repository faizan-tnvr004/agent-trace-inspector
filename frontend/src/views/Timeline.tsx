/**
 * Timeline view (FR-24, FR-25).
 *
 * One row per step in `seq` order, colour-coded by agent, collapsed by default
 * so a 40-step run stays scannable. Filters compose: agent and event-type
 * selections intersect, and "critical only" restricts to the exact step ids
 * returned by `/critical?k=5` rather than recomputing anything client-side.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { getCritical } from '../api'
import type { Run, ScoreBreakdown, Step } from '../types'
import {
  EventBadge,
  EmptyState,
  agentColour,
  formatMs,
  truncate,
} from '../components/ui'

/** Above this many rows the list is windowed rather than fully rendered (NFR-2). */
const VIRTUALISE_THRESHOLD = 100
const ROW_HEIGHT = 44
const OVERSCAN = 8

function StepRow({
  step,
  expanded,
  onToggle,
  score,
}: {
  step: Step
  expanded: boolean
  onToggle: () => void
  score?: ScoreBreakdown
}) {
  const colour = agentColour(step.agent_id)
  return (
    <li className="border-b border-line last:border-b-0">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="flex w-full items-center gap-3 px-3 py-2.5 text-left hover:bg-gray-50"
      >
        <span className={`h-8 w-1 shrink-0 rounded ${colour.bar}`} aria-hidden />
        <span className="w-8 shrink-0 font-mono text-xs text-muted">
          {step.seq}
        </span>
        <span
          className={`w-24 shrink-0 truncate rounded px-1.5 py-0.5 text-xs ${colour.chip}`}
          title={`${step.agent_id} (${step.agent_role})`}
        >
          {step.agent_role}
        </span>
        <span className="w-24 shrink-0">
          <EventBadge eventType={step.event_type} />
        </span>
        <span className="w-16 shrink-0 text-right font-mono text-xs text-muted">
          {formatMs(step.latency_ms)}
        </span>
        {score && (
          <span
            className="w-14 shrink-0 text-right font-mono text-xs text-violet-700"
            title="critical score"
          >
            {score.critical_score.toFixed(2)}
          </span>
        )}
        <span className="min-w-0 flex-1 truncate text-xs text-gray-700">
          {truncate(step.output || step.input, 140)}
        </span>
        <span className="shrink-0 text-xs text-muted">{expanded ? '▾' : '▸'}</span>
      </button>

      {expanded && (
        <div className="space-y-3 border-t border-line bg-gray-50 px-4 py-3 text-xs">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-4">
            <Meta label="agent_id" value={step.agent_id} />
            <Meta label="model" value={step.model} />
            <Meta
              label="tokens"
              value={`${step.prompt_tokens} in / ${step.completion_tokens} out`}
            />
            <Meta label="cost" value={`$${step.cost_usd.toFixed(8)}`} />
            {step.retry_of && <Meta label="retry_of" value={step.retry_of} />}
            {step.rejection_outcome && (
              <Meta label="rejection outcome" value={step.rejection_outcome} />
            )}
            {step.evidence_refs.length > 0 && (
              <Meta
                label="evidence_refs"
                value={step.evidence_refs.join(', ')}
                wide
              />
            )}
          </dl>

          {step.error && (
            <p className="rounded border border-danger/30 bg-dangerBg px-3 py-2 text-danger">
              <strong className="font-mono">{step.error.error_type}</strong>{' '}
              {step.error.message}
            </p>
          )}

          {score && (
            <div className="rounded border border-line bg-white px-3 py-2">
              <p className="mb-1 font-semibold">Why this step scored</p>
              <p className="font-mono text-[11px] text-muted">
                evidence {score.evidence_survival.toFixed(3)} × {score.weights.evidence_survival}
                {' + '}branch {score.branch.toFixed(1)} × {score.weights.branch}
                {' + '}error {score.error.toFixed(1)} × {score.weights.error}
                {' = '}
                {score.critical_score.toFixed(3)}
              </p>
              {score.reasons.length > 0 && (
                <ul className="mt-1 list-inside list-disc text-[11px] text-gray-700">
                  {score.reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <Block label="Input" text={step.input} />
          <Block label="Output" text={step.output} />
        </div>
      )}
    </li>
  )
}

function Meta({
  label,
  value,
  wide,
}: {
  label: string
  value: string
  wide?: boolean
}) {
  return (
    <div className={wide ? 'col-span-2 sm:col-span-4' : ''}>
      <dt className="text-[11px] uppercase tracking-wide text-muted">{label}</dt>
      <dd className="break-all font-mono text-[11px]">{value}</dd>
    </div>
  )
}

function Block({ label, text }: { label: string; text: string }) {
  return (
    <div>
      <p className="mb-1 text-[11px] uppercase tracking-wide text-muted">{label}</p>
      <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded border border-line bg-white px-3 py-2 font-mono text-[11px] leading-relaxed">
        {text || '(empty)'}
      </pre>
    </div>
  )
}

function Chips<T extends string>({
  label,
  options,
  selected,
  onChange,
}: {
  label: string
  options: T[]
  selected: Set<T>
  onChange: (next: Set<T>) => void
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-[11px] uppercase tracking-wide text-muted">{label}</span>
      {options.map((option) => {
        const active = selected.has(option)
        return (
          <button
            key={option}
            type="button"
            aria-pressed={active}
            onClick={() => {
              const next = new Set(selected)
              if (active) next.delete(option)
              else next.add(option)
              onChange(next)
            }}
            className={`rounded border px-2 py-0.5 font-mono text-[11px] ${
              active
                ? 'border-ink bg-ink text-white'
                : 'border-line bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            {option}
          </button>
        )
      })}
      {selected.size > 0 && (
        <button
          type="button"
          onClick={() => onChange(new Set())}
          className="text-[11px] text-muted underline"
        >
          clear
        </button>
      )}
    </div>
  )
}

export default function Timeline({ run }: { run: Run }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [agents, setAgents] = useState<Set<string>>(new Set())
  const [eventTypes, setEventTypes] = useState<Set<string>>(new Set())
  const [criticalOnly, setCriticalOnly] = useState(false)
  const [critical, setCritical] = useState<ScoreBreakdown[] | null>(null)
  const [criticalError, setCriticalError] = useState<string | null>(null)

  const scrollRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)

  // Reset per-run state so filters from a previous run do not silently hide
  // steps in the next one.
  useEffect(() => {
    setExpanded(new Set())
    setAgents(new Set())
    setEventTypes(new Set())
    setCriticalOnly(false)
    setCritical(null)
    setCriticalError(null)
    setScrollTop(0)
  }, [run.run_id])

  useEffect(() => {
    if (!criticalOnly || critical) return
    let cancelled = false
    getCritical(run.run_id, 5)
      .then((response) => {
        if (!cancelled) setCritical(response.steps)
      })
      .catch((error: Error) => {
        if (!cancelled) setCriticalError(error.message)
      })
    return () => {
      cancelled = true
    }
  }, [criticalOnly, critical, run.run_id])

  const allAgents = useMemo(
    () => [...new Set(run.steps.map((s) => s.agent_id))].sort(),
    [run.steps],
  )
  const allEventTypes = useMemo(
    () => [...new Set(run.steps.map((s) => s.event_type))].sort(),
    [run.steps],
  )
  const scoreByStepId = useMemo(
    () => new Map((critical ?? []).map((s) => [s.step_id, s])),
    [critical],
  )

  const visible = useMemo(() => {
    let steps = run.steps
    if (agents.size > 0) steps = steps.filter((s) => agents.has(s.agent_id))
    if (eventTypes.size > 0)
      steps = steps.filter((s) => eventTypes.has(s.event_type))
    if (criticalOnly && critical) {
      const ids = new Set(critical.map((s) => s.step_id))
      steps = steps.filter((s) => ids.has(s.step_id))
    }
    return steps
  }, [run.steps, agents, eventTypes, criticalOnly, critical])

  const virtualise = visible.length > VIRTUALISE_THRESHOLD
  const windowed = useMemo(() => {
    if (!virtualise) return { rows: visible, padTop: 0, padBottom: 0 }
    const viewport = scrollRef.current?.clientHeight ?? 600
    const first = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
    const count = Math.ceil(viewport / ROW_HEIGHT) + OVERSCAN * 2
    const last = Math.min(visible.length, first + count)
    return {
      rows: visible.slice(first, last),
      padTop: first * ROW_HEIGHT,
      padBottom: (visible.length - last) * ROW_HEIGHT,
    }
  }, [virtualise, visible, scrollTop])

  const toggle = (stepId: string) => {
    const next = new Set(expanded)
    if (next.has(stepId)) next.delete(stepId)
    else next.add(stepId)
    setExpanded(next)
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-2 rounded-lg border border-line bg-white p-3">
        <Chips
          label="agent"
          options={allAgents}
          selected={agents}
          onChange={setAgents}
        />
        <Chips
          label="event"
          options={allEventTypes}
          selected={eventTypes}
          onChange={setEventTypes}
        />
        <div className="flex flex-wrap items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={criticalOnly}
              onChange={(event) => setCriticalOnly(event.target.checked)}
              className="h-3.5 w-3.5"
            />
            <span>Critical only (top 5)</span>
          </label>
          <p className="font-mono text-[11px] text-muted">
            showing {visible.length} of {run.steps.length} steps
            {virtualise && ' · virtualised'}
          </p>
        </div>
        {criticalError && (
          <p className="text-[11px] text-danger">
            Could not load critical steps: {criticalError}
          </p>
        )}
      </div>

      {visible.length === 0 ? (
        <EmptyState>
          No steps match these filters. Clear a filter to see steps again.
        </EmptyState>
      ) : (
        <div
          ref={scrollRef}
          onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
          className={`rounded-lg border border-line bg-white ${
            virtualise ? 'max-h-[70vh] overflow-y-auto' : ''
          }`}
        >
          {windowed.padTop > 0 && <div style={{ height: windowed.padTop }} />}
          <ul>
            {windowed.rows.map((step) => (
              <StepRow
                key={step.step_id}
                step={step}
                expanded={expanded.has(step.step_id)}
                onToggle={() => toggle(step.step_id)}
                score={scoreByStepId.get(step.step_id)}
              />
            ))}
          </ul>
          {windowed.padBottom > 0 && <div style={{ height: windowed.padBottom }} />}
        </div>
      )}
    </div>
  )
}
