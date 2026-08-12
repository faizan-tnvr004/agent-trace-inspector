/**
 * Cost view (FR-28).
 *
 * Two bar charts, cost by agent and cost by event type, plus a totals table
 * that reconciles against `run.total_cost_usd`. The reconciliation is shown
 * rather than assumed: the backend sums the same steps independently and reports
 * whether the two agree, so a mismatch is visible instead of hidden.
 *
 * Bars are plain divs. A charting library would be a dependency for two bar
 * charts over at most a dozen categories.
 */

import { useEffect, useState } from 'react'
import { getCost } from '../api'
import type { CostBucket, CostResponse, Run } from '../types'
import {
  EmptyState,
  Panel,
  agentColour,
  formatMs,
  formatUsd,
} from '../components/ui'

function BarChart({
  title,
  data,
  colourFor,
}: {
  title: string
  data: Record<string, CostBucket>
  colourFor: (key: string) => string
}) {
  const entries = Object.entries(data).sort(
    (a, b) => b[1].cost_usd - a[1].cost_usd,
  )
  const max = Math.max(...entries.map(([, bucket]) => bucket.cost_usd), 0)

  return (
    <Panel title={title}>
      {entries.length === 0 ? (
        <EmptyState>No steps recorded.</EmptyState>
      ) : (
        <ul className="space-y-2">
          {entries.map(([key, bucket]) => (
            <li key={key}>
              <div className="mb-0.5 flex items-baseline justify-between gap-2 text-xs">
                <span className="truncate font-mono">{key}</span>
                <span className="shrink-0 font-mono text-muted">
                  {formatUsd(bucket.cost_usd)} · {bucket.steps} step
                  {bucket.steps === 1 ? '' : 's'}
                </span>
              </div>
              <div
                className="h-3 w-full overflow-hidden rounded bg-gray-100"
                role="img"
                aria-label={`${key}: ${formatUsd(bucket.cost_usd)}`}
              >
                <div
                  className={`h-full rounded ${colourFor(key)}`}
                  style={{
                    width: max > 0 ? `${Math.max((bucket.cost_usd / max) * 100, 1)}%` : '0%',
                  }}
                />
              </div>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}

const EVENT_BAR = 'bg-slate-500'

export default function Cost({ run }: { run: Run }) {
  const [data, setData] = useState<CostResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setData(null)
    setError(null)
    getCost(run.run_id)
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

  if (error) return <EmptyState>Could not load cost: {error}</EmptyState>
  if (!data) return <EmptyState>Loading cost…</EmptyState>

  return (
    <div className="space-y-3">
      <div
        className={`rounded border px-4 py-2 text-xs ${
          data.reconciles
            ? 'border-success/40 bg-successBg text-success'
            : 'border-danger/40 bg-dangerBg text-danger'
        }`}
      >
        {data.reconciles
          ? `Totals reconcile with the run's recorded total of ${formatUsd(
              data.run_total_cost_usd,
            )}.`
          : `Totals do not reconcile: steps sum to ${formatUsd(
              data.total.cost_usd,
            )} but the run records ${formatUsd(data.run_total_cost_usd)}.`}
      </div>

      <p className="rounded border border-line bg-gray-50 px-4 py-2 text-[11px] leading-relaxed text-muted">
        Cost basis: {data.cost_basis}. Corpus generation ran on a free tier, so
        no money was spent. These figures show what the same token usage would
        cost at the model's published rate.
      </p>

      <div className="grid gap-3 lg:grid-cols-2">
        <BarChart
          title="Cost by agent"
          data={data.by_agent}
          colourFor={(key) => agentColour(key).bar}
        />
        <BarChart
          title="Cost by event type"
          data={data.by_event_type}
          colourFor={() => EVENT_BAR}
        />
      </div>

      <Panel title="Totals">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-line text-left text-muted">
                <th className="py-1.5 pr-4 font-medium">Measure</th>
                <th className="py-1.5 pr-4 text-right font-medium">Value</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              <Row label="Steps" value={String(data.total.steps)} />
              <Row label="Prompt tokens" value={String(data.total.prompt_tokens)} />
              <Row
                label="Completion tokens"
                value={String(data.total.completion_tokens)}
              />
              <Row label="Total tokens" value={String(data.total.total_tokens)} />
              <Row
                label="Total latency"
                value={formatMs(data.total.latency_ms)}
              />
              <Row
                label="Cost summed from steps"
                value={formatUsd(data.total.cost_usd)}
              />
              <Row
                label="Cost recorded on the run"
                value={formatUsd(data.run_total_cost_usd)}
              />
              <Row
                label="Tokens recorded on the run"
                value={String(run.total_tokens)}
              />
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <tr className="border-b border-line/60 last:border-b-0">
      <td className="py-1.5 pr-4 font-sans">{label}</td>
      <td className="py-1.5 pr-4 text-right">{value}</td>
    </tr>
  )
}
