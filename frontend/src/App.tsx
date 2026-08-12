/**
 * Application shell: run list, run header, and the four views.
 *
 * The run list carries the filters from `GET /runs` and marks failed and
 * fault-injected runs, because the first thing a user of this tool wants is a
 * failed run with known ground truth. A developer opening this cold should be
 * able to reach a failure cause without reading the codebase (NFR-8).
 *
 * The selected run and view are held in the URL hash, so a particular view of a
 * particular run can be linked to and survives a reload (FR-30).
 */

import { useCallback, useEffect, useState } from 'react'
import { exportUrl, getRun, listRuns } from './api'
import type { Run, RunSummary } from './types'
import { EmptyState, formatUsd, truncate } from './components/ui'
import Attribution from './views/Attribution'
import Cost from './views/Cost'
import Provenance from './views/Provenance'
import Timeline from './views/Timeline'

const VIEWS = ['timeline', 'provenance', 'attribution', 'cost'] as const
type ViewName = (typeof VIEWS)[number]

type OutcomeFilter = 'all' | 'failed' | 'succeeded'

function readHash(): { runId: string | null; view: ViewName } {
  const [runId, view] = window.location.hash.replace(/^#\/?/, '').split('/')
  const parsed = VIEWS.includes(view as ViewName) ? (view as ViewName) : 'timeline'
  return { runId: runId || null, view: parsed }
}

function RunListItem({
  summary,
  selected,
  onSelect,
}: {
  summary: RunSummary
  selected: boolean
  onSelect: () => void
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        className={`w-full border-l-2 px-3 py-2 text-left text-xs hover:bg-gray-50 ${
          selected
            ? 'border-ink bg-gray-50'
            : 'border-transparent'
        }`}
      >
        <span className="flex items-center gap-1.5">
          <span
            className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${
              summary.success ? 'bg-success' : 'bg-danger'
            }`}
            aria-hidden
          />
          <span className="font-mono text-[11px] text-muted">
            {summary.workflow_type === 'rag_qa' ? 'rag' : 'rev'}
          </span>
          {summary.has_injected_fault && (
            <span
              className="rounded bg-amber-100 px-1 text-[10px] text-amber-900"
              title={summary.fault_type ?? undefined}
            >
              fault
            </span>
          )}
          <span className="ml-auto font-mono text-[10px] text-muted">
            {summary.step_count} steps
          </span>
        </span>
        <span className="mt-0.5 block truncate text-gray-700">
          {truncate(summary.task_input, 70)}
        </span>
      </button>
    </li>
  )
}

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [total, setTotal] = useState(0)
  const [listError, setListError] = useState<string | null>(null)
  const [outcome, setOutcome] = useState<OutcomeFilter>('all')
  const [workflow, setWorkflow] = useState<string>('')

  const [{ runId, view }, setRoute] = useState(readHash)
  const [run, setRun] = useState<Run | null>(null)
  const [runError, setRunError] = useState<string | null>(null)

  useEffect(() => {
    const onHashChange = () => setRoute(readHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const navigate = useCallback((nextRunId: string, nextView: ViewName) => {
    window.location.hash = `#/${nextRunId}/${nextView}`
  }, [])

  useEffect(() => {
    let cancelled = false
    setListError(null)
    listRuns({
      success: outcome === 'all' ? undefined : outcome === 'succeeded',
      workflowType: workflow || undefined,
    })
      .then((response) => {
        if (cancelled) return
        setRuns(response.items)
        setTotal(response.total)
      })
      .catch((error: Error) => {
        if (!cancelled) setListError(error.message)
      })
    return () => {
      cancelled = true
    }
  }, [outcome, workflow])

  useEffect(() => {
    if (!runId) {
      setRun(null)
      return
    }
    let cancelled = false
    setRun(null)
    setRunError(null)
    getRun(runId)
      .then((response) => {
        if (!cancelled) setRun(response)
      })
      .catch((error: Error) => {
        if (!cancelled) setRunError(error.message)
      })
    return () => {
      cancelled = true
    }
  }, [runId])

  return (
    <div className="flex h-screen flex-col">
      <header className="flex shrink-0 items-baseline gap-3 border-b border-line px-4 py-2.5">
        <h1 className="text-sm font-semibold">
          Multi-Agent Execution Trace Inspector
        </h1>
        <p className="text-[11px] text-muted">
          Which steps mattered, where it broke, what the answer rests on
        </p>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside className="flex w-80 shrink-0 flex-col border-r border-line">
          <div className="shrink-0 space-y-2 border-b border-line p-3">
            <div className="flex gap-1">
              {(['all', 'failed', 'succeeded'] as OutcomeFilter[]).map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={outcome === value}
                  onClick={() => setOutcome(value)}
                  className={`flex-1 rounded border px-2 py-1 text-[11px] ${
                    outcome === value
                      ? 'border-ink bg-ink text-white'
                      : 'border-line bg-white hover:bg-gray-50'
                  }`}
                >
                  {value}
                </button>
              ))}
            </div>
            <select
              value={workflow}
              onChange={(event) => setWorkflow(event.target.value)}
              className="w-full rounded border border-line px-2 py-1 text-[11px]"
            >
              <option value="">all workflows</option>
              <option value="rag_qa">rag_qa</option>
              <option value="reviewer_pipeline">reviewer_pipeline</option>
            </select>
            <p className="font-mono text-[10px] text-muted">
              {runs.length} shown · {total} matching
            </p>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto">
            {listError ? (
              <p className="p-3 text-xs text-danger">{listError}</p>
            ) : (
              <ul>
                {runs.map((summary) => (
                  <RunListItem
                    key={summary.run_id}
                    summary={summary}
                    selected={summary.run_id === runId}
                    onSelect={() => navigate(summary.run_id, view)}
                  />
                ))}
              </ul>
            )}
          </div>
        </aside>

        <main className="min-w-0 flex-1 overflow-y-auto bg-gray-50 p-4">
          {!runId ? (
            <EmptyState>
              Select a run. Runs marked <strong>fault</strong> with a red dot are
              failed runs with a known cause, which is where attribution can be
              checked against ground truth.
            </EmptyState>
          ) : runError ? (
            <EmptyState>Could not load run: {runError}</EmptyState>
          ) : !run ? (
            <EmptyState>Loading run…</EmptyState>
          ) : (
            <div className="space-y-3">
              <section className="rounded-lg border border-line bg-white p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded px-2 py-0.5 text-[11px] font-medium ${
                      run.success
                        ? 'bg-successBg text-success'
                        : 'bg-dangerBg text-danger'
                    }`}
                  >
                    {run.success ? 'succeeded' : 'failed'}
                  </span>
                  <span className="rounded bg-gray-100 px-2 py-0.5 font-mono text-[11px]">
                    {run.workflow_type} v{run.workflow_version}
                  </span>
                  {run.injected_fault && (
                    <span
                      className="rounded bg-amber-100 px-2 py-0.5 font-mono text-[11px] text-amber-900"
                      title={run.injected_fault.description}
                    >
                      {run.injected_fault.fault_type} @ seq{' '}
                      {run.injected_fault.target_step_seq}
                    </span>
                  )}
                  <span className="font-mono text-[11px] text-muted">
                    {run.steps.length} steps · {run.total_tokens} tokens ·{' '}
                    {formatUsd(run.total_cost_usd)}
                  </span>
                  <a
                    href={exportUrl(run.run_id)}
                    target="_blank"
                    rel="noreferrer"
                    className="ml-auto rounded border border-line px-2 py-0.5 text-[11px] hover:bg-gray-50"
                  >
                    audit export
                  </a>
                </div>

                <dl className="mt-3 grid gap-3 text-xs sm:grid-cols-3">
                  <div>
                    <dt className="uppercase tracking-wide text-muted">Task</dt>
                    <dd className="mt-0.5">{run.task_input}</dd>
                  </div>
                  <div>
                    <dt className="uppercase tracking-wide text-muted">
                      Expected
                    </dt>
                    <dd className="mt-0.5 font-mono">
                      {run.ground_truth ?? '(unknown)'}
                    </dd>
                  </div>
                  <div>
                    <dt className="uppercase tracking-wide text-muted">
                      Final output
                    </dt>
                    <dd className="mt-0.5">{truncate(run.final_output, 200)}</dd>
                  </div>
                </dl>
              </section>

              <nav className="flex gap-1">
                {VIEWS.map((name) => (
                  <button
                    key={name}
                    type="button"
                    aria-current={view === name}
                    onClick={() => navigate(run.run_id, name)}
                    className={`rounded border px-3 py-1 text-xs capitalize ${
                      view === name
                        ? 'border-ink bg-white font-medium'
                        : 'border-transparent text-muted hover:bg-white'
                    }`}
                  >
                    {name}
                  </button>
                ))}
              </nav>

              {view === 'timeline' && <Timeline run={run} />}
              {view === 'provenance' && <Provenance run={run} />}
              {view === 'attribution' && <Attribution run={run} />}
              {view === 'cost' && <Cost run={run} />}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
