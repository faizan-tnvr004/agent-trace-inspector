/**
 * Shared presentation primitives.
 *
 * Kept in one place so the four views stay visually consistent: the agent
 * colour assignment in particular must be identical across Timeline,
 * Attribution and Cost, or the same agent would appear in different colours
 * depending on which view you were looking at.
 */

import type { ReactNode } from 'react'
import type { EventType } from '../types'

/**
 * Fixed palette. Colours are assigned to agents deterministically from a stable
 * hash of the agent id, so a given agent is the same colour in every view and
 * across reloads. Assigning by first-seen order would make the colour depend on
 * which steps happened to be filtered in.
 */
const AGENT_PALETTE = [
  { bar: 'bg-sky-500', chip: 'bg-sky-100 text-sky-900', ring: 'ring-sky-300' },
  {
    bar: 'bg-violet-500',
    chip: 'bg-violet-100 text-violet-900',
    ring: 'ring-violet-300',
  },
  {
    bar: 'bg-amber-500',
    chip: 'bg-amber-100 text-amber-900',
    ring: 'ring-amber-300',
  },
  { bar: 'bg-teal-500', chip: 'bg-teal-100 text-teal-900', ring: 'ring-teal-300' },
  { bar: 'bg-rose-500', chip: 'bg-rose-100 text-rose-900', ring: 'ring-rose-300' },
  {
    bar: 'bg-indigo-500',
    chip: 'bg-indigo-100 text-indigo-900',
    ring: 'ring-indigo-300',
  },
  { bar: 'bg-lime-600', chip: 'bg-lime-100 text-lime-900', ring: 'ring-lime-300' },
  {
    bar: 'bg-fuchsia-500',
    chip: 'bg-fuchsia-100 text-fuchsia-900',
    ring: 'ring-fuchsia-300',
  },
]

export function agentColour(agentId: string) {
  let hash = 0
  for (let i = 0; i < agentId.length; i += 1) {
    hash = (hash * 31 + agentId.charCodeAt(i)) % 100000
  }
  return AGENT_PALETTE[hash % AGENT_PALETTE.length]
}

/** Event types that carry a warning meaning get a colour; the rest stay neutral. */
const EVENT_STYLES: Partial<Record<EventType, string>> = {
  error: 'bg-dangerBg text-danger border-danger/30',
  retry: 'bg-amber-50 text-amber-800 border-amber-300',
  critique: 'bg-violet-50 text-violet-800 border-violet-300',
  revision: 'bg-sky-50 text-sky-800 border-sky-300',
  final: 'bg-successBg text-success border-success/30',
  retrieval: 'bg-teal-50 text-teal-800 border-teal-300',
}

export function EventBadge({ eventType }: { eventType: EventType }) {
  const style = EVENT_STYLES[eventType] ?? 'bg-gray-50 text-gray-700 border-line'
  return (
    <span
      className={`inline-block rounded border px-1.5 py-0.5 font-mono text-[11px] leading-none ${style}`}
    >
      {eventType}
    </span>
  )
}

export function Panel({
  title,
  children,
  right,
}: {
  title: string
  children: ReactNode
  right?: ReactNode
}) {
  return (
    <section className="rounded-lg border border-line bg-white">
      <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <h2 className="text-sm font-semibold">{title}</h2>
        {right}
      </header>
      <div className="p-4">{children}</div>
    </section>
  )
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <p className="rounded border border-dashed border-line px-4 py-8 text-center text-sm text-muted">
      {children}
    </p>
  )
}

export function WarningIcon({ className = '' }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 16 16"
      aria-hidden="true"
      className={`h-4 w-4 shrink-0 ${className}`}
      fill="currentColor"
    >
      <path d="M8 1.5 15 14H1L8 1.5Zm0 3.6a.75.75 0 0 0-.75.75v3.3a.75.75 0 0 0 1.5 0V5.85A.75.75 0 0 0 8 5.1Zm0 6.05a.85.85 0 1 0 0 1.7.85.85 0 0 0 0-1.7Z" />
    </svg>
  )
}

export const formatUsd = (value: number) => `$${value.toFixed(6)}`

export const formatMs = (ms: number) =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms}ms`

export function truncate(text: string, max = 120) {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length <= max ? flat : `${flat.slice(0, max)}…`
}
