/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Single source of truth for semantic colour. The danger colour is
        // required by FR-26 for unsupported claims and by FR-27 for an
        // incorrect attribution, so it is named rather than repeated.
        danger: '#b91c1c',
        dangerBg: '#fef2f2',
        success: '#15803d',
        successBg: '#f0fdf4',
        ink: '#111827',
        muted: '#6b7280',
        line: '#e5e7eb',
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
