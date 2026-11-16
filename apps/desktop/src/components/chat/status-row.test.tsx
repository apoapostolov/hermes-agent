import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { Codicon } from '@/components/ui/codicon'

import { StatusRow } from './status-row'

vi.stubGlobal(
  'ResizeObserver',
  class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
)

afterEach(cleanup)

it('keeps dismiss and nested controls independent from row activation', () => {
  const activate = vi.fn()
  const dismiss = vi.fn()
  const action = vi.fn()

  const { container } = render(
    <div data-slot="composer-status-stack">
      <StatusRow
        dismiss={{ label: 'Dismiss task', onDismiss: dismiss }}
        leading={<Codicon name="comment" />}
        onActivate={activate}
        trailing={
          <button
            onClick={event => {
              event.stopPropagation()
              action()
            }}
          >
            Edit task
          </button>
        }
      >
        <span>Task title</span>
      </StatusRow>
    </div>
  )

  const row = container.querySelector<HTMLElement>('[data-slot="status-row"]')!
  const close = screen.getByRole('button', { name: 'Dismiss task' })
  fireEvent.keyDown(close, { key: 'Enter' })
  fireEvent.click(close)
  expect(dismiss).toHaveBeenCalledOnce()
  expect(activate).not.toHaveBeenCalled()
  fireEvent.keyDown(screen.getByRole('button', { name: 'Edit task' }), { key: ' ' })
  fireEvent.click(screen.getByRole('button', { name: 'Edit task' }))
  expect(action).toHaveBeenCalledOnce()
  expect(activate).not.toHaveBeenCalled()
  fireEvent.keyDown(row, { key: 'Enter' })
  fireEvent.keyDown(row, { key: ' ' })
  expect(activate).toHaveBeenCalledTimes(2)
})

// #156548: a queued prompt's controls act on the prompt itself, so a row that opts
// in draws them before the content and keeps them on screen at rest. Background and
// subagent rows keep the trailing, hover-revealed cluster.
it('leads with its actions, visible at rest, when the row opts in', () => {
  const { container } = render(
    <div data-slot="composer-status-stack">
      <StatusRow actionsFirst leading={<Codicon name="comment" />} trailing={<button type="button">Edit task</button>}>
        <span>Queued prompt</span>
      </StatusRow>
    </div>
  )

  const row = container.querySelector<HTMLElement>('[data-slot="status-row"]')!
  const actions = row.querySelector<HTMLElement>('.status-row-actions')!
  const content = row.querySelector<HTMLElement>('.status-row-content')!

  expect(row.getAttribute('data-status-actions')).toBe('leading')
  // DOM order, not just the grid columns: keyboard and screen readers follow it.
  expect(actions.nextElementSibling).toBe(content)
  expect(actions.className).not.toContain('opacity-0')
})

it('keeps its actions trailing and hover-revealed by default', () => {
  const { container } = render(
    <div data-slot="composer-status-stack">
      <StatusRow leading={<Codicon name="comment" />} trailing={<button type="button">Stop task</button>}>
        <span>Background command</span>
      </StatusRow>
    </div>
  )

  const row = container.querySelector<HTMLElement>('[data-slot="status-row"]')!
  const actions = row.querySelector<HTMLElement>('.status-row-actions')!
  const content = row.querySelector<HTMLElement>('.status-row-content')!

  expect(row.hasAttribute('data-status-actions')).toBe(false)
  expect(content.nextElementSibling).toBe(actions)
  expect(actions.className).toContain('opacity-0')
})
