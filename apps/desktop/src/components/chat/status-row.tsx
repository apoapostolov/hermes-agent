import './status-stack.css'

import { type CSSProperties, type KeyboardEvent, type MouseEvent, type ReactNode, type Ref } from 'react'

import { cn } from '@/lib/utils'

import { StatusDismissButton } from './status-dismiss-button'

interface StatusRowProps {
  children: ReactNode
  className?: string
  depth?: number
  dismiss?: { label: string; onDismiss: () => void }
  expanded?: boolean
  /** Leading glyph slot (spinner / status dot / selection circle). */
  leading?: ReactNode
  /** Makes the whole row activatable (adds `cursor-pointer` + keyboard a11y).
   *  Receives the originating event so consumers can branch on modifier keys
   *  (e.g. ⌘/Ctrl-click). Trailing-slot buttons should `stopPropagation` so
   *  they don't also fire it. */
  onActivate?: (event: KeyboardEvent | MouseEvent) => void
  /** Right-aligned actions. Revealed on row hover/focus unless `trailingVisible`. */
  trailing?: ReactNode
  trailingVisible?: boolean
  /** Draw the actions cluster before the content instead of after it, and keep it
   *  visible (dimmed) while the row is at rest. For rows whose controls act on the
   *  content itself — a queued prompt you can still edit — trailing hover-only
   *  controls sit past the end of a long line. Background and subagent rows keep
   *  the trailing reveal. */
  actionsFirst?: boolean
  /** Forwarded to the row's root — lets a wrapper (e.g. a context-menu trigger
   *  using `asChild`) attach `ref` / `onContextMenu` to the real DOM node. */
  ref?: Ref<HTMLDivElement>
  onContextMenu?: (event: MouseEvent) => void
}

/**
 * Shared row chrome for everything in the composer status stack — status items
 * (subagents, background) and queued prompts. Icons and dismiss controls align
 * with the first text line; section nesting and row padding share one CSS grid.
 */
export function StatusRow({
  actionsFirst = false,
  children,
  className,
  depth = 0,
  dismiss,
  expanded,
  leading,
  onActivate,
  onContextMenu,
  ref,
  trailing,
  trailingVisible = false
}: StatusRowProps) {
  const actions = trailing && (
    <div
      className={cn(
        'status-row-actions flex shrink-0 items-center gap-0.5',
        // A leading cluster advertises itself at rest; a trailing one waits for
        // the row's hover, which is right for background and subagent rows.
        !trailingVisible &&
          (actionsFirst
            ? 'opacity-60 group-hover/status-row:opacity-100 group-focus-within/status-row:opacity-100'
            : 'opacity-0 group-hover/status-row:opacity-100 group-focus-within/status-row:opacity-100')
      )}
    >
      {trailing}
    </div>
  )

  return (
    <div
      aria-expanded={expanded}
      className={cn(
        'group/status-row status-row flex min-h-6 items-center gap-2 rounded-md px-1.5 py-1',
        // row-hover bundles cursor:pointer — only when the row actually activates.
        onActivate ? 'row-hover' : 'hover:bg-(--ui-row-hover-background)',
        className
      )}
      data-slot="status-row"
      data-status-actions={actionsFirst ? 'leading' : undefined}
      data-status-icon={leading !== undefined ? '' : undefined}
      onClick={onActivate}
      onContextMenu={onContextMenu}
      onKeyDown={
        onActivate
          ? event => {
              if (event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ')) {
                event.preventDefault()
                onActivate(event)
              }
            }
          : undefined
      }
      ref={ref}
      role={onActivate ? 'button' : undefined}
      style={{ '--status-row-depth': depth } as CSSProperties}
      tabIndex={onActivate ? 0 : undefined}
    >
      {dismiss && (
        <span className="status-row-dismiss">
          <StatusDismissButton label={dismiss.label} onDismiss={dismiss.onDismiss} />
        </span>
      )}
      {leading !== undefined && (
        <span className="status-row-icon flex size-3.5 shrink-0 items-center justify-center">{leading}</span>
      )}
      {actionsFirst && actions}
      <div className="status-row-content flex min-w-0 flex-1 items-center gap-2">{children}</div>
      {!actionsFirst && actions}
    </div>
  )
}
