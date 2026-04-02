import { useCallback, useEffect, useRef, useState } from 'react'

const KNOB_PX = 54
/** Drag at least this fraction of usable half-width toward an edge to commit */
const COMMIT_FRACTION = 0.38

export function IncomingVideoCallOverlay({
  callerName = 'Lilly',
  profileImageSrc,
  onAccept,
  onDecline,
}: {
  callerName?: string
  profileImageSrc: string | null
  onAccept: () => void
  onDecline: () => void
}) {
  const railRef = useRef<HTMLDivElement>(null)
  const [knobPx, setKnobPx] = useState(0)
  const [dragging, setDragging] = useState(false)
  const [locked, setLocked] = useState(false)
  const [imageFailed, setImageFailed] = useState(false)
  const knobPxRef = useRef(0)
  const dragRef = useRef<{
    pointerId: number
    startClientX: number
    startKnob: number
    maxAbs: number
  } | null>(null)

  const measureMaxAbs = useCallback(() => {
    const rail = railRef.current
    if (!rail) return 0
    const w = rail.clientWidth
    return Math.max(0, (w - KNOB_PX) / 2 - 12)
  }, [])

  useEffect(() => {
    const rail = railRef.current
    if (!rail) return
    const ro = new ResizeObserver(() => {
      const m = measureMaxAbs()
      setKnobPx((k) => Math.min(m, Math.max(-m, k)))
    })
    ro.observe(rail)
    return () => ro.disconnect()
  }, [measureMaxAbs])

  useEffect(() => {
    knobPxRef.current = knobPx
  }, [knobPx])

  useEffect(() => {
    setImageFailed(false)
  }, [profileImageSrc])

  const finishDrag = useCallback(
    (finalKnob: number) => {
      const maxAbs = measureMaxAbs()
      setKnobPx(finalKnob)
      if (maxAbs <= 0) {
        setKnobPx(0)
        return
      }
      const threshold = maxAbs * COMMIT_FRACTION
      if (finalKnob >= threshold) {
        setLocked(true)
        setKnobPx(maxAbs)
        window.setTimeout(onAccept, 160)
        return
      }
      if (finalKnob <= -threshold) {
        setLocked(true)
        setKnobPx(-maxAbs)
        window.setTimeout(onDecline, 160)
        return
      }
      setKnobPx(0)
    },
    [measureMaxAbs, onAccept, onDecline],
  )

  const onKnobPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    e.preventDefault()
    const maxAbs = measureMaxAbs()
    dragRef.current = {
      pointerId: e.pointerId,
      startClientX: e.clientX,
      startKnob: knobPxRef.current,
      maxAbs,
    }
    setDragging(true)
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }

  const onKnobPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current
    if (!d || e.pointerId !== d.pointerId) return
    const dx = e.clientX - d.startClientX
    const next = Math.min(d.maxAbs, Math.max(-d.maxAbs, d.startKnob + dx))
    setKnobPx(next)
  }

  const onKnobPointerUp = (e: React.PointerEvent) => {
    const d = dragRef.current
    if (!d || e.pointerId !== d.pointerId) return
    const dx = e.clientX - d.startClientX
    const finalKnob = Math.min(d.maxAbs, Math.max(-d.maxAbs, d.startKnob + dx))
    dragRef.current = null
    setDragging(false)
    try {
      ;(e.target as HTMLElement).releasePointerCapture(e.pointerId)
    } catch {
      /* */
    }
    finishDrag(finalKnob)
  }

  const maxAbs = measureMaxAbs()
  const acceptHint = maxAbs > 0 && knobPx > maxAbs * 0.2
  const declineHint = maxAbs > 0 && knobPx < -maxAbs * 0.2

  return (
    <div
      className={`incoming-call-overlay${locked ? ' incoming-call-overlay--locked' : ''}`}
      role="dialog"
      aria-modal="true"
      aria-labelledby="incoming-call-title"
    >
      <p className="sr-only">
        Incoming video call from {callerName}. Drag the glass handle to the right to accept or to the
        left to decline.
      </p>
      <div className="incoming-call-backdrop" aria-hidden />
      <div className="incoming-call-card">
        <div className="incoming-call-calling" aria-hidden>
          <span className="incoming-call-calling-dot" />
          <span className="incoming-call-calling-dot" />
          <span className="incoming-call-calling-dot" />
        </div>

        <div className="incoming-call-avatar-shell">
          <div className="incoming-call-rings" aria-hidden>
            <span className="incoming-call-ring incoming-call-ring--1" />
            <span className="incoming-call-ring incoming-call-ring--2" />
            <span className="incoming-call-ring incoming-call-ring--3" />
          </div>
          <div className="incoming-call-avatar">
            {!imageFailed && profileImageSrc ? (
              <img
                key={profileImageSrc}
                src={profileImageSrc}
                alt=""
                className="incoming-call-avatar-img"
                onError={() => setImageFailed(true)}
              />
            ) : null}
            {(imageFailed || !profileImageSrc) ? (
              <span className="incoming-call-avatar-fallback" aria-hidden>
                {callerName.slice(0, 1).toUpperCase()}
              </span>
            ) : null}
          </div>
        </div>

        <p className="incoming-call-label">Incoming video call</p>
        <h2 id="incoming-call-title" className="incoming-call-name">
          {callerName}
        </h2>
        <p className="incoming-call-hint">
          Slide the glass handle — right to answer, left to decline
        </p>

        <div
          className={`incoming-call-track${acceptHint ? ' incoming-call-track--accept' : ''}${declineHint ? ' incoming-call-track--decline' : ''}`}
        >
          <span className="incoming-call-track-label incoming-call-track-label--left">Decline</span>
          <span className="incoming-call-track-label incoming-call-track-label--right">Accept</span>
          <div className="incoming-call-track-rail" ref={railRef}>
            <button
              type="button"
              className="incoming-call-knob"
              style={{
                transform: `translate(calc(-50% + ${knobPx}px), -50%)`,
                transition: dragging ? 'none' : 'transform 0.22s cubic-bezier(0.34, 1.2, 0.64, 1)',
              }}
              aria-label="Drag right to accept call, left to decline"
              onPointerDown={onKnobPointerDown}
              onPointerMove={onKnobPointerMove}
              onPointerUp={onKnobPointerUp}
              onPointerCancel={onKnobPointerUp}
            >
              <span className="incoming-call-knob-grip" aria-hidden />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
