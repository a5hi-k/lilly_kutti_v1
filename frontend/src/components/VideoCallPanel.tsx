import { useEffect, useRef, useState, type SyntheticEvent } from 'react'

/** Start swap prep this far before clip end so the veil overlaps real tail footage. */
const LEAD_SECONDS = 0.7
const DOUBLE_TAP_MS = 420
const DRAG_EDGE_KEEP_PX = 56

function clampDragOffset(
  ox: number,
  oy: number,
  panel: HTMLElement,
): { x: number; y: number } {
  const main = panel.offsetParent as HTMLElement | null
  if (!main) return { x: ox, y: oy }
  const pl = panel.offsetLeft
  const pt = panel.offsetTop
  const pw = panel.offsetWidth
  const ph = panel.offsetHeight
  const W = main.clientWidth
  const H = main.clientHeight
  const k = DRAG_EDGE_KEEP_PX
  const minX = k - pl - pw
  const maxX = W - pl - k
  const minY = k - pt - ph
  const maxY = H - pt - k
  const clamp1 = (v: number, lo: number, hi: number) =>
    lo <= hi ? Math.min(hi, Math.max(lo, v)) : v
  return {
    x: clamp1(ox, minX, maxX),
    y: clamp1(oy, minY, maxY),
  }
}

function waitForFirstFrame(video: HTMLVideoElement): Promise<void> {
  return new Promise((resolve) => {
    let settled = false
    const finish = () => {
      if (settled) return
      settled = true
      resolve()
    }
    const v = video as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: () => void) => number
    }
    if (typeof v.requestVideoFrameCallback === 'function') {
      v.requestVideoFrameCallback(finish)
      return
    }
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      requestAnimationFrame(finish)
    } else {
      video.addEventListener('loadeddata', finish, { once: true })
    }
  })
}

function toUrl(file: string) {
  return `/avatar-videos/${encodeURIComponent(file)}`
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * Veil stays on the outgoing clip (preSwap), then through the cut, then on the
 * incoming clip’s start (postSwap) so both ends feel covered.
 */
function glitchTransitionMs(): { preSwap: number; postSwap: number } {
  const preSwap = 550 + Math.random() * 500
  const postSwap = 550 + Math.random() * 500
  return { preSwap, postSwap }
}

export function VideoCallPanel({
  avatarFiles,
  pickRandom,
  onClose,
  lipsyncVideoUrl,
  onLipsyncEnded,
}: {
  avatarFiles: string[]
  pickRandom: (exclude?: string) => string
  onClose: () => void
  lipsyncVideoUrl: string | null
  onLipsyncEnded: () => void
}) {
  const [showNext, setShowNext] = useState(false)
  const avatarARef = useRef<HTMLVideoElement>(null)
  const avatarBRef = useRef<HTMLVideoElement>(null)
  const lipsyncRef = useRef<HTMLVideoElement>(null)
  const camRef = useRef<HTMLVideoElement>(null)
  const camStreamRef = useRef<MediaStream | null>(null)

  const activeFileRef = useRef('')
  const nextFileRef = useRef('')
  const swapScheduledRef = useRef(false)
  const showNextRef = useRef(showNext)
  const pickRandomRef = useRef(pickRandom)
  const lipsyncUrlRef = useRef(lipsyncVideoUrl)
  const onLipsyncEndedRef = useRef(onLipsyncEnded)
  const initRef = useRef(false)

  const paneRef = useRef<HTMLElement | null>(null)
  const stageRef = useRef<HTMLDivElement | null>(null)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const dragOffsetRef = useRef(dragOffset)
  const [dragging, setDragging] = useState(false)
  const pointerDownTimesRef = useRef<number[]>([])
  const activeDragRef = useRef<{
    pointerId: number
    startClientX: number
    startClientY: number
    originX: number
    originY: number
  } | null>(null)

  pickRandomRef.current = pickRandom
  lipsyncUrlRef.current = lipsyncVideoUrl
  onLipsyncEndedRef.current = onLipsyncEnded
  showNextRef.current = showNext
  dragOffsetRef.current = dragOffset

  const stopWebcam = () => {
    camStreamRef.current?.getTracks().forEach((t) => t.stop())
    camStreamRef.current = null
    if (camRef.current) camRef.current.srcObject = null
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        camStreamRef.current = stream
        if (camRef.current) {
          camRef.current.srcObject = stream
          await camRef.current.play().catch(() => {})
        }
      } catch {
        // ignore
      }
    })()
    return () => {
      cancelled = true
      stopWebcam()
    }
  }, [])

  const primeHidden = (file: string) => {
    const hidden = showNextRef.current ? avatarARef.current : avatarBRef.current
    if (!hidden || !file) return
    hidden.src = toUrl(file)
    hidden.load()
  }

  useEffect(() => {
    if (avatarFiles.length === 0 || initRef.current) return
    initRef.current = true

    const first = pickRandomRef.current()
    activeFileRef.current = first
    const a = avatarARef.current
    if (a) {
      a.src = toUrl(first)
      a.load()
      a.play().catch(() => {})
    }

    const n = pickRandomRef.current(first)
    nextFileRef.current = n
    if (n) primeHidden(n)
  }, [avatarFiles.length])

  const primeNextAfterSwap = (newActiveFile: string) => {
    const n = pickRandomRef.current(newActiveFile)
    if (!n) return
    nextFileRef.current = n
    primeHidden(n)
  }

  const getFrontBack = () => {
    const a = avatarARef.current
    const b = avatarBRef.current
    if (!a || !b) return { front: null, back: null }
    return showNextRef.current
      ? { front: b, back: a }
      : { front: a, back: b }
  }

  const performSwapCore = async (): Promise<boolean> => {
    const { front, back } = getFrontBack()
    if (!front || !back || !nextFileRef.current) return false

    const incoming = back
    const outgoing = front
    const file = nextFileRef.current

    try {
      incoming.currentTime = 0
      await incoming.play()
      await waitForFirstFrame(incoming)
    } catch {
      return false
    }

    outgoing.pause()
    activeFileRef.current = file
    const nextTop = !showNextRef.current
    showNextRef.current = nextTop
    const a = avatarARef.current
    const b = avatarBRef.current
    if (a && b) {
      a.style.zIndex = nextTop ? '1' : '2'
      b.style.zIndex = nextTop ? '2' : '1'
    }
    setShowNext(nextTop)

    primeNextAfterSwap(file)
    return true
  }

  const performSwapWithGlitch = async () => {
    const { front, back } = getFrontBack()
    if (!front || !back || !nextFileRef.current) {
      swapScheduledRef.current = false
      return
    }

    const stage = stageRef.current
    const { preSwap, postSwap } = glitchTransitionMs()
    stage?.classList.add('video-stage--glitch')

    await sleep(preSwap)
    const ok = await performSwapCore()
    if (!ok) {
      stage?.classList.remove('video-stage--glitch')
      swapScheduledRef.current = false
      return
    }

    await sleep(postSwap)
    stage?.classList.remove('video-stage--glitch')
    swapScheduledRef.current = false
  }

  useEffect(() => {
    if (!lipsyncVideoUrl) return
    stageRef.current?.classList.remove('video-stage--glitch')
    avatarARef.current?.pause()
    avatarBRef.current?.pause()
    swapScheduledRef.current = false

    const el = lipsyncRef.current
    if (!el) return
    el.src = lipsyncVideoUrl
    el.muted = false
    el.load()
    void el.play().catch(() => {})
  }, [lipsyncVideoUrl])

  const handleLipsyncEnded = () => {
    onLipsyncEndedRef.current()
    const { front } = getFrontBack()
    void front?.play().catch(() => {})
  }

  const handleAvatarEnded = (e: SyntheticEvent<HTMLVideoElement>) => {
    const { front } = getFrontBack()
    if (e.currentTarget !== front) return
    if (swapScheduledRef.current) return
    swapScheduledRef.current = true
    void performSwapWithGlitch()
  }

  useEffect(() => {
    let raf = 0
    const tick = () => {
      if (lipsyncUrlRef.current) {
        raf = requestAnimationFrame(tick)
        return
      }
      if (swapScheduledRef.current) {
        raf = requestAnimationFrame(tick)
        return
      }

      const { front, back } = getFrontBack()
      if (!front || !back || !nextFileRef.current) {
        raf = requestAnimationFrame(tick)
        return
      }

      const dur = front.duration
      const t = front.currentTime
      if (!Number.isFinite(dur) || dur <= 0) {
        raf = requestAnimationFrame(tick)
        return
      }

      const lead = Math.min(LEAD_SECONDS, Math.max(0.1, dur * 0.12))
      if (dur - t <= lead && back.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
        swapScheduledRef.current = true
        void performSwapWithGlitch()
      }

      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  const endDrag = (el: HTMLElement, pointerId: number) => {
    if (activeDragRef.current?.pointerId !== pointerId) return
    activeDragRef.current = null
    setDragging(false)
    try {
      el.releasePointerCapture(pointerId)
    } catch {
      /* already released */
    }
  }

  const handlePanePointerDown = (e: React.PointerEvent<HTMLElement>) => {
    if (e.button !== 0) return
    const target = e.target as HTMLElement
    if (target.closest('button')) return

    const t = performance.now()
    const recent = pointerDownTimesRef.current.filter((t0) => t - t0 < DOUBLE_TAP_MS)
    recent.push(t)
    pointerDownTimesRef.current = recent

    if (recent.length < 2) return

    pointerDownTimesRef.current = []
    e.preventDefault()

    activeDragRef.current = {
      pointerId: e.pointerId,
      startClientX: e.clientX,
      startClientY: e.clientY,
      originX: dragOffsetRef.current.x,
      originY: dragOffsetRef.current.y,
    }
    setDragging(true)
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  const handlePanePointerMove = (e: React.PointerEvent<HTMLElement>) => {
    const dr = activeDragRef.current
    if (!dr || e.pointerId !== dr.pointerId) return
    const panel = paneRef.current
    if (!panel) return
    const dx = e.clientX - dr.startClientX
    const dy = e.clientY - dr.startClientY
    setDragOffset(clampDragOffset(dr.originX + dx, dr.originY + dy, panel))
  }

  const handlePanePointerUp = (e: React.PointerEvent<HTMLElement>) => {
    endDrag(e.currentTarget, e.pointerId)
  }

  const handlePaneLostCapture = (e: React.PointerEvent<HTMLElement>) => {
    if (activeDragRef.current?.pointerId === e.pointerId) {
      activeDragRef.current = null
      setDragging(false)
    }
  }

  return (
    <section
      ref={paneRef}
      className={`video-call-pane${dragging ? ' video-call-pane--dragging' : ''}`}
      style={
        dragOffset.x !== 0 || dragOffset.y !== 0
          ? { transform: `translate(${dragOffset.x}px, ${dragOffset.y}px)` }
          : undefined
      }
      aria-label="Video call simulation"
      title="Double-click, hold, and drag to move the call window"
      onPointerDown={handlePanePointerDown}
      onPointerMove={handlePanePointerMove}
      onPointerUp={handlePanePointerUp}
      onPointerCancel={handlePanePointerUp}
      onLostPointerCapture={handlePaneLostCapture}
    >
      <div className="video-call-header">
        <div className="video-call-title">
          <span className="video-dot" />
          <span>Video call</span>
        </div>
        <button
          className="video-call-end"
          onClick={() => {
            stopWebcam()
            onClose()
          }}
        >
          End
        </button>
      </div>

      <div ref={stageRef} className="video-stage">
        <video
          ref={avatarARef}
          className="avatar-layer"
          style={{ zIndex: showNext ? 1 : 2 }}
          playsInline
          muted
          preload="auto"
          onEnded={handleAvatarEnded}
        />
        <video
          ref={avatarBRef}
          className="avatar-layer"
          style={{ zIndex: showNext ? 2 : 1 }}
          playsInline
          muted
          preload="auto"
          onEnded={handleAvatarEnded}
        />

        {lipsyncVideoUrl ? (
          <video
            ref={lipsyncRef}
            className="lipsync-overlay"
            playsInline
            onEnded={handleLipsyncEnded}
          />
        ) : null}

        <video ref={camRef} className="webcam-pip" playsInline muted autoPlay />

        {avatarFiles.length === 0 && (
          <div className="video-call-empty">
            <p>
              No avatar videos found. Put MP4 files in{' '}
              <code>backend/videos</code>.
            </p>
          </div>
        )}
      </div>
    </section>
  )
}
