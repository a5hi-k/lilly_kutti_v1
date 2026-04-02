import { useEffect, useRef, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import './App.css'

import { ChatMessages } from './components/ChatMessages'
import { StatusPill } from './components/StatusPill'
import type { ChatMessage, SocketAssistantPayload } from './components/types'
import { IncomingVideoCallOverlay } from './components/IncomingVideoCallOverlay'
import { ProfileImageLightbox } from './components/ProfileImageLightbox'
import { VideoCallPanel } from './components/VideoCallPanel'
import { VoiceButton } from './components/VoiceButton'
import { useProfileRefImageSrc } from './hooks/useProfileRefImageSrc'

function getOrCreateSessionId() {
  const key = 'gf_lilly_session_id'
  const legacyKey = 'gf_juliet_session_id'
  const existing = window.localStorage.getItem(key) ?? window.localStorage.getItem(legacyKey)
  if (existing) {
    window.localStorage.setItem(key, existing)
    return existing
  }
  const fresh = crypto.randomUUID()
  window.localStorage.setItem(key, fresh)
  return fresh
}

function fileToBase64Payload(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const r = typeof reader.result === 'string' ? reader.result : ''
      const i = r.indexOf(',')
      resolve(i >= 0 ? r.slice(i + 1) : r)
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

function appendAssistantBubbles(
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>,
  opts: { content: string; imageUrl?: string },
) {
  const text = String(opts.content ?? '').trim()
  const img = String(opts.imageUrl ?? '').trim()
  if (!text && !img) return
  setMessages((prev) => {
    const next = [...prev]
    if (text) {
      next.push({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: text,
      })
    }
    if (img) {
      next.push({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
        imageUrl: img,
      })
    }
    return next
  })
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [connected, setConnected] = useState(false)
  const [malMode, setMalMode] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const paneRef = useRef<HTMLDivElement | null>(null)
  const sessionIdRef = useRef<string>(getOrCreateSessionId())
  const [isRecording, setIsRecording] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [videoCallRinging, setVideoCallRinging] = useState(false)
  const [isVideoCallOpen, setIsVideoCallOpen] = useState(false)
  const [lipsyncVideoUrl, setLipsyncVideoUrl] = useState<string | null>(null)
  const [avatarFiles, setAvatarFiles] = useState<string[]>([])
  const [headerAvatarFailed, setHeaderAvatarFailed] = useState(false)
  const [profileLightboxOpen, setProfileLightboxOpen] = useState(false)
  const [pendingImage, setPendingImage] = useState<{ file: File; preview: string } | null>(
    null,
  )
  const [workerMode, setWorkerMode] = useState(false)
  const imageInputRef = useRef<HTMLInputElement | null>(null)
  const profileImageSrc = useProfileRefImageSrc()
  const headerImageOk = Boolean(profileImageSrc && !headerAvatarFailed)

  useEffect(() => {
    setHeaderAvatarFailed(false)
  }, [profileImageSrc])

  useEffect(() => {
    let ws: WebSocket
    let reconnectTimeout: ReturnType<typeof setTimeout>

    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = window.location.host
      ws = new WebSocket(`${protocol}//${host}/ws/chat`)
      socketRef.current = ws

      ws.onopen = () => {
        setConnected(true)
      }

      ws.onmessage = (event) => {
        const raw = String(event.data)
        let parsed: SocketAssistantPayload | null = null
        try {
          parsed = JSON.parse(raw) as SocketAssistantPayload
        } catch {
          parsed = null
        }

        if (parsed?.kind === 'assistant') {
          const content = String(parsed.content ?? '')
          const evtType = parsed.ui_event?.type

          let imageUrl: string | undefined
          if (evtType === 'share_photo') {
            imageUrl = String((parsed.ui_event as any)?.image_url ?? '') || undefined
          } else if (
            evtType === 'worker_mode_active' ||
            evtType === 'worker_mode_payment_qr'
          ) {
            imageUrl = String((parsed.ui_event as any)?.image_url ?? '') || undefined
          }

          appendAssistantBubbles(setMessages, { content, imageUrl })

          if (evtType === 'worker_mode_enter') {
            setWorkerMode(true)
          } else if (evtType === 'worker_mode_exit') {
            setWorkerMode(false)
          }

          if (evtType === 'video_call_start') {
            setVideoCallRinging(true)
          }
          return
        }

        setMessages((prev) => [
          ...prev,
          { id: crypto.randomUUID(), role: 'assistant', content: raw },
        ])
      }

      ws.onclose = () => {
        setConnected(false)
        socketRef.current = null
        reconnectTimeout = setTimeout(connect, 2000)
      }

      ws.onerror = () => {
        setConnected(false)
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimeout)
      ws?.close()
    }
  }, [])

  useEffect(() => {
    if (paneRef.current) {
      paneRef.current.scrollTop = paneRef.current.scrollHeight
    }
  }, [messages])

  useEffect(() => {
    if (!videoCallRinging && !isVideoCallOpen) return
    if (avatarFiles.length > 0) return

    ;(async () => {
      try {
        const res = await fetch('/api/videos/list')
        const data = (await res.json()) as { files?: string[] }
        const files = Array.isArray(data.files) ? data.files.filter(Boolean) : []
        setAvatarFiles(files)
      } catch {
        setAvatarFiles([])
      }
    })()
  }, [videoCallRinging, isVideoCallOpen, avatarFiles.length])

  const pickRandomAvatar = (exclude?: string) => {
    const options = avatarFiles.filter((f) => f && f !== exclude)
    if (options.length === 0) return avatarFiles[0] ?? ''
    return options[Math.floor(Math.random() * options.length)] ?? ''
  }

  const sendMessage = () => {
    void (async () => {
      const text = input.trim()
      if (
        (!text && !pendingImage) ||
        !socketRef.current ||
        socketRef.current.readyState !== WebSocket.OPEN
      ) {
        return
      }

      let imageBase64: string | undefined
      let mimeType: string | undefined
      if (pendingImage) {
        try {
          imageBase64 = await fileToBase64Payload(pendingImage.file)
          mimeType = pendingImage.file.type || 'image/jpeg'
        } catch {
          return
        }
      }

      const displayText = text || (pendingImage ? '(photo)' : '')
      const previewUrl = pendingImage?.preview
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'user',
          content: displayText,
          ...(previewUrl ? { imageUrl: previewUrl } : {}),
        },
      ])

      const payload: Record<string, string> = {
        session_id: sessionIdRef.current,
        role: 'user',
        content: text,
      }
      if (imageBase64) {
        payload.image_base64 = imageBase64
        payload.mime_type = mimeType || 'image/jpeg'
      }

      socketRef.current.send(JSON.stringify(payload))
      setInput('')
      if (pendingImage) {
        URL.revokeObjectURL(pendingImage.preview)
        setPendingImage(null)
      }
    })()
  }

  return (
    <div className="chat-app">
      <div className="chat-shell">
        <header className="app-header">
          <div className="app-title app-title--lilly">
            <button
              type="button"
              className="header-profile-bubble"
              onClick={() => setProfileLightboxOpen(true)}
              aria-label="View Lilly profile photo"
            >
              {!headerAvatarFailed && profileImageSrc ? (
                <img
                  key={profileImageSrc}
                  src={profileImageSrc}
                  alt=""
                  className="header-profile-img"
                  onError={() => setHeaderAvatarFailed(true)}
                />
              ) : null}
              {headerAvatarFailed || !profileImageSrc ? (
                <span className="header-profile-fallback" aria-hidden>
                  J
                </span>
              ) : null}
            </button>
            <span className="header-profile-name">Lilly</span>
          </div>
          <div className="header-controls">
            <StatusPill connected={connected} />
            <button
              className={`mal-toggle ${malMode ? 'active' : ''}`}
              onClick={() => setMalMode((v) => !v)}
              title={
                malMode
                  ? 'Live: Malayalam voice replies ON'
                  : 'Live: Malayalam voice replies OFF'
              }
            >
              <span className="mal-indicator" />
              <span>Live</span>
            </button>
          </div>
        </header>

        {workerMode && (
          <div className="worker-mode-banner">
            <span className="worker-mode-dot" />
            <span>WORK MODE</span>
            <span className="worker-mode-hint">Type "end" to exit</span>
          </div>
        )}

        <ProfileImageLightbox
          open={profileLightboxOpen}
          onClose={() => setProfileLightboxOpen(false)}
          imageSrc={profileImageSrc}
          imageOk={headerImageOk}
          displayName="Lilly"
        />

        <main className="chat-main">
          {videoCallRinging && (
            <IncomingVideoCallOverlay
              callerName="Lilly"
              profileImageSrc={profileImageSrc}
              onAccept={() => {
                setVideoCallRinging(false)
                setIsVideoCallOpen(true)
              }}
              onDecline={() => {
                setVideoCallRinging(false)
                setLipsyncVideoUrl(null)
              }}
            />
          )}
          {isVideoCallOpen && (
            <VideoCallPanel
              avatarFiles={avatarFiles}
              pickRandom={pickRandomAvatar}
              onClose={() => {
                setLipsyncVideoUrl(null)
                setIsVideoCallOpen(false)
              }}
              lipsyncVideoUrl={lipsyncVideoUrl}
              onLipsyncEnded={() => setLipsyncVideoUrl(null)}
            />
          )}
          <ChatMessages messages={messages} paneRef={paneRef} />
        </main>

        <div className="input-region">
          <div className="floating-cta">
            <VoiceButton
              isUploading={isUploading}
              isRecording={isRecording}
              setIsRecording={setIsRecording}
              setIsUploading={setIsUploading}
              sessionId={sessionIdRef.current}
              malMode={malMode}
              videoCallSimulation={isVideoCallOpen || videoCallRinging}
              onVideoCallStart={() => setVideoCallRinging(true)}
              onLipsyncVideo={(url) => setLipsyncVideoUrl(url)}
              onTranscription={(transcription) =>
                setMessages((prev) => [
                  ...prev,
                  { id: crypto.randomUUID(), role: 'user', content: transcription },
                ])
              }
              onReply={(reply, imageUrl) =>
                appendAssistantBubbles(setMessages, { content: reply, imageUrl })
              }
              onErrorMessage={(msg) =>
                setMessages((prev) => [
                  ...prev,
                  { id: crypto.randomUUID(), role: 'assistant', content: msg },
                ])
              }
            />
          </div>

          <div className="type-area">
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              className="visually-hidden"
              aria-label="Attach image"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (!f || !f.type.startsWith('image/')) return
                setPendingImage((prev) => {
                  if (prev) URL.revokeObjectURL(prev.preview)
                  return { file: f, preview: URL.createObjectURL(f) }
                })
                e.target.value = ''
              }}
            />
            <button
              type="button"
              className="attach-image-btn"
              title="Attach image"
              disabled={!connected}
              onClick={() => imageInputRef.current?.click()}
            >
              Image
            </button>
            <input
              type="text"
              placeholder={
                !connected
                  ? 'Connecting to server...'
                  : workerMode
                    ? 'Work mode — describe your project...'
                    : 'Type your message...'
              }
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  sendMessage()
                }
              }}
              disabled={!connected}
            />
            <button
              className="send-btn"
              onClick={sendMessage}
              disabled={!connected || (!input.trim() && !pendingImage)}
            >
              Send
            </button>
          </div>
          {pendingImage ? (
            <div className="pending-image-strip">
              <img src={pendingImage.preview} alt="" className="pending-thumb" />
              <span>Ready to send</span>
              <button
                type="button"
                className="pending-remove"
                onClick={() => {
                  URL.revokeObjectURL(pendingImage.preview)
                  setPendingImage(null)
                }}
              >
                Remove
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

export default App
