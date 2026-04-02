import { useRef } from 'react'
import type { AudioUploadResponse } from './types'

export function VoiceButton({
  isUploading,
  isRecording,
  setIsRecording,
  setIsUploading,
  onTranscription,
  onReply,
  onErrorMessage,
  onVideoCallStart,
  onLipsyncVideo,
  sessionId,
  malMode,
  videoCallSimulation,
}: {
  isUploading: boolean
  isRecording: boolean
  setIsRecording: (v: boolean) => void
  setIsUploading: (v: boolean) => void
  onTranscription: (text: string) => void
  onReply: (text: string, imageUrl?: string) => void
  onErrorMessage: (text: string) => void
  /** Fires when backend detects video-call intent (matches typed chat WebSocket behavior) */
  onVideoCallStart?: () => void
  /** When live + video call: backend runs lip-sync; URL is played in the call panel */
  onLipsyncVideo?: (url: string) => void
  sessionId: string
  malMode: boolean
  videoCallSimulation?: boolean
}) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const playBase64Audio = (base64: string) => {
    try {
      const audio = new Audio(`data:audio/wav;base64,${base64}`)
      audio.play().catch((err) => {
        console.warn('Audio playback failed:', err)
      })
    } catch (err) {
      console.warn('Could not create audio from base64:', err)
    }
  }

  return (
    <button
      className="tap-speak-btn"
      disabled={isUploading}
      onClick={async () => {
        if (isUploading) return

        if (isRecording) {
          const recorder = mediaRecorderRef.current
          if (!recorder || recorder.state !== 'recording') return
          recorder.stop()
          setIsRecording(false)
          return
        }

        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true },
          })
          const mimeType = MediaRecorder.isTypeSupported('audio/webm')
            ? 'audio/webm'
            : 'audio/mp4'
          const recorder = new MediaRecorder(stream)
          mediaRecorderRef.current = recorder
          chunksRef.current = []

          recorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunksRef.current.push(e.data)
          }

          recorder.onstop = async () => {
            stream.getTracks().forEach((t) => t.stop())
            if (chunksRef.current.length === 0) return
            setIsUploading(true)
            try {
              const blob = new Blob(chunksRef.current, { type: mimeType })
              const form = new FormData()
              form.append(
                'file',
                new File([blob], 'recording.webm', { type: mimeType }),
              )
              form.append('session_id', sessionId)
              form.append('mal_mode', malMode ? 'true' : 'false')
              if (malMode && videoCallSimulation) {
                form.append('video_call_simulation', 'true')
              }

              const res = await fetch('/api/audio/upload', {
                method: 'POST',
                body: form,
              })
              const text = await res.text()
              let data: AudioUploadResponse
              try {
                data = JSON.parse(text)
              } catch {
                throw new Error(res.ok ? 'Invalid response' : text || res.statusText)
              }

              if (!res.ok) {
                const detail = data?.detail ?? text ?? 'Request failed'
                throw new Error(String(detail))
              }

              const transcription = String(data.transcription ?? '')
              const reply = String(data.reply ?? '')
              const sharePhotoUrl =
                data.ui_event?.type === 'share_photo'
                  ? String(data.ui_event.image_url ?? '')
                  : ''

              if (transcription) onTranscription(transcription)
              if (reply || sharePhotoUrl) onReply(reply, sharePhotoUrl || undefined)

              const lipsyncUrl = data.lipsync_video_url
                ? String(data.lipsync_video_url)
                : ''
              const videoSim = Boolean(videoCallSimulation)
              const videoCallFromResponse =
                data.ui_event?.type === 'video_call_start'

              if (videoCallFromResponse) {
                onVideoCallStart?.()
              }

              if (lipsyncUrl) {
                onLipsyncVideo?.(lipsyncUrl)
              } else if (
                malMode &&
                data.mal_audio &&
                !videoSim &&
                !videoCallFromResponse
              ) {
                playBase64Audio(data.mal_audio)
              }

              if (!transcription && !reply && !sharePhotoUrl) {
                onErrorMessage("I couldn't catch that—try speaking again.")
              }
            } catch (err) {
              const msg =
                err instanceof Error ? err.message : 'Could not process your voice.'
              onErrorMessage(msg)
            } finally {
              setIsUploading(false)
              chunksRef.current = []
            }
          }

          recorder.start()
          setIsRecording(true)
        } catch {
          onErrorMessage(
            'Microphone access denied. Please allow microphone and try again.',
          )
        }
      }}
    >
      <span className={`btn-core ${isRecording ? 'recording' : ''}`}>
        <span className="wave-bar" />
        <span className="wave-bar" />
        <span className="wave-bar" />
      </span>
      <span className="btn-label">
        {isUploading
          ? 'Sending…'
          : isRecording
            ? 'Stop'
            : malMode
              ? 'Tap & Speak (Live)'
              : 'Tap & Speak'}
      </span>
    </button>
  )
}
