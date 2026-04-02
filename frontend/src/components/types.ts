export type Role = 'user' | 'assistant' | 'system'

export type ChatMessage = {
  id: string
  role: Role
  content: string
  /** Assistant reference photo from /ref-assets/ when Lilly shares a picture */
  imageUrl?: string
}

export type SocketAssistantPayload = {
  kind: 'assistant'
  content: string
  ui_event?:
    | { type: 'video_call_start' }
    | { type: 'share_photo'; image_url?: string }
    | { type: 'instagram_posted' }
    | { type: 'worker_mode_enter' }
    | { type: 'worker_mode_exit' }
    | { type: 'worker_mode_active'; image_url?: string }
    | { type: 'worker_mode_payment_qr'; image_url?: string }
}

export type AudioUploadResponse = {
  session_id: string
  transcription?: string
  reply?: string
  mal_text?: string
  mal_audio?: string
  /** Served under /lipsync-tmp/ when live + video call + Gradio lip-sync succeeds */
  lipsync_video_url?: string
  detail?: string
  /** Same shape as WebSocket assistant payload (video call or shared / generated photo). */
  ui_event?:
    | { type: 'video_call_start' }
    | { type: 'share_photo'; image_url?: string }
    | { type: 'instagram_posted' }
    | { type: 'worker_mode_enter' }
    | { type: 'worker_mode_exit' }
    | { type: 'worker_mode_active'; image_url?: string }
    | { type: 'worker_mode_payment_qr'; image_url?: string }
}
