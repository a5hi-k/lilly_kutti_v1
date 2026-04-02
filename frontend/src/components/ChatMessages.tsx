import type { ChatMessage } from './types'

export function ChatMessages({
  messages,
  paneRef,
}: {
  messages: ChatMessage[]
  paneRef: React.RefObject<HTMLDivElement | null>
}) {
  return (
    <div className="messages-pane" ref={paneRef}>
      {messages.map((m) => {
        const hasText = Boolean(m.content?.trim())
        const hasImage = Boolean(m.imageUrl)
        const bubbleKind =
          hasImage && !hasText ? 'bubble bubble--media' : 'bubble bubble--text'
        return (
          <div
            key={m.id}
            className={`message ${m.role === 'user' ? 'message-user' : 'message-ai'}`}
          >
            <div className={bubbleKind}>
              {hasImage ? (
                <img
                  className={`chat-image${hasText ? ' chat-image--with-text' : ''}`}
                  src={m.imageUrl}
                  alt=""
                  loading="lazy"
                />
              ) : null}
              {hasText ? <p>{m.content}</p> : null}
            </div>
          </div>
        )
      })}
    </div>
  )
}

