import { useEffect } from 'react'
import { createPortal } from 'react-dom'

export function ProfileImageLightbox({
  open,
  onClose,
  imageSrc,
  imageOk,
  displayName,
}: {
  open: boolean
  onClose: () => void
  imageSrc: string | null
  imageOk: boolean
  displayName: string
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  const initial = displayName.slice(0, 1).toUpperCase()

  return createPortal(
    <div
      className="profile-lightbox"
      role="dialog"
      aria-modal="true"
      aria-labelledby="profile-lightbox-title"
    >
      <button
        type="button"
        className="profile-lightbox-backdrop"
        aria-label="Close profile photo"
        onClick={onClose}
      />
      <div className="profile-lightbox-panel">
        <button type="button" className="profile-lightbox-close" onClick={onClose} aria-label="Close">
          ×
        </button>
        <div className="profile-lightbox-visual">
          {imageOk && imageSrc ? (
            <img key={imageSrc} src={imageSrc} alt={displayName} className="profile-lightbox-img" />
          ) : (
            <span className="profile-lightbox-fallback" aria-hidden>
              {initial}
            </span>
          )}
        </div>
        <p id="profile-lightbox-title" className="profile-lightbox-title">
          {displayName}
        </p>
      </div>
    </div>,
    document.body,
  )
}
