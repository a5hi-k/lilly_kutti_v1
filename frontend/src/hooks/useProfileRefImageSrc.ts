import { useEffect, useState } from 'react'

/**
 * Profile image with revision in the URL (mtime_ns) so replacing the file always loads fresh bytes.
 * Refetches revision when the window regains focus (e.g. after editing ref.jpg).
 */
export function useProfileRefImageSrc(): string | null {
  const [src, setSrc] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = () => {
      fetch('/api/profile-ref/revision', { cache: 'no-store' })
        .then((res) => res.json())
        .then((d: { revision: string }) => {
          if (!cancelled) {
            setSrc(`/api/profile-ref/image?r=${encodeURIComponent(d.revision)}`)
          }
        })
        .catch(() => {
          if (!cancelled) setSrc('/api/profile-ref/image?r=0')
        })
    }
    load()
    window.addEventListener('focus', load)
    return () => {
      cancelled = true
      window.removeEventListener('focus', load)
    }
  }, [])

  return src
}
