"""Persist user-sent chat images under backend/session_uploads/."""

from __future__ import annotations

import base64
import re
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_UPLOAD_DIR = _BACKEND_DIR / "session_uploads"

_MAX_BYTES = 15 * 1024 * 1024


def session_uploads_dir() -> Path:
    return _UPLOAD_DIR


def _safe_session_fragment(session_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", session_id)[:80]
    return s or "session"


def save_user_image_base64(session_id: str, b64: str, mime_type: str) -> Path:
    """Decode base64, write to disk, return absolute path (PNG/JPEG/WebP for IG)."""
    s = str(b64).strip()
    if "," in s and s.lower().startswith("data:"):
        header, s = s.split(",", 1)
        hl = header.lower()
        if "png" in hl:
            mime_type = "image/png"
        elif "webp" in hl:
            mime_type = "image/webp"
        elif "gif" in hl:
            mime_type = "image/gif"
        else:
            mime_type = "image/jpeg"
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception:
        raw = base64.b64decode(s, validate=False)
    if len(raw) > _MAX_BYTES:
        raise ValueError("Image is too large (max 15 MB).")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{_safe_session_fragment(session_id)}_{uuid.uuid4().hex}"
    mt = (mime_type or "").lower()

    if "gif" in mt:
        im = Image.open(BytesIO(raw))
        im = im.convert("RGBA")
        path = _UPLOAD_DIR / f"{base}.png"
        im.save(path, format="PNG")
        return path

    if "png" in mt:
        ext = ".png"
    elif "webp" in mt:
        ext = ".webp"
    else:
        ext = ".jpg"

    path = _UPLOAD_DIR / f"{base}{ext}"
    path.write_bytes(raw)
    return path
