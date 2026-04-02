"""Profile reference image (header + incoming call). Bypasses stale browser cache via revision in URL."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.core.config import get_settings

router = APIRouter(prefix="/profile-ref", tags=["profile-ref"])

_REF_DIR = Path(__file__).resolve().parents[1] / "ReferImage"
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*\.(jpe?g|png|webp)$")


def _image_path() -> Path:
    name = (get_settings().profile_ref_image_filename or "ref.jpg").strip()
    if not _SAFE_NAME.match(name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid profile image filename.",
        )
    return _REF_DIR / name


@router.get("/revision")
async def profile_ref_revision() -> dict:
    """Use `revision` as a query param on `/profile-ref/image` so swaps invalidate cache."""
    p = _image_path()
    if not p.is_file():
        return {"revision": "0", "missing": True}
    st = p.stat()
    return {"revision": str(int(st.st_mtime_ns)), "missing": False}


@router.get("/image")
async def profile_ref_image(
    r: str | None = Query(None, description="Revision from /profile-ref/revision; busts browser cache."),
) -> FileResponse:
    """Query `r` is not used for lookup; it only makes each on-disk version a distinct URL."""
    p = _image_path()
    if not p.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile image not found.")
    st = p.stat()
    suffix = p.suffix.lower()
    media = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png" if suffix == ".png" else "image/webp"
    return FileResponse(
        path=p,
        media_type=media,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "ETag": f'W/"{st.st_mtime_ns}-{st.st_size}"',
        },
    )
