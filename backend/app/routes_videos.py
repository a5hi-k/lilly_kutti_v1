from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException, status

from app.services.lipsync_service import SAFE_LIPSYNC_FILENAME, safe_delete_lipsync_file


router = APIRouter(prefix="/videos", tags=["videos"])


def _videos_dir() -> Path:
    # Keep avatar mp4 files here: <repo>/backend/videos
    return Path(__file__).resolve().parents[1] / "videos"


@router.get("/list")
async def list_videos() -> Dict[str, List[str]]:
    videos_dir = _videos_dir()
    if not videos_dir.exists():
        return {"files": []}

    files = sorted(
        [p.name for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]
    )
    return {"files": files}


@router.delete("/lipsync/{filename}")
async def delete_lipsync_video(filename: str) -> Dict[str, bool]:
    """Remove a generated lip-sync clip from tmp after playback (or on cancel)."""
    if not SAFE_LIPSYNC_FILENAME.match(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )
    if not safe_delete_lipsync_file(filename):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found.",
        )
    return {"ok": True}

