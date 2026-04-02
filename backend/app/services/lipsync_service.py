"""Gradio lip-sync: drive avatar video with generated TTS audio."""

from __future__ import annotations

import base64
import logging
import random
import re
import shutil
import threading
import uuid
from pathlib import Path

from app.core.config import get_settings

try:
    from gradio_client import Client, handle_file
except ImportError:
    Client = None  # type: ignore[misc, assignment]
    handle_file = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

SAFE_LIPSYNC_FILENAME = re.compile(r"^[a-f0-9]{32}\.mp4$")

_gradio_client: Client | None = None  # type: ignore[type-arg]
_client_lock = threading.Lock()
_client_url: str = ""


def _get_gradio_client(url: str) -> Client | None:
    """Return a reusable Gradio Client, creating it once with a generous timeout."""
    global _gradio_client, _client_url
    if Client is None:
        return None
    with _client_lock:
        if _gradio_client is not None and _client_url == url:
            return _gradio_client
        settings = get_settings()
        logger.info("Creating Gradio Client for %s (timeout=%.0fs) …", url, settings.lipsync_timeout)
        import httpx as _httpx
        _gradio_client = Client(
            url,
            httpx_kwargs={"timeout": _httpx.Timeout(settings.lipsync_timeout)},
        )
        _client_url = url
        return _gradio_client


def lipsync_tmp_dir() -> Path:
    p = Path(__file__).resolve().parents[2] / "tmp" / "lipsync"
    p.mkdir(parents=True, exist_ok=True)
    return p


def videos_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "videos"


def pick_random_mp4() -> Path | None:
    d = videos_dir()
    if not d.is_dir():
        return None
    mp4s = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]
    if not mp4s:
        return None
    return random.choice(mp4s)


def _extract_video_path(result: object) -> Path | None:
    if result is None:
        return None
    if isinstance(result, dict):
        v = result.get("video")
        if v is None:
            return None
        if hasattr(v, "path"):
            p = getattr(v, "path", None)
            if p:
                return Path(str(p))
        if isinstance(v, dict):
            inner = v.get("path") or v.get("video")
            if inner:
                return Path(str(inner))
        return Path(str(v))
    if isinstance(result, (list, tuple)) and result:
        return Path(str(result[0]))
    p = Path(str(result))
    return p if p.exists() else None


def process_lipsync_to_tmp(
    *,
    audio_base64_wav: str,
    source_video: Path,
) -> Path | None:
    """
    Call the local Gradio /process_video app, copy the output into lipsync_tmp_dir().
    Returns the destination path or None on failure.
    """
    settings = get_settings()
    base = (settings.lipsync_gradio_url or "").strip()
    if not base:
        return None

    if Client is None or handle_file is None:
        logger.warning(
            "gradio_client is not installed; lip-sync disabled. "
            "Install: pip install gradio_client (in the same venv as uvicorn)."
        )
        return None

    tmp = lipsync_tmp_dir()
    wav_path = tmp / f"in_{uuid.uuid4().hex}.wav"
    out_name = f"{uuid.uuid4().hex}.mp4"
    dest = tmp / out_name

    try:
        wav_path.write_bytes(base64.b64decode(audio_base64_wav))
    except Exception:
        logger.exception("Failed to decode TTS audio for lip-sync")
        return None

    try:
        client = _get_gradio_client(base)
        if client is None:
            logger.warning("gradio_client not installed; lip-sync disabled.")
            return None

        job = client.submit(
            video_path={
                "video": handle_file(str(source_video)),
                "subtitles": None,
            },
            audio_path=handle_file(str(wav_path)),
            guidance_scale=float(settings.lipsync_guidance_scale),
            inference_steps=int(settings.lipsync_inference_steps),
            seed=int(settings.lipsync_seed),
            api_name="/process_video",
        )
        logger.info("Lip-sync job submitted, waiting up to %.0fs …", settings.lipsync_timeout)
        result = job.result(timeout=settings.lipsync_timeout)
        logger.info("Lip-sync job completed, result type=%s", type(result).__name__)
        src = _extract_video_path(result)
        if not src or not src.is_file():
            logger.warning("Lip-sync returned no usable video path: %r", result)
            return None
        shutil.copyfile(src, dest)
        return dest
    except Exception:
        logger.exception("Gradio lip-sync failed")
        return None
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass


def safe_delete_lipsync_file(filename: str) -> bool:
    """Remove a file under lipsync_tmp_dir() if the name is safe. Returns True if deleted."""
    if not SAFE_LIPSYNC_FILENAME.match(filename):
        return False
    path = lipsync_tmp_dir() / filename
    try:
        path.resolve().relative_to(lipsync_tmp_dir().resolve())
    except ValueError:
        return False
    if not path.is_file():
        return False
    path.unlink()
    return True
