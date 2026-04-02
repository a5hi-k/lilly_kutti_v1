"""
Instagram DM bridge using the official Meta Graph API.

Polls /{PAGE_ID}/conversations?platform=instagram for new messages,
processes text / image / audio, and replies via /{IG_USER_ID}/messages.

Media sending (images, audio) requires IG_DM_PUBLIC_BASE_URL pointing to
this FastAPI backend over HTTPS (ngrok, Cloudflare Tunnel, or production).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings
from app.services import gemini_service
from app.services.reference_image_service import resolve_dm_photo_url_to_path
from app.services.session_latest_image import set_latest
from app.services.worker_mode_state import WorkerStep, get_worker_session, update_step
from app.workflows.chat_graph import get_shared_workflow

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]

_replied_message_ids: set[str] = set()
_running = False


def _parse_fb_created_time(ts: str | None) -> float:
    """Parse Graph API created_time for sorting (handles +0000 without colon)."""
    if not ts:
        return 0.0
    try:
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        else:
            m = re.search(r"([+-]\d{2})(\d{2})$", s)
            if m:
                s = s[: -len(m.group(0))] + m.group(1) + ":" + m.group(2)
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def _sniff_bytes_look_like_voice_container(head: bytes) -> bool:
    """True if bytes look like MP4/M4A/WebM/ID3, not JPEG/PNG/GIF (voice mis-tagged as image)."""
    if len(head) < 12:
        return False
    if head[:3] == b"\xff\xd8\xff":
        return False
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return False
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return False
    if len(head) >= 8 and head[4:8] == b"ftyp":
        return True
    if head[:3] == b"ID3":
        return True
    if head[:4] == b"OggS":
        return True
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return True
    return False


async def _peek_url_head(url: str, limit: int = 65536) -> bytes:
    """Download enough bytes to sniff container type (voice notes may be mislabeled as image)."""
    _ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": _ua},
        ) as client:
            r = await client.get(url, headers={"Range": f"bytes=0-{limit - 1}"})
            if r.status_code in (200, 206):
                return r.content
            r2 = await client.get(url)
            r2.raise_for_status()
            return r2.content[:limit]
    except Exception:
        return b""


def _graph_base(version: str | None = None) -> str:
    v = version or get_settings().ig_graph_api_version or "v25.0"
    return f"https://graph.facebook.com/{v}"


def _log_graph_error(label: str, error: dict) -> None:
    logger.warning(
        "Graph API %s error: code=%s subcode=%s type=%s message=%s fbtrace=%s",
        label,
        error.get("code"),
        error.get("error_subcode"),
        error.get("type"),
        error.get("message"),
        error.get("fbtrace_id"),
    )


# ── send helpers ──────────────────────────────────────────────────────────
# Instagram Messaging API requires sending via /{PAGE_ID}/messages, NOT /{IG_USER_ID}/messages.

async def _send_text_dm(recipient_id: str, text: str) -> bool:
    msg = (text or "").strip()
    if not msg:
        return False
    settings = get_settings()
    url = f"{_graph_base()}/{settings.ig_graph_page_id}/messages"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": msg},
        "access_token": settings.ig_graph_access_token,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            if "error" in data:
                _log_graph_error("send_text", data["error"])
                return False
            return True
    except Exception:
        logger.exception("Graph API send_text failed to %s", recipient_id)
        return False


async def _send_media_dm(
    recipient_id: str,
    media_type: str,
    public_url: str,
) -> bool:
    """Send an image or audio DM via Graph API. media_type = 'image' | 'audio'."""
    settings = get_settings()
    url = f"{_graph_base()}/{settings.ig_graph_page_id}/messages"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {
            "attachment": {
                "type": media_type,
                "payload": {"url": public_url},
            }
        },
        "access_token": settings.ig_graph_access_token,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            if "error" in data:
                _log_graph_error(f"send_{media_type}", data["error"])
                return False
            logger.info("Graph API: sent %s to %s", media_type, recipient_id)
            return True
    except Exception:
        logger.exception("Graph API send_%s failed", media_type)
        return False


# ── public-URL helpers ────────────────────────────────────────────────────

def _relative_url_for_served_file(path: Path) -> str | None:
    """Map a local file to a FastAPI-served URL path."""
    try:
        p = path.resolve()
    except OSError:
        p = path
    if not p.is_file():
        return None
    from app.services.freepic_service import generated_costumes_dir

    try:
        gen_dir = generated_costumes_dir().resolve()
        if p.parent == gen_dir:
            return f"/generated-costumes/{p.name}"
    except OSError:
        pass
    pay = (_BACKEND_DIR / "PaymentQR").resolve()
    if p.parent == pay:
        return f"/payment-qr/{p.name}"
    ref = (_BACKEND_DIR / "ReferImage").resolve()
    if p.parent == ref:
        return f"/ref-assets/{p.name}"
    uploads = (_BACKEND_DIR / "session_uploads").resolve()
    if p.parent == uploads:
        return f"/session-uploads/{p.name}"
    return None


def _public_url(path: Path) -> str | None:
    base = (get_settings().ig_dm_public_base_url or "").strip().rstrip("/")
    rel = _relative_url_for_served_file(path)
    if not base or not rel:
        return None
    return f"{base}{rel}"


def _public_url_from_relative(rel_url: str) -> str | None:
    base = (get_settings().ig_dm_public_base_url or "").strip().rstrip("/")
    if not base or not rel_url:
        return None
    return f"{base}{rel_url}"


# ── image optimisation for DM ─────────────────────────────────────────────

_DM_IMAGE_MAX_SIDE = 1080
_DM_IMAGE_MAX_BYTES = 1_000_000  # 1 MB


async def _optimise_image_for_dm(src: Path) -> Path:
    """Return a DM-friendly copy (≤1080 px longest side, ≤1 MB, sRGB JPEG).

    If the source is already small enough it is returned as-is.
    The optimised copy is placed in session_uploads so it can be served.
    """
    if not src.is_file():
        return src
    if src.stat().st_size <= _DM_IMAGE_MAX_BYTES:
        from PIL import Image as _Img
        try:
            with _Img.open(src) as im:
                if max(im.size) <= _DM_IMAGE_MAX_SIDE:
                    return src
        except Exception:
            return src

    def _resize() -> Path:
        from PIL import Image as _Img
        uploads = _BACKEND_DIR / "session_uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / f"dm_{uuid.uuid4().hex}.jpg"

        with _Img.open(src) as im:
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            elif im.mode != "RGB":
                im = im.convert("RGB")
            im.thumbnail((_DM_IMAGE_MAX_SIDE, _DM_IMAGE_MAX_SIDE))
            im.save(dest, format="JPEG", quality=85, optimize=True)
        return dest

    return await asyncio.to_thread(_resize)


# ── send image / audio DM (resolve local path → public URL) ──────────────

async def _send_photo_dm(recipient_id: str, photo_path: Path, label: str = "image") -> bool:
    try:
        optimised = await _optimise_image_for_dm(photo_path)
    except Exception:
        logger.exception("Failed to optimise image for DM: %s", photo_path)
        optimised = photo_path

    pub = _public_url(optimised)
    if not pub:
        logger.warning("Cannot send %s DM — no IG_DM_PUBLIC_BASE_URL or file not served: %s", label, photo_path)
        await _send_text_dm(
            recipient_id,
            f"[I have a {label} ready but need IG_DM_PUBLIC_BASE_URL configured to send it in DM.]",
        )
        return False
    return await _send_media_dm(recipient_id, "image", pub)


async def _send_audio_dm(recipient_id: str, audio_path: Path) -> bool:
    pub = _public_url(audio_path)
    if not pub:
        logger.warning("Cannot send audio DM — no public URL for %s", audio_path)
        return False
    return await _send_media_dm(recipient_id, "audio", pub)


# ── assistant payload parsing ─────────────────────────────────────────────

def _parse_assistant_payload(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("kind") == "assistant":
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return {"kind": "assistant", "content": (raw or "").strip(), "ui_event": None}


# ── image / media DM attachment sending ───────────────────────────────────

async def _maybe_send_chat_image_dm(recipient_id: str, ui: Any) -> bool:
    """Send generated images, payment QR, or worker-mode images after assistant text."""
    try:
        if not isinstance(ui, dict):
            return False

        ui_type = ui.get("type")
        image_url = str(ui.get("image_url") or "")

        if ui_type == "worker_mode_payment_qr" and image_url:
            qr_path = _BACKEND_DIR / "PaymentQR" / Path(image_url).name
            if qr_path.is_file():
                return await _send_photo_dm(recipient_id, qr_path, "payment QR")
            return False

        if ui_type in ("worker_mode_active",) and image_url:
            path = resolve_dm_photo_url_to_path(image_url)
            if path is None:
                from app.services.freepic_service import resolve_generated_costume_url_to_path
                path = resolve_generated_costume_url_to_path(image_url)
            if path and path.is_file():
                return await _send_photo_dm(recipient_id, path, "worker-mode generated")
            return False

        if ui_type != "share_photo":
            return False

        path = resolve_dm_photo_url_to_path(image_url)
        if path is None:
            return False

        set_latest(f"ig_{recipient_id}", path)
        return await _send_photo_dm(recipient_id, path, "chat")
    except Exception:
        logger.exception("_maybe_send_chat_image_dm failed for %s", recipient_id)
        return False


# ── handle incoming voice ─────────────────────────────────────────────────

def _wav_base64_to_m4a(wav_base64: str) -> Path | None:
    """Decode WAV/audio base64, convert to M4A (AAC in MP4) via ffmpeg for Graph API audio DM."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found — cannot convert audio for DM")
        return None
    try:
        raw = base64.b64decode(wav_base64, validate=False)
    except Exception:
        return None
    if not raw:
        return None

    tmpdir = Path(tempfile.mkdtemp(prefix="ig_voice_"))
    wav_path = tmpdir / "tts.wav"
    m4a_path = tmpdir / "voice.m4a"
    wav_path.write_bytes(raw)

    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-c:a", "aac", "-b:a", "64k", "-ar", "44100", "-ac", "1",
        "-movflags", "+faststart", str(m4a_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
    except Exception as e:
        logger.warning("ffmpeg WAV→M4A failed: %s", e)
        _cleanup_dir(tmpdir)
        return None

    if not m4a_path.is_file() or m4a_path.stat().st_size == 0:
        _cleanup_dir(tmpdir)
        return None
    return m4a_path


def _cleanup_dir(directory: Path | None) -> None:
    if not directory:
        return
    try:
        for p in directory.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        directory.rmdir()
    except OSError:
        pass


async def _save_audio_for_serving(m4a_path: Path) -> Path | None:
    """Copy the M4A into session_uploads so it is served by FastAPI."""
    uploads = _BACKEND_DIR / "session_uploads"
    await asyncio.to_thread(uploads.mkdir, parents=True, exist_ok=True)
    dest = uploads / f"voice_{uuid.uuid4().hex}.m4a"
    try:
        await asyncio.to_thread(shutil.copy2, m4a_path, dest)
        return dest
    except Exception:
        logger.exception("Failed to copy M4A to session_uploads")
        return None


def _mime_from_voice_download(url: str, content_type: str | None) -> tuple[str, str]:
    """Pick filename suffix + mime for audio_pipeline (matches /audio/upload behavior)."""
    ext = Path(urlparse(url).path).suffix.lower()
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct.startswith("audio/"):
        mime = ct
        if ext not in (".mp4", ".m4a", ".aac", ".webm", ".opus", ".wav", ".mp3", ".ogg"):
            if "wav" in ct:
                ext = ".wav"
            elif "mpeg" in ct or "mp3" in ct:
                ext = ".mp3"
            elif "webm" in ct:
                ext = ".webm"
            elif any(x in ct for x in ("mp4", "aac", "m4a")):
                ext = ".m4a"
            else:
                ext = ".m4a"
        return f"ig_voice{ext}", mime
    if ct.startswith("video/"):
        # Voice notes are often video/mp4 containers
        mime = "audio/mp4"
        if ext not in (".mp4", ".m4a", ".mov", ".webm"):
            ext = ".mp4"
        return f"ig_voice{ext}", mime
    if ext in (".m4a", ".aac"):
        return f"ig_voice{ext}", "audio/mp4"
    if ext in (".webm", ".opus", ".ogg"):
        return f"ig_voice{ext}", "audio/webm"
    if ext in (".wav",):
        return f"ig_voice{ext}", "audio/wav"
    if ext in (".mp3",):
        return f"ig_voice{ext}", "audio/mpeg"
    if ext in (".mp4", ".mov"):
        # Instagram voice notes are often MP4 (video container); Gemini accepts audio/mp4.
        return f"ig_voice{ext}", "audio/mp4"
    return "ig_voice.m4a", "audio/mp4"


async def _reply_voice_message(recipient_id: str, audio_url: str) -> bool:
    """Download voice attachment; same pipeline as web: process_voice_dm_reply_bytes (STT → graph → TTS)."""
    from app.services.audio_pipeline import process_voice_dm_reply_bytes

    _ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    try:
        async with httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
            headers={"User-Agent": _ua},
        ) as client:
            r = await client.get(audio_url)
            r.raise_for_status()
            raw = r.content
            ct = r.headers.get("content-type")
    except Exception:
        logger.exception("Failed to download voice attachment from %s", audio_url)
        return False

    if not raw:
        return await _send_text_dm(
            recipient_id,
            "I couldn't download that voice message — please try sending it again.",
        )

    fname, mime = _mime_from_voice_download(audio_url, ct)
    sid = f"ig_{recipient_id}"
    logger.info(
        "IG DM: voice pipeline bytes=%d fname=%s mime=%s user=%s",
        len(raw), fname, mime, recipient_id,
    )

    try:
        result = await process_voice_dm_reply_bytes(
            raw, filename=fname, mime_type=mime,
            session_id=sid, video_sim=False,
        )
    except Exception:
        logger.exception("Voice DM pipeline failed user=%s", recipient_id)
        return await _send_text_dm(
            recipient_id,
            "Sorry — I couldn't process that voice note. Try again or type your message.",
        )

    tts_b64 = result.get("tts_audio") or result.get("mal_audio")
    reply_text = (result.get("reply") or "").strip()
    mal_text = (result.get("mal_text") or "").strip()
    fallback = mal_text or reply_text

    audio_sent = False
    if tts_b64:
        m4a_path = await asyncio.to_thread(_wav_base64_to_m4a, tts_b64)
        if m4a_path:
            served = await _save_audio_for_serving(m4a_path)
            _cleanup_dir(m4a_path.parent)
            if served:
                audio_sent = await _send_audio_dm(recipient_id, served)

    sent = audio_sent
    if fallback and not audio_sent:
        ok = await _send_text_dm(recipient_id, fallback)
        sent = sent or ok

    await _maybe_send_chat_image_dm(recipient_id, result.get("ui_event"))
    return sent


# ── handle incoming image ─────────────────────────────────────────────────

async def _reply_incoming_image(recipient_id: str, image_url: str) -> bool:
    """Download IG attachment, reply (worker ref or Gemini). Always completes in a defined way.

    On hard failures after download we still try to ack the user and return True so the same
    message id is not retried forever while newer texts are ignored.
    """
    from app.services.worker_mode_state import add_reference_image, is_worker_mode

    session_id = f"ig_{recipient_id}"

    _ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    try:
        async with httpx.AsyncClient(
            timeout=90.0,
            follow_redirects=True,
            headers={"User-Agent": _ua},
        ) as client:
            r = await client.get(image_url)
            r.raise_for_status()
            img_bytes = r.content
    except httpx.RequestError as e:
        logger.warning("Could not download DM image (%s): %s", image_url[:80], e)
        return False
    except Exception as e:
        logger.warning("Could not download DM image: %s", e)
        return False

    if not img_bytes:
        return await _send_text_dm(
            recipient_id,
            "I got an empty image from Instagram — try sending the photo again?",
        )

    ext = Path(urlparse(image_url).path).suffix.lower()
    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    fname = Path(urlparse(image_url).path).name or "dm.jpg"

    upload_dir = _BACKEND_DIR / "session_uploads"
    await asyncio.to_thread(upload_dir.mkdir, parents=True, exist_ok=True)
    local_path = upload_dir / f"ig_{recipient_id}_{uuid.uuid4().hex}.jpg"
    await asyncio.to_thread(local_path.write_bytes, img_bytes)

    if is_worker_mode(session_id):
        add_reference_image(session_id, local_path)
        ws = get_worker_session(session_id)
        n = len(ws.reference_images)
        ack_en = (
            f"Reference image #{n} saved. "
            "You can send more images or describe your requirements."
        )
        wf = get_shared_workflow()
        anchor = wf.worker_language_anchor(session_id, "")
        ack = await gemini_service.worker_mode_localize_reply(anchor, ack_en)
        sent = await _send_text_dm(recipient_id, ack)
        logger.info("Graph API: worker mode — tracked reference image #%d user=%s", n, recipient_id)
        return sent

    set_latest(session_id, local_path)

    try:
        text = await gemini_service.girlfriend_reply_to_incoming_image(
            img_bytes, filename=fname, mime_type=mime,
        )
        text = (text or "").strip()
        if not text:
            text = "Aww, cute pic — thanks for sharing!"

        wf = get_shared_workflow()
        await wf.record_exchange(
            session_id=session_id,
            user_content="[User sent a photo]",
            assistant_content=text,
        )

        sent = await _send_text_dm(recipient_id, text)
        if sent:
            logger.info("Graph API: girlfriend reply to incoming photo user=%s", recipient_id)
        return sent
    except Exception:
        logger.exception("Gemini / send failed for incoming DM photo user=%s", recipient_id)
        fallback = (
            "Sorry — I had trouble processing that photo. "
            "Try another image or describe it in a text message?"
        )
        return await _send_text_dm(recipient_id, fallback)


# ── handle incoming text ──────────────────────────────────────────────────

async def _reply_text_dm(orchestrator: Any, recipient_id: str, user_query: str) -> bool:
    try:
        raw = await orchestrator.handle_user_message(
            {"content": user_query, "session_id": f"ig_{recipient_id}"}
        )
        payload = _parse_assistant_payload(raw)
        reply_text = str(payload.get("content") or "").strip()
        ui = payload.get("ui_event")

        sent = False
        if reply_text:
            ok = await _send_text_dm(recipient_id, reply_text)
            sent = sent or ok

        if await _maybe_send_chat_image_dm(recipient_id, ui):
            sent = True

        return sent
    except Exception:
        logger.exception("Orchestrator failed for DM text user=%s", recipient_id)
        return await _send_text_dm(
            recipient_id,
            "Sorry — something glitched on my side. Please send your message again in a moment.",
        )


# ── Graph API polling ─────────────────────────────────────────────────────

async def _fetch_conversations(
    access_token: str, page_id: str,
) -> tuple[list[dict], bool]:
    """Fetch recent Instagram DM conversations. Returns (data, graph_reachable).

    graph_reachable False means DNS/network failure — caller should back off, not spam tracebacks.
    """
    url = f"{_graph_base()}/{page_id}/conversations"
    params = {
        "platform": "instagram",
        "folder": "inbox",
        "access_token": access_token,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
            if "error" in data:
                _log_graph_error("conversations", data["error"])
                return [], True
            return data.get("data", []), True
    except httpx.RequestError as e:
        logger.warning("Graph API: cannot reach Facebook (conversations): %s", e)
        return [], False
    except Exception as e:
        logger.warning("Graph API: conversations request failed: %s", e)
        return [], False


# Nested attachment fields — plain `attachments` often omits URLs for Instagram DMs.
# `type` helps distinguish AUDIO vs VIDEO voice notes; `name` hints at .m4a / .mp4.
_MESSAGE_FIELDS = (
    "id,message,from,created_time,"
    "attachments{mime_type,type,name,file_url,image_data{url},video_data{url}}"
)
_MESSAGE_FIELDS_FALLBACK = "id,message,from,created_time,attachments"


async def _fetch_messages(
    access_token: str, conversation_id: str,
) -> tuple[list[dict], bool]:
    """Fetch messages in a conversation, newest first. Returns (data, graph_reachable)."""
    url = f"{_graph_base()}/{conversation_id}/messages"
    params = {
        "fields": _MESSAGE_FIELDS,
        "access_token": access_token,
        "limit": "10",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
            if "error" in data:
                err = data.get("error") or {}
                msg_e = (err.get("message") or "").lower()
                if err.get("code") in (100, 2500) or "field" in msg_e or "invalid" in msg_e:
                    params_fb = {**params, "fields": _MESSAGE_FIELDS_FALLBACK}
                    resp = await client.get(url, params=params_fb)
                    data = resp.json()
                if "error" in data:
                    _log_graph_error("messages", data["error"])
                    return [], True
            out = data.get("data", []) or []
            out.sort(key=lambda m: _parse_fb_created_time(m.get("created_time")), reverse=True)
            return out, True
    except httpx.RequestError as e:
        logger.warning(
            "Graph API: cannot reach Facebook (messages): %s",
            e,
        )
        return [], False
    except Exception as e:
        logger.warning("Graph API: messages request failed: %s", e)
        return [], False


async def _fetch_message_by_id(
    access_token: str, message_id: str,
) -> dict | None:
    """GET /{message-id} with full attachment fields (sometimes richer than conversation edge)."""
    url = f"{_graph_base()}/{message_id}"
    params = {
        "fields": _MESSAGE_FIELDS,
        "access_token": access_token,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
            if "error" in data:
                _log_graph_error("message_by_id", data["error"])
                return None
            return data if isinstance(data, dict) else None
    except httpx.RequestError:
        return None
    except Exception:
        logger.warning("Graph API: message_by_id failed for %s", message_id[:40])
        return None


def _attachment_data_list(msg: dict) -> list[dict]:
    raw = msg.get("attachments")
    out: list[dict] = []
    if isinstance(raw, list):
        out.extend(a for a in raw if isinstance(a, dict))
    elif isinstance(raw, dict):
        out.extend(raw.get("data") or [])
    sing = msg.get("attachment")
    if isinstance(sing, dict):
        out.append(sing)
    return out


def _infer_media_kind_from_url(url: str) -> str | None:
    """Guess audio / video / image from CDN path when mime_type is missing or octet-stream."""
    path = urlparse(url).path.lower()
    if path.endswith(
        (".m4a", ".aac", ".mp3", ".opus", ".wav", ".ogg", ".flac", ".amr", ".caf"),
    ):
        return "audio"
    if path.endswith(".webm"):
        return "audio"
    if path.endswith((".mp4", ".mov", ".m4v")):
        # Instagram voice notes are often MP4 containers; treat as video → same voice pipeline.
        return "video"
    if path.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
        return "image"
    return None


def _resolve_file_url_media_type(
    att: dict, mime: str, file_url: str,
) -> str:
    """Classify file_url attachment: voice notes often use octet-stream + opaque URLs."""
    inferred = _infer_media_kind_from_url(file_url)
    if inferred:
        return inferred
    if "audio" in mime:
        return "audio"
    if "video" in mime:
        return "video"
    if "image" in mime:
        return "image"
    name = (att.get("name") or "").lower()
    if any(name.endswith(ext) for ext in (".m4a", ".aac", ".mp3", ".opus", ".wav", ".ogg")):
        return "audio"
    if any(name.endswith(ext) for ext in (".mp4", ".mov", ".webm", ".m4v")):
        return "video"
    # Opaque CDN paths with no extension: voice pipeline accepts MP4/AAC blobs
    if "octet-stream" in mime or not mime:
        return "video"
    return "image"


def _pick_best_media_pair(
    pairs: list[tuple[str | None, str]],
) -> tuple[str | None, str | None]:
    """Prefer audio (voice) over video over image when Graph returns multiple parts."""
    if not pairs:
        return None, None
    for priority in ("audio", "video", "image"):
        for t, u in pairs:
            if t == priority and u:
                return t, u
    t0, u0 = pairs[0]
    return t0, u0


def _extract_attachment_info_from_dict(att: dict) -> tuple[str | None, str | None]:
    """Single attachment object from Graph API (inline or /attachments edge).

    Voice notes often include **image_data** (waveform/preview JPEG) *and* **file_url** (real audio).
    We must prefer file_url / video_data over image_data or voice is misrouted to the image pipeline.
    """
    mime = (att.get("mime_type") or "").lower()
    raw = (att.get("type") or "").lower() or None
    if raw in ("audio_message", "animated_audio"):
        raw = "audio"
    elif raw in ("ig_reel", "share", "fallback"):
        raw = None

    # MIME is more reliable than `type` (voice previews sometimes report type=image).
    if "audio" in mime:
        att_type: str | None = "audio"
    elif "video" in mime:
        att_type = "video"
    elif "image" in mime:
        att_type = "image"
    elif raw == "image":
        att_type = "image"
    elif raw == "audio":
        att_type = "audio"
    elif raw == "video":
        att_type = "video"
    else:
        att_type = raw

    file_url = att.get("file_url")
    if file_url:
        if att_type not in ("image", "audio", "video"):
            att_type = _resolve_file_url_media_type(att, mime, file_url)
        elif att_type == "image" and (
            "audio" in mime
            or "video" in mime
            or _infer_media_kind_from_url(file_url) in ("audio", "video")
        ):
            att_type = _resolve_file_url_media_type(att, mime, file_url)
        return att_type, file_url

    video_data = att.get("video_data") or {}
    if isinstance(video_data, dict) and video_data.get("url"):
        url = video_data["url"]
        t = "video" if att_type is None else att_type
        if t == "image":
            t = "video"
        return t, url

    image_data = att.get("image_data") or {}
    if isinstance(image_data, dict) and image_data.get("url"):
        url = image_data["url"]
        t = "image" if att_type is None else att_type
        return t, url

    if att.get("url"):
        u = att["url"]
        if att_type is None:
            att_type = _resolve_file_url_media_type(att, mime, u)
        return att_type, u

    payload = att.get("payload") or {}
    if isinstance(payload, dict) and payload.get("url"):
        u = payload["url"]
        if att_type is None:
            att_type = _resolve_file_url_media_type(att, mime, u)
        return att_type, u

    return None, None


async def _fetch_message_attachments_edge(
    access_token: str, message_id: str,
) -> list[dict]:
    """Fallback: GET /{message-id}/attachments (returns file_url for many DM types)."""
    url = f"{_graph_base()}/{message_id}/attachments"
    params = {
        "fields": "mime_type,type,name,file_url,image_data{url},video_data{url},payload",
        "access_token": access_token,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
            if "error" in data:
                _log_graph_error("message_attachments", data["error"])
                return []
            return data.get("data", [])
    except httpx.RequestError:
        return []
    except Exception:
        logger.warning("Graph API: message_attachments failed for %s", message_id[:32])
        return []


def _append_pair_dedupe(
    pairs: list[tuple[str | None, str]],
    seen: set[str],
    t: str | None,
    u: str,
) -> None:
    if u and u not in seen:
        seen.add(u)
        pairs.append((t, u))


async def _resolve_incoming_attachment(
    access_token: str, msg: dict,
) -> tuple[str | None, str | None]:
    """Merge inline + /message-id/attachments + GET /message-id, then audio > video > image."""
    pairs: list[tuple[str | None, str]] = []
    seen_urls: set[str] = set()
    for att in _attachment_data_list(msg):
        if isinstance(att, dict):
            t, u = _extract_attachment_info_from_dict(att)
            if u:
                _append_pair_dedupe(pairs, seen_urls, t, u)
    mid = msg.get("id")
    if mid:
        extra = await _fetch_message_attachments_edge(access_token, mid)
        for att in extra:
            if isinstance(att, dict):
                t2, u2 = _extract_attachment_info_from_dict(att)
                if u2:
                    _append_pair_dedupe(pairs, seen_urls, t2, u2)

        if not pairs:
            full = await _fetch_message_by_id(access_token, mid)
            if full:
                for att in _attachment_data_list(full):
                    if isinstance(att, dict):
                        t3, u3 = _extract_attachment_info_from_dict(att)
                        if u3:
                            _append_pair_dedupe(pairs, seen_urls, t3, u3)
        else:
            best_t, best_u = _pick_best_media_pair(pairs)
            if best_t == "image" and best_u:
                full = await _fetch_message_by_id(access_token, mid)
                if full:
                    for att in _attachment_data_list(full):
                        if isinstance(att, dict):
                            t4, u4 = _extract_attachment_info_from_dict(att)
                            if u4:
                                _append_pair_dedupe(pairs, seen_urls, t4, u4)

    return _pick_best_media_pair(pairs)


async def _poll_once(orchestrator: Any, settings: Any) -> bool:
    """Poll inbox once. Returns False if Facebook was unreachable (caller should sleep longer)."""
    access_token = settings.ig_graph_access_token
    page_id = settings.ig_graph_page_id
    ig_user_id = settings.ig_graph_user_id

    conversations, conv_ok = await _fetch_conversations(access_token, page_id)
    if not conv_ok:
        return False

    for convo in conversations:
        convo_id = convo.get("id")
        if not convo_id:
            continue

        messages, msg_ok = await _fetch_messages(access_token, convo_id)
        if not msg_ok:
            return False
        if not messages:
            continue

        # Newest-first API order: handle only the first actionable user message, then stop.
        # (If it fails with ok=False we do not fall through to older messages in the same poll.)
        for newest in messages:
            mid = newest.get("id")
            if not mid or mid in _replied_message_ids:
                continue

            sender = (newest.get("from") or {})
            sender_id = sender.get("id", "")

            if sender_id == ig_user_id or sender_id == page_id:
                _replied_message_ids.add(mid)
                continue

            text = (newest.get("message") or "").strip()
            att_type, att_url = await _resolve_incoming_attachment(access_token, newest)

            ok = False

            if att_type == "audio" and att_url:
                ok = await _reply_voice_message(sender_id, att_url)
            elif att_type == "image" and att_url:
                # Voice notes are sometimes labeled image (waveform JPEG) while file_url is audio;
                # if we only got the MP4 URL, sniff bytes — JPEG/PNG → image, ftyp/ID3 → voice.
                if not text:
                    peek = await _peek_url_head(att_url)
                    if _sniff_bytes_look_like_voice_container(peek):
                        logger.info(
                            "IG DM: attachment typed image but bytes look like voice/video; "
                            "using voice pipeline mid=%s",
                            mid[:48] if mid else "",
                        )
                        ok = await _reply_voice_message(sender_id, att_url)
                    else:
                        ok = await _reply_incoming_image(sender_id, att_url)
                else:
                    ok = await _reply_incoming_image(sender_id, att_url)
            elif att_type == "video" and att_url:
                ok = await _reply_voice_message(sender_id, att_url)
            elif text:
                ok = await _reply_text_dm(orchestrator, sender_id, text)
            else:
                att_list = _attachment_data_list(newest)
                if att_list and not text:
                    logger.warning(
                        "IG DM: media message but no usable URL mid=%s attachment_keys=%s",
                        mid,
                        [list(a.keys()) for a in att_list if isinstance(a, dict)],
                    )
                    ok = await _send_text_dm(
                        sender_id,
                        "I got your media but Instagram didn't give me a download link. "
                        "Try sending again as a normal photo, or describe it in text.",
                    )
                    if ok:
                        _replied_message_ids.add(mid)
                    break
                logger.info(
                    "Graph API: skipping message mid=%s (no text/attachment), sender=%s",
                    mid, sender_id,
                )
                _replied_message_ids.add(mid)
                continue

            if ok:
                _replied_message_ids.add(mid)
            break

    if len(_replied_message_ids) > 5000:
        excess = list(_replied_message_ids)[:2500]
        for eid in excess:
            _replied_message_ids.discard(eid)

    return True


async def _listener_loop(orchestrator: Any) -> None:
    global _running

    logger.info("Instagram DM listener started (Graph API)")
    while _running:
        settings = get_settings()
        base_interval = max(3.0, settings.ig_dm_poll_interval)
        try:
            graph_ok = await _poll_once(orchestrator, settings)
        except Exception:
            logger.exception("Instagram DM poll error")
            await asyncio.sleep(min(120.0, max(15.0, base_interval * 3)))
            continue

        if graph_ok:
            await asyncio.sleep(base_interval)
        else:
            # DNS / connectivity — avoid hammering Graph API and flooding logs
            wait = min(120.0, max(30.0, base_interval * 4))
            logger.warning("Instagram DM: backing off %.0fs (network unreachable)", wait)
            await asyncio.sleep(wait)


async def start_instagram_dm_listener() -> asyncio.Task[None] | None:
    global _running, _replied_message_ids

    settings = get_settings()
    token = (settings.ig_graph_access_token or "").strip()
    ig_uid = (settings.ig_graph_user_id or "").strip()
    page_id = (settings.ig_graph_page_id or "").strip()

    if not token or not ig_uid or not page_id:
        logger.info(
            "Instagram DM off: set IG_GRAPH_ACCESS_TOKEN, IG_GRAPH_USER_ID, IG_GRAPH_PAGE_ID in .env"
        )
        return None

    pub_base = (settings.ig_dm_public_base_url or "").strip()
    if not pub_base:
        logger.warning(
            "IG_DM_PUBLIC_BASE_URL not set — image and audio DMs will not work. "
            "Set it to your backend's public HTTPS URL (e.g. ngrok)."
        )

    from app.services.chat_manager import orchestrator

    _running = True
    _replied_message_ids = set()

    # Pre-seed with all existing message IDs so we only reply to NEW messages
    convos, conv_ok = await _fetch_conversations(token, page_id)
    if not conv_ok:
        logger.warning(
            "Instagram DM: could not fetch conversations at startup (network). "
            "Seeding skipped — restart when online or risk replying to old threads."
        )
    seeded = 0
    for convo in convos:
        cid = convo.get("id")
        if not cid:
            continue
        msgs, _ = await _fetch_messages(token, cid)
        for m in msgs:
            mid = m.get("id")
            if mid:
                _replied_message_ids.add(mid)
                seeded += 1

    logger.info(
        "Instagram Graph API DM listener starting — IG user=%s, page=%s, seeded %d existing messages",
        ig_uid, page_id, seeded,
    )
    return asyncio.create_task(_listener_loop(orchestrator))


async def stop_instagram_dm_listener(task: asyncio.Task[None] | None) -> None:
    global _running
    _running = False
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
