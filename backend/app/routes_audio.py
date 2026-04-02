from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.services import audio_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    mal_mode: str | None = Form(None),
    video_call_simulation: str | None = Form(None),
) -> Dict[str, Any]:
    """
    Accept an audio file and process it through the appropriate workflow.

    If mal_mode is "true":
      Malayalam audio → Gemini (multimodal speech→English) → Orchestrator
      → English response → Gemini (en→colloquial ml) → Sarvam TTS → audio

    Otherwise:
      Audio → Gemini (upload + transcript) → English text → Orchestrator → response
    """
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload an audio file.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file.",
        )

    sid = session_id or "audio-" + (file.filename or "default")
    is_mal = mal_mode == "true"
    video_sim = video_call_simulation == "true"

    if is_mal:
        return await _handle_malayalam_flow(raw, file, sid, video_sim=video_sim)
    return await _handle_english_flow(raw, file, sid)


async def _handle_english_flow(
    raw: bytes,
    file: UploadFile,
    sid: str,
) -> Dict[str, Any]:
    """Standard flow: Gemini audio transcription → Orchestrator → text response."""
    fname = file.filename or "recording.webm"
    mime = file.content_type or "audio/webm"

    try:
        return await audio_pipeline.process_english_voice_bytes(
            raw,
            filename=fname,
            mime_type=mime,
            session_id=sid,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Gemini transcription failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Transcription failed: {e}",
        ) from e


async def _handle_malayalam_flow(
    raw: bytes,
    file: UploadFile,
    sid: str,
    *,
    video_sim: bool = False,
) -> Dict[str, Any]:
    """
    Malayalam flow:
      1. Gemini multimodal   (Malayalam/mixed audio → English text)
      2. Orchestrator         (English text   → English response)
      3. Gemini translate     (English resp   → colloquial Malayalam text)
      4. Sarvam TTS           (Malayalam text  → base64 audio)
    """
    try:
        return await audio_pipeline.process_malayalam_live_voice_bytes(
            raw,
            filename=file.filename or "audio.webm",
            mime_type=file.content_type or "audio/webm",
            session_id=sid,
            video_sim=video_sim,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Gemini Malayalam speech→English failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Malayalam speech recognition failed: {e}",
        ) from e
