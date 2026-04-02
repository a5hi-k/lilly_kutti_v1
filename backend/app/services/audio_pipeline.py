"""
Shared audio processing for /audio/upload and Instagram DM voice notes.

Malayalam live path matches "Tap & Speak (Live)" in the web app (mal_mode=true).
Instagram voice DMs route by detected language: Malayalam → ml TTS, English → en-IN TTS.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Literal

from app.services import gemini_service, lipsync_service, sarvam_service
from app.workflows.chat_graph import get_shared_workflow

logger = logging.getLogger(__name__)

# Sarvam Bulbul speaker for English TTS (en-IN)
_EN_TTS_SPEAKER = "anushka"


def _attach_intent_ui_events(out: Dict[str, Any], state: dict) -> None:
    intent = state.get("intent")
    if intent == "video_call":
        out["ui_event"] = {"type": "video_call_start"}
        return
    if intent == "post_instagram":
        out["ui_event"] = {"type": "instagram_posted"}
        return
    if intent in ("share_photo", "costume_tryon"):
        img = state.get("image_url")
        if img:
            out["ui_event"] = {"type": "share_photo", "image_url": str(img)}


async def process_english_voice_bytes(
    raw: bytes,
    *,
    filename: str,
    mime_type: str,
    session_id: str,
) -> Dict[str, Any]:
    """English tap & speak: transcribe → orchestrator → text reply."""
    text = await gemini_service.transcribe_audio(
        audio_bytes=raw,
        filename=filename,
        mime_type=mime_type,
    )
    if not text:
        raise ValueError("Transcription did not return any text.")

    workflow = get_shared_workflow()
    state = await workflow.run_turn_state(session_id=session_id, user_text=text)
    reply = state.get("last_assistant") or ""

    out: Dict[str, Any] = {
        "session_id": session_id,
        "transcription": text,
        "reply": reply,
    }
    _attach_intent_ui_events(out, state)
    return out


async def process_malayalam_live_voice_bytes(
    raw: bytes,
    *,
    filename: str,
    mime_type: str,
    session_id: str,
    video_sim: bool = False,
) -> Dict[str, Any]:
    """
    Live Malayalam flow (same as VoiceButton with malMode on):
      MAL speech → EN → orchestrator → EN reply → EN→ML text → Sarvam TTS (WAV base64).
    """
    english_text = await gemini_service.transcribe_malayalam_speech_to_english(
        audio_bytes=raw,
        filename=filename,
        mime_type=mime_type,
    )
    if not english_text:
        raise ValueError("Could not understand the Malayalam audio.")

    workflow = get_shared_workflow()
    graph_state = await workflow.run_turn_state(
        session_id=session_id, user_text=english_text
    )
    english_reply = graph_state.get("last_assistant") or ""

    mal_text: str | None = None
    try:
        mal_text = await gemini_service.translate_english_to_malayalam(english_reply)
    except Exception:
        logger.exception("Gemini en→ml translation failed, falling back to English")

    mal_audio: str | None = None
    if mal_text:
        try:
            mal_audio = await sarvam_service.text_to_speech(
                text=mal_text,
                language_code="ml-IN",
            )
        except Exception:
            logger.exception("Sarvam TTS failed, no audio will be returned")

    lipsync_video_url: str | None = None
    if video_sim and mal_audio:
        src = lipsync_service.pick_random_mp4()
        if src is not None:
            lip = await asyncio.to_thread(
                lipsync_service.process_lipsync_to_tmp,
                audio_base64_wav=mal_audio,
                source_video=src,
            )
            if lip is not None:
                lipsync_video_url = f"/lipsync-tmp/{lip.name}"
                mal_audio = None

    out: Dict[str, Any] = {
        "session_id": session_id,
        "transcription": english_text,
        "reply": english_reply,
        "mal_text": mal_text,
        "mal_audio": mal_audio,
    }
    if lipsync_video_url:
        out["lipsync_video_url"] = lipsync_video_url
    _attach_intent_ui_events(out, graph_state)
    return out


async def process_english_voice_with_tts_bytes(
    raw: bytes,
    *,
    filename: str,
    mime_type: str,
    session_id: str,
) -> Dict[str, Any]:
    """English voice: transcribe → orchestrator → Sarvam TTS (en-IN WAV base64)."""
    text = await gemini_service.transcribe_audio(
        audio_bytes=raw,
        filename=filename,
        mime_type=mime_type,
    )
    if not text:
        raise ValueError("Transcription did not return any text.")

    workflow = get_shared_workflow()
    state = await workflow.run_turn_state(session_id=session_id, user_text=text)
    reply = state.get("last_assistant") or ""

    tts_audio: str | None = None
    if reply.strip():
        try:
            tts_audio = await sarvam_service.text_to_speech(
                text=reply,
                language_code="en-IN",
                speaker=_EN_TTS_SPEAKER,
            )
        except Exception:
            logger.exception("Sarvam English TTS failed")

    out: Dict[str, Any] = {
        "session_id": session_id,
        "transcription": text,
        "reply": reply,
        "voice_language": "en",
        "tts_audio": tts_audio,
    }
    _attach_intent_ui_events(out, state)
    return out


async def process_voice_dm_reply_bytes(
    raw: bytes,
    *,
    filename: str,
    mime_type: str,
    session_id: str,
    video_sim: bool = False,
) -> Dict[str, Any]:
    """
    Instagram / bilingual voice: detect language, then Malayalam live pipeline or English+TTS.
    """
    lang: Literal["en", "ml"] = await gemini_service.detect_voice_language_from_audio(
        raw, filename=filename, mime_type=mime_type
    )
    logger.info("Voice DM language detected: %s", lang)

    if lang == "ml":
        out = await process_malayalam_live_voice_bytes(
            raw,
            filename=filename,
            mime_type=mime_type,
            session_id=session_id,
            video_sim=video_sim,
        )
        out["voice_language"] = "ml"
        out["tts_audio"] = out.get("mal_audio")
        return out

    out = await process_english_voice_with_tts_bytes(
        raw,
        filename=filename,
        mime_type=mime_type,
        session_id=session_id,
    )
    return out
