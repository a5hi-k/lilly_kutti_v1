"""Sarvam AI API — Malayalam text-to-speech only."""

from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SARVAM_BASE_URL = "https://api.sarvam.ai"


def _headers() -> dict:
    settings = get_settings()
    return {"api-subscription-key": settings.sarvam_api_key}


async def text_to_speech(
    text: str,
    language_code: str = "ml-IN",
    speaker: str = "kavya",
) -> str:
    """
    Convert text to speech using Sarvam TTS API.
    Returns a base64-encoded WAV audio string.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "text": text,
            "target_language_code": language_code,
            "speaker": speaker,
            "model": "bulbul:v3",
            "pace": 1.0,
        }
        resp = await client.post(
            f"{SARVAM_BASE_URL}/text-to-speech",
            headers={**_headers(), "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        result = resp.json()
        audios = result.get("audios", [])
        if not audios:
            raise ValueError("Sarvam TTS returned no audio")
        logger.info("Sarvam TTS: generated audio successfully")
        return audios[0]
