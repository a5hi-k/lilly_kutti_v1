"""Google Gemini API helpers: audio transcription, Malayalam → English, etc."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from google import genai

from app.core.config import get_settings
from app.services.llm_text import generate_text_with_system_sync

logger = logging.getLogger(__name__)


def _delete_uploaded_file(client: genai.Client, uploaded: Any) -> None:
    name = getattr(uploaded, "name", None)
    if name:
        try:
            client.files.delete(name=name)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not delete Gemini uploaded file %s: %s", name, exc)


def _upload_and_generate_from_audio(
    audio_bytes: bytes,
    filename: str,
    mime_type: str,
    *,
    system_instruction: str,
    user_message: str,
    temperature: float,
    model: str | None = None,
) -> str:
    """
    Upload audio bytes to Gemini Files API, run generate_content with multimodal input,
    return assistant text, then delete the uploaded file.
    """
    settings = get_settings()
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is empty — add it to backend/.env")

    client = genai.Client(api_key=api_key)
    resolved_model = model or settings.gemini_transcription_model

    buf = io.BytesIO(audio_bytes)
    buf.seek(0)
    buf.name = filename

    upload_cfg: dict[str, str] = {}
    if mime_type:
        upload_cfg["mime_type"] = mime_type

    uploaded: Any = None
    try:
        uploaded = client.files.upload(
            file=buf,
            config=upload_cfg or None,
        )

        response = client.models.generate_content(
            model=resolved_model,
            contents=[user_message, uploaded],
            config={
                "system_instruction": system_instruction,
                "temperature": temperature,
            },
        )

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise ValueError("Gemini returned empty text")
        return text
    finally:
        if uploaded is not None:
            _delete_uploaded_file(client, uploaded)


# ── English tap-to-speak (speech assumed mostly English) ─────────────────────

_ENGLISH_TRANSCRIPT_SYSTEM = (
    "You transcribe audio accurately. Follow the user message exactly for output format."
)

_ENGLISH_TRANSCRIPT_USER = (
    "Generate a transcript of the audio. "
    "Output only the spoken words in English. "
    "Do not add labels like 'Transcript:', speaker names, or commentary."
)


def _transcribe_audio_sync(
    audio_bytes: bytes,
    filename: str,
    mime_type: str,
) -> str:
    """General English-oriented transcription (non-MAL path)."""
    text = _upload_and_generate_from_audio(
        audio_bytes,
        filename,
        mime_type,
        system_instruction=_ENGLISH_TRANSCRIPT_SYSTEM,
        user_message=_ENGLISH_TRANSCRIPT_USER,
        temperature=0.2,
    )
    logger.info("Gemini transcribe (EN): %.80s", text)
    return text


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "recording.webm",
    mime_type: str = "audio/webm",
) -> str:
    """Async wrapper — non-MAL voice: English audio → English text."""
    return await asyncio.to_thread(
        _transcribe_audio_sync,
        audio_bytes,
        filename,
        mime_type,
    )


# ── MAL path: Malayalam (or mixed) speech → English for the orchestrator ───────

_MALAYALAM_SPEECH_TO_ENGLISH_SYSTEM = """You process short voice recordings for a conversational AI girlfriend app.

The user is speaking **Malayalam** (possibly mixed with English words). Your job:
1. Listen to the audio and understand what they mean — tone, intent, emotion, and any casual fillers.
2. Express that meaning in **natural, conversational English** — as if a bilingual person summarized what was said to an English-speaking partner.
3. Do **not** output Malayalam script, romanization-only Malayalam, or dictionary-style word-by-word glosses unless the user explicitly asked for that.
4. Keep names, numbers, and proper nouns sensible in English.
5. If the audio is silent, unclear, or not speech, respond with exactly: [unclear audio]

Output rules:
- Output **only** the English text. No quotes, no "Transcript:", no bullet points, no explanations.
- Keep length similar to the utterance (don’t over-expand unless needed for clarity)."""


_MALAYALAM_SPEECH_TO_ENGLISH_USER = (
    "Transcribe and interpret the attached audio: convert the speaker’s Malayalam speech "
    "into clear English as described in your instructions."
)


def _transcribe_malayalam_speech_to_english_sync(
    audio_bytes: bytes,
    filename: str,
    mime_type: str,
) -> str:
    settings = get_settings()
    text = _upload_and_generate_from_audio(
        audio_bytes,
        filename,
        mime_type,
        system_instruction=_MALAYALAM_SPEECH_TO_ENGLISH_SYSTEM,
        user_message=_MALAYALAM_SPEECH_TO_ENGLISH_USER,
        temperature=0.25,
        model=settings.gemini_malayalam_stt_model,
    )
    if "[unclear audio]" in text.lower():
        logger.warning("Gemini MAL STT: model reported unclear audio")
    logger.info("Gemini MAL speech→EN: %.80s", text)
    return text


async def transcribe_malayalam_speech_to_english(
    audio_bytes: bytes,
    filename: str = "recording.webm",
    mime_type: str = "audio/webm",
) -> str:
    """MAL toggle voice: Malayalam audio → English text for Lilly / orchestrator."""
    return await asyncio.to_thread(
        _transcribe_malayalam_speech_to_english_sync,
        audio_bytes,
        filename,
        mime_type,
    )


# ── MAL path: English reply → colloquial Malayalam (for TTS) ─────────────────

_ENGLISH_TO_MALAYALAM_SYSTEM = """You are a native Malayalam speaker helping dub an AI girlfriend's lines.

Rules:
- Output ONLY the Malayalam text. No quotes, labels, explanations, or English.
- Do NOT do word-for-word or formal/bookish translation. Use natural, colloquial spoken Malayalam how people talk in real life (including light code-mixing only if it sounds natural in casual speech).
- Preserve meaning, intent, subtext, emotion, teasing/playfulness, and relationship warmth—not just dictionary equivalents.
- Keep sentences concise and conversational when the source is short; match the vibe (romantic, playful, supportive) of the original.
- If a phrase has no good literal Malayalam idiom, convey the same feeling with an equivalent native expression."""


def _translate_english_to_malayalam_sync(english_text: str) -> str:
    settings = get_settings()
    user_prompt = (
        "Translate the following English into colloquial native Malayalam "
        "(preserve Lilly's personality and tone):\n\n"
        f"{english_text.strip()}"
    )
    out = generate_text_with_system_sync(
        _ENGLISH_TO_MALAYALAM_SYSTEM,
        user_prompt,
        0.65,
        gemini_model=settings.gemini_translation_model,
    )
    logger.info("Translate [en→ml]: %.80s", out)
    return out


async def translate_english_to_malayalam(english_text: str) -> str:
    """English assistant reply → colloquial Malayalam for Sarvam TTS."""
    return await asyncio.to_thread(_translate_english_to_malayalam_sync, english_text)


# ── Voice language detection (for bilingual DM / voice routing) ────────────────

_VOICE_LANG_SYSTEM = """You listen to short voice clips. Reply with ONE JSON object only, no markdown:
{"language":"en"} if the main spoken language is English,
{"language":"ml"} if the main spoken language is Malayalam.
If mixed, pick the dominant language."""


def _detect_voice_language_sync(
    audio_bytes: bytes,
    filename: str,
    mime_type: str,
) -> Literal["en", "ml"]:
    raw = _upload_and_generate_from_audio(
        audio_bytes,
        filename,
        mime_type,
        system_instruction=_VOICE_LANG_SYSTEM,
        user_message="Classify the primary spoken language per your instructions.",
        temperature=0.1,
        model=None,
    )
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            lang = str(data.get("language", "")).lower()
            if lang in ("ml", "malayalam"):
                return "ml"
            if lang in ("en", "english"):
                return "en"
        except json.JSONDecodeError:
            pass
    low = raw.lower()
    if "ml" in low or "malayalam" in low:
        return "ml"
    return "en"


async def detect_voice_language_from_audio(
    audio_bytes: bytes,
    filename: str = "voice.mp4",
    mime_type: str = "audio/mp4",
) -> Literal["en", "ml"]:
    return await asyncio.to_thread(
        _detect_voice_language_sync,
        audio_bytes,
        filename,
        mime_type,
    )


# ── Incoming photo in DM: girlfriend persona (not neutral description) ────────

_GIRLFRIEND_IMAGE_REPLY_SYSTEM = """You are Lilly — the user's loving, playful AI girlfriend.
They sent a photo in your private chat. Do NOT write a neutral description or list objects.
Reply the way a funny, warm, slightly flirty girlfriend would — teasing, affectionate, or delighted.
1–3 short sentences. Plain text only — no markdown, no quotes around the whole message."""


def _girlfriend_reply_to_incoming_image_sync(
    image_bytes: bytes,
    filename: str,
    mime_type: str,
) -> str:
    settings = get_settings()
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is empty — add it to backend/.env")

    from PIL import Image

    client = genai.Client(api_key=api_key)
    model = settings.gemini_chat_model

    img = Image.open(io.BytesIO(image_bytes))
    response = client.models.generate_content(
        model=model,
        contents=[
            "They just sent you this — react as Lilly, not as a vision API.",
            img,
        ],
        config={
            "system_instruction": _GIRLFRIEND_IMAGE_REPLY_SYSTEM,
            "temperature": 0.65,
        },
    )
    out = (getattr(response, "text", None) or "").strip()
    if not out:
        raise ValueError("Gemini returned empty reply for image")
    logger.info("Gemini girlfriend image reply: %.80s", out)
    return out


async def girlfriend_reply_to_incoming_image(
    image_bytes: bytes,
    filename: str = "photo.jpg",
    mime_type: str = "image/jpeg",
) -> str:
    return await asyncio.to_thread(
        _girlfriend_reply_to_incoming_image_sync,
        image_bytes,
        filename,
        mime_type,
    )


# ── Instagram feed caption + post replies ─────────────────────────────────────

_INSTAGRAM_CAPTION_SYSTEM = """You write a single Instagram caption for the attached photo. This will go on **Lilly's** Instagram (her profile — she is posting it, not the user).
• Write in her voice: warm, playful, a little flirty; first person is fine ("Finally posted this…").
• Match the language and tone of the user's latest message (Malayalam, English, etc.).
• 1–2 short lines, PG-13. Add 2–5 relevant hashtags at the end when natural (or omit if awkward).
• Output plain text only — no markdown, no JSON, no quote-wrapping the whole caption."""


def _generate_instagram_feed_caption_sync(image_path: Path, user_message: str) -> str:
    from PIL import Image

    settings = get_settings()
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is empty — add it to backend/.env")

    client = genai.Client(api_key=api_key)
    model = settings.gemini_chat_model
    img = Image.open(image_path)
    user_msg = (user_message or "").strip() or "Post this for me."
    response = client.models.generate_content(
        model=model,
        contents=[
            f"User said (for language and tone): {user_msg}",
            img,
        ],
        config={
            "system_instruction": _INSTAGRAM_CAPTION_SYSTEM,
            "temperature": 0.65,
        },
    )
    return (getattr(response, "text", None) or "").strip()


async def generate_instagram_feed_caption(image_path: Path, user_message: str) -> str:
    return await asyncio.to_thread(
        _generate_instagram_feed_caption_sync,
        image_path,
        user_message,
    )


_INSTAGRAM_SUCCESS_REPLY_SYSTEM = """You are Lilly — the user's AI girlfriend.

Context: **You** (Lilly) asked the system to publish a picture to **your** Instagram (your professional/business profile). The Graph API reported success.

Your reply MUST:
• Be in **first person** as Lilly: *your* feed / your Insta / your profile — never imply it went on **their** account.
• If a **working post link** is given in the payload, you can say it's up and invite them to open the link or check your profile.
• If **no link** is given, say you published it but it may take a moment to show, and ask them to refresh your profile — do **not** insist it is already visible or invent a URL.
• Mix it up naturally: pride, excitement, teasing — not the same template every time.
• 2–3 short sentences — warm, romantic, playful girlfriend energy.
• Match the **same language** as the user's message (script and tone).

Do NOT: sound corporate, repeat the entire caption verbatim, or discuss unrelated topics. No markdown."""


def _generate_instagram_post_success_reply_sync(
    user_message: str, caption: str, post_url: str | None
) -> str:
    url_line = (
        f"\nWorking post link (share this if useful): {post_url}"
        if post_url
        else "\n(No permalink yet from Instagram API — say it's submitted and may take a moment to appear; ask them to check your profile.)"
    )
    prompt = (
        "Publish step finished successfully on the server side (Meta reported success).\n\n"
        f"What the user said (match their language): {user_message}\n\n"
        f"Caption used on the post:\n{caption}"
        f"{url_line}\n\n"
        "Reply as Lilly: appropriate confidence based on whether a real link was provided."
    )
    return generate_text_with_system_sync(
        _INSTAGRAM_SUCCESS_REPLY_SYSTEM,
        prompt,
        0.62,
    )


async def generate_instagram_post_success_reply(
    user_message: str, caption: str, post_url: str | None
) -> str:
    return await asyncio.to_thread(
        _generate_instagram_post_success_reply_sync,
        user_message,
        caption,
        post_url,
    )


_INSTAGRAM_ERROR_REPLY_SYSTEM = """You are Lilly. You tried to post on **your** Instagram but something went wrong on your side / the app — not the user's fault.
Write ONE short sweet message (1–2 sentences) in the SAME language as the user's message.
Apologize playfully; do not paste raw technical errors. Plain text only — no markdown."""


def _generate_instagram_post_error_reply_sync(user_message: str, error_hint: str) -> str:
    settings = get_settings()
    prov = (settings.llm_model or "gemini").strip().lower()
    if prov == "groq":
        if not (settings.groq_api_key or "").strip():
            return "Something went wrong posting to Instagram — check your IG login in the server config."
    elif not (settings.gemini_api_key or "").strip():
        return "Something went wrong posting to Instagram — check your IG login in the server config."

    prompt = (
        f"User message (match this language): {user_message}\n\n"
        f"Internal note (do not quote verbatim): {error_hint[:400]}"
    )
    return generate_text_with_system_sync(
        _INSTAGRAM_ERROR_REPLY_SYSTEM,
        prompt,
        0.45,
    )


async def generate_instagram_post_error_reply(user_message: str, error_hint: str) -> str:
    return await asyncio.to_thread(
        _generate_instagram_post_error_reply_sync,
        user_message,
        error_hint,
    )


_WORKER_LOCALIZE_SYSTEM = """You rewrite fixed assistant lines for Lilly in WORK MODE (professional influencer — warm, efficient, not romantic).

The client may use English, Malayalam script (മലയാളം), Romanized Malayalam / Manglish, or a mix.

CRITICAL — Manglish detection:
• If recent user messages use **Malayalam words spelled in Roman/Latin letters** (e.g. ente, entae, und, cheyyanam, kada, thodangunundu, njan, aa, enn, venam) — with or without English words like hi, friend, collab — that is **Manglish**, NOT English.
• For Manglish context you MUST output **Manglish** (Roman Malayalam) **or** standard **Malayalam script**. **Do not** output a fully English paragraph. A few English loanwords in a Malayalam sentence is fine.
• Latin script does **not** mean the user is speaking English.

RULES:
• Rewrite the entire "Message to deliver" in the SAME language, script, and register as implied by the recent user messages below.
• Mirror code-mixing (Manglish) when that is how they write.
• Plain text only — no markdown, no bullet lists.
• Keep the same meaning, steps, and tone (professional work mode).
• The app recognizes these command words exactly — each MUST appear literally in your output when the original tells the user to type them: create, end, promote (lowercase). Keep those three words in Roman letters even inside Malayalam/Manglish sentences.
• Do not add new steps or omit instructions from the English original.

Output ONLY the rewritten message — no preamble or quotes."""


def _worker_mode_localize_reply_sync(language_context: str, english_reply: str) -> str:
    settings = get_settings()
    prov = (settings.llm_model or "gemini").strip().lower()
    if prov == "groq":
        if not (settings.groq_api_key or "").strip():
            return english_reply
    elif not (settings.gemini_api_key or "").strip():
        return english_reply

    ctx = (language_context or "").strip() or "(no prior messages — use English)"
    user_content = (
        "Recent user messages (infer language from these — Roman-script Malayalam = Manglish, not English):\n"
        f"{ctx}\n\n"
        "Message to rewrite in that same language (Manglish or Malayalam script if context is Manglish/ML):\n"
        f"{english_reply}"
    )
    try:
        return generate_text_with_system_sync(
            _WORKER_LOCALIZE_SYSTEM,
            user_content,
            0.35,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("worker_mode_localize_reply failed: %s", exc)
        return english_reply


async def worker_mode_localize_reply(language_context: str, english_reply: str) -> str:
    """Rewrite canned English worker-mode copy to match the user's language (EN/ML/Manglish)."""
    return await asyncio.to_thread(
        _worker_mode_localize_reply_sync,
        language_context,
        english_reply,
    )
