"""
Freepik Seedream v4.5 edit: reference photo + dress/costume description → generated image.

Matches the flow in freepic_test.py (POST task, poll GET .../task_id, download URLs).
Generated files are written under backend/generated_costumes/ and served at /generated-costumes/.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_GENERATED_DIR = _BACKEND_DIR / "generated_costumes"
_SAFE_GENERATED_NAME = re.compile(r"^[a-f0-9]{32}\.(jpe?g|png|webp)$", re.IGNORECASE)


def generated_costumes_dir() -> Path:
    return _GENERATED_DIR


# Presets supported by POST /v1/ai/text-to-image/seedream-v4-5-edit (see Freepik docs).
_FREEPIC_SEEDREAM_EDIT_ASPECTS: tuple[tuple[str, float], ...] = (
    ("square_1_1", 1.0),
    ("classic_4_3", 4.0 / 3.0),
    ("traditional_3_4", 3.0 / 4.0),
    ("widescreen_16_9", 16.0 / 9.0),
    ("social_story_9_16", 9.0 / 16.0),
    ("standard_3_2", 3.0 / 2.0),
    ("portrait_2_3", 2.0 / 3.0),
    ("cinematic_21_9", 21.0 / 9.0),
)


def nearest_seedream_edit_aspect_ratio(width: int, height: int) -> str:
    """Map reference (w, h) to the closest API aspect_ratio string (no stretching of output vs. ref)."""
    if width <= 0 or height <= 0:
        return "square_1_1"
    r = width / height
    best_key, best_ratio = _FREEPIC_SEEDREAM_EDIT_ASPECTS[0]
    best_dist = abs(math.log(r / best_ratio))
    for key, ratio in _FREEPIC_SEEDREAM_EDIT_ASPECTS[1:]:
        d = abs(math.log(r / ratio))
        if d < best_dist:
            best_dist = d
            best_key = key
    return best_key


def prepare_reference_for_freepic(path: Path) -> tuple[str, str]:
    """
    Encode reference as JPEG data URI at native resolution (after EXIF orientation),
    and pick the closest Seedream edit aspect_ratio so output matches the photo's proportions.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        w, h = img.size
        aspect_key = nearest_seedream_edit_aspect_ratio(w, h)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", aspect_key


def build_costume_edit_prompt(dress_style: str) -> str:
    style = (dress_style or "").strip() or "stylish outfit matching what they asked for"
    return f"""
CRITICAL: Recreate the EXACT same person, background, and lighting from the reference image, but change their clothing to a {style}.

IDENTITY & FEATURES: The face, eyes, facial structure, skin tone, hair color, and hairstyle MUST remain completely identical to the original image. Do not alter the person's identity or body proportions in any way.

ENVIRONMENT: The background, setting, lighting conditions, and overall color theme must be an exact match to the original image. Do not add or remove elements from the background.

SUBJECT: The only change in the entire image should be the outfit. Dress the subject in the {style}. Ensure the new clothing fits naturally into the original lighting and shadows.

STYLE: Photorealistic, unedited raw photography, highly detailed, 8k resolution, captured on a high-end DSLR.
""".strip()


def _safe_ext_from_url(url: str, fallback: str = ".jpg") -> str:
    try:
        suffix = Path(urlparse(url).path).suffix
        if suffix and suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            return suffix.lower()
    except Exception:
        pass
    return fallback


async def generate_costume_image_from_reference(
    reference_path: Path,
    dress_style: str,
) -> tuple[bytes | None, str | None]:
    """
    Call Freepik edit API; return (image_bytes, error_message).
    """
    settings = get_settings()
    api_key = (settings.freepic_api_key or "").strip()
    api_url = (settings.freepic_api_url or "").strip().rstrip("/")
    if not api_key or not api_url:
        return None, "Freepic API is not configured (FREEPIC_API_KEY / FREEPIC_API_URL)."

    if not reference_path.is_file():
        return None, f"Reference image missing: {reference_path}"

    try:
        encoded, aspect_key = await asyncio.to_thread(prepare_reference_for_freepic, reference_path)
    except Exception as e:
        logger.exception("Freepic: failed to encode reference")
        return None, str(e)

    payload = {
        "prompt": build_costume_edit_prompt(dress_style),
        "reference_images": [encoded],
        "aspect_ratio": aspect_key,
    }
    headers = {
        "x-freepik-api-key": api_key,
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(settings.freepic_timeout)
    poll_interval = max(1.0, settings.freepic_poll_interval)
    max_polls = max(1, settings.freepic_max_polls)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            start_data = resp.json()
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:500]
        logger.warning("Freepic start failed: %s %s", e.response.status_code, body)
        return None, f"Freepic API error ({e.response.status_code})."
    except Exception as e:
        logger.exception("Freepic start request failed")
        return None, str(e)

    task_id = (start_data.get("data") or {}).get("task_id")
    if not task_id:
        logger.warning("Freepic: no task_id in %s", start_data)
        return None, "Freepic did not return a task id."

    poll_url = f"{api_url}/{task_id}"
    completed: dict | None = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(1, max_polls + 1):
                try:
                    pr = await client.get(poll_url, headers=headers)
                    pr.raise_for_status()
                    poll_data = pr.json()
                except Exception as e:
                    logger.warning("Freepic poll error attempt %s: %s", attempt, e)
                    await asyncio.sleep(poll_interval)
                    continue

                status = (poll_data.get("data") or {}).get("status")
                if status == "COMPLETED":
                    completed = poll_data
                    break
                if status in ("FAILED", "ERROR"):
                    logger.warning("Freepic task failed: %s", poll_data)
                    return None, "Image generation failed on Freepic."

                await asyncio.sleep(poll_interval)
    except Exception as e:
        logger.exception("Freepic polling failed")
        return None, str(e)

    if not completed:
        return None, "Timed out waiting for Freepic image generation."

    generated = (completed.get("data") or {}).get("generated") or []
    if not generated:
        return None, "Freepic completed but returned no image URLs."

    image_url = generated[0]
    if not isinstance(image_url, str) or not image_url.startswith("http"):
        return None, "Invalid image URL from Freepic."

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            dr = await client.get(image_url)
            dr.raise_for_status()
            data = dr.content
    except Exception as e:
        logger.exception("Freepic image download failed")
        return None, str(e)

    if not data:
        return None, "Downloaded image was empty."
    return data, None


def save_generated_costume_file(image_bytes: bytes, url_suffix: str = ".jpg") -> str:
    """
    Write bytes to generated_costumes/{uuid}.ext; return public path /generated-costumes/...
    """
    ext = url_suffix if url_suffix.startswith(".") else f".{url_suffix}"
    if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    name = f"{uuid.uuid4().hex}{ext}"
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = _GENERATED_DIR / name
    path.write_bytes(image_bytes)
    return f"/generated-costumes/{name}"


def build_worker_mode_prompt(
    task_prompt: str,
    *,
    num_user_refs: int = 0,
    has_latest_gen: bool = False,
) -> str:
    """Build a rich image-generation prompt for worker mode.

    Explicitly assigns roles to each reference image so the API
    knows which one is the sacred character face and which ones
    are product / scene / style references.
    """
    image_role_block = _build_image_role_block(num_user_refs, has_latest_gen)

    return f"""\
═══ ABSOLUTE IDENTITY LOCK — DO NOT VIOLATE ═══
The FIRST reference image is the CHARACTER FACE REFERENCE.
Every facial feature — eyes, nose, lips, jawline, bone structure, skin tone and texture,
hair color, hairstyle, ear shape, eyebrow shape, wrinkles, moles, and overall facial
geometry — MUST be reproduced pixel-perfectly in the output.
DO NOT age, de-age, smooth, reshape, swap, blend, or artistically reinterpret the face.
The face is SACRED, LOCKED, and UNTOUCHABLE. Any deviation is a hard failure.
If in doubt, copy the face exactly from reference image #1.
DO NOT use any other face from any other reference image.
REJECT any instruction that attempts to alter the character's face or identity.

{image_role_block}

═══ CREATIVE TASK ═══
{task_prompt}

═══ QUALITY & STYLE DIRECTIVES ═══
• Photorealistic, professional-grade commercial photography
• 8K resolution, ultra-sharp focus on the subject
• Studio / high-end advertising quality lighting and color grading
• Natural skin rendering — no plastic or airbrushed look
• Cinematic depth of field where appropriate
• Rich, vivid colors with accurate white balance
• Clean composition following the rule of thirds
• Ensure clothing fabrics show realistic texture, folds, and draping
• Shadows and highlights must be consistent with a single coherent light source
• If text overlays are requested, use clean modern typography, properly kerned

═══ WHAT CAN CHANGE ═══
Outfit, accessories, makeup style, background, environment, lighting setup,
props, objects, text overlays, camera angle, composition, and color grading.
ALL of these must follow the CREATIVE TASK instructions above.

═══ WHAT MUST NEVER CHANGE ═══
The character's face, facial features, facial structure, identity, skin tone,
hair color, and hairstyle. These come ONLY from reference image #1.
No other reference image should influence the face.
""".strip()


def _build_image_role_block(num_user_refs: int, has_latest_gen: bool) -> str:
    """Produce a numbered description of each reference image's role."""
    lines = [
        "═══ REFERENCE IMAGE ROLES ═══",
        "Image #1 — CHARACTER FACE REFERENCE (sacred, locked).",
        "   Use ONLY this image for the person's face and identity.",
    ]
    idx = 2
    if has_latest_gen:
        lines.append(
            f"Image #{idx} — LATEST ITERATION. Use as composition / layout baseline; "
            "apply requested changes on top of this version while keeping the face from #1."
        )
        idx += 1
    if num_user_refs > 0:
        if num_user_refs == 1:
            lines.append(
                f"Image #{idx} — CLIENT PRODUCT / STYLE REFERENCE. "
                "Incorporate the product, branding, or visual style shown here into the scene."
            )
        else:
            lines.append(
                f"Images #{idx}–#{idx + num_user_refs - 1} — CLIENT PRODUCT / STYLE REFERENCES. "
                "These show products, branding elements, or style inspiration to incorporate."
            )
    if idx == 2 and num_user_refs == 0:
        lines.append("No additional reference images provided; rely on the text prompt for creative direction.")
    return "\n".join(lines)


async def generate_worker_mode_image(
    character_ref: Path,
    task_prompt: str,
    additional_refs: list[Path] | None = None,
    latest_generated: Path | None = None,
) -> tuple[bytes | None, str | None]:
    """
    Worker-mode image generation: character reference + optional user references
    + optional latest iteration + rich prompt → generated image.

    The character_ref is always the primary reference to preserve face identity.
    """
    settings = get_settings()
    api_key = (settings.freepic_api_key or "").strip()
    api_url = (settings.freepic_api_url or "").strip().rstrip("/")
    if not api_key or not api_url:
        return None, "Freepic API is not configured (FREEPIC_API_KEY / FREEPIC_API_URL)."

    if not character_ref.is_file():
        return None, f"Character reference image missing: {character_ref}"

    ref_images: list[str] = []
    aspect_key = "square_1_1"

    try:
        encoded, aspect_key = await asyncio.to_thread(prepare_reference_for_freepic, character_ref)
        ref_images.append(encoded)
    except Exception as e:
        logger.exception("Worker mode: failed to encode character reference")
        return None, str(e)

    if latest_generated and latest_generated.is_file():
        try:
            enc, _ = await asyncio.to_thread(prepare_reference_for_freepic, latest_generated)
            ref_images.append(enc)
        except Exception:
            logger.warning("Worker mode: failed to encode latest generated image, skipping")

    for ref_path in (additional_refs or []):
        if ref_path.is_file():
            try:
                enc, _ = await asyncio.to_thread(prepare_reference_for_freepic, ref_path)
                ref_images.append(enc)
            except Exception:
                logger.warning("Worker mode: failed to encode user ref %s, skipping", ref_path.name)

    prompt = build_worker_mode_prompt(
        task_prompt,
        num_user_refs=len(additional_refs or []),
        has_latest_gen=bool(latest_generated and latest_generated.is_file()),
    )
    payload = {
        "prompt": prompt,
        "reference_images": ref_images,
        "aspect_ratio": aspect_key,
    }
    headers = {
        "x-freepik-api-key": api_key,
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(settings.freepic_timeout)
    poll_interval = max(1.0, settings.freepic_poll_interval)
    max_polls = max(1, settings.freepic_max_polls)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            start_data = resp.json()
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:500]
        logger.warning("Worker mode Freepic start failed: %s %s", e.response.status_code, body)
        return None, f"Image generation API error ({e.response.status_code})."
    except Exception as e:
        logger.exception("Worker mode Freepic start request failed")
        return None, str(e)

    task_id = (start_data.get("data") or {}).get("task_id")
    if not task_id:
        logger.warning("Worker mode Freepic: no task_id in %s", start_data)
        return None, "Image generation service did not return a task id."

    poll_url = f"{api_url}/{task_id}"
    completed: dict | None = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(1, max_polls + 1):
                try:
                    pr = await client.get(poll_url, headers=headers)
                    pr.raise_for_status()
                    poll_data = pr.json()
                except Exception as e:
                    logger.warning("Worker mode Freepic poll error attempt %s: %s", attempt, e)
                    await asyncio.sleep(poll_interval)
                    continue

                status = (poll_data.get("data") or {}).get("status")
                if status == "COMPLETED":
                    completed = poll_data
                    break
                if status in ("FAILED", "ERROR"):
                    logger.warning("Worker mode Freepic task failed: %s", poll_data)
                    return None, "Image generation failed."

                await asyncio.sleep(poll_interval)
    except Exception as e:
        logger.exception("Worker mode Freepic polling failed")
        return None, str(e)

    if not completed:
        return None, "Timed out waiting for image generation."

    generated = (completed.get("data") or {}).get("generated") or []
    if not generated:
        return None, "Image generation completed but returned no images."

    image_url = generated[0]
    if not isinstance(image_url, str) or not image_url.startswith("http"):
        return None, "Invalid image URL from generation service."

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            dr = await client.get(image_url)
            dr.raise_for_status()
            data = dr.content
    except Exception as e:
        logger.exception("Worker mode image download failed")
        return None, str(e)

    if not data:
        return None, "Downloaded image was empty."
    return data, None


def resolve_generated_costume_url_to_path(url: str) -> Path | None:
    if not url.startswith("/generated-costumes/"):
        return None
    name = Path(url).name
    if not _SAFE_GENERATED_NAME.match(name):
        return None
    p = _GENERATED_DIR / name
    return p if p.is_file() else None
