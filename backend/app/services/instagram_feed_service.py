"""Post the session's latest image to the Instagram feed via Meta Graph API."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.services import gemini_service
from app.services.session_latest_image import clear_latest, get_latest_path
from app.services.user_image_upload import session_uploads_dir

logger = logging.getLogger(__name__)

# Instagram container must finish processing before media_publish; otherwise error 9007
# ("Media ID is not available") is common even for images on slow or loaded CDN checks.
_CONTAINER_POLL_INTERVAL_SEC = 2.0
_CONTAINER_POLL_MAX_SEC = 120.0
# Meta often returns HTTP 500 + OAuthException code 2, is_transient — must retry.
_IG_CREATE_MAX_ATTEMPTS = 6
_IG_PUBLISH_MAX_ATTEMPTS = 5
_IG_RETRY_BACKOFF_START = 2.0
_IG_RETRY_BACKOFF_MAX = 28.0
# Instagram caption hard limit
_IG_CAPTION_MAX_CHARS = 2200


def _truncate_ig_caption(caption: str, max_chars: int = _IG_CAPTION_MAX_CHARS) -> str:
    if len(caption) <= max_chars:
        return caption
    return caption[: max_chars - 1].rstrip() + "…"


def _ig_error_is_transient(data: dict, http_status: int) -> bool:
    if http_status >= 500 or http_status == 429:
        return True
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return False
    if err.get("is_transient") is True:
        return True
    # Meta "unexpected error — please retry" (common on create media)
    if err.get("code") == 2:
        return True
    return False


async def _stage_generated_image_for_ig_fetch(path: Path) -> tuple[Path, Path | None]:
    """Copy generated_costumes files into session_uploads so IG fetches the same URL pattern as DM photos."""
    from app.services.freepic_service import generated_costumes_dir

    try:
        if path.resolve().parent != generated_costumes_dir().resolve():
            return path, None
    except OSError:
        return path, None
    suffix = path.suffix if path.suffix else ".jpg"
    dest = session_uploads_dir() / f"ig_feed_{uuid.uuid4().hex}{suffix}"

    def _copy() -> None:
        shutil.copy2(path, dest)

    await asyncio.to_thread(_copy)
    logger.info(
        "Instagram: staged generated image for Meta fetch (session_uploads): %s",
        dest.name,
    )
    return dest, dest


async def _ig_post_form_with_retry(
    client: httpx.AsyncClient,
    url: str,
    data: dict,
    *,
    label: str,
    max_attempts: int,
) -> tuple[dict, int]:
    """POST application/x-www-form-urlencoded; retry on transient Meta failures."""
    backoff = _IG_RETRY_BACKOFF_START
    last_body: dict = {}
    last_status = 0
    for attempt in range(max_attempts):
        resp = await client.post(url, data=data)
        last_status = resp.status_code
        try:
            last_body = resp.json()
        except Exception:
            last_body = {}
        ok = (
            resp.status_code < 400
            and "error" not in last_body
            and bool(last_body.get("id"))
        )
        if ok:
            return last_body, last_status
        if attempt < max_attempts - 1 and _ig_error_is_transient(last_body, resp.status_code):
            logger.warning(
                "%s transient failure (%s/%s) HTTP=%s err=%s — retry in %.1fs",
                label,
                attempt + 1,
                max_attempts,
                resp.status_code,
                last_body.get("error"),
                backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, _IG_RETRY_BACKOFF_MAX)
            continue
        return last_body, last_status
    return last_body, last_status


async def _wait_for_ig_media_container_ready(
    client: httpx.AsyncClient,
    *,
    container_id: str,
    api_version: str,
    token: str,
) -> None:
    """Poll Graph API until the media container is ready to publish."""
    deadline = time.monotonic() + _CONTAINER_POLL_MAX_SEC
    url = f"https://graph.facebook.com/{api_version}/{container_id}"
    while time.monotonic() < deadline:
        st_resp = await client.get(
            url,
            params={"fields": "status_code", "access_token": token},
        )
        st_data = st_resp.json()
        if "error" in st_data:
            raise RuntimeError(st_data["error"].get("message", "Unknown error"))
        code = (st_data.get("status_code") or "").upper()
        if code == "FINISHED":
            return
        if code == "ERROR":
            raise RuntimeError("Instagram media container processing failed (status ERROR).")
        if code == "EXPIRED":
            raise RuntimeError("Instagram media container expired before publish.")
        await asyncio.sleep(_CONTAINER_POLL_INTERVAL_SEC)
    raise RuntimeError(
        "Instagram media container did not become ready in time; try again.",
    )


async def _fetch_ig_published_permalink(
    client: httpx.AsyncClient,
    *,
    ig_media_id: str,
    api_version: str,
    token: str,
) -> str | None:
    """Resolve web URL for a published IG media; media_publish id is NOT the /p/ shortcode."""
    url = f"https://graph.facebook.com/{api_version}/{ig_media_id}"
    delay = 1.0
    for attempt in range(6):
        resp = await client.get(
            url,
            params={"fields": "permalink,shortcode", "access_token": token},
        )
        try:
            data = resp.json()
        except Exception:
            data = {}
        if "error" in data:
            logger.warning(
                "Instagram permalink fetch attempt %s: %s",
                attempt + 1,
                data.get("error", {}).get("message", data),
            )
        else:
            per = (data.get("permalink") or "").strip()
            if per:
                return per
            sc = (data.get("shortcode") or "").strip()
            if sc:
                return f"https://www.instagram.com/p/{sc}/"
        await asyncio.sleep(delay)
        delay = min(delay + 1.0, 4.0)
    return None


def _path_is_under_session_uploads(path: Path) -> bool:
    try:
        path.resolve().relative_to(session_uploads_dir().resolve())
    except ValueError:
        return False
    return True


async def post_latest_image_to_instagram_feed(session_id: str, user_message: str) -> str:
    """Upload the latest tracked image with a Gemini caption; return Lilly's reply for the user."""
    path = get_latest_path(session_id)
    if not path or not path.is_file():
        return ""

    caption = await gemini_service.generate_instagram_feed_caption(path, user_message)
    if not caption.strip():
        caption = "💕"
    caption = _truncate_ig_caption(caption.strip())

    settings = get_settings()
    token = (settings.ig_graph_access_token or "").strip()
    ig_uid = (settings.ig_graph_user_id or "").strip()
    pub_base = (settings.ig_dm_public_base_url or "").strip().rstrip("/")

    if not token or not ig_uid:
        return await gemini_service.generate_instagram_post_error_reply(
            user_message, "IG_GRAPH_ACCESS_TOKEN or IG_GRAPH_USER_ID not set",
        )

    from app.services.instagram_dm_service import _relative_url_for_served_file

    if not pub_base:
        return await gemini_service.generate_instagram_post_error_reply(
            user_message,
            "IG_DM_PUBLIC_BASE_URL not set — Graph API needs a public image URL to post.",
        )

    api_version = settings.ig_graph_api_version or "v25.0"
    staging_path: Path | None = None

    try:
        path_for_url, staging_path = await _stage_generated_image_for_ig_fetch(path)
        rel = _relative_url_for_served_file(path_for_url)
        if not rel:
            return await gemini_service.generate_instagram_post_error_reply(
                user_message,
                "Image path could not be mapped to a public URL for Instagram.",
            )
        image_url = f"{pub_base}{rel}"

        async with httpx.AsyncClient(timeout=90.0) as client:
            create_url = f"https://graph.facebook.com/{api_version}/{ig_uid}/media"
            create_data, create_status = await _ig_post_form_with_retry(
                client,
                create_url,
                {
                    "image_url": image_url,
                    "caption": caption,
                    "access_token": token,
                },
                label="Instagram create media",
                max_attempts=_IG_CREATE_MAX_ATTEMPTS,
            )
            if create_status >= 400:
                logger.warning(
                    "Instagram create media HTTP %s: %s",
                    create_status,
                    create_data,
                )
            if "error" in create_data:
                raise RuntimeError(create_data["error"].get("message", "Unknown error"))
            creation_id = create_data.get("id")
            if not creation_id:
                raise RuntimeError("Instagram create media returned no container id.")

            await _wait_for_ig_media_container_ready(
                client,
                container_id=str(creation_id),
                api_version=api_version,
                token=token,
            )

            # Meta has pulled the image; safe to drop staging copy (worker-mode generated file).
            if staging_path is not None and staging_path.is_file():
                try:
                    await asyncio.to_thread(staging_path.unlink)
                except OSError as exc:
                    logger.warning("Could not remove staged IG file %s: %s", staging_path, exc)
                staging_path = None

            pub_url = f"https://graph.facebook.com/{api_version}/{ig_uid}/media_publish"
            pub_data, pub_status = await _ig_post_form_with_retry(
                client,
                pub_url,
                {
                    "creation_id": creation_id,
                    "access_token": token,
                },
                label="Instagram media_publish",
                max_attempts=_IG_PUBLISH_MAX_ATTEMPTS,
            )
            if pub_status >= 400:
                logger.warning(
                    "Instagram media_publish HTTP %s: %s",
                    pub_status,
                    pub_data,
                )
            if "error" in pub_data:
                raise RuntimeError(pub_data["error"].get("message", "Unknown error"))

            ig_media_id = pub_data.get("id")
            if not ig_media_id:
                raise RuntimeError(
                    "Instagram media_publish returned no media id — post was not created.",
                )

            post_url = await _fetch_ig_published_permalink(
                client,
                ig_media_id=str(ig_media_id),
                api_version=api_version,
                token=token,
            )
            if post_url:
                logger.info("Instagram feed post published media_id=%s url=%s", ig_media_id, post_url)
            else:
                logger.warning(
                    "Instagram media_publish returned id=%s but permalink not available yet; "
                    "post may still appear on the profile shortly.",
                    ig_media_id,
                )

        reply = await gemini_service.generate_instagram_post_success_reply(
            user_message, caption, post_url,
        )

        if _path_is_under_session_uploads(path):
            try:
                await asyncio.to_thread(path.unlink)
            except OSError as exc:
                logger.warning("Could not delete session upload %s: %s", path, exc)
        clear_latest(session_id)
        return reply
    except Exception as e:
        logger.exception("Instagram feed upload failed")
        return await gemini_service.generate_instagram_post_error_reply(user_message, str(e))
    finally:
        if staging_path is not None and staging_path.is_file():
            try:
                await asyncio.to_thread(staging_path.unlink)
            except OSError:
                pass