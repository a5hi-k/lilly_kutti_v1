"""Random reference images from backend/ReferImage (served at /ref-assets/)."""

from __future__ import annotations

import random
import re
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REF_DIR = _BACKEND_DIR / "ReferImage"
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*\.(jpe?g|png|webp)$")
_ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".webp"}


def refer_image_dir() -> Path:
    return _REF_DIR


def list_reference_images() -> list[Path]:
    if not _REF_DIR.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(_REF_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in _ALLOWED_SUFFIX and _SAFE_NAME.match(p.name):
            out.append(p)
    return out


def pick_random_reference_image() -> Path | None:
    files = list_reference_images()
    if not files:
        return None
    return random.choice(files)


def ref_asset_public_url(filename: str) -> str | None:
    if not _SAFE_NAME.match(filename):
        return None
    return f"/ref-assets/{filename}"


def resolve_ref_asset_url_to_path(url: str) -> Path | None:
    """Map `/ref-assets/foo.jpg` to a file under ReferImage if it exists."""
    if not url.startswith("/ref-assets/"):
        return None
    name = Path(url).name
    if not _SAFE_NAME.match(name):
        return None
    p = _REF_DIR / name
    return p if p.is_file() else None


def resolve_public_image_url_to_path(url: str) -> Path | None:
    """Map `/ref-assets/...` or `/generated-costumes/...` to a local file on disk."""
    p = resolve_ref_asset_url_to_path(url)
    if p is not None:
        return p
    from app.services.freepic_service import resolve_generated_costume_url_to_path

    return resolve_generated_costume_url_to_path(url)


def resolve_dm_photo_url_to_path(url: str) -> Path | None:
    """Reference stills (`/ref-assets/`) or Freepic costume outputs (`/generated-costumes/`)."""
    return resolve_public_image_url_to_path(url)
