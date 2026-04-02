"""Per-session pointer to the latest image (user upload or assistant-shared) for Instagram posting."""

from __future__ import annotations

import threading
from pathlib import Path

_lock = threading.Lock()
_latest: dict[str, str] = {}


def set_latest(session_id: str, path: Path) -> None:
    """Remember the local file for the most recent image in this session."""
    with _lock:
        _latest[str(session_id)] = str(path.resolve())


def get_latest_path(session_id: str) -> Path | None:
    """Return the path if it still exists, else None."""
    with _lock:
        p = _latest.get(str(session_id))
    if not p:
        return None
    path = Path(p)
    return path if path.is_file() else None


def clear_latest(session_id: str) -> None:
    """Forget the stored latest image path for this session (e.g. after posting or cleanup)."""
    with _lock:
        _latest.pop(str(session_id), None)
