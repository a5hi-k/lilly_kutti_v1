"""Per-session worker-mode state machine.

Tracks the multi-step onboarding + image-generation workflow
when Lilly operates as a professional influencer / fashion model.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List


class WorkerStep(str, Enum):
    COLLECTING_NAME = "collecting_name"
    COLLECTING_EMAIL = "collecting_email"
    COLLECTING_PAYMENT = "collecting_payment"
    COLLECTING_TASK = "collecting_task"
    READY_TO_CREATE = "ready_to_create"
    GENERATING = "generating"
    ITERATING = "iterating"


@dataclass
class WorkerSession:
    active: bool = False
    step: WorkerStep = WorkerStep.COLLECTING_NAME
    client_name: str | None = None
    client_email: str | None = None
    transaction_id: str | None = None
    task_description: str | None = None
    refined_prompt: str | None = None
    reference_images: List[Path] = field(default_factory=list)
    character_ref_image: Path | None = None
    latest_generated: Path | None = None
    all_generated_images: List[Path] = field(default_factory=list)
    iteration_history: List[str] = field(default_factory=list)


_lock = threading.Lock()
_sessions: Dict[str, WorkerSession] = {}


def get_worker_session(session_id: str) -> WorkerSession:
    with _lock:
        sess = _sessions.get(session_id)
        if sess is None:
            sess = WorkerSession()
            _sessions[session_id] = sess
        return sess


def activate_worker_mode(session_id: str) -> WorkerSession:
    with _lock:
        sess = _sessions.get(session_id)
        if sess is None:
            sess = WorkerSession()
            _sessions[session_id] = sess
        sess.active = True
        sess.step = WorkerStep.COLLECTING_NAME
        return sess


def deactivate_worker_mode(session_id: str) -> List[Path]:
    """Deactivate and return list of ALL tracked images (generated + user refs) for cleanup."""
    with _lock:
        sess = _sessions.pop(session_id, None)
        if sess is None:
            return []
        images = list(sess.all_generated_images) + list(sess.reference_images)
        return images


def is_worker_mode(session_id: str) -> bool:
    with _lock:
        sess = _sessions.get(session_id)
        return sess is not None and sess.active


def update_step(session_id: str, step: WorkerStep) -> None:
    with _lock:
        sess = _sessions.get(session_id)
        if sess:
            sess.step = step


def add_reference_image(session_id: str, path: Path) -> None:
    with _lock:
        sess = _sessions.get(session_id)
        if sess:
            sess.reference_images.append(path)


def set_latest_generated(session_id: str, path: Path) -> None:
    with _lock:
        sess = _sessions.get(session_id)
        if sess:
            sess.latest_generated = path
            sess.all_generated_images.append(path)


def cleanup_generated_images(images: List[Path]) -> None:
    """Delete generated images from disk."""
    for p in images:
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass
