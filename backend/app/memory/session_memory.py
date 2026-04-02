from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, TypedDict

# Last N chat messages (user + assistant turns) kept per session (web + Instagram).
MAX_MESSAGES_PER_SESSION = 10


Role = Literal["user", "assistant", "system"]


class Message(TypedDict):
    role: Role
    content: str


@dataclass
class SessionMemory:
    """
    Simple in‑memory, per‑session conversation history.
    In production, back this with Redis or a database.
    """

    _sessions: Dict[str, List[Message]] = field(default_factory=dict)

    def get_history(self, session_id: str) -> List[Message]:
        return self._sessions.get(session_id, [])

    def _trim(self, session_id: str) -> None:
        history = self._sessions.get(session_id)
        if not history or len(history) <= MAX_MESSAGES_PER_SESSION:
            return
        self._sessions[session_id] = history[-MAX_MESSAGES_PER_SESSION:]

    def append(self, session_id: str, message: Message) -> None:
        history = self._sessions.setdefault(session_id, [])
        history.append(message)
        self._trim(session_id)

