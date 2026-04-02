from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal

from app.core.config import get_settings
from app.services.llm_text import generate_text_with_system_sync


Role = Literal["user", "assistant", "system"]


class BaseAgent(ABC):
    """
    Base class for orchestrator / router / persona agents.
    Text generation uses LLM_MODEL: gemini (google-genai) or groq (GROQ_*).
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        temperature: float = 0.4,
    ) -> None:
        self.name = name
        self.settings = get_settings()
        self._system_prompt = system_prompt
        self._temperature = temperature

    def _gemini_generate_sync(self, history_text: str, user_input: str) -> str:
        user_content = f"Context:\n{history_text}\n\nUser: {user_input}"
        return generate_text_with_system_sync(
            self._system_prompt,
            user_content,
            self._temperature,
        )

    async def _generate_text(self, history_text: str, user_input: str) -> str:
        """Run LLM text generation without blocking the event loop."""
        return await asyncio.to_thread(
            self._gemini_generate_sync,
            history_text,
            user_input,
        )

    @abstractmethod
    async def run(
        self,
        *,
        history: List[Dict[str, str]],
        user_input: str,
        session_id: str,
    ) -> Any:
        """Execute the agent; return type depends on the subclass."""
