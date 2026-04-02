from __future__ import annotations

from typing import Dict, List, Literal, TypedDict

from app.agents.base_agent import BaseAgent
from app.memory.session_memory import MAX_MESSAGES_PER_SESSION


RouterIntent = Literal["chat", "tool", "mcp", "human_review", "video_call"]


ROUTER_SYSTEM_PROMPT = """
You are an intent router for Lilly, an AI girlfriend.

Your job is to look at the user's latest message and decide:
- whether this is normal romantic / casual conversation (chat),
- or a request that should use a TOOL (calendar, reminders, music, small utilities),
- or a request that should use an MCP / external service (e.g. some backend system),
- or a request to start a video call simulation UI (video_call),
- or something that should be escalated to a human reviewer.

You must respond with STRICT JSON only, no extra text.

Schema:
{{
  "intent": "chat" | "tool" | "mcp" | "human_review" | "video_call",
  "tool_name": "reminder" | "calendar" | "music" | null,
  "tool_args": {{ ... free form JSON, arguments for the tool ... }},
  "mcp_target": "name-of-mcp-server-or-resource" | null,
  "reason": "short natural language explanation"
}}

Guidelines:
- If the user is just flirting, venting, or chatting -> intent = "chat".
- If the user is asking to call / video call / meet online / "I want to see you now" (including semantic equivalents) -> intent = "video_call".
- If the user clearly asks to remember something, set a reminder, keep track of a task or preference -> intent = "tool", tool_name = "reminder".
- If the user asks about dates/times, schedules, or appointments -> intent = "tool", tool_name = "calendar".
- If the user asks for songs, playlists, or mood music suggestions -> intent = "tool", tool_name = "music".
- If the user explicitly says to call an MCP server or external system, or refers to system-level actions -> intent = "mcp" and set mcp_target accordingly.
- If the user requests something safety‑sensitive or that requires manual approval -> intent = "human_review".
""".strip()


class RouterDecision(TypedDict):
    intent: RouterIntent
    tool_name: str | None
    tool_args: Dict | None
    mcp_target: str | None
    reason: str


class RouterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="router", system_prompt=ROUTER_SYSTEM_PROMPT, temperature=0.2)

    async def run(
        self,
        *,
        history: List[Dict[str, str]],
        user_input: str,
        session_id: str,
    ) -> RouterDecision:
        history_text = "\n".join(
            [
                f"{m.get('role','user')}: {m.get('content','')}"
                for m in history[-MAX_MESSAGES_PER_SESSION:]
            ]
        )

        raw = (await self._generate_text(history_text, user_input)).strip()

        # Be defensive around JSON parsing; the model may wrap in ```json ... ```
        import json
        import re
        # Strip ```json ... ``` if present
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if m:
            raw = m.group(1).strip()
        try:
            data = json.loads(raw)
        except Exception:
            # Fallback to plain chat if parsing fails.
            return {
                "intent": "chat",
                "tool_name": None,
                "tool_args": None,
                "mcp_target": None,
                "reason": "Failed to parse router JSON, defaulting to chat.",
            }

        intent: RouterIntent = data.get("intent", "chat")
        return {
            "intent": intent,
            "tool_name": data.get("tool_name"),
            "tool_args": data.get("tool_args") or None,
            "mcp_target": data.get("mcp_target"),
            "reason": data.get("reason", ""),
        }

