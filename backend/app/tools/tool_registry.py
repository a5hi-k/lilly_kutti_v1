from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict


@dataclass
class ToolResult:
    name: str
    summary: str
    raw: Any | None = None


def reminder_tool(args: Dict[str, Any]) -> ToolResult:
    text = str(args.get("text") or args.get("message") or "")
    when = str(args.get("when") or "sometime soon")
    if not text:
        text = "something sweet you asked me to remember"
    summary = f"I’ll remember this for you around {when}: {text}"
    return ToolResult(name="reminder", summary=summary, raw={"text": text, "when": when})


def calendar_tool(args: Dict[str, Any]) -> ToolResult:
    date = str(args.get("date") or args.get("day") or "soon")
    what = str(args.get("event") or args.get("title") or "a little moment together")
    summary = f"I would add a calendar entry on {date} for: {what}."
    return ToolResult(name="calendar", summary=summary, raw={"date": date, "event": what})


def music_tool(args: Dict[str, Any]) -> ToolResult:
    mood = str(args.get("mood") or "romantic")
    summary = (
        f"For a {mood} mood, I’d put on soft lo‑fi or gentle acoustic tracks. "
        "Imagine us sharing headphones while it plays."
    )
    return ToolResult(name="music", summary=summary, raw={"mood": mood})


TOOL_REGISTRY: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
    "reminder": reminder_tool,
    "calendar": calendar_tool,
    "music": music_tool,
}


def run_tool(name: str, args: Dict[str, Any] | None) -> ToolResult:
    func = TOOL_REGISTRY.get(name)
    if not func:
        return ToolResult(
            name=name,
            summary="I wanted to use a tool for you, but that one is not wired up yet.",
            raw=None,
        )
    return func(args or {})

