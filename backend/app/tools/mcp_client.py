from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MCPResult:
    target: str
    summary: str
    raw: Any | None = None


async def call_mcp(target: str, args: Dict[str, Any] | None = None) -> MCPResult:
    """
    Placeholder for MCP / external integrations.
    In a real system, this would:
    - Look up the configured MCP server or client by `target`
    - Execute the appropriate command
    - Return a structured result
    """
    summary = (
        f"I would talk to the MCP server '{target}' with your request and then "
        "bring the results back into our conversation, but right now it is only a stub."
    )
    return MCPResult(target=target, summary=summary, raw=args or {})

