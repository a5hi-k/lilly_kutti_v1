import asyncio
import json
import logging
from typing import Any, Dict, List

from fastapi import WebSocket

from app.services import gemini_service
from app.services.worker_mode_state import add_reference_image, get_worker_session, is_worker_mode
from app.workflows.chat_graph import get_shared_workflow

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal(self, websocket: WebSocket, message: str) -> None:
        try:
            await websocket.send_text(message)
        except Exception:
            self.disconnect(websocket)

    async def broadcast(self, message: str) -> None:
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)


class MultiAgentOrchestrator:
    """
    Entry point for the orchestrator-based workflow.
    Routes all messages through the central Orchestrator agent.
    """

    def __init__(self) -> None:
        self._workflow = get_shared_workflow()

    async def handle_user_message(self, payload: Dict[str, Any]) -> str:
        session_id = str(payload.get("session_id") or "default")
        text = str(payload.get("content") or "").strip()
        img_raw = payload.get("image_base64")
        has_image = img_raw is not None and str(img_raw).strip() != ""

        if has_image:
            from app.services.session_latest_image import set_latest
            from app.services.user_image_upload import save_user_image_base64

            mime = str(payload.get("mime_type") or "image/jpeg")
            try:
                path = await asyncio.to_thread(
                    save_user_image_base64,
                    session_id,
                    str(img_raw),
                    mime,
                )
            except Exception as e:
                logger.exception("User image save failed")
                return json.dumps(
                    {
                        "kind": "assistant",
                        "content": f"I couldn't read that image. ({e})",
                    }
                )

            if is_worker_mode(session_id):
                add_reference_image(session_id, path)
                if not text:
                    ws = get_worker_session(session_id)
                    n = len(ws.reference_images)
                    ack_en = (
                        f"Reference image #{n} saved. "
                        "You can send more images or describe your requirements."
                    )
                    anchor = self._workflow.worker_language_anchor(session_id, "")
                    ack = await gemini_service.worker_mode_localize_reply(anchor, ack_en)
                    return json.dumps({
                        "kind": "assistant",
                        "content": ack,
                        "ui_event": {"type": "worker_mode_active"},
                    })
                text = f"{text}\n\n[Client attached reference image #{len(get_worker_session(session_id).reference_images)}.]"
            else:
                set_latest(session_id, path)
                if not text:
                    from app.services import gemini_service

                    suf = path.suffix.lower()
                    mime_for_g = "image/jpeg"
                    if suf == ".png":
                        mime_for_g = "image/png"
                    elif suf == ".webp":
                        mime_for_g = "image/webp"
                    raw = path.read_bytes()
                    reply = await gemini_service.girlfriend_reply_to_incoming_image(
                        raw,
                        filename=path.name,
                        mime_type=mime_for_g,
                    )
                    await self._workflow.record_exchange(
                        session_id=session_id,
                        user_content="[User sent a photo]",
                        assistant_content=reply,
                    )
                    return json.dumps({"kind": "assistant", "content": reply})
                text = f"{text}\n\n[They attached a photo in this chat.]"

        if not text:
            return ""

        state = await self._workflow.run_turn_state(session_id=session_id, user_text=text)
        reply = state.get("last_assistant") or ""

        intent = state.get("intent", "chat")
        graph_ui_event = state.get("ui_event")

        if graph_ui_event:
            evt = dict(graph_ui_event)
            img = state.get("image_url")
            if img:
                evt["image_url"] = img
            return json.dumps({"kind": "assistant", "content": reply, "ui_event": evt})

        if intent == "video_call":
            return json.dumps(
                {
                    "kind": "assistant",
                    "content": reply,
                    "ui_event": {"type": "video_call_start"},
                }
            )

        if intent == "post_instagram":
            return json.dumps(
                {
                    "kind": "assistant",
                    "content": reply,
                    "ui_event": {"type": "instagram_posted"},
                }
            )

        if intent in ("share_photo", "costume_tryon"):
            ui: dict = {"type": "share_photo"}
            img = state.get("image_url")
            if img:
                ui["image_url"] = img
            return json.dumps(
                {
                    "kind": "assistant",
                    "content": reply,
                    "ui_event": ui,
                }
            )

        return json.dumps({"kind": "assistant", "content": reply})


# Single instance shared by WebSocket chat and optional Instagram DM listener.
orchestrator = MultiAgentOrchestrator()
