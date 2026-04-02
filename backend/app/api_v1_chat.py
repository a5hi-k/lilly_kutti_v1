import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.chat_manager import ConnectionManager, orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()
manager = ConnectionManager()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            try:
                raw = await websocket.receive()
            except RuntimeError as e:
                if "receive" in str(e) and "disconnect" in str(e).lower():
                    break  # Client disconnected; exit cleanly
                raise
            # Support both JSON and plain text
            if "text" not in raw:
                continue

            text = raw["text"]
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                data = {"content": text}

            if isinstance(data, str):
                payload = {"content": data}
            else:
                payload = data

            try:
                reply = await orchestrator.handle_user_message(payload)
                if not reply:
                    continue
                await manager.send_personal(websocket, reply)
            except Exception as e:
                logger.exception("Error processing message: %s", e)
                err_msg = f"Something went wrong. Please try again. ({type(e).__name__})"
                try:
                    await manager.send_personal(websocket, err_msg)
                except Exception:
                    manager.disconnect(websocket)
                    raise
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "receive" not in str(e) or "disconnect" not in str(e).lower():
            logger.exception("WebSocket error: %s", e)
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        manager.disconnect(websocket)

