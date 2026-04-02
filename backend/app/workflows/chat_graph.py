"""
Orchestrator-centric chat workflow.

Flow:  orchestrator → (conditional: tool?) → tool_node → finalize
                                           ↘ finalize → END

Worker-mode bypass: when a session is in worker_mode the normal graph is
skipped and a dedicated worker pipeline handles message processing.

The OrchestratorAgent handles intent classification, personality, and
response generation in a single pass.  Tools are executed as side-effects
when the orchestrator requests them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.orchestrator import OrchestratorAgent, OrchestratorDecision
from app.agents.worker_agent import WorkerAgent
from app.core.config import get_settings
from app.memory.session_memory import Message, SessionMemory
from app.services import freepic_service, gemini_service
from app.services.instagram_feed_service import post_latest_image_to_instagram_feed
from app.services.reference_image_service import (
    pick_random_reference_image,
    ref_asset_public_url,
    resolve_public_image_url_to_path,
)
from app.services.session_latest_image import get_latest_path, set_latest
from app.services.worker_mode_state import (
    WorkerSession,
    WorkerStep,
    activate_worker_mode,
    cleanup_generated_images,
    deactivate_worker_mode,
    get_worker_session,
    is_worker_mode,
    set_latest_generated,
    update_step,
)
from app.tools import ToolResult, run_tool

logger = logging.getLogger(__name__)


def _user_requests_worker_instagram_promo(user_text: str) -> bool:
    """True when the client wants the latest work-mode image posted to Lilly's Instagram."""
    t = user_text.strip().lower()
    if not t:
        return False
    if any(
        neg in t
        for neg in (
            "don't",
            "do not",
            "dont ",
            "no thanks",
            "not interested",
            "don't want",
            "dont want",
            "skip",
        )
    ):
        return False
    if t in ("no", "nope", "nah"):
        return False

    exact = frozenset({
        "promote",
        "post instagram",
        "post to instagram",
        "post on instagram",
        "share on instagram",
        "instagram",
        "please promote",
        "please post",
    })
    if t in exact:
        return True

    if "instagram" in t and any(
        w in t for w in ("post", "share", "promote", "upload", "publish")
    ):
        return True
    if "promote" in t and any(w in t for w in ("instagram", "ig ", " ig", "feed")):
        return True
    if t.startswith("yes") and any(
        w in t for w in ("instagram", "post", "promote", "share")
    ):
        return True
    return False


Intent = Literal[
    "chat",
    "tool",
    "video_call",
    "share_photo",
    "costume_tryon",
    "post_instagram",
    "worker_mode",
]


class ChatState(TypedDict):
    session_id: str
    messages: List[Message]
    last_user: str
    last_assistant: str | None
    intent: Intent
    tool_result: str | None
    tool_name: str | None
    tool_args: Dict | None
    image_url: str | None
    ui_event: Dict[str, Any] | None


class ChatWorkflow:
    """
    Orchestrator-centric LangGraph workflow.

    The orchestrator decides intent, generates the personality-driven response,
    and optionally routes to tool execution — all in one streamlined graph.
    Worker-mode sessions are handled through a dedicated pipeline.
    """

    def __init__(self, memory: SessionMemory | None = None) -> None:
        self._memory = memory or SessionMemory()
        self._orchestrator = OrchestratorAgent()
        self._worker_agent = WorkerAgent()
        self._graph = self._build_graph()
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _session_turn_lock(self, session_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    async def record_exchange(
        self,
        *,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """
        Append one user/assistant pair with the same locking as graph turns
        (e.g. Instagram image DMs that skip the LangGraph pipeline).
        """
        lock = await self._session_turn_lock(session_id)
        async with lock:
            self._memory.append(session_id, {"role": "user", "content": user_content})
            self._memory.append(session_id, {"role": "assistant", "content": assistant_content})

    def worker_language_anchor(self, session_id: str, latest_user_text: str = "") -> str:
        """Recent user lines so the localizer can infer English / Malayalam / Manglish."""
        hist = self._memory.get_history(session_id)
        chunks: list[str] = []
        for m in hist[-15:]:
            if m.get("role") == "user":
                t = (m.get("content") or "").strip()
                if t:
                    chunks.append(t)
        tail = (latest_user_text or "").strip()
        if tail:
            chunks.append(tail)
        joined = "\n".join(chunks[-6:])
        return joined if joined else tail or "(no prior messages — use English)"

    async def _worker_localize(self, session_id: str, latest_user_text: str, english_reply: str) -> str:
        return await gemini_service.worker_mode_localize_reply(
            self.worker_language_anchor(session_id, latest_user_text),
            english_reply,
        )

    def _build_graph(self):
        graph = StateGraph(ChatState)

        async def orchestrator_node(state: ChatState) -> ChatState:
            session_id = state["session_id"]
            history = self._memory.get_history(session_id)

            decision: OrchestratorDecision = await self._orchestrator.run(
                history=history,
                user_input=state["last_user"],
                session_id=session_id,
            )

            if decision["intent"] == "post_instagram":
                lp = get_latest_path(session_id)
                if lp is not None and lp.is_file():
                    posted_text = await post_latest_image_to_instagram_feed(
                        session_id, state["last_user"]
                    )
                    assistant_text = posted_text
                else:
                    assistant_text = decision["response"]
                return {
                    **state,
                    "intent": "post_instagram",
                    "tool_name": None,
                    "tool_args": None,
                    "last_assistant": assistant_text,
                    "image_url": None,
                    "ui_event": None,
                }

            image_url: str | None = None
            if decision["intent"] == "share_photo":
                path = pick_random_reference_image()
                if path is not None:
                    image_url = ref_asset_public_url(path.name)

            elif decision["intent"] == "costume_tryon":
                settings = get_settings()
                ref = pick_random_reference_image()
                dress = (decision.get("dress_style") or "").strip()
                if not dress:
                    dress = "stylish outfit matching what they asked for"
                if ref and (settings.freepic_api_key or "").strip():
                    img_bytes, err = await freepic_service.generate_costume_image_from_reference(
                        ref, dress
                    )
                    if img_bytes:
                        image_url = await asyncio.to_thread(
                            freepic_service.save_generated_costume_file,
                            img_bytes,
                            ".jpg",
                        )
                    elif err:
                        logger.warning("costume_tryon: %s", err)
                elif not ref:
                    logger.warning("costume_tryon: no files in ReferImage")
                elif not (settings.freepic_api_key or "").strip():
                    logger.warning("costume_tryon: FREEPIC_API_KEY not set")

            if decision["intent"] == "worker_mode":
                activate_worker_mode(session_id)
                return {
                    **state,
                    "intent": "worker_mode",
                    "tool_name": None,
                    "tool_args": None,
                    "last_assistant": decision["response"],
                    "image_url": None,
                    "ui_event": {"type": "worker_mode_enter"},
                }

            if image_url:
                p = resolve_public_image_url_to_path(image_url)
                if p is not None:
                    set_latest(session_id, p)

            return {
                **state,
                "intent": decision["intent"],
                "tool_name": decision.get("tool_name"),
                "tool_args": decision.get("tool_args") or None,
                "last_assistant": decision["response"],
                "image_url": image_url,
                "ui_event": None,
            }

        async def tool_node(state: ChatState) -> ChatState:
            name = state.get("tool_name")
            args = state.get("tool_args") or {}
            if not name:
                return {**state, "tool_result": None}
            result: ToolResult = run_tool(name, args)
            return {**state, "tool_result": result.summary}

        async def finalize_node(state: ChatState) -> ChatState:
            session_id = state["session_id"]

            self._memory.append(
                session_id,
                {"role": "user", "content": state["last_user"]},
            )
            self._memory.append(
                session_id,
                {"role": "assistant", "content": state["last_assistant"] or ""},
            )

            return {
                **state,
                "messages": self._memory.get_history(session_id),
            }

        graph.add_node("orchestrator", orchestrator_node)
        graph.add_node("tools", tool_node)
        graph.add_node("finalize", finalize_node)

        graph.set_entry_point("orchestrator")

        def route_from_orchestrator(state: ChatState) -> str:
            if state.get("intent") == "tool" and state.get("tool_name"):
                return "tools"
            return "finalize"

        graph.add_conditional_edges("orchestrator", route_from_orchestrator)
        graph.add_edge("tools", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    # ── worker-mode pipeline ──────────────────────────────────

    async def _handle_worker_turn(self, session_id: str, user_text: str) -> ChatState:
        """Process one turn inside worker mode — bypasses the LangGraph pipeline."""
        ws = get_worker_session(session_id)
        text_lower = user_text.strip().lower()

        if text_lower == "end":
            images = deactivate_worker_mode(session_id)
            await asyncio.to_thread(cleanup_generated_images, images)
            self._memory.append(session_id, {"role": "user", "content": user_text})
            reply_en = (
                "Work session ended! All generated images have been cleaned up. "
                "I'm back to being your Lilly now — what's up, babe?"
            )
            reply = await self._worker_localize(session_id, user_text, reply_en)
            self._memory.append(session_id, {"role": "assistant", "content": reply})
            return {
                "session_id": session_id,
                "messages": self._memory.get_history(session_id),
                "last_user": user_text,
                "last_assistant": reply,
                "intent": "chat",
                "tool_result": None,
                "tool_name": None,
                "tool_args": None,
                "image_url": None,
                "ui_event": {"type": "worker_mode_exit"},
            }

        if text_lower == "create" and ws.step in (
            WorkerStep.COLLECTING_TASK,
            WorkerStep.READY_TO_CREATE,
            WorkerStep.ITERATING,
        ):
            return await self._worker_generate_image(session_id, ws, user_text)

        if ws.step == WorkerStep.ITERATING and _user_requests_worker_instagram_promo(
            user_text,
        ):
            lp = get_latest_path(session_id)
            if (not lp or not lp.is_file()) and ws.latest_generated and ws.latest_generated.is_file():
                set_latest(session_id, ws.latest_generated)
                lp = ws.latest_generated
            self._memory.append(session_id, {"role": "user", "content": user_text})
            if lp and lp.is_file():
                posted = await post_latest_image_to_instagram_feed(session_id, user_text)
                if posted:
                    assistant_text = posted
                else:
                    assistant_text = await self._worker_localize(
                        session_id,
                        user_text,
                        "I couldn't post to Instagram right now — please check settings or try again later.",
                    )
            else:
                assistant_text = await self._worker_localize(
                    session_id,
                    user_text,
                    "I don't have a generated image to post yet. "
                    "Type create to generate one, then ask me to promote it.",
                )
            self._memory.append(session_id, {"role": "assistant", "content": assistant_text})
            return {
                "session_id": session_id,
                "messages": self._memory.get_history(session_id),
                "last_user": user_text,
                "last_assistant": assistant_text,
                "intent": "worker_mode",
                "tool_result": None,
                "tool_name": None,
                "tool_args": None,
                "image_url": None,
                "ui_event": {"type": "worker_mode_active"},
            }

        history = self._memory.get_history(session_id)
        result = await self._worker_agent.run(
            history=history,
            user_input=user_text,
            session_id=session_id,
            worker_session=ws,
        )

        reply = result.get("response", "")
        valid = result.get("valid", False)
        extracted = result.get("extracted_value")
        ui_event: Dict[str, Any] | None = {"type": "worker_mode_active"}

        if ws.step == WorkerStep.COLLECTING_NAME and valid and extracted:
            ws.client_name = str(extracted).strip()
            update_step(session_id, WorkerStep.COLLECTING_EMAIL)

        elif ws.step == WorkerStep.COLLECTING_EMAIL and valid and extracted:
            ws.client_email = str(extracted).strip()
            update_step(session_id, WorkerStep.COLLECTING_PAYMENT)
            ui_event = {"type": "worker_mode_payment_qr", "image_url": "/payment-qr/GooglePay_QR.png"}

        elif ws.step == WorkerStep.COLLECTING_PAYMENT and valid and extracted:
            ws.transaction_id = str(extracted).strip()
            update_step(session_id, WorkerStep.COLLECTING_TASK)
            task_hint_en = (
                "Now let's get to the creative part! Please describe in detail "
                "what you'd like me to create — the concept, my outfit/costume, "
                "background setting, any text, objects, lighting, and style. "
                "You can also attach reference images. "
                "Once you're happy with the brief, type create to start generating the image."
            )
            reply += "\n\n" + await self._worker_localize(session_id, user_text, task_hint_en)

        elif ws.step == WorkerStep.COLLECTING_TASK:
            refined = result.get("refined_prompt")
            if refined:
                ws.task_description = user_text
                ws.refined_prompt = refined

        elif ws.step == WorkerStep.ITERATING:
            refined = result.get("refined_prompt")
            if refined:
                ws.iteration_history.append(user_text)
                ws.refined_prompt = refined

        self._memory.append(session_id, {"role": "user", "content": user_text})
        self._memory.append(session_id, {"role": "assistant", "content": reply})

        return {
            "session_id": session_id,
            "messages": self._memory.get_history(session_id),
            "last_user": user_text,
            "last_assistant": reply,
            "intent": "worker_mode",
            "tool_result": None,
            "tool_name": None,
            "tool_args": None,
            "image_url": None,
            "ui_event": ui_event,
        }

    async def _worker_generate_image(
        self, session_id: str, ws: WorkerSession, user_text: str,
    ) -> ChatState:
        """Run the Freepik generation for worker mode."""
        if not ws.refined_prompt:
            reply = await self._worker_localize(
                session_id,
                user_text,
                "I don't have enough details yet. Please describe what you'd like me to create first!",
            )
            self._memory.append(session_id, {"role": "user", "content": user_text})
            self._memory.append(session_id, {"role": "assistant", "content": reply})
            return {
                "session_id": session_id,
                "messages": self._memory.get_history(session_id),
                "last_user": user_text,
                "last_assistant": reply,
                "intent": "worker_mode",
                "tool_result": None,
                "tool_name": None,
                "tool_args": None,
                "image_url": None,
                "ui_event": {"type": "worker_mode_active"},
            }

        if not ws.character_ref_image:
            ref = pick_random_reference_image()
            if ref is None:
                reply = await self._worker_localize(
                    session_id,
                    user_text,
                    "I'm sorry, there are no model reference images available right now. Please contact support.",
                )
                self._memory.append(session_id, {"role": "user", "content": user_text})
                self._memory.append(session_id, {"role": "assistant", "content": reply})
                return {
                    "session_id": session_id,
                    "messages": self._memory.get_history(session_id),
                    "last_user": user_text,
                    "last_assistant": reply,
                    "intent": "worker_mode",
                    "tool_result": None,
                    "tool_name": None,
                    "tool_args": None,
                    "image_url": None,
                    "ui_event": {"type": "worker_mode_active"},
                }
            ws.character_ref_image = ref

        update_step(session_id, WorkerStep.GENERATING)
        self._memory.append(session_id, {"role": "user", "content": user_text})

        generating_msg = await self._worker_localize(
            session_id,
            user_text,
            "Got it! I'm generating your image now — this may take a moment...",
        )
        self._memory.append(session_id, {"role": "assistant", "content": generating_msg})

        settings = get_settings()
        image_url: str | None = None
        if (settings.freepic_api_key or "").strip():
            img_bytes, err = await freepic_service.generate_worker_mode_image(
                character_ref=ws.character_ref_image,
                task_prompt=ws.refined_prompt,
                additional_refs=ws.reference_images or None,
                latest_generated=ws.latest_generated,
            )
            if img_bytes:
                image_url = await asyncio.to_thread(
                    freepic_service.save_generated_costume_file, img_bytes, ".jpg"
                )
                gen_path = freepic_service.resolve_generated_costume_url_to_path(image_url)
                if gen_path:
                    set_latest_generated(session_id, gen_path)
                    set_latest(session_id, gen_path)
            elif err:
                logger.warning("Worker mode generation error: %s", err)
        else:
            logger.warning("Worker mode: FREEPIC_API_KEY not set")

        update_step(session_id, WorkerStep.ITERATING)

        if image_url:
            reply = await self._worker_localize(
                session_id,
                user_text,
                "Here's your generated image! Take a look and let me know if you'd like any changes. "
                "You can describe modifications and type create to regenerate, "
                "or type end to finish the session. "
                "If you'd like extra reach, I can help promote your ad on my Instagram — "
                "say promote or ask me to post it on Instagram and I'll share this image there.",
            )
        else:
            reply = await self._worker_localize(
                session_id,
                user_text,
                "I wasn't able to generate the image this time. "
                "Please try describing your requirements again and type create.",
            )
            update_step(session_id, WorkerStep.COLLECTING_TASK)

        self._memory.append(
            session_id, {"role": "assistant", "content": reply}
        )

        return {
            "session_id": session_id,
            "messages": self._memory.get_history(session_id),
            "last_user": user_text,
            "last_assistant": generating_msg + "\n\n" + reply,
            "intent": "worker_mode",
            "tool_result": None,
            "tool_name": None,
            "tool_args": None,
            "image_url": image_url,
            "ui_event": {"type": "worker_mode_active"},
        }

    # ── public API ──────────────────────────────────────────────

    async def run_turn(self, *, session_id: str, user_text: str) -> str:
        """Run one conversational turn and return the assistant reply."""
        result = await self._run(session_id, user_text)
        return result["last_assistant"] or ""

    async def run_turn_state(self, *, session_id: str, user_text: str) -> ChatState:
        """Run one conversational turn and return the full graph state."""
        return await self._run(session_id, user_text)

    async def _run(self, session_id: str, user_text: str) -> ChatState:
        lock = await self._session_turn_lock(session_id)
        async with lock:
            if is_worker_mode(session_id):
                return await self._handle_worker_turn(session_id, user_text)

            initial_state: ChatState = {
                "session_id": session_id,
                "messages": self._memory.get_history(session_id),
                "last_user": user_text,
                "last_assistant": None,
                "intent": "chat",
                "tool_result": None,
                "tool_name": None,
                "tool_args": None,
                "image_url": None,
                "ui_event": None,
            }
            return await self._graph.ainvoke(initial_state)


_shared_instance: ChatWorkflow | None = None


def get_shared_workflow() -> ChatWorkflow:
    """Singleton accessor so WebSocket and HTTP endpoints share memory."""
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = ChatWorkflow()
    return _shared_instance
