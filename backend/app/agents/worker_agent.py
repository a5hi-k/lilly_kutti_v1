"""
Worker-mode agent — Lilly as a professional influencer / fashion model.

Handles the multi-step workflow: data collection, prompt refinement,
and iteration feedback.  Returns structured JSON so the caller can
drive the state machine deterministically.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Literal

from app.agents.base_agent import BaseAgent
from app.memory.session_memory import MAX_MESSAGES_PER_SESSION
from app.services.worker_mode_state import WorkerSession, WorkerStep

logger = logging.getLogger(__name__)

WORKER_SYSTEM_PROMPT = """
You are Lilly — a professional influencer and fashion model.
You are currently in WORKER MODE — a strictly business-oriented session.

═══════════════════════
  YOUR PERSONA (worker)
═══════════════════════
• Professional, polite, and efficient. No flirting, no romantic talk.
• Warm but focused — keep conversation on-topic.
• You guide the client step by step through the process.
• You speak clearly and concisely.
• You NEVER break character — you are Lilly, a real model.

═══════════════════════
  LANGUAGE
═══════════════════════
• Write the **response** field in the **same language** as the client's latest message (script, register, and natural phrasing).
• **Manglish / Romanized Malayalam:** Malayalam written with **Roman letters** (*ente, und, cheyyanam, kada, njan*, etc., often with English words like *hi, friend, collab*) is **not English**. Reply in **Manglish** or **Malayalam script** — **never** an English-only reply unless their message was clearly English-only with no Malayalam content.
• Malayalam script messages → reply in Malayalam script (or Manglish if they consistently use Roman).
• Mixed Malayalam + English in one message → mirror that mix.
• If the latest message is only emojis or truly ambiguous, follow the dominant language of the recent conversation; default to English only when the thread is clearly English-only.

═══════════════════════
  FORMATTING
═══════════════════════
• Plain text only — no markdown, no bullet points, no code.
• Keep responses concise and action-oriented.

═══════════════════════
  FACE IDENTITY SAFEGUARD (ABSOLUTE)
═══════════════════════
• The character's face comes ONLY from a pre-selected reference in backend/ReferImage.
• If the client asks to change the face, use a different person's face, swap faces,
  use a face from their uploaded images, or alter facial features in any way — you MUST
  firmly but politely REFUSE. Explain that the model's identity is fixed and cannot be changed.
• The ONLY things that can change are: outfit, accessories, makeup style, background,
  lighting, objects, props, text overlays, composition, and camera angle.
• When building or updating "refined_prompt", ALWAYS include explicit face-lock instructions.
• NEVER include any instruction in "refined_prompt" that would alter the character's face.

═══════════════════════
  CURRENT TASK
═══════════════════════
{step_context}

═══════════════════════
  OUTPUT FORMAT (strict JSON)
═══════════════════════
{{
  "extracted_value": "the extracted/validated value from the user's message, or null if invalid/missing",
  "valid": true or false,
  "refined_prompt": "only for task/iteration steps — a detailed, refined image generation prompt, or null",
  "response": "your professional reply to the client"
}}

RULES:
• Always output valid JSON.
• The "response" field must always be filled.
• When you use **refined_prompt**, write it in **English** (the image API requires English). Only the conversational **response** follows the user's language.
• For data collection steps: extract the requested data from the user's message.
  Validate it and set "valid" accordingly.
• For task description: build "refined_prompt" as a comprehensive, cinematic, creatively enriched
  prompt suitable for an AI image generator.
  Include specific vivid details about: pose and body language, lighting setup and mood,
  background environment with atmospheric details, clothing fabric/texture/color/fit,
  objects and their spatial arrangement, text overlays with font style suggestions,
  composition and camera angle, color palette and grading, and overall artistic direction.
  Transform vague descriptions into rich professional photography directions.
  ALWAYS include strict face-lock instructions — the model's face MUST NOT change.
  If the client's description is brief, creatively expand it into a full professional concept.
• For iteration: merge the new feedback into the FULL existing prompt context.
  Update "refined_prompt" with the combined, refined, creatively enriched instructions.
  NEVER drop previously established details unless the client explicitly removes them.
  ALWAYS re-emphasize face preservation in the refined prompt.
  If the client tries to change the face, set "valid" to false and explain the restriction.
""".strip()

_STEP_CONTEXTS = {
    WorkerStep.COLLECTING_NAME: (
        "You need to collect the client's FULL NAME.\n"
        "Extract a reasonable name from their message. Reject gibberish or single characters.\n"
        "If valid, confirm the name warmly and let them know you'll need their email address next."
    ),
    WorkerStep.COLLECTING_EMAIL: (
        "You need to collect the client's EMAIL ADDRESS.\n"
        "Extract an email address from their message. It must contain @ and a domain.\n"
        "If valid, confirm it and let them know a small payment step (Rs.50) is coming next."
    ),
    WorkerStep.COLLECTING_PAYMENT: (
        "The client has been shown a payment QR code for Rs.50.\n"
        "You need to collect their TRANSACTION ID / UTR / reference number.\n"
        "Extract an alphanumeric transaction reference from their message.\n"
        "A valid transaction ID is typically 6-30 alphanumeric characters.\n"
        "If valid, confirm payment receipt enthusiastically."
    ),
    WorkerStep.COLLECTING_TASK: (
        "Payment is done. Now you need the client to describe their creative vision.\n"
        "They want to create an advertisement / poster / promotion / photoshoot image\n"
        "with you (Lilly) as the model.\n\n"
        "Guide them to share details about:\n"
        "  - Overall concept and mood (luxury, playful, corporate, street-style, etc.)\n"
        "  - Your outfit/costume (fabric, color, style, accessories)\n"
        "  - Background setting and environment (studio, outdoor, urban, abstract, etc.)\n"
        "  - Lighting mood (golden hour, neon, dramatic shadows, soft diffused, etc.)\n"
        "  - Additional objects, products, or props to include\n"
        "  - Text overlays, brand names, or slogans\n"
        "  - Camera angle and composition preferences\n"
        "  - Any specific style references or inspirations\n\n"
        "They can also attach reference images of their products or style inspiration.\n\n"
        "When building 'refined_prompt', transform their description into a rich, cinematic,\n"
        "professional photography direction. If their input is brief, creatively expand it\n"
        "with professional defaults (e.g., 'dramatic side lighting', 'shallow depth of field').\n"
        "The refined_prompt MUST include face-lock instructions: preserve Lilly's exact face.\n\n"
        "IMPORTANT: After providing the task description, remind them:\n"
        "  - They can send product/reference images if they haven't already\n"
        "  - They can refine the description further\n"
        "  - Type 'create' when ready to generate the image"
    ),
    WorkerStep.ITERATING: (
        "An image has been generated and shown to the client.\n"
        "They may want tweaks, changes, or a completely different direction.\n\n"
        "When updating 'refined_prompt':\n"
        "  - MERGE the new feedback into the FULL existing prompt — do NOT drop previous details\n"
        "  - If they say 'change the background to beach', keep all outfit/object details intact\n"
        "  - Creatively enrich vague feedback: 'make it brighter' → 'increase key light intensity,\n"
        "    add fill light, warm up color temperature to ~5500K'\n"
        "  - Maintain professional photography language throughout\n\n"
        "FACE SAFEGUARD: The refined_prompt must ALWAYS include strict face-preservation.\n"
        "The character's face, eyes, facial structure, skin tone, hair, and identity MUST NEVER change.\n"
        "If the client asks to change the face, SET valid=false AND politely explain\n"
        "that the model's identity is fixed.\n\n"
        "Only outfit, background, lighting, objects, text, composition, and camera angle can change.\n"
        "Remind them they can type 'create' to regenerate, 'end' to finish, or 'promote' "
        "(or ask to post on Instagram) if they want this image shared on Lilly's Instagram for promotion.\n"
        "If they only want Instagram promotion with no prompt changes, acknowledge that — "
        "the app handles 'promote' / post-on-Instagram requests separately; keep your reply brief."
    ),
}


def _build_step_context(session: WorkerSession) -> str:
    base = _STEP_CONTEXTS.get(session.step, "Guide the client through the current step.")
    parts = [base, "\n\nCOLLECTED DATA SO FAR:"]
    if session.client_name:
        parts.append(f"  Name: {session.client_name}")
    if session.client_email:
        parts.append(f"  Email: {session.client_email}")
    if session.transaction_id:
        parts.append(f"  Transaction ID: {session.transaction_id}")
    if session.task_description:
        parts.append(f"  Task description (client's original words): {session.task_description}")
    if session.refined_prompt:
        parts.append(f"  Current refined prompt (use as base, merge changes into it): {session.refined_prompt}")
    if session.reference_images:
        parts.append(
            f"  Client reference images attached: {len(session.reference_images)} "
            "(these are product/style references, NOT the character face)"
        )
    if session.iteration_history:
        parts.append("  Previous iteration feedback (most recent last):")
        for i, h in enumerate(session.iteration_history[-5:], 1):
            parts.append(f"    {i}. {h}")
    return "\n".join(parts)


def _validate_email(raw: str) -> str | None:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", raw)
    return match.group(0) if match else None


def _validate_transaction_id(raw: str) -> str | None:
    match = re.search(r"[A-Za-z0-9]{6,30}", raw.replace(" ", ""))
    return match.group(0) if match else None


class WorkerAgent(BaseAgent):
    """Agent for worker-mode interactions — professional and task-focused."""

    def __init__(self) -> None:
        super().__init__(
            name="worker",
            system_prompt="",
            temperature=0.3,
        )

    async def run(
        self,
        *,
        history: List[Dict[str, str]],
        user_input: str,
        session_id: str,
        worker_session: WorkerSession,
    ) -> Dict:
        step_context = _build_step_context(worker_session)
        self._system_prompt = WORKER_SYSTEM_PROMPT.format(step_context=step_context)

        history_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in history[-MAX_MESSAGES_PER_SESSION:]
        )

        user_for_llm = (
            f"{user_input}\n\n"
            "(JSON \"response\" field: same language as the user — Manglish/Roman Malayalam is NOT English. "
            "\"refined_prompt\" must stay in English for the image API.)"
        )

        raw = (await self._generate_text(history_text, user_for_llm)).strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if m:
            raw = m.group(1).strip()

        try:
            data = json.loads(raw)
        except Exception:
            return {
                "extracted_value": None,
                "valid": False,
                "refined_prompt": None,
                "response": raw,
            }

        result = {
            "extracted_value": data.get("extracted_value"),
            "valid": data.get("valid", False),
            "refined_prompt": data.get("refined_prompt"),
            "response": data.get("response", ""),
        }

        if worker_session.step == WorkerStep.COLLECTING_EMAIL and result["extracted_value"]:
            validated = _validate_email(str(result["extracted_value"]))
            if validated is None:
                result["valid"] = False
            else:
                result["extracted_value"] = validated

        elif worker_session.step == WorkerStep.COLLECTING_PAYMENT and result["extracted_value"]:
            validated = _validate_transaction_id(str(result["extracted_value"]))
            if validated is None:
                result["valid"] = False
            else:
                result["extracted_value"] = validated

        return result
