"""Text-only completions: Gemini or Groq based on LLM_MODEL in settings."""

from __future__ import annotations

from google import genai

from app.core.config import get_settings


def _provider() -> str:
    s = get_settings()
    v = (getattr(s, "llm_model", None) or "gemini").strip().lower()
    return "groq" if v == "groq" else "gemini"


def generate_text_with_system_sync(
    system_instruction: str,
    user_content: str,
    temperature: float,
    *,
    gemini_model: str | None = None,
) -> str:
    settings = get_settings()
    if _provider() == "groq":
        api_key = (settings.groq_api_key or "").strip()
        if not api_key:
            raise RuntimeError("LLM_MODEL=groq but GROQ_API_KEY is empty — add it to backend/.env")
        model = (settings.groq_model or "").strip() or "llama-3.3-70b-versatile"
        from groq import Groq

        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("Groq returned empty text")
        return text

    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is empty — add it to backend/.env")
    client = genai.Client(api_key=api_key)
    model = gemini_model or settings.gemini_chat_model
    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config={
            "system_instruction": system_instruction,
            "temperature": temperature,
        },
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ValueError("Gemini returned empty text")
    return text
