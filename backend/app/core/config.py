from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import AliasChoices, AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env path: config.py -> core -> app -> backend; .env lives in backend/
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _BACKEND_DIR / ".env"
# override=True: values in backend/.env win over empty/mistaken shell exports
load_dotenv(dotenv_path=_ENV_PATH, override=True)


class Settings(BaseSettings):
    app_name: str = "GF Chat Backend"
    debug: bool = True

    backend_cors_origins: List[AnyHttpUrl] = [
        "http://localhost:5173",
    ]

    # LLM_MODEL=gemini (default) or groq — text chat / text-only helpers use this provider.
    # Audio + image flows still use GEMINI_API_KEY (multimodal).
    llm_model: str = "gemini"
    groq_api_key: str = ""
    groq_model: str = ""

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.1-flash-lite-preview"

    sarvam_api_key: str = ""

    gemini_translation_model: str = "gemini-3.1-flash-lite-preview"
    gemini_transcription_model: str = "gemini-3.1-flash-lite-preview"
    gemini_malayalam_stt_model: str = "gemini-3.1-flash-lite-preview"

    # File under backend/ReferImage/ — served via /profile-ref/image (revision-busted URL)
    profile_ref_image_filename: str = "ref.jpg"

    # Local Gradio lip-sync (empty string = disabled)
    lipsync_gradio_url: str = "http://127.0.0.1:7860/"
    lipsync_guidance_scale: float = 1.5
    # LatentSync argparse expects int for these (not 10.0 strings)
    lipsync_inference_steps: int = 10
    lipsync_seed: int = 1247
    lipsync_timeout: float = 600.0  # seconds to wait for Gradio inference

    # Instagram DM — Meta Graph API (official). Requires a Facebook Page linked to an IG Business account.
    ig_graph_access_token: str = ""
    ig_graph_user_id: str = ""       # IG Business Account ID
    ig_graph_page_id: str = ""       # Facebook Page ID
    ig_graph_api_version: str = "v25.0"
    # Public HTTPS base URL of this backend (ngrok / tunnel / prod).
    # Required for sending images and audio in DMs (Graph API needs public URLs for media attachments).
    ig_dm_public_base_url: str = ""
    ig_dm_poll_interval: float = 3.0

    # Freepik Seedream edit API — costume / outfit try-on from ReferImage reference.
    # Alias FREEPIK_* for the vendor spelling; avoid @lru_cache on get_settings so .env edits apply.
    freepic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("FREEPIC_API_KEY", "FREEPIK_API_KEY"),
    )
    freepic_api_url: str = Field(
        default="https://api.freepik.com/v1/ai/text-to-image/seedream-v4-5-edit",
        validation_alias=AliasChoices("FREEPIC_API_URL", "FREEPIK_API_URL"),
    )
    freepic_timeout: float = 90.0
    freepic_poll_interval: float = 5.0
    freepic_max_polls: int = 24

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )


def get_settings() -> Settings:
    """Reload .env each call so keys added while the server runs are picked up (uvicorn --reload does not watch .env)."""
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    return Settings()
