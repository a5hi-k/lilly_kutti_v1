from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from starlette.requests import Request

from app.core.config import get_settings
from app.api_v1_chat import router as chat_router
from app.routes_audio import router as audio_router
from app.routes_profile_ref import router as profile_ref_router
from app.routes_videos import router as videos_router
from app.services.instagram_dm_service import (
    start_instagram_dm_listener,
    stop_instagram_dm_listener,
)
from app.services.freepic_service import generated_costumes_dir
from app.services.lipsync_service import lipsync_tmp_dir


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ig_task = await start_instagram_dm_listener()
    try:
        yield
    finally:
        await stop_instagram_dm_listener(ig_task)


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.backend_cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def refer_image_no_store_cache(request: Request, call_next):
    """Same URL (e.g. ref.jpg) must not be stuck in browser cache after file replace."""
    response = await call_next(request)
    if (
        request.url.path.startswith("/ref-assets")
        or request.url.path.startswith("/generated-costumes")
        or request.url.path.startswith("/payment-qr")
        or request.url.path.startswith("/session-uploads")
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok"}


app.include_router(chat_router)
app.include_router(audio_router)
app.include_router(videos_router)
app.include_router(profile_ref_router)

# Static hosting for avatar video-call simulation clips
app.mount(
    "/avatar-videos",
    StaticFiles(directory=str(Path(__file__).resolve().parents[1] / "videos")),
    name="avatar-videos",
)
app.mount(
    "/lipsync-tmp",
    StaticFiles(directory=str(lipsync_tmp_dir())),
    name="lipsync-tmp",
)

_ref_dir = Path(__file__).resolve().parents[1] / "ReferImage"
_ref_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/ref-assets",
    StaticFiles(directory=str(_ref_dir)),
    name="ref-assets",
)

_gen_costumes = generated_costumes_dir()
_gen_costumes.mkdir(parents=True, exist_ok=True)
app.mount(
    "/generated-costumes",
    StaticFiles(directory=str(_gen_costumes)),
    name="generated-costumes",
)

_payment_qr_dir = Path(__file__).resolve().parents[1] / "PaymentQR"
_payment_qr_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/payment-qr",
    StaticFiles(directory=str(_payment_qr_dir)),
    name="payment-qr",
)

_session_uploads_dir = Path(__file__).resolve().parents[1] / "session_uploads"
_session_uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/session-uploads",
    StaticFiles(directory=str(_session_uploads_dir)),
    name="session-uploads",
)