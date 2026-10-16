"""Main FastAPI Application Entrypoint (Person C).

Configures lifespan events, static files, Jinja2 review UI templates, and REST APIs.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app_service.abdm.sync_worker import sync_worker
from app_service.api.routes import router as api_router
from app_service.config import settings
from app_service.storage.database import db
from app_service.storage.models import SessionDAO

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db.init_db()
    sync_worker.start()
    yield
    # Shutdown
    sync_worker.stop()


app = FastAPI(
    title="Edge-AI Ambient Clinical Scribe (Indian OPDs)",
    description="Person C: Application, Integration, Clinician Review UI, and ABDM Sync Service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include API Router
app.include_router(api_router)


# ---------------------------------------------------------------------------
# UI View Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Renders main clinician dashboard and session launcher."""
    with db.cursor() as cur:
        cur.execute("SELECT session_id, clinician_id, patient_identifier, status, updated_at FROM sessions ORDER BY updated_at DESC LIMIT 15;")
        sessions = cur.fetchall()

    queue_summary = sync_worker.get_queue_summary()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "sessions": sessions,
            "queue": queue_summary,
            "mock_mode": settings.abdm_mock_mode,
        },
    )


@app.get("/review/{session_id}", response_class=HTMLResponse)
async def review_page(request: Request, session_id: str):
    """Renders side-by-side clinician review screen with transcript source highlighting."""
    session = SessionDAO.get_session(session_id)
    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "session_id": session_id,
            "session": session,
            "mock_mode": settings.abdm_mock_mode,
        },
    )


@app.get("/health")
def health_check():
    """Service liveness and readiness probe."""
    return {
        "status": "healthy",
        "service": "ambient_clinical_scribe",
        "sync_worker_alive": sync_worker._thread.is_alive() if sync_worker._thread else False,
        "abdm_online": sync_worker.client.is_online(),
    }


if __name__ == "__main__":
    uvicorn.run("app_service.main:app", host=settings.host, port=settings.port, reload=settings.debug)
