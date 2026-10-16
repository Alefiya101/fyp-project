#!/usr/bin/env python3
"""run_app.py

Production entrypoint for Person C: Ambient Clinical Scribe Web Application.
Launches the FastAPI backend, clinician review UI, and background ABDM sync worker.

Usage:
    python run_app.py
    python run_app.py --port 8000 --seed-demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from app_service.config import settings
from app_service.orchestrator.orchestrator import orchestrator
from app_service.storage.database import db
from app_service.storage.models import SessionDAO


def seed_demo_session(session_id: str = "OPD_DEMO_001") -> None:
    """Pre-seeds a complete Hinglish OPD consultation draft for live evaluation."""
    print(f"[INFO] Pre-seeding demo consultation session '{session_id}'...")
    existing = SessionDAO.get_session(session_id)
    if existing:
        print(f"[INFO] Demo session '{session_id}' already exists.")
        return

    sample_transcript = {
        "session_id": session_id,
        "language_mix": "hi-en",
        "segments": [
            {"start": 0.0, "end": 3.0, "speaker": "doctor", "text": "Aapko kab se bukhar hai?", "lang": "hi-en", "confidence": 0.95},
            {"start": 3.1, "end": 8.0, "speaker": "patient", "text": "Teen din se tez bukhar hai aur badan me dard hai. Khansi bilkul nahi hai.", "lang": "hi-en", "confidence": 0.92},
            {"start": 8.1, "end": 12.0, "speaker": "doctor", "text": "Temperature 101.4 F hai. Pulse 88 hai.", "lang": "hi-en", "confidence": 0.96},
            {"start": 12.1, "end": 15.0, "speaker": "doctor", "text": "Yeh viral fever lag raha hai.", "lang": "hi-en", "confidence": 0.94},
            {"start": 15.1, "end": 20.0, "speaker": "doctor", "text": "Aap Paracetamol 650 mg TDS lijiye 3 din tak khana khane ke baad.", "lang": "hi-en", "confidence": 0.97},
            {"start": 20.1, "end": 24.0, "speaker": "doctor", "text": "Agar aaram na mile to Complete Blood Count (CBC) test karwa lena.", "lang": "hi-en", "confidence": 0.91},
        ],
    }

    SessionDAO.create_session(session_id, "dr.sharma@opd", "ABHA_1234_5678_9012")
    orchestrator.process_transcript_input(session_id, sample_transcript, actor_id="dr.sharma")
    print(f"[INFO] Demo session '{session_id}' is ready for review at http://127.0.0.1:{settings.port}/review/{session_id}")


def find_available_port(host: str, preferred_port: int) -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, preferred_port)) != 0:
            return preferred_port
    for p in range(preferred_port + 1, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, p)) != 0:
                print(f"[WARN] Port {preferred_port} in use. Switching to port {p}.")
                return p
    return preferred_port


def main():
    parser = argparse.ArgumentParser(description="Run Ambient Clinical Scribe Application")
    parser.add_argument("--host", default=settings.host, help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--seed-demo", action="store_true", help="Pre-seed an OPD demo consultation into the database")
    args = parser.parse_args()

    port = find_available_port(args.host, args.port)
    db.init_db()

    if args.seed_demo:
        seed_demo_session()

    print("\n" + "=" * 65)
    print("  EDGE-AI AMBIENT CLINICAL SCRIBE (INDIAN OPDS)")
    print("  Person C Application, Clinician Review UI & ABDM Sync")
    print("=" * 65)
    print(f"  • Web Dashboard   : http://{args.host}:{port}/")
    print(f"  • API Docs (OpenAPI): http://{args.host}:{port}/docs")
    print(f"  • Offline Storage : {settings.db_path}")
    print(f"  • ABDM Mode       : {'MOCK SANDBOX' if settings.abdm_mock_mode else 'LIVE GATEWAY'}")
    print("=" * 65 + "\n")

    uvicorn.run("app_service.main:app", host=args.host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
