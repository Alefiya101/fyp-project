"""Application configuration for Person C (Application, Integration & Evaluation).

Loads settings from environment variables or .env file with production-ready defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

class AppConfig:
    """Configuration manager for the Ambient Clinical Scribe Application."""

    def __init__(self):
        # Database & Storage
        self.db_path: Path = Path(os.environ.get("DB_PATH", str(BASE_DIR / "data" / "ambient_scribe.db")))
        self.encryption_key: str = os.environ.get("ENCRYPTION_KEY", "")
        self.auto_delete_audio: bool = os.environ.get("AUTO_DELETE_AUDIO", "false").lower() in ("true", "1", "yes")
        self.audio_upload_dir: Path = Path(os.environ.get("AUDIO_UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))

        # ABDM Integration
        self.abdm_mock_mode: bool = os.environ.get("ABDM_MOCK_MODE", "true").lower() in ("true", "1", "yes")
        self.abdm_gateway_url: str = os.environ.get("ABDM_GATEWAY_URL", "https://dev.abdm.gov.in/gateway")
        self.abdm_client_id: str = os.environ.get("ABDM_CLIENT_ID", "DEMO_HIP_CLIENT_001")
        self.abdm_client_secret: str = os.environ.get("ABDM_CLIENT_SECRET", "DEMO_SECRET_KEY")
        self.abdm_hip_id: str = os.environ.get("ABDM_HIP_ID", "IN010000001")

        # Sync Worker
        self.sync_poll_interval_sec: float = float(os.environ.get("SYNC_POLL_INTERVAL_SEC", "10.0"))
        self.sync_max_retries: int = int(os.environ.get("SYNC_MAX_RETRIES", "5"))

        # Server
        self.host: str = os.environ.get("HOST", "127.0.0.1")
        self.port: int = int(os.environ.get("PORT", "8000"))
        self.debug: bool = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")

        # Clinical & Speech Service Integration
        self.extractor_backend: str = os.environ.get("EXTRACTOR_BACKEND", "rule_based")
        self.model_path: str | None = os.environ.get("MODEL_PATH", None)
        self.allow_mock_fallback: bool = os.environ.get("ALLOW_MOCK_FALLBACK", "false").lower() in ("true", "1", "yes")

        # Ensure required directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_upload_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = AppConfig()
