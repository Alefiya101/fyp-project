"""ABDM Gateway and HIP Adapter Client.

Handles:
1. ABDM M2M OAuth authentication session tokens.
2. Health Information Provider (HIP) care-context linking and OPConsultRecord transfer.
3. High-fidelity Mock ABDM Gateway mode when sandbox credentials are not configured.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from app_service.config import settings

logger = logging.getLogger(__name__)


class ABDMClient:
    """Client for ABDM Milestone 1-3 HIP flows (Care Context Linking & Health Data Push)."""

    def __init__(
        self,
        gateway_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        hip_id: str | None = None,
        mock_mode: bool | None = None,
    ):
        self.gateway_url = (gateway_url or settings.abdm_gateway_url).rstrip("/")
        self.client_id = client_id or settings.abdm_client_id
        self.client_secret = client_secret or settings.abdm_client_secret
        self.hip_id = hip_id or settings.abdm_hip_id
        self.mock_mode = settings.abdm_mock_mode if mock_mode is None else mock_mode

        self._session_token: str | None = None
        self._token_expiry: float = 0.0

    def is_online(self) -> bool:
        """Checks network reachability of ABDM gateway or external connectivity."""
        if self.mock_mode:
            return True
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.gateway_url}/v0.5/heartbeat")
                return res.status_code in (200, 204)
        except Exception:
            return False

    def push_op_consult_record(
        self,
        session_id: str,
        fhir_bundle: dict[str, Any],
        patient_id: str = "patient-opd-001",
    ) -> dict[str, Any]:
        """Pushes an approved ABDM OPConsultRecord document bundle to ABDM Gateway."""
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).isoformat()

        if self.mock_mode:
            logger.info("ABDMClient [MOCK MODE]: Simulating bundle push for session '%s'", session_id)
            return {
                "status": "SUCCESS",
                "mode": "MOCK_SANDBOX",
                "requestId": request_id,
                "timestamp": timestamp,
                "hipId": self.hip_id,
                "careContextReference": f"OPD-{session_id}",
                "bundleId": fhir_bundle.get("id", f"bundle-{session_id}"),
                "message": "Care context linked and OPConsultRecord FHIR bundle synced successfully to ABDM sandbox",
            }

        # Real ABDM Sandbox Flow
        token = self._get_auth_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "X-CM-ID": "sbx",
            "Content-Type": "application/json",
        }
        payload = {
            "requestId": request_id,
            "timestamp": timestamp,
            "hipId": self.hip_id,
            "patientId": patient_id,
            "careContext": {
                "referenceNumber": f"OPD-{session_id}",
                "display": f"Outpatient Consultation - {session_id}",
            },
            "content": fhir_bundle,
        }

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"{self.gateway_url}/v0.5/health-information/hip/push",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    def _get_auth_token(self) -> str:
        """Retrieves or refreshes M2M JWT session token from ABDM Gateway."""
        if self._session_token:
            return self._session_token

        with httpx.Client(timeout=10.0) as client:
            res = client.post(
                f"{self.gateway_url}/v0.5/sessions",
                json={"clientId": self.client_id, "clientSecret": self.client_secret},
            )
            res.raise_for_status()
            data = res.json()
            self._session_token = data.get("accessToken")
            return self._session_token


# Singleton
abdm_client = ABDMClient()
