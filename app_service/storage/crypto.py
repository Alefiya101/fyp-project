"""Cryptographic utilities for Person C storage and audit.

Implements AES-256-GCM authenticated encryption and HMAC-SHA256 signatures for:
1. Field-level encryption of consultation transcripts, extraction drafts, and approved FHIR bundles.
2. Tamper-evident audit trail signatures.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app_service.config import settings


class CryptoManager:
    """Manages AES-256-GCM authenticated encryption and integrity signing."""

    def __init__(self, key: bytes | str | None = None):
        self._key = self._resolve_key(key)
        self._aesgcm = AESGCM(self._key)

    def _resolve_key(self, key_override: bytes | str | None) -> bytes:
        if key_override:
            if isinstance(key_override, str):
                try:
                    raw = bytes.fromhex(key_override)
                    if len(raw) == 32:
                        return raw
                except ValueError:
                    pass
                try:
                    raw = base64.b64decode(key_override)
                    if len(raw) == 32:
                        return raw
                except Exception:
                    pass
                return hashlib.sha256(key_override.encode("utf-8")).digest()
            return key_override

        if settings.encryption_key:
            return self._resolve_key(settings.encryption_key)

        # Local master key file
        keyfile = settings.db_path.parent / ".secret_master.key"
        if keyfile.is_file():
            try:
                data = keyfile.read_bytes()
                if len(data) == 32:
                    return data
            except OSError:
                pass

        # Generate fresh 256-bit key
        new_key = AESGCM.generate_key(bit_length=256)
        try:
            keyfile.parent.mkdir(parents=True, exist_ok=True)
            keyfile.write_bytes(new_key)
        except OSError:
            pass
        return new_key

    def encrypt_bytes(self, plaintext: bytes, associated_data: bytes = b"") -> str:
        """Encrypts plaintext bytes with AES-256-GCM. Returns base64 nonce+ciphertext."""
        nonce = os.urandom(12)  # 96-bit standard nonce
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data)
        return base64.b64encode(nonce + ciphertext).decode("ascii")

    def decrypt_bytes(self, payload_b64: str, associated_data: bytes = b"") -> bytes:
        """Decrypts base64 nonce+ciphertext with AES-256-GCM."""
        raw = base64.b64decode(payload_b64.encode("ascii"))
        if len(raw) < 28:  # 12 nonce + 16 tag minimum
            raise ValueError("Ciphertext payload too short")
        nonce = raw[:12]
        ciphertext = raw[12:]
        return self._aesgcm.decrypt(nonce, ciphertext, associated_data)

    def encrypt_text(self, text: str, associated_data: str = "") -> str:
        """Encrypts UTF-8 string."""
        return self.encrypt_bytes(text.encode("utf-8"), associated_data.encode("utf-8"))

    def decrypt_text(self, payload_b64: str, associated_data: str = "") -> str:
        """Decrypts to UTF-8 string."""
        raw = self.decrypt_bytes(payload_b64, associated_data.encode("utf-8"))
        return raw.decode("utf-8")

    def encrypt_json(self, data: Any, associated_data: str = "") -> str:
        """Serializes and encrypts JSON data."""
        serialized = json.dumps(data, default=str, ensure_ascii=False)
        return self.encrypt_text(serialized, associated_data)

    def decrypt_json(self, payload_b64: str, associated_data: str = "") -> Any:
        """Decrypts and parses JSON data."""
        text = self.decrypt_text(payload_b64, associated_data)
        return json.loads(text)

    def sign_record(self, data: str | bytes) -> str:
        """Computes HMAC-SHA256 signature for audit integrity."""
        raw = data.encode("utf-8") if isinstance(data, str) else data
        return hmac.new(self._key, raw, hashlib.sha256).hexdigest()

    def verify_signature(self, data: str | bytes, signature_hex: str) -> bool:
        """Verifies HMAC-SHA256 signature against data."""
        expected = self.sign_record(data)
        return hmac.compare_digest(expected, signature_hex)


# Singleton
crypto = CryptoManager()
