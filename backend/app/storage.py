"""Encrypted-at-rest storage of uploaded images (Fernet: AES-128-CBC + HMAC-SHA256).

Uploads are kept so that user-labelled images can later be added to training
data, but they are never written to disk in plain form.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class EncryptedStore:
    def __init__(self, directory: str, key: str = ""):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        if not key:
            key = Fernet.generate_key().decode()
            logger.warning(
                "ENCRYPTION_KEY not set; generated an ephemeral key (stored uploads "
                "will be unreadable after restart)"
            )
        self.fernet = Fernet(key.encode() if isinstance(key, str) else key)

    def save(self, item_id: str, data: bytes) -> Path:
        path = self.dir / f"{item_id}.bin"
        path.write_bytes(self.fernet.encrypt(data))
        return path

    def load(self, item_id: str) -> bytes:
        return self.fernet.decrypt((self.dir / f"{item_id}.bin").read_bytes())
