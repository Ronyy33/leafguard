"""Backend configuration. Every value can be overridden with an environment variable,
so the same image runs unchanged in dev, test and docker-compose."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # MLflow registry + model to serve
    mlflow_tracking_uri: str = _env("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name: str = _env("MODEL_NAME", "leafguard")
    model_alias: str = _env("MODEL_ALIAS", "production")

    # "mlflow_serve": predictions via REST call to `mlflow models serve` (MODEL_SERVER_URL)
    # "local"       : predictions with the in-process model (also used as fallback)
    inference_mode: str = _env("INFERENCE_MODE", "mlflow_serve")
    model_server_url: str = _env("MODEL_SERVER_URL", "http://localhost:5001")
    model_server_timeout_s: float = float(_env("MODEL_SERVER_TIMEOUT_S", "5"))

    # trainer service (runs `dvc repro` on demand)
    trainer_url: str = _env("TRAINER_URL", "http://localhost:8001")

    # uploads
    max_upload_mb: float = float(_env("MAX_UPLOAD_MB", "5"))
    upload_dir: str = _env("UPLOAD_DIR", "uploads")
    encryption_key: str = _env("ENCRYPTION_KEY", "")  # Fernet key; generated at startup if empty
    store_uploads: bool = _env("STORE_UPLOADS", "true").lower() == "true"

    # monitoring
    drift_window: int = int(_env("DRIFT_WINDOW", "50"))
    low_confidence_threshold: float = float(_env("LOW_CONFIDENCE_THRESHOLD", "0.6"))

    # CORS: the frontend runs on a different origin (separate container)
    cors_origins: list[str] = field(
        default_factory=lambda: _env("CORS_ORIGINS", "*").split(",")
    )
    log_level: str = _env("LOG_LEVEL", "INFO")


settings = Settings()
