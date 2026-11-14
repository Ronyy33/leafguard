"""Shared fixtures. Unit/integration tests run fully offline: no MLflow server, no
model server, no pretrained-weight downloads (an untrained LeafNet is used)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

LABELS = ["Apple___Apple_scab", "Apple___healthy", "Tomato___Late_blight", "Tomato___healthy"]
SAMPLES = ROOT / "tests" / "samples"


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def leaf_like() -> np.ndarray:
    """Deterministic textured green 128x128 image."""
    rng = np.random.default_rng(0)
    img = np.zeros((128, 128, 3), np.uint8)
    img[..., 1] = 140
    img[..., 0] = 60
    img[..., 2] = 40
    noise = rng.integers(-40, 40, (128, 128, 1))
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def sample_jpg() -> bytes:
    return (SAMPLES / "tomato_healthy.jpg").read_bytes()


@pytest.fixture
def api(tmp_path, monkeypatch):
    """FastAPI TestClient wired to an in-process, untrained model (startup hooks NOT run)."""
    import torch
    from fastapi.testclient import TestClient

    from app import main
    from app.config import Settings
    from app.model_service import LoadedModel
    from app.storage import EncryptedStore
    from src.monitoring.drift import Baseline
    from src.train.model import LeafNet

    torch.manual_seed(0)
    model = LeafNet(len(LABELS), "mobilenet_v2", pretrained=False).eval()
    baseline = Baseline(
        mean={"brightness": 100.0, "blur": 1000.0, "mean_r": 90.0, "mean_g": 110.0, "mean_b": 80.0},
        std={"brightness": 20.0, "blur": 400.0, "mean_r": 20.0, "mean_g": 20.0, "mean_b": 20.0},
        n=100,
    )
    settings = Settings(
        inference_mode="local", model_server_url="http://127.0.0.1:9", trainer_url="http://127.0.0.1:9", max_upload_mb=1
    )
    svc = main.service
    monkeypatch.setattr(svc, "s", settings)
    monkeypatch.setattr(main, "settings", settings)
    svc._loaded = LoadedModel(
        model=model,
        version="7",
        run_id="testrun",
        labels=LABELS,
        baseline=baseline,
        image_size=128,
        backbone="mobilenet_v2",
        test_macro_f1=0.95,
    )
    monkeypatch.setattr(svc, "production_version", lambda: "7")
    main.monitor.set_baseline(baseline)
    monkeypatch.setattr(main, "store", EncryptedStore(str(tmp_path / "uploads")))
    yield TestClient(main.app, raise_server_exceptions=False)
    svc._loaded = None
