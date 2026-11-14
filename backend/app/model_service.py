"""ModelService: everything the API needs to know about the model.

Responsibilities
  * resolve `models:/<name>@<alias>` in the MLflow registry and load that version
  * fetch the run's metadata artifacts (class labels, drift baseline)
  * predict either through the MLflow model server (REST) or in-process
  * Grad-CAM explanations (always in-process, needs gradients)
  * hot-reload when the registry alias moves (after retraining / rollback)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mlflow
import numpy as np
import requests
import torch
from mlflow import MlflowClient
from PIL import Image, UnidentifiedImageError

from app import metrics as M
from app.config import Settings
from src.monitoring.drift import Baseline

logger = logging.getLogger(__name__)


class InvalidImageError(ValueError):
    """Uploaded bytes are not a decodable image."""


class ModelNotLoadedError(RuntimeError):
    """No model available (registry empty or unreachable)."""


def display_name(label: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato – Late blight'."""
    crop, _, disease = label.partition("___")
    return f"{crop.replace('_', ' ')} – {disease.replace('_', ' ').strip()}"


@dataclass
class LoadedModel:
    model: torch.nn.Module
    version: str
    run_id: str
    labels: list[str]
    baseline: Baseline
    image_size: int
    backbone: str
    test_macro_f1: float | None


class ModelService:
    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.RLock()
        self._loaded: LoadedModel | None = None
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient(settings.mlflow_tracking_uri)
        torch.set_num_threads(2)

    # ------------------------------------------------------------------ loading
    @property
    def loaded(self) -> LoadedModel:
        if self._loaded is None:
            raise ModelNotLoadedError("model not loaded yet")
        return self._loaded

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def production_version(self) -> str:
        return self.client.get_model_version_by_alias(self.s.model_name, self.s.model_alias).version

    def load(self, reason: str = "startup") -> LoadedModel:
        """Load whatever version the alias currently points to."""
        mv = self.client.get_model_version_by_alias(self.s.model_name, self.s.model_alias)
        run = self.client.get_run(mv.run_id)
        meta_dir = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="meta"))
        labels_map = json.loads((meta_dir / "labels.json").read_text())
        labels = [labels_map[str(i)] for i in range(len(labels_map))]

        model = mlflow.pytorch.load_model(f"models:/{self.s.model_name}/{mv.version}", map_location="cpu")
        model.eval()

        loaded = LoadedModel(
            model=model,
            version=str(mv.version),
            run_id=mv.run_id,
            labels=labels,
            baseline=Baseline.load(meta_dir / "baseline_stats.json"),
            image_size=int(run.data.params.get("data.image_size", 128)),
            backbone=run.data.tags.get("backbone", "unknown"),
            test_macro_f1=run.data.metrics.get("test_macro_f1"),
        )
        with self._lock:
            self._loaded = loaded
        M.MODEL_INFO.info({"name": self.s.model_name, "version": loaded.version,
                           "run_id": loaded.run_id, "backbone": loaded.backbone})
        M.MODEL_VERSION.set(float(loaded.version))
        M.MODEL_RELOADS.labels(reason=reason).inc()
        logger.info("loaded %s v%s (run %s) reason=%s", self.s.model_name, loaded.version, loaded.run_id, reason)
        return loaded

    def reload_if_changed(self) -> bool:
        """Reload if the registry alias now points to a different version."""
        try:
            if self._loaded is None or self.production_version() != self._loaded.version:
                self.load(reason="alias_changed")
                return True
        except Exception as exc:
            logger.warning("reload check failed: %s", exc)
        return False

    # ------------------------------------------------------------- inference
    def decode(self, data: bytes) -> np.ndarray:
        """Bytes -> resized RGB uint8 [H, W, 3] at the model's training resolution."""
        try:
            with Image.open(io.BytesIO(data)) as im:
                im = im.convert("RGB")
                size = self.loaded.image_size
                return np.asarray(im.resize((size, size), Image.BILINEAR))
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("file is not a valid image") from exc

    @staticmethod
    def to_tensor(rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    def _predict_remote(self, x: torch.Tensor) -> np.ndarray:
        resp = requests.post(
            f"{self.s.model_server_url}/invocations",
            json={"inputs": x.numpy().tolist()},
            timeout=self.s.model_server_timeout_s,
        )
        resp.raise_for_status()
        return np.asarray(resp.json()["predictions"], dtype=np.float32)[0]

    def _predict_local(self, x: torch.Tensor) -> np.ndarray:
        with self._lock, torch.no_grad():
            return self.loaded.model(x)[0].numpy()

    def predict_logits(self, x: torch.Tensor) -> tuple[np.ndarray, str]:
        """Logits for one image. Uses the MLflow model server, falls back to local."""
        if self.s.inference_mode == "mlflow_serve":
            t0 = time.perf_counter()
            try:
                logits = self._predict_remote(x)
                M.INFERENCE_LATENCY.labels(backend="mlflow_serve").observe(time.perf_counter() - t0)
                M.MODEL_SERVER_UP.set(1)
                return logits, "mlflow_serve"
            except Exception as exc:
                logger.warning("model server failed (%s); falling back to local model", exc)
                M.MODEL_SERVER_UP.set(0)
                M.FALLBACKS.inc()
                M.ERRORS.labels(type="model_server").inc()
        t0 = time.perf_counter()
        logits = self._predict_local(x)
        M.INFERENCE_LATENCY.labels(backend="local").observe(time.perf_counter() - t0)
        return logits, "local"

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max())
        return e / e.sum()

    # --------------------------------------------------------- explainability
    def gradcam_png(self, rgb: np.ndarray, class_idx: int) -> str:
        """Grad-CAM heatmap over the last conv block, blended on the image, base64 PNG."""
        with self._lock:
            model = self.loaded.model
            acts: dict[str, torch.Tensor] = {}
            hook = model.features.register_forward_hook(lambda _m, _i, out: acts.__setitem__("a", out))
            try:
                x = self.to_tensor(rgb).requires_grad_(True)  # backbone is frozen -> need input grad
                logits = model(x)
                grads = torch.autograd.grad(logits[0, class_idx], acts["a"])[0]
            finally:
                hook.remove()
        a = acts["a"].detach()[0]
        weights = grads[0].mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * a).sum(0)).numpy()
        cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]  # BGR->RGB
        overlay = np.uint8(0.55 * rgb + 0.45 * heat)
        buf = io.BytesIO()
        Image.fromarray(overlay).resize((256, 256)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # ---------------------------------------------------------- registry ops
    def rollback(self) -> str:
        """Point the alias at the newest registered version older than the current one."""
        current = int(self.production_version())
        versions = sorted(
            int(v.version) for v in self.client.search_model_versions(f"name='{self.s.model_name}'")
        )
        older = [v for v in versions if v < current]
        if not older:
            raise RuntimeError(f"no version older than v{current} to roll back to")
        target = str(older[-1])
        self.client.set_registered_model_alias(self.s.model_name, self.s.model_alias, target)
        self.client.set_model_version_tag(self.s.model_name, str(current), "rolled_back", "true")
        self.load(reason="rollback")
        return target
