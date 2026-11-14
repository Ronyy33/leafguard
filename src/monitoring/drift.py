"""Image statistics and data-drift detection for LeafGuard.

The same `compute_image_stats` function is used in three places so the numbers
are always comparable:
  * EDA (src/data/eda.py)                 -> exploratory reports
  * preprocessing (src/data/preprocess.py) -> training baseline (baseline_stats.json)
  * the live API (backend)                -> rolling drift scores exported to Prometheus
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

STAT_KEYS = ("brightness", "blur", "mean_r", "mean_g", "mean_b")


def compute_image_stats(rgb: np.ndarray) -> dict[str, float]:
    """Return simple, interpretable statistics for one RGB uint8 image.

    brightness : mean of the grayscale image (0-255)
    blur       : variance of the Laplacian; LOW values mean a blurry image
    mean_r/g/b : per-channel means (0-255), sensitive to colour/lighting shifts
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB array, got shape {rgb.shape}")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "brightness": float(gray.mean()),
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_r": float(rgb[..., 0].mean()),
        "mean_g": float(rgb[..., 1].mean()),
        "mean_b": float(rgb[..., 2].mean()),
    }


@dataclass
class Baseline:
    """Mean and standard deviation of each statistic over the training set."""

    mean: dict[str, float]
    std: dict[str, float]
    n: int

    @classmethod
    def from_stats(cls, rows: list[dict[str, float]]) -> "Baseline":
        arr = {k: np.array([r[k] for r in rows], dtype=float) for k in STAT_KEYS}
        return cls(
            mean={k: float(v.mean()) for k, v in arr.items()},
            std={k: float(v.std() or 1e-6) for k, v in arr.items()},
            n=len(rows),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        return cls(**json.loads(Path(path).read_text()))


class DriftDetector:
    """Rolling-window drift detector.

    Keeps the last `window` request statistics and reports, per statistic, the
    z-score of the window mean relative to the training baseline:

        z = (window_mean - baseline_mean) / (baseline_std / sqrt(window_size))

    A large |z| means recent inputs look systematically different from training
    data (e.g. darker photos or blurrier photos).
    """

    def __init__(self, baseline: Baseline, window: int = 50):
        self.baseline = baseline
        self.buffer: deque[dict[str, float]] = deque(maxlen=window)

    def update(self, stats: dict[str, float]) -> dict[str, float]:
        self.buffer.append(stats)
        return self.scores()

    def scores(self) -> dict[str, float]:
        if not self.buffer:
            return {k: 0.0 for k in STAT_KEYS}
        n = len(self.buffer)
        out = {}
        for k in STAT_KEYS:
            window_mean = float(np.mean([s[k] for s in self.buffer]))
            se = self.baseline.std[k] / np.sqrt(n)
            out[k] = (window_mean - self.baseline.mean[k]) / se
        return out
