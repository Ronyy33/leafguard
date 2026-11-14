"""Runtime monitoring state: data drift (vs training baseline) and live accuracy from feedback."""

from __future__ import annotations

import threading
from collections import OrderedDict, deque

from app import metrics as M
from src.monitoring.drift import Baseline, DriftDetector, compute_image_stats

DRIFT_FEATURES = ("brightness", "blur")  # the features we alert on


class RuntimeMonitor:
    def __init__(self, window: int, z_threshold: float = 3.0, feedback_window: int = 100,
                 max_tracked_predictions: int = 5000):
        self.window = window
        self.z_threshold = z_threshold
        self.detector: DriftDetector | None = None
        self._recent_preds: OrderedDict[str, str] = OrderedDict()  # prediction_id -> predicted label
        self._max_tracked = max_tracked_predictions
        self._feedback: deque[bool] = deque(maxlen=feedback_window)
        self._lock = threading.Lock()

    def set_baseline(self, baseline: Baseline) -> None:
        """Called on (re)load: a new model brings its own training baseline."""
        with self._lock:
            self.detector = DriftDetector(baseline, window=self.window)
        self._export(self.detector.scores(), 0)

    def observe_image(self, rgb) -> dict[str, float]:
        stats = compute_image_stats(rgb)
        for k, v in stats.items():
            M.INPUT_STAT.labels(feature=k).set(v)
        with self._lock:
            if self.detector is None:
                return stats
            scores = self.detector.update(stats)
            n = len(self.detector.buffer)
        self._export(scores, n)
        return stats

    @staticmethod
    def _export(scores: dict[str, float], n: int) -> None:
        for k, z in scores.items():
            M.DRIFT_Z.labels(feature=k).set(z)
        M.DRIFT_WINDOW_FILL.set(n)

    def drift_status(self) -> dict:
        with self._lock:
            scores = self.detector.scores() if self.detector else {}
            n = len(self.detector.buffer) if self.detector else 0
        drifting = [k for k in DRIFT_FEATURES if abs(scores.get(k, 0.0)) > self.z_threshold]
        return {"window_size": self.window, "samples_in_window": n, "zscores": scores,
                "drifting_features": drifting, "threshold": self.z_threshold}

    def reset_window(self) -> None:
        with self._lock:
            if self.detector:
                self.detector.buffer.clear()
        self._export({k: 0.0 for k in ("brightness", "blur", "mean_r", "mean_g", "mean_b")}, 0)

    # ---------------------------------------------------------------- feedback
    def remember_prediction(self, prediction_id: str, label: str) -> None:
        with self._lock:
            self._recent_preds[prediction_id] = label
            while len(self._recent_preds) > self._max_tracked:
                self._recent_preds.popitem(last=False)

    def record_feedback(self, prediction_id: str, true_label: str) -> tuple[bool, float, int]:
        with self._lock:
            if prediction_id not in self._recent_preds:
                raise KeyError(prediction_id)
            correct = self._recent_preds.pop(prediction_id) == true_label
            self._feedback.append(correct)
            acc = sum(self._feedback) / len(self._feedback)
            n = len(self._feedback)
        M.FEEDBACK.labels(correct=str(correct).lower()).inc()
        M.ROLLING_ACC.set(acc)
        return correct, acc, n
