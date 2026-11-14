"""Unit tests: image statistics + drift detector (src/monitoring/drift.py)."""

import cv2
import numpy as np
import pytest

from src.monitoring.drift import STAT_KEYS, Baseline, DriftDetector, compute_image_stats


def test_stats_keys_and_ranges(leaf_like):
    s = compute_image_stats(leaf_like)
    assert set(s) == set(STAT_KEYS)
    assert 0 <= s["brightness"] <= 255
    assert s["blur"] >= 0


def test_darker_image_has_lower_brightness(leaf_like):
    dark = (leaf_like * 0.3).astype(np.uint8)
    assert compute_image_stats(dark)["brightness"] < compute_image_stats(leaf_like)["brightness"] * 0.5


def test_blurred_image_has_lower_sharpness(leaf_like):
    blurred = cv2.GaussianBlur(leaf_like, (15, 15), 5)
    assert compute_image_stats(blurred)["blur"] < compute_image_stats(leaf_like)["blur"] * 0.2


def test_rejects_non_rgb():
    with pytest.raises(ValueError):
        compute_image_stats(np.zeros((10, 10), np.uint8))


def test_baseline_roundtrip(tmp_path, leaf_like):
    rows = [compute_image_stats(leaf_like) for _ in range(5)]
    b = Baseline.from_stats(rows)
    b.save(tmp_path / "b.json")
    b2 = Baseline.load(tmp_path / "b.json")
    assert b2.n == 5 and b2.mean == b.mean
    assert all(v > 0 for v in b2.std.values())  # std floor avoids division by zero


def test_drift_detector_flags_dark_inputs(leaf_like):
    rng = np.random.default_rng(1)
    train = [
        compute_image_stats(np.clip(leaf_like * rng.uniform(0.9, 1.1), 0, 255).astype(np.uint8)) for _ in range(30)
    ]
    det = DriftDetector(Baseline.from_stats(train), window=20)
    for row in train[:20]:  # in-distribution
        z = det.update(row)
    assert abs(z["brightness"]) < 3
    dark = compute_image_stats((leaf_like * 0.4).astype(np.uint8))
    for _ in range(20):
        z = det.update(dark)
    assert z["brightness"] < -3


def test_empty_detector_scores_zero():
    b = Baseline(mean={k: 1.0 for k in STAT_KEYS}, std={k: 1.0 for k in STAT_KEYS}, n=1)
    assert DriftDetector(b).scores() == {k: 0.0 for k in STAT_KEYS}
