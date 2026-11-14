"""Unit tests: data validation and preprocessing transforms (src/data/*)."""

import numpy as np
import pandas as pd
from PIL import Image

from src.data.preprocess import make_transform
from src.data.validate import check_image


def _save(tmp_path, name, size=(256, 256), mode="RGB"):
    p = tmp_path / name
    Image.new(mode, size, color=120 if mode == "L" else (60, 140, 40)).save(p)
    return p


def test_valid_image_passes(tmp_path):
    assert check_image(str(_save(tmp_path, "ok.jpg")), 64, 64) == ""


def test_grayscale_image_is_accepted(tmp_path):
    assert check_image(str(_save(tmp_path, "g.png", mode="L")), 64, 64) == ""


def test_small_image_rejected(tmp_path):
    assert check_image(str(_save(tmp_path, "s.jpg", size=(32, 32))), 64, 64).startswith("too_small")


def test_corrupt_file_rejected(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"definitely not a jpeg")
    assert check_image(str(p), 64, 64).startswith("unreadable")


def test_missing_file_rejected(tmp_path):
    assert check_image(str(tmp_path / "nope.jpg"), 64, 64).startswith("unreadable")


def test_transform_resizes_and_adds_stats(tmp_path):
    paths = [str(_save(tmp_path, f"{i}.jpg")) for i in range(3)]
    batch = pd.DataFrame({"path": paths, "label": ["a", "b", "a"]})
    out = make_transform(64)(batch)
    assert len(out) == 3
    assert {"image", "brightness", "blur", "mean_r", "mean_g", "mean_b"} <= set(out.columns)
    img = out["image"].iloc[0]
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
