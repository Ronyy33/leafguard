"""Stage 2 - Validate (Ray Data).

Data-quality checks on every ingested file, run in parallel with Ray:
  * file decodes as an image
  * colour mode convertible to RGB
  * width/height >= configured minimum
  * no exact duplicates (same sha256)
  * every class has at least one valid image

Bad files are dropped (not fatal); the run FAILS if more than 5 % of files are
bad or any class ends up empty, so a broken data drop stops the DVC pipeline.

Output:  data/interim/valid.csv
Metrics: reports/validation.json

Usage:   python -m src.data.validate
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import ray
from PIL import Image

from src.utils import get_logger, load_params, write_json

logger = get_logger("validate")
IN = Path("data/interim/manifest.csv")
OUT = Path("data/interim/valid.csv")
MAX_BAD_FRACTION = 0.05


def check_image(path: str, min_w: int, min_h: int) -> str:
    """Return '' if the image is OK, otherwise a short reason string."""
    try:
        with Image.open(path) as im:
            im.verify()  # structural check
        with Image.open(path) as im:  # verify() invalidates the handle; reopen to decode
            im = im.convert("RGB")
            if im.width < min_w or im.height < min_h:
                return f"too_small_{im.width}x{im.height}"
    except Exception as exc:
        return f"unreadable:{type(exc).__name__}"
    return ""


def main() -> None:
    params = load_params()["data"]
    min_w, min_h = params["min_width"], params["min_height"]
    manifest = pd.read_csv(IN)

    ray.init(ignore_reinit_error=True, num_cpus=4, object_store_memory=300 * 1024**2,
             include_dashboard=False, log_to_driver=False)
    t0 = time.time()
    ds = ray.data.from_pandas(manifest).repartition(8).map(
        lambda r: {**r, "issue": check_image(r["path"], min_w, min_h)}
    )
    df = ds.to_pandas().sort_values("path").reset_index(drop=True)  # deterministic dedupe
    elapsed = time.time() - t0
    ray.shutdown()

    dup_mask = df.duplicated("sha256", keep="first")
    df.loc[dup_mask & (df["issue"] == ""), "issue"] = "duplicate"

    bad = df[df["issue"] != ""]
    good = df[df["issue"] == ""].drop(columns=["issue"]).sort_values("path")
    expected_classes = set(manifest["label"])
    empty_classes = sorted(expected_classes - set(good["label"]))

    report = {
        "validated_files": int(len(df)),
        "valid_files": int(len(good)),
        "bad_files": int(len(bad)),
        "bad_fraction": round(len(bad) / max(1, len(df)), 4),
        "issues_by_type": bad["issue"].str.split(":").str[0].value_counts().to_dict(),
        "bad_examples": bad[["path", "issue"]].head(20).to_dict("records"),
        "empty_classes": empty_classes,
        "class_counts": good["label"].value_counts().sort_index().to_dict(),
        "validate_seconds": round(elapsed, 2),
        "validate_files_per_sec": round(len(df) / elapsed, 1),
    }
    write_json(report, "reports/validation.json")

    if report["bad_fraction"] > MAX_BAD_FRACTION or empty_classes:
        raise SystemExit(f"data validation FAILED: {report['bad_fraction']:.1%} bad, empty={empty_classes}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    good.to_csv(OUT, index=False)
    logger.info("validation passed: %d valid / %d bad", len(good), len(bad))


if __name__ == "__main__":
    main()
