"""Stage 3 - Preprocess & feature engineering (Ray Data).

  * decode + resize every valid image to image_size x image_size (Ray map_batches)
  * compute drift statistics (brightness, blur, RGB means) per image
  * stratified train / val / test split
  * save arrays + label map + the TRAINING-set drift baseline

Outputs (data/processed/):
  train.npz, val.npz, test.npz   X: uint8 [N,H,W,3], y: int [N]
  labels.json                    index -> class name
  baseline_stats.json            mean/std of drift features on the training split
Metrics: reports/preprocess_metrics.json

Usage:   python -m src.data.preprocess
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from PIL import Image
from sklearn.model_selection import train_test_split

from src.monitoring.drift import Baseline, compute_image_stats
from src.utils import get_logger, load_params, write_json

logger = get_logger("preprocess")
IN = Path("data/interim/valid.csv")


def make_transform(size: int):
    """Build the Ray batch function (closure over the target size)."""

    def transform(batch: pd.DataFrame) -> pd.DataFrame:
        images, stats = [], []
        for p in batch["path"]:
            with Image.open(p) as im:
                rgb = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
            images.append(rgb)
            stats.append(compute_image_stats(rgb))
        out = batch[["path", "label"]].copy()
        out["image"] = images
        stats_df = pd.DataFrame(stats, index=out.index)
        return pd.concat([out, stats_df], axis=1)

    return transform


def main() -> None:
    params = load_params()["data"]
    size, seed = params["image_size"], params["seed"]
    out_dir = Path(params["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = pd.read_csv(IN)
    classes = sorted(valid["label"].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}

    ray.init(ignore_reinit_error=True, num_cpus=4, object_store_memory=400 * 1024**2,
             include_dashboard=False, log_to_driver=False)
    t0 = time.time()
    ds = ray.data.from_pandas(valid[["path", "label"]]).repartition(8).map_batches(
        make_transform(size), batch_format="pandas", batch_size=64
    )
    df = ds.to_pandas().sort_values("path").reset_index(drop=True)
    elapsed = time.time() - t0
    ray.shutdown()

    X = np.stack(df["image"].to_numpy()).astype(np.uint8)
    y = df["label"].map(label_to_idx).to_numpy()
    idx = np.arange(len(df))

    test_frac = params["test_split"]
    val_frac = params["val_split"] / (1 - test_frac)
    idx_trval, idx_test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    idx_train, idx_val = train_test_split(idx_trval, test_size=val_frac, stratify=y[idx_trval], random_state=seed)

    for name, ids in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
        np.savez_compressed(out_dir / f"{name}.npz", X=X[ids], y=y[ids])

    (out_dir / "labels.json").write_text(json.dumps({i: c for c, i in label_to_idx.items()}, indent=2))

    stat_cols = ["brightness", "blur", "mean_r", "mean_g", "mean_b"]
    baseline = Baseline.from_stats(df.loc[idx_train, stat_cols].to_dict("records"))
    baseline.save(out_dir / "baseline_stats.json")

    metrics = {
        "preprocess_images": int(len(df)),
        "train_size": int(len(idx_train)),
        "val_size": int(len(idx_val)),
        "test_size": int(len(idx_test)),
        "num_classes": len(classes),
        "image_size": size,
        "preprocess_seconds": round(elapsed, 2),
        "preprocess_images_per_sec": round(len(df) / elapsed, 1),
    }
    write_json(metrics, "reports/preprocess_metrics.json")
    logger.info("preprocess done %s", metrics)


if __name__ == "__main__":
    main()
