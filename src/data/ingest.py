"""Stage 1 - Ingest (Ray Data).

Scans the raw dataset (one sub-folder per class) in parallel with Ray, and
writes a manifest with one row per file: path, label, size in bytes, sha256.

Output:  data/interim/manifest.csv
Metrics: reports/ingest_metrics.json  (count, seconds, files/sec)

Usage:   python -m src.data.ingest
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import ray

from src.utils import get_logger, load_params, write_json

logger = get_logger("ingest")
OUT = Path("data/interim/manifest.csv")
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def _describe(row: dict) -> dict:
    """Ray map fn: turn one binary file row into a manifest row."""
    path = Path(row["path"])
    data = row["bytes"]
    return {
        "path": str(path),
        "label": path.parent.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    params = load_params()
    raw_dir = Path(params["data"]["raw_dir"]).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw data dir not found: {raw_dir} (did you run `dvc pull`?)")

    files = [str(p) for p in sorted(raw_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXT]
    if not files:
        raise RuntimeError(f"no images under {raw_dir}")

    ray.init(ignore_reinit_error=True, num_cpus=4, object_store_memory=300 * 1024**2,
             include_dashboard=False, log_to_driver=False)
    t0 = time.time()
    ds = ray.data.read_binary_files(files, include_paths=True).map(_describe)
    df = ds.to_pandas()
    elapsed = time.time() - t0
    ray.shutdown()

    # store paths relative to the repo so the manifest is portable
    df["path"] = df["path"].map(lambda p: str(Path(p).relative_to(Path.cwd())))
    df = df.sort_values("path").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    metrics = {
        "ingest_files": int(len(df)),
        "ingest_classes": int(df["label"].nunique()),
        "ingest_seconds": round(elapsed, 2),
        "ingest_files_per_sec": round(len(df) / elapsed, 1),
    }
    write_json(metrics, "reports/ingest_metrics.json")
    logger.info("ingested %s", metrics)


if __name__ == "__main__":
    main()
