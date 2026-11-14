"""Exploratory data analysis for the raw LeafGuard dataset.

Produces (all under reports/eda/):
  eda_summary.json      class counts, image sizes/modes, stats per class, issues found
  class_balance.png     images per class
  brightness_blur.png   brightness and blur distributions (the drift features)

Usage:  python -m src.data.eda
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from src.monitoring.drift import STAT_KEYS, compute_image_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eda")

OUT_DIR = Path("reports/eda")


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    raw_dir = Path(params["data"]["raw_dir"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    per_class: dict[str, list[dict[str, float]]] = defaultdict(list)
    issues: list[dict[str, str]] = []

    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for img_path in sorted(class_dir.iterdir()):
            try:
                with Image.open(img_path) as im:
                    sizes[f"{im.width}x{im.height}"] += 1
                    modes[im.mode] += 1
                    rgb = np.asarray(im.convert("RGB"))
                per_class[class_dir.name].append(compute_image_stats(rgb))
                counts[class_dir.name] += 1
            except Exception as exc:  # corrupt / unreadable file
                issues.append({"file": str(img_path), "error": repr(exc)})

    total = sum(counts.values())
    elapsed = time.time() - t0
    class_stats = {
        c: {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in STAT_KEYS}
        for c, rows in per_class.items()
    }
    summary = {
        "total_images": total,
        "num_classes": len(counts),
        "class_counts": dict(counts),
        "imbalance_ratio": max(counts.values()) / max(1, min(counts.values())),
        "image_sizes": dict(sizes),
        "image_modes": dict(modes),
        "unreadable_files": issues,
        "per_class_stats": class_stats,
        "seconds": round(elapsed, 2),
        "images_per_second": round(total / elapsed, 1) if elapsed else None,
    }
    (OUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # Plot 1: class balance
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(counts.keys())
    ax.barh(names, [counts[n] for n in names], color="#3a7d44")
    ax.set_xlabel("images")
    ax.set_title("Images per class")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_balance.png", dpi=120)
    plt.close(fig)

    # Plot 2: brightness & blur distributions (the drift features)
    all_rows = [r for rows in per_class.values() for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist([r["brightness"] for r in all_rows], bins=40, color="#3a7d44")
    axes[0].set_title("Brightness (grayscale mean)")
    axes[1].hist(np.log10([r["blur"] + 1 for r in all_rows]), bins=40, color="#8a5a00")
    axes[1].set_title("Sharpness: log10(Laplacian variance)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "brightness_blur.png", dpi=120)
    plt.close(fig)

    logger.info("EDA done: %d images, %d classes, %d issues, %.1fs", total, len(counts), len(issues), elapsed)


if __name__ == "__main__":
    main()
