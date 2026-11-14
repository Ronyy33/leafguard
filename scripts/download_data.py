"""Rebuild data/raw from the public PlantVillage repository (no DVC remote needed).

data/plantvillage_subset.txt lists the exact 2,852 files used (class/filename).
This script downloads them from github.com/spMohanty/PlantVillage-Dataset
(raw/color/...) and DVC then proves the result is byte-identical:

    python scripts/download_data.py
    dvc status data/raw.dvc        # "Data and pipelines are up to date." = same md5 as ours
    dvc repro

Use this when you cloned the repo but cannot reach the project's DVC remote.
"""

from __future__ import annotations

import argparse
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/"


def fetch(rel: str, out_dir: Path, retries: int = 5) -> str:
    dest = out_dir / rel
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + urllib.parse.quote(rel)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return "downloaded"
        except Exception:
            time.sleep(2**attempt)
    return "failed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/plantvillage_subset.txt")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    files = [line.strip() for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    out = Path(args.out)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda rel: fetch(rel, out), files))
    counts = {k: results.count(k) for k in ("downloaded", "skipped", "failed")}
    print(f"{len(files)} files: {counts} in {time.time() - t0:.0f}s")
    if counts["failed"]:
        raise SystemExit("some downloads failed - rerun the script (it resumes)")
    print("now run: dvc status data/raw.dvc   (should report up to date)")


if __name__ == "__main__":
    main()
