"""Send realistic traffic to the LeafGuard API (load test + drift demo).

Picks random images from data/raw, optionally degrades them to simulate
field conditions, calls POST /predict, and sends the true label to POST /feedback.

  python scripts/simulate_traffic.py -n 100                  # normal traffic
  python scripts/simulate_traffic.py -n 60 --darken 0.35     # dark photos  -> brightness drift alert
  python scripts/simulate_traffic.py -n 60 --blur 9          # blurry photos -> blur drift alert
  python scripts/simulate_traffic.py -n 200 --concurrency 8  # throughput test

Prints accuracy, latency percentiles and throughput at the end.
"""

from __future__ import annotations

import argparse
import io
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageEnhance, ImageFilter


def degrade(path: Path, darken: float, blur: float) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if darken < 1.0:
            im = ImageEnhance.Brightness(im).enhance(darken)
        if blur > 0:
            im = im.filter(ImageFilter.GaussianBlur(blur))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def one(api: str, path: Path, args) -> dict:
    data = degrade(path, args.darken, args.blur)
    true_label = path.parent.name
    t0 = time.perf_counter()
    r = requests.post(f"{api}/predict", params={"explain": str(not args.no_explain).lower()},
                      files={"file": (path.name, data, "image/jpeg")}, timeout=60)
    latency = time.perf_counter() - t0
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "latency": latency}
    pred = r.json()
    if not args.no_feedback:
        requests.post(f"{api}/feedback", json={"prediction_id": pred["prediction_id"], "true_label": true_label},
                      timeout=10)
    return {"ok": True, "correct": pred["label"] == true_label, "latency": latency,
            "backend": pred["inference_backend"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--darken", type=float, default=1.0, help="brightness factor, e.g. 0.35")
    ap.add_argument("--blur", type=float, default=0.0, help="gaussian blur radius, e.g. 9")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--no-explain", action="store_true", help="skip Grad-CAM (faster)")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = [p for p in Path(args.data).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sample = [rng.choice(files) for _ in range(args.n)]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(lambda p: one(args.api, p, args), sample))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    lat = sorted(r["latency"] * 1000 for r in ok)
    if not lat:
        print("all requests failed:", {r.get("status") for r in results})
        return
    pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]  # noqa: E731
    print(f"requests      {len(results)}  ok={len(ok)}  failed={len(results) - len(ok)}")
    print(f"accuracy      {sum(r['correct'] for r in ok) / len(ok):.3f}")
    print(f"latency ms    p50={pct(.5):.0f}  p95={pct(.95):.0f}  p99={pct(.99):.0f}  mean={statistics.mean(lat):.0f}")
    print(f"throughput    {len(results) / wall:.1f} req/s  (concurrency {args.concurrency}, {wall:.1f}s)")
    print(f"backends      {sorted({r['backend'] for r in ok})}")


if __name__ == "__main__":
    main()
