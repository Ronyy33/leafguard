"""LeafGuard trainer service: runs the DVC pipeline (the CI pipeline) on demand.

POST /retrain   start `dvc repro -f evaluate` in the repo (202), or 409 if busy / cooling down
GET  /status    current state, history of runs, per-stage durations, log tail
GET  /metrics   Prometheus metrics: run counts, durations, pipeline throughput

Triggered by the backend (manual button or Alertmanager drift/accuracy alerts).

Run:  uvicorn trainer.app:app --host 0.0.0.0 --port 8001   (from the repo root)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, make_asgi_app
from pydantic import BaseModel

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("leafguard.trainer")

REPO = Path(os.getenv("REPO_DIR", ".")).resolve()
COMMAND = os.getenv("RETRAIN_COMMAND", "dvc repro -f evaluate").split()
COOLDOWN_S = int(os.getenv("RETRAIN_COOLDOWN_S", "600"))
STAGE_RE = re.compile(r"Running stage '([^']+)'")

RUNS = Counter("leafguard_pipeline_runs_total", "Pipeline runs", ["status"])
RUNNING = Gauge("leafguard_pipeline_running", "1 while the pipeline is running")
LAST_DURATION = Gauge("leafguard_pipeline_last_duration_seconds", "Duration of the last pipeline run")
LAST_SUCCESS = Gauge("leafguard_pipeline_last_success_timestamp", "Unix time of the last successful run")
STAGE_SECONDS = Gauge("leafguard_pipeline_stage_seconds", "Duration of each stage in the last run", ["stage"])
THROUGHPUT = Gauge("leafguard_pipeline_throughput", "Pipeline throughput (items/sec) per stage", ["stage"])
DATA_QUALITY = Gauge("leafguard_data_quality", "Data validation results of the last run", ["kind"])
EVAL_F1 = Gauge("leafguard_last_eval_macro_f1", "Test macro-F1 of the last trained model")
PROMOTED = Gauge("leafguard_last_eval_promoted", "1 if the last trained model was promoted")


class RetrainRequest(BaseModel):
    reason: str = "manual"


class Trainer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = "idle"  # idle | running | succeeded | failed
        self.current: dict | None = None
        self.history: deque[dict] = deque(maxlen=20)
        self.log_tail: deque[str] = deque(maxlen=200)
        self.last_finished = 0.0
        self.export_reports()

    def start(self, reason: str) -> tuple[bool, str]:
        with self.lock:
            if self.state == "running":
                return False, "a pipeline run is already in progress"
            wait = COOLDOWN_S - (time.time() - self.last_finished)
            if self.last_finished and wait > 0 and not reason.startswith("manual"):
                return False, f"cooldown active ({int(wait)}s left) - prevents retrain storms"
            self.state = "running"
            self.current = {"reason": reason, "started": time.time(), "stages": {}, "status": "running"}
        threading.Thread(target=self._run, daemon=True).start()
        return True, f"pipeline started: {' '.join(COMMAND)}"

    def _run(self) -> None:
        RUNNING.set(1)
        run = self.current
        logger.info("pipeline start reason=%s", run["reason"])
        stage, stage_t0 = None, time.time()
        try:
            proc = subprocess.Popen(COMMAND, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                line = line.rstrip()
                if not line or "row/s" in line:  # skip Ray progress-bar noise
                    continue
                self.log_tail.append(line)
                m = STAGE_RE.search(line)
                if m:
                    if stage:
                        run["stages"][stage] = round(time.time() - stage_t0, 1)
                    stage, stage_t0 = m.group(1), time.time()
            rc = proc.wait()
            if stage:
                run["stages"][stage] = round(time.time() - stage_t0, 1)
            status = "succeeded" if rc == 0 else "failed"
        except Exception as exc:
            logger.exception("pipeline crashed")
            self.log_tail.append(f"ERROR: {exc}")
            status = "failed"
        run["status"] = status
        run["finished"] = time.time()
        run["duration_s"] = round(run["finished"] - run["started"], 1)
        run["eval"] = self._read_json("reports/eval_metrics.json")
        with self.lock:
            self.state = status
            self.history.appendleft(run)
            self.current = None
            self.last_finished = run["finished"]
        RUNS.labels(status=status).inc()
        RUNNING.set(0)
        LAST_DURATION.set(run["duration_s"])
        for s, sec in run["stages"].items():
            STAGE_SECONDS.labels(stage=s).set(sec)
        if status == "succeeded":
            LAST_SUCCESS.set(run["finished"])
        self.export_reports()
        logger.info("pipeline %s in %.1fs", status, run["duration_s"])

    @staticmethod
    def _read_json(rel: str) -> dict:
        try:
            return json.loads((REPO / rel).read_text())
        except Exception:
            return {}

    def export_reports(self) -> None:
        """Publish the DVC metrics files as Prometheus gauges."""
        ing = self._read_json("reports/ingest_metrics.json")
        val = self._read_json("reports/validation.json")
        pre = self._read_json("reports/preprocess_metrics.json")
        ev = self._read_json("reports/eval_metrics.json")
        tr = self._read_json("reports/train_metrics.json")
        if ing:
            THROUGHPUT.labels(stage="ingest").set(ing.get("ingest_files_per_sec", 0))
        if val:
            THROUGHPUT.labels(stage="validate").set(val.get("validate_files_per_sec", 0))
            DATA_QUALITY.labels(kind="valid_files").set(val.get("valid_files", 0))
            DATA_QUALITY.labels(kind="bad_files").set(val.get("bad_files", 0))
        if pre:
            THROUGHPUT.labels(stage="preprocess").set(pre.get("preprocess_images_per_sec", 0))
        if tr and tr.get("train_total_seconds"):
            train_size = pre.get("train_size", 0) if pre else 0
            THROUGHPUT.labels(stage="train").set(round(train_size / max(1, tr["train_total_seconds"]), 1))
        if ev:
            EVAL_F1.set(ev.get("test_macro_f1", 0))
            PROMOTED.set(1 if ev.get("promoted") else 0)

    def status(self) -> dict:
        with self.lock:
            return {
                "state": self.state,
                "current": self.current,
                "history": list(self.history),
                "log_tail": list(self.log_tail)[-60:],
                "cooldown_s": COOLDOWN_S,
                "command": " ".join(COMMAND),
                "reports": {k: self._read_json(f"reports/{k}.json") for k in
                            ("ingest_metrics", "validation", "preprocess_metrics", "train_metrics",
                             "eval_metrics")},
            }


trainer = Trainer()
app = FastAPI(title="LeafGuard Trainer", version="1.0.0")
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrain", status_code=202)
def retrain(body: RetrainRequest):
    ok, detail = trainer.start(body.reason)
    if not ok:
        return JSONResponse(status_code=409, content={"accepted": False, "detail": detail, "state": trainer.status()})
    return {"accepted": True, "detail": detail, "state": {"state": "running"}}


@app.get("/status")
def status():
    return trainer.status()
