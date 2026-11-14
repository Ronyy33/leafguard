"""End-to-end tests against a RUNNING stack (docker compose up).

Skipped unless LEAFGUARD_E2E=1.  Run:  LEAFGUARD_E2E=1 pytest tests/e2e -v
"""

import os
import re
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.skipif(os.getenv("LEAFGUARD_E2E") != "1", reason="set LEAFGUARD_E2E=1 with the stack running")

API = os.getenv("API_BASE_URL", "http://localhost:8000")
UI = os.getenv("UI_URL", "http://localhost:8080")
PROM = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _counter(name_regex: str) -> float:
    text = requests.get(f"{API}/metrics/", timeout=5).text
    return sum(float(m.group(1)) for m in re.finditer(rf"^{name_regex}\S* ([0-9.e+]+)$", text, re.M))


def test_ui_served_and_configured():
    assert "Check a plant leaf" in requests.get(UI, timeout=5).text
    assert "API_BASE_URL" in requests.get(f"{UI}/config.js", timeout=5).text


def test_backend_ready_via_model_server():
    r = requests.get(f"{API}/ready", timeout=5).json()
    assert r["ready"] and r["model_server_reachable"]


@pytest.mark.parametrize(
    "fname,expected",
    [
        ("tomato_healthy.jpg", "Tomato___healthy"),
        ("potato_late_blight.jpg", "Potato___Late_blight"),
        ("apple_scab.jpg", "Apple___Apple_scab"),
    ],
)
def test_real_leaf_predictions(fname, expected):
    before = _counter("leafguard_predictions_total")
    r = requests.post(
        f"{API}/predict", files={"file": (fname, (SAMPLES / fname).read_bytes(), "image/jpeg")}, timeout=30
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == expected
    assert body["inference_backend"] == "mlflow_serve"
    assert _counter("leafguard_predictions_total") == before + 1


def test_prometheus_scrapes_all_targets():
    targets = requests.get(f"{PROM}/api/v1/targets", timeout=5).json()["data"]["activeTargets"]
    health = {t["labels"]["job"]: t["health"] for t in targets}
    assert all(health.get(j) == "up" for j in ("backend", "trainer", "node", "prometheus")), health


def test_pipeline_status_lists_production_model():
    r = requests.get(f"{API}/pipeline/status", timeout=10).json()
    assert any(v["is_production"] for v in r["model_versions"])
    assert r["trainer"]["state"] in {"idle", "running", "succeeded", "failed"}
