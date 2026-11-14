"""Integration tests: FastAPI backend endpoints against the LLD contract (docs/LLD.md)."""

import base64
import io

import numpy as np
from PIL import Image

from tests.conftest import LABELS, png_bytes


def _predict(api, data, ctype="image/jpeg", explain=True):
    return api.post(f"/predict?explain={str(explain).lower()}", files={"file": ("leaf.jpg", data, ctype)})


# ---------------------------------------------------------------- ops
def test_health(api):
    assert api.get("/health").json() == {"status": "ok"}


def test_ready_with_model_but_no_model_server(api):
    r = api.get("/ready").json()
    assert r["ready"] is True and r["model_loaded"] is True and r["model_server_reachable"] is False


def test_model_info(api):
    r = api.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "7" and body["classes"] == LABELS and body["num_classes"] == len(LABELS)


def test_classes(api):
    r = api.get("/classes").json()
    assert [c["label"] for c in r] == LABELS
    assert r[0]["display_name"] == "Apple – Apple scab"


# ------------------------------------------------------------ predict
def test_predict_happy_path(api, sample_jpg):
    r = _predict(api, sample_jpg)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["label"] in LABELS
    assert 0 <= b["confidence"] <= 1
    assert len(b["top3"]) == 3
    assert abs(sum(c["probability"] for c in b["top3"]) - 1) < 1  # top-3 is a subset of a distribution
    assert b["top3"][0]["label"] == b["label"]
    assert b["inference_backend"] == "local"
    assert b["model_version"] == "7"
    assert set(b["input_quality"]) == {"brightness", "blur", "mean_r", "mean_g", "mean_b"}
    png = base64.b64decode(b["heatmap_png_base64"])
    assert Image.open(io.BytesIO(png)).size == (256, 256)


def test_predict_without_explanation(api, sample_jpg):
    assert _predict(api, sample_jpg, explain=False).json()["heatmap_png_base64"] is None


def test_predict_png(api, leaf_like):
    assert _predict(api, png_bytes(leaf_like), ctype="image/png").status_code == 200


def test_predict_rejects_wrong_type(api):
    r = _predict(api, b"hello", ctype="text/plain")
    assert r.status_code == 415 and "JPG" in r.json()["detail"]


def test_predict_rejects_empty(api):
    assert _predict(api, b"").status_code == 400


def test_predict_rejects_corrupt_image(api):
    r = _predict(api, b"\xff\xd8\xff garbage")
    assert r.status_code == 400 and "image" in r.json()["detail"]


def test_predict_rejects_too_large(api):
    big = np.random.default_rng(0).integers(0, 255, (1200, 1200, 3), dtype=np.uint8)
    r = _predict(api, png_bytes(big), ctype="image/png")  # ~4 MB > 1 MB test limit
    assert r.status_code == 413


def test_dark_photo_warns(api, sample_jpg):
    dark = (np.asarray(Image.open(io.BytesIO(sample_jpg))) * 0.1).astype(np.uint8)
    r = _predict(api, png_bytes(dark), ctype="image/png").json()
    assert any("dark" in w for w in r["warnings"])


def test_upload_is_stored_encrypted(api, sample_jpg):
    from app import main

    pid = _predict(api, sample_jpg).json()["prediction_id"]
    raw = (main.store.dir / f"{pid}.bin").read_bytes()
    assert sample_jpg[:20] not in raw  # not plaintext on disk
    assert main.store.load(pid) == sample_jpg  # but decryptable


# ----------------------------------------------------------- feedback
def test_feedback_flow(api, sample_jpg):
    pred = _predict(api, sample_jpg).json()
    r = api.post("/feedback", json={"prediction_id": pred["prediction_id"], "true_label": pred["label"]})
    assert r.status_code == 200
    assert r.json()["correct"] is True and r.json()["feedback_count"] >= 1
    # same prediction cannot be rated twice
    again = api.post("/feedback", json={"prediction_id": pred["prediction_id"], "true_label": pred["label"]})
    assert again.status_code == 404


def test_feedback_queues_labelled_photo_for_retraining(api, sample_jpg, tmp_path, monkeypatch):
    from app import main
    from app.config import Settings

    monkeypatch.setattr(main, "settings", Settings(inference_mode="local", feedback_dir=str(tmp_path / "fb")))
    pid = _predict(api, sample_jpg).json()["prediction_id"]
    api.post("/feedback", json={"prediction_id": pid, "true_label": "Tomato___healthy"})
    saved = tmp_path / "fb" / "Tomato___healthy" / f"{pid}.bin"
    assert saved.exists() and main.store.fernet.decrypt(saved.read_bytes()) == sample_jpg


def test_feedback_unknown_label(api, sample_jpg):
    pid = _predict(api, sample_jpg).json()["prediction_id"]
    assert api.post("/feedback", json={"prediction_id": pid, "true_label": "Banana___x"}).status_code == 422


def test_feedback_unknown_prediction(api):
    assert api.post("/feedback", json={"prediction_id": "nope", "true_label": LABELS[0]}).status_code == 404


def test_feedback_validation_error(api):
    assert api.post("/feedback", json={"true_label": LABELS[0]}).status_code == 422


# ------------------------------------------------------- monitoring
def test_drift_window_fills_and_resets(api, sample_jpg):
    api.post("/drift/reset")
    for _ in range(3):
        _predict(api, sample_jpg, explain=False)
    d = api.get("/drift").json()
    assert d["samples_in_window"] == 3 and d["threshold"] == 3.0
    assert api.post("/drift/reset").json()["samples_in_window"] == 0


def test_metrics_exposed(api, sample_jpg):
    _predict(api, sample_jpg, explain=False)
    text = api.get("/metrics/").text
    for name in (
        "leafguard_requests_total",
        "leafguard_request_latency_seconds_bucket",
        "leafguard_predictions_total",
        "leafguard_drift_zscore",
        "leafguard_input_bytes",
    ):
        assert name in text


# --------------------------------------------------- pipeline hooks
def test_retrain_trigger_when_trainer_down(api):
    r = api.post("/retrain/trigger").json()
    assert r["accepted"] is False and "unreachable" in r["detail"]


def test_alert_webhook_ignores_non_retrain_alerts(api):
    payload = {"alerts": [{"status": "firing", "labels": {"alertname": "HighPredictLatency"}}]}
    assert api.post("/alerts/webhook", json=payload).json()["retrain"] is False


def test_alert_webhook_triggers_retrain_on_drift(api, monkeypatch):
    from app import main

    calls = []

    class Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"detail": "pipeline started", "state": {"state": "running"}}

    monkeypatch.setattr(main.requests, "post", lambda url, **kw: calls.append((url, kw)) or Resp())
    payload = {
        "alerts": [
            {"status": "firing", "labels": {"alertname": "InputBrightnessDrift"}},
            {"status": "resolved", "labels": {"alertname": "AccuracyDrop"}},
        ]
    }
    r = api.post("/alerts/webhook", json=payload).json()
    assert r["retrain"] is True
    assert calls[0][0].endswith("/retrain") and "InputBrightnessDrift" in calls[0][1]["json"]["reason"]


def test_model_not_loaded_returns_503(api, sample_jpg):
    from app import main

    loaded, main.service._loaded = main.service._loaded, None
    try:
        r = _predict(api, sample_jpg)
        assert r.status_code == 503 and r.json()["error"] == "model_unavailable"
    finally:
        main.service._loaded = loaded
