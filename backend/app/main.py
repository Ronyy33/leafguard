"""LeafGuard backend: FastAPI service between the web UI and the model.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000   (from the backend/ dir,
      with the repo root on PYTHONPATH so `src.*` imports resolve)

Endpoint contract: docs/LLD.md (also live at /docs).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

import requests
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from prometheus_client import make_asgi_app

from app import metrics as M
from app.config import settings
from app.model_service import InvalidImageError, ModelNotLoadedError, ModelService, display_name
from app.monitor import RuntimeMonitor
from app.schemas import (ClassScore, DriftStatus, ErrorResponse, FeedbackRequest, FeedbackResponse,
                         HealthResponse, ModelActionResponse, ModelInfo, PredictionResponse, ReadyResponse,
                         RetrainResponse)
from app.storage import EncryptedStore

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("leafguard.api")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
RETRAIN_ALERTS = {"InputBrightnessDrift", "InputBlurDrift", "AccuracyDrop"}
REGISTRY_POLL_S = 30

app = FastAPI(
    title="LeafGuard API",
    version="1.0.0",
    description="Plant leaf disease detection with drift monitoring and automated retraining.",
    responses={500: {"model": ErrorResponse}},
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
app.mount("/metrics", make_asgi_app())

service = ModelService(settings)
monitor = RuntimeMonitor(window=settings.drift_window)
store = EncryptedStore(settings.upload_dir, settings.encryption_key)


# ----------------------------------------------------------------- lifecycle
def _try_load(reason: str) -> None:
    try:
        loaded = service.load(reason=reason)
        monitor.set_baseline(loaded.baseline)
    except Exception as exc:
        logger.error("could not load model (%s): %s", reason, exc)
        M.ERRORS.labels(type="model_load").inc()


def _registry_poller() -> None:
    """Continuous deployment: pick up a newly promoted / rolled-back version automatically."""
    while True:
        time.sleep(REGISTRY_POLL_S)
        if not service.is_loaded:
            _try_load("retry")
        elif service.reload_if_changed():
            monitor.set_baseline(service.loaded.baseline)
        try:
            requests.get(f"{settings.model_server_url}/ping", timeout=2).raise_for_status()
            M.MODEL_SERVER_UP.set(1)
        except Exception:
            M.MODEL_SERVER_UP.set(0)


@app.on_event("startup")
def startup() -> None:
    _try_load("startup")
    threading.Thread(target=_registry_poller, daemon=True, name="registry-poller").start()


# ---------------------------------------------------------------- middleware
@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Count + time every request. Labels use the route template to keep cardinality bounded."""
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        if not endpoint.startswith("/metrics"):
            M.REQUESTS.labels(endpoint=endpoint, method=request.method, status=str(status)).inc()
            M.LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t0)


# ------------------------------------------------------------ error handling
@app.exception_handler(ModelNotLoadedError)
async def _no_model(_: Request, exc: ModelNotLoadedError):
    M.ERRORS.labels(type="model_not_loaded").inc()
    return JSONResponse(status_code=503, content={"error": "model_unavailable",
                                                  "detail": "The model is not loaded yet. Try again shortly."})


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    logger.exception("unhandled error")
    M.ERRORS.labels(type="unhandled").inc()
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})


# -------------------------------------------------------------------- health
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready():
    """Readiness: a model is loaded (model server reachability is informational; local fallback exists)."""
    try:
        requests.get(f"{settings.model_server_url}/ping", timeout=2).raise_for_status()
        reachable = True
    except Exception:
        reachable = False
    loaded = service.is_loaded
    return {"ready": loaded, "model_loaded": loaded, "model_server_reachable": reachable,
            "detail": "" if loaded else "model not loaded"}


# --------------------------------------------------------------------- model
@app.get("/classes", response_model=list[ClassScore], tags=["model"])
def classes():
    """All classes the current model can predict (probability field is 0)."""
    return [ClassScore(label=lbl, display_name=display_name(lbl), probability=0.0)
            for lbl in service.loaded.labels]


@app.get("/model/info", response_model=ModelInfo, tags=["model"])
def model_info():
    m = service.loaded
    return ModelInfo(name=settings.model_name, alias=settings.model_alias, version=m.version, run_id=m.run_id,
                     backbone=m.backbone, num_classes=len(m.labels), image_size=m.image_size,
                     test_macro_f1=m.test_macro_f1, classes=m.labels)


@app.post("/model/reload", response_model=ModelActionResponse, tags=["model"])
def model_reload():
    """Reload whatever version the registry alias points to now."""
    loaded = service.load(reason="manual")
    monitor.set_baseline(loaded.baseline)
    return {"ok": True, "detail": "reloaded", "version": loaded.version}


@app.post("/model/rollback", response_model=ModelActionResponse, tags=["model"])
def model_rollback():
    """Move @production to the previous registered version and load it."""
    try:
        version = service.rollback()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    monitor.set_baseline(service.loaded.baseline)
    return {"ok": True, "detail": f"rolled back to v{version}", "version": version}


# ---------------------------------------------------------------- inference
@app.post("/predict", response_model=PredictionResponse, tags=["inference"],
          responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                     415: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def predict(file: UploadFile = File(..., description="Leaf photo (JPG/PNG/WEBP, max 5 MB)"),
                  explain: bool = Query(True, description="Include a Grad-CAM heatmap")):
    t0 = time.perf_counter()
    if file.content_type not in ALLOWED_TYPES:
        M.ERRORS.labels(type="bad_content_type").inc()
        raise HTTPException(415, f"Unsupported file type {file.content_type!r}. Please upload a JPG or PNG photo.")
    data = await file.read()
    M.INPUT_BYTES.observe(len(data))
    if not data:
        M.ERRORS.labels(type="empty_file").inc()
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        M.ERRORS.labels(type="too_large").inc()
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb:g} MB.")

    try:
        rgb = service.decode(data)
    except InvalidImageError:
        M.ERRORS.labels(type="invalid_image").inc()
        raise HTTPException(400, "The file could not be read as an image.")

    quality = monitor.observe_image(rgb)
    logits, backend = service.predict_logits(service.to_tensor(rgb))
    probs = service.softmax(logits)
    labels = service.loaded.labels
    order = probs.argsort()[::-1]
    top = int(order[0])
    label, conf = labels[top], float(probs[top])

    warnings = []
    base = service.loaded.baseline
    if quality["brightness"] < base.mean["brightness"] - 3 * base.std["brightness"]:
        warnings.append("Photo looks very dark; try better lighting.")
    if quality["blur"] < base.mean["blur"] - 2 * base.std["blur"] and quality["blur"] < 100:
        warnings.append("Photo looks blurry; hold the camera steady and focus on the leaf.")
    low = conf < settings.low_confidence_threshold
    if low:
        M.LOW_CONFIDENCE.inc()
        warnings.append("The model is not confident. Consider retaking the photo or asking an expert.")

    heatmap = service.gradcam_png(rgb, top) if explain else None
    pid = uuid.uuid4().hex
    monitor.remember_prediction(pid, label)
    if settings.store_uploads:
        try:
            store.save(pid, data)
        except Exception as exc:  # storage must never break inference
            logger.warning("could not store upload: %s", exc)
            M.ERRORS.labels(type="storage").inc()

    M.PREDICTIONS.labels(label=label).inc()
    M.CONFIDENCE.observe(conf)
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("predict id=%s label=%s conf=%.3f backend=%s %.1fms", pid, label, conf, backend, latency_ms)
    return PredictionResponse(
        prediction_id=pid, label=label, display_name=display_name(label),
        is_healthy=label.endswith("healthy"), confidence=conf, low_confidence=low,
        top3=[ClassScore(label=labels[i], display_name=display_name(labels[i]), probability=float(probs[i]))
              for i in order[:3]],
        heatmap_png_base64=heatmap, model_version=service.loaded.version, inference_backend=backend,
        latency_ms=round(latency_ms, 1), input_quality={k: round(v, 2) for k, v in quality.items()},
        warnings=warnings,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["inference"],
          responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def feedback(body: FeedbackRequest):
    """User tells us the correct class -> live accuracy metric (drives the AccuracyDrop alert)."""
    if body.true_label not in service.loaded.labels:
        raise HTTPException(422, f"Unknown class {body.true_label!r}")
    try:
        correct, acc, n = monitor.record_feedback(body.prediction_id, body.true_label)
    except KeyError:
        raise HTTPException(404, "Unknown or already-rated prediction_id")
    return FeedbackResponse(accepted=True, correct=correct, rolling_accuracy=round(acc, 4), feedback_count=n)


# --------------------------------------------------------------- monitoring
@app.get("/drift", response_model=DriftStatus, tags=["monitoring"])
def drift():
    return monitor.drift_status()


@app.post("/drift/reset", response_model=DriftStatus, tags=["monitoring"])
def drift_reset():
    monitor.reset_window()
    return monitor.drift_status()


# ------------------------------------------------------- pipeline / retrain
def _trigger_retrain(reason: str, source: str) -> dict:
    try:
        r = requests.post(f"{settings.trainer_url}/retrain", json={"reason": reason}, timeout=5)
        body = r.json()
        accepted = r.status_code == 202
    except Exception as exc:
        M.ERRORS.labels(type="trainer_unreachable").inc()
        body, accepted = {"detail": f"trainer unreachable: {exc}", "state": {}}, False
    M.RETRAIN_REQUESTS.labels(source=source, accepted=str(accepted).lower()).inc()
    return {"accepted": accepted, "detail": body.get("detail", ""), "state": body.get("state", {})}


@app.post("/retrain/trigger", response_model=RetrainResponse, tags=["pipeline"])
def retrain_trigger(reason: str = Query("manual", max_length=200)):
    """Start `dvc repro` on the trainer service (409-style accepted=false if already running)."""
    return _trigger_retrain(reason, source="manual")


@app.post("/alerts/webhook", tags=["pipeline"])
async def alert_webhook(request: Request):
    """Alertmanager webhook receiver: firing drift/accuracy alerts trigger retraining."""
    payload = await request.json()
    firing = [a["labels"].get("alertname") for a in payload.get("alerts", []) if a.get("status") == "firing"]
    relevant = sorted(set(firing) & RETRAIN_ALERTS)
    logger.warning("alert webhook: firing=%s", firing)
    if not relevant:
        return {"retrain": False, "firing": firing}
    result = _trigger_retrain(f"alert:{','.join(relevant)}", source="alertmanager")
    return {"retrain": result["accepted"], "firing": firing, "trainer": result}


@app.get("/pipeline/status", tags=["pipeline"])
def pipeline_status(limit: int = Query(10, ge=1, le=50)):
    """Pipeline console data: trainer state + recent MLflow runs + registry versions."""
    try:
        trainer = requests.get(f"{settings.trainer_url}/status", timeout=3).json()
    except Exception as exc:
        trainer = {"state": "unreachable", "detail": str(exc)}

    runs, versions = [], []
    try:
        client = MlflowClient(settings.mlflow_tracking_uri)
        exp = client.get_experiment_by_name("leafguard")
        if exp:
            for r in client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=limit):
                end = r.info.end_time or int(time.time() * 1000)
                runs.append({
                    "run_id": r.info.run_id, "name": r.info.run_name, "status": r.info.status,
                    "start_time": r.info.start_time, "duration_s": round((end - r.info.start_time) / 1000, 1),
                    "git_commit": r.data.tags.get("git_commit", "")[:8],
                    "val_macro_f1": r.data.metrics.get("best_val_macro_f1"),
                    "test_macro_f1": r.data.metrics.get("test_macro_f1"),
                    "train_seconds": r.data.metrics.get("train_total_seconds"),
                })
        prod = service.production_version() if service.is_loaded else None
        for v in client.search_model_versions(f"name='{settings.model_name}'"):
            versions.append({"version": v.version, "run_id": v.run_id, "created": v.creation_timestamp,
                             "test_macro_f1": v.tags.get("test_macro_f1"),
                             "rolled_back": v.tags.get("rolled_back") == "true",
                             "is_production": v.version == prod})
        versions.sort(key=lambda v: int(v["version"]), reverse=True)
    except Exception as exc:
        logger.warning("mlflow query failed: %s", exc)
        M.ERRORS.labels(type="mlflow_query").inc()
    return {"trainer": trainer, "runs": runs, "model_versions": versions,
            "served_version": service.loaded.version if service.is_loaded else None}
