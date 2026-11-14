# LeafGuard: Low-Level Design

**Rohan Sirsewad (DA24B023)** · DA3408 AI Operations (Prof. Sudarsun) · IIT Madras

The machine-readable contract is [`openapi.json`](openapi.json), exported from the running backend. It's also browsable live at `http://localhost:8000/docs`. This document is the human-readable version.

## 1. Module design (OO)

| Module | Class / function | Responsibility |
|---|---|---|
| `backend/app/config.py` | `Settings` (frozen dataclass) | All config from env vars (MLflow URI, model alias, model server URL, trainer URL, upload limit, encryption key, CORS, drift window). |
| `backend/app/model_service.py` | `ModelService` | Resolves `models:/leafguard@production`, loads the model and its metadata (labels, baseline), runs remote/local inference with fallback, Grad-CAM, hot reload, rollback. |
| | `LoadedModel` (dataclass) | Immutable snapshot of the served version: model, version, run_id, labels, baseline, image_size, backbone, test F1. |
| `backend/app/monitor.py` | `RuntimeMonitor` | Drift window (`DriftDetector`), recent-prediction registry for feedback, rolling accuracy, Prometheus export. |
| `backend/app/storage.py` | `EncryptedStore` | Fernet encrypt/decrypt of uploads. |
| `backend/app/metrics.py` | — | Every Prometheus metric definition. |
| `backend/app/main.py` | FastAPI `app` | Routes, request-metrics middleware, exception handlers, registry poller thread. |
| `src/monitoring/drift.py` | `compute_image_stats`, `Baseline`, `DriftDetector` | Shared image statistics and z-score drift. |
| `src/train/model.py` | `LeafNet(nn.Module)` | MobileNetV2/ResNet18 transfer model with built-in normalization. |
| `trainer/app.py` | `Trainer` | Pipeline run state machine (`idle → running → succeeded/failed`), log capture, stage timing, cooldown, metrics. |

## 2. Backend REST API (`http://<host>:8000`)

**Conventions:**
- All bodies are JSON unless stated otherwise.
- Errors are `{"detail": "<human readable message>"}` (FastAPI `HTTPException`), or `{"error": "<code>", "detail": "..."}` from the global handlers.
- Each request is counted in `leafguard_requests_total{endpoint,method,status}` and timed in `leafguard_request_latency_seconds{endpoint}`.

### 2.1 `POST /predict`: diagnose a leaf
**Request:** `multipart/form-data`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `file` | binary | yes | `image/jpeg`, `image/png` or `image/webp`; at most `MAX_UPLOAD_MB` (5 MB); must decode as an image |
| `explain` (query) | bool | no, default `true` | include a Grad-CAM heatmap |

**Response 200** (`PredictionResponse`):
```json
{
  "prediction_id": "365e3e5d2b8a4012afd3f1dc1996b324",
  "label": "Tomato___Late_blight",
  "display_name": "Tomato – Late blight",
  "is_healthy": false,
  "confidence": 0.94,
  "low_confidence": false,
  "top3": [{"label": "Tomato___Late_blight", "display_name": "Tomato – Late blight", "probability": 0.94}, "..."],
  "heatmap_png_base64": "iVBORw0KGgo...",
  "model_version": "2",
  "inference_backend": "mlflow_serve",
  "latency_ms": 113.4,
  "input_quality": {"brightness": 88.7, "blur": 401.1, "mean_r": 92.7, "mean_g": 88.8, "mean_b": 77.4},
  "warnings": []
}
```

| Field | Type | Meaning |
|---|---|---|
| `prediction_id` | string (32 hex) | Use this when sending feedback |
| `label` / `display_name` | string | Class id / human-readable name |
| `is_healthy` | bool | The label ends with `healthy` |
| `looks_like_leaf` | bool | `false` if under 0.5% of the pixels are leaf-coloured (HSV hue 10°–100°, saturated, not black). The UI then shows a neutral ❓ verdict and a warning. |
| `confidence` | float [0,1] | Top-1 softmax probability |
| `low_confidence` | bool | `confidence < LOW_CONFIDENCE_THRESHOLD` (0.6) |
| `top3` | ClassScore[3] | The three most likely classes |
| `heatmap_png_base64` | string or null | 256×256 Grad-CAM overlay PNG; null if `explain=false` |
| `model_version` | string | Registry version that answered |
| `inference_backend` | `mlflow_serve` or `local` | `local` means the model server was unavailable and the fallback was used |
| `latency_ms` | float | Server-side latency |
| `input_quality` | object | Brightness, blur (Laplacian variance) and RGB means at 128 px |
| `warnings` | string[] | User-facing tips: dark photo, blurry photo, low confidence |

**Errors:**

| Code | When | Example `detail` |
|---|---|---|
| 400 | Empty file, or bytes that don't decode as an image | "The file could not be read as an image." |
| 413 | File larger than the limit | "File is larger than 5 MB." |
| 415 | Wrong content type | "Unsupported file type 'text/plain'. Please upload a JPG or PNG photo." |
| 422 | `file` missing | FastAPI validation error |
| 503 | No model loaded | `{"error": "model_unavailable", ...}` |

**Side effects:**
- Updates the drift window and drift gauges.
- Stores the upload encrypted as `UPLOAD_DIR/<prediction_id>.bin`.
- Increments `leafguard_predictions_total{label}` and the confidence histogram.

### 2.2 `POST /feedback`: report the true class
**Request:** `{"prediction_id": "<id from /predict>", "true_label": "<class id from GET /classes>"}`

**Response 200:** `{"accepted": true, "correct": false, "rolling_accuracy": 0.93, "feedback_count": 57}`

| Code | When |
|---|---|
| 404 | Unknown `prediction_id`, or it was already rated (each prediction can be rated once) |
| 422 | `true_label` isn't a known class, or the body is invalid |

**Side effects:**
- Updates `leafguard_feedback_total{correct}` and `leafguard_rolling_accuracy` (last 100 feedbacks). The `AccuracyDrop` alert depends on these.
- Copies the encrypted upload to `FEEDBACK_DIR/<true_label>/<prediction_id>.bin` and increments `leafguard_labelled_samples_total`. The next retrain imports it into the training data.

### 2.3 Read-only model endpoints

| Method and path | Response |
|---|---|
| `GET /health` | `{"status": "ok"}` (liveness) |
| `GET /ready` | `{"ready": bool, "model_loaded": bool, "model_server_reachable": bool, "detail": str}` |
| `GET /classes` | `[{"label", "display_name", "probability": 0}]` for every class the model knows |
| `GET /model/info` | `{"name", "alias", "version", "run_id", "backbone", "num_classes", "image_size", "test_macro_f1", "classes": [...]}`; 503 if no model is loaded |
| `GET /drift` | `{"window_size": 50, "samples_in_window": n, "zscores": {"brightness": z, "blur": z, "mean_r": z, "mean_g": z, "mean_b": z}, "drifting_features": ["brightness"], "threshold": 3.0}` |
| `GET /metrics/` | Prometheus text exposition format |

### 2.4 Operator endpoints

| Method and path | Request | Response | Notes |
|---|---|---|---|
| `POST /model/reload` | — | `{"ok": true, "detail": "reloaded", "version": "2"}` | Reloads whatever `@production` points to |
| `POST /model/rollback` | — | `{"ok": true, "detail": "rolled back to v1", "version": "1"}` | 409 if there's no older version. Tags the current version `rolled_back=true`. |
| `POST /drift/reset` | — | Same as `GET /drift` | Clears the rolling window, e.g. after retraining |
| `POST /retrain/trigger?reason=<text>` | query `reason` (≤200 chars) | `{"accepted": bool, "detail": str, "state": {...}}` | Proxies to the trainer. `accepted=false` if a run is in progress or the trainer is unreachable. |
| `GET /pipeline/status?limit=10` | query `limit` 1–50 | `{"trainer": {...}, "runs": [...], "model_versions": [...], "experiment_id": str, "served_version": str}` | Data for the pipeline console |
| `POST /alerts/webhook` | Alertmanager webhook payload v4 | `{"retrain": bool, "firing": [...], "trainer": {...}}` | Retrains only when `InputBrightnessDrift`, `InputBlurDrift` or `AccuracyDrop` is *firing* |

`runs[]` items have these fields: `run_id`, `name`, `status`, `start_time` (epoch ms), `duration_s`, `git_commit`, `val_macro_f1`, `test_macro_f1`, `train_seconds`.

`model_versions[]` items have these fields: `version`, `run_id`, `created`, `test_macro_f1`, `rolled_back`, `is_production`.

## 3. Trainer REST API (`http://<host>:8001`)

| Method and path | Request | Response |
|---|---|---|
| `POST /retrain` | `{"reason": "manual"}` | **202** `{"accepted": true, "detail": "pipeline started: bash scripts/retrain.sh", ...}`, or **409** `{"accepted": false, "detail": "a pipeline run is already in progress" \| "cooldown active (Ns left)", ...}` |
| `GET /status` | — | `{"state": "idle\|running\|succeeded\|failed", "current": {...}, "history": [{"reason", "started", "finished", "duration_s", "status", "stages": {"ingest": 9.8, ...}, "eval": {...}}], "log_tail": [...], "reports": {...}}` |
| `GET /health` | — | `{"status": "ok"}` |
| `GET /metrics/` | — | `leafguard_pipeline_*`, `leafguard_data_quality`, `leafguard_last_eval_*` |

Cooldown (`RETRAIN_COOLDOWN_S`, default 600 s) applies to automatic triggers only. A reason that starts with `manual` always runs.

## 4. Model server (`http://<host>:5001`, MLflow scoring protocol)

| Method and path | Request | Response |
|---|---|---|
| `GET /ping` | — | 200 when healthy |
| `POST /invocations` | `{"inputs": [[[[float]]]]}`, a float32 tensor of shape `[N, 3, 128, 128]` with values in [0,1] | `{"predictions": [[logit × 10]]}` |

The signature is inferred at log time (`mlflow.models.infer_signature`) and stored in the model's `MLmodel` file.

## 5. Prometheus metrics

| Metric | Type | Labels |
|---|---|---|
| `leafguard_requests_total` | counter | endpoint, method, status |
| `leafguard_request_latency_seconds` | histogram | endpoint |
| `leafguard_inference_latency_seconds` | histogram | backend |
| `leafguard_errors_total` | counter | type |
| `leafguard_input_bytes` | summary | — |
| `leafguard_predictions_total` | counter | label |
| `leafguard_prediction_confidence` | histogram | — |
| `leafguard_low_confidence_total` | counter | — |
| `leafguard_inference_fallback_total` | counter | — |
| `leafguard_input_stat` | gauge | feature |
| `leafguard_drift_zscore` | gauge | feature |
| `leafguard_drift_window_samples` | gauge | — |
| `leafguard_feedback_total` | counter | correct |
| `leafguard_rolling_accuracy` | gauge | — |
| `leafguard_model_info` | info | name, version, run_id, backbone |
| `leafguard_model_version` | gauge | — |
| `leafguard_model_reloads_total` | counter | reason |
| `leafguard_model_server_up` | gauge | — |
| `leafguard_retrain_requests_total` | counter | source, accepted |
| `leafguard_labelled_samples_total` | counter | — |
| `leafguard_not_leaf_total` | counter | — |
| `leafguard_mlflow_up` | gauge | — |
| `leafguard_pipeline_runs_total` | counter (trainer) | status |
| `leafguard_pipeline_running` | gauge (trainer) | — |
| `leafguard_pipeline_last_duration_seconds` | gauge (trainer) | — |
| `leafguard_pipeline_last_success_timestamp` | gauge (trainer) | — |
| `leafguard_pipeline_stage_seconds` | gauge (trainer) | stage |
| `leafguard_pipeline_throughput` | gauge (trainer) | stage |
| `leafguard_data_quality` | gauge (trainer) | kind |
| `leafguard_last_eval_macro_f1` | gauge (trainer) | — |
| `leafguard_last_eval_promoted` | gauge (trainer) | — |

## 6. Configuration (environment variables)

| Variable | Default | Used by |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | backend, trainer, model-server |
| `MODEL_NAME` / `MODEL_ALIAS` | `leafguard` / `production` | backend, model-server |
| `INFERENCE_MODE` | `mlflow_serve` | backend (`local` disables the model server) |
| `MODEL_SERVER_URL` | `http://localhost:5001` | backend |
| `TRAINER_URL` | `http://localhost:8001` | backend |
| `MAX_UPLOAD_MB` | `5` | backend |
| `ENCRYPTION_KEY` | from `.env` (gitignored; template in `.env.example`) | backend, trainer |
| `DRIFT_WINDOW` | `50` | backend |
| `LOW_CONFIDENCE_THRESHOLD` | `0.6` | backend |
| `CORS_ORIGINS` | `*` | backend |
| `API_BASE_URL` | `http://localhost:8000` | frontend (injected into `config.js`) |
| `FEEDBACK_DIR` | `feedback` (backend), `data/feedback` (trainer) | backend, trainer |
| `RETRAIN_COMMAND` | `bash scripts/retrain.sh` | trainer |
| `RETRAIN_COOLDOWN_S` | `600` | trainer |
