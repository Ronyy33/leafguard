# LeafGuard: End-to-End Test Report

**Rohan Sirsewad (DA24B023)** · DA3408 AI Operations (Prof. Sudarsun) · IIT Madras

**Date:** 2026-11-14
**Environment:** `docker compose up -d --build` on a single aarch64 machine (6 CPUs, 7 GB RAM, no GPU), 9 services.

## 1. Automated E2E suite
```
$ LEAFGUARD_E2E=1 pytest tests/e2e -v
7 passed in 0.75s
```

| Test | Result |
|---|---|
| `test_ui_served_and_configured`: nginx serves the UI, and `config.js` is injected from env | ✅ |
| `test_backend_ready_via_model_server`: backend ready, MLflow model server reachable | ✅ |
| `test_real_leaf_predictions[tomato_healthy]`: correct class, served by `mlflow_serve`, Prometheus counter +1 | ✅ |
| `test_real_leaf_predictions[potato_late_blight]` | ✅ |
| `test_real_leaf_predictions[apple_scab]` | ✅ |
| `test_prometheus_scrapes_all_targets`: backend, trainer, node and prometheus all `up` | ✅ |
| `test_pipeline_status_lists_production_model`: registry shows a production version | ✅ |

## 2. Closed-loop scenario: drift, alert, automatic retrain, hot-swap
No manual step was involved after sending the traffic.

| Time | Event | Evidence |
|---|---|---|
| 22:07:39 | 30 normal uploads: accuracy 1.00, p95 147 ms | `simulate_traffic.py` |
| 22:07:43 | 40 dark uploads (brightness ×0.35): accuracy drops to **0.825** | `simulate_traffic.py --darken 0.35` |
| 22:08:01 | `InputBrightnessDrift` and `InputBlurDrift` **pending** (brightness z = −24.4, blur z = −7.3) | Prometheus `/api/v1/alerts` |
| 22:08:31 | Both alerts **firing** (`for: 30s`) | Prometheus |
| 22:08:36 | Alertmanager posts to `backend:8000/alerts/webhook`, and the backend calls `trainer:8001/retrain` | backend log `alert webhook: firing=['InputBrightnessDrift']` |
| 22:08:36 | Second alert (`InputBlurDrift`) arrives while a run is active and is **rejected** (single-run lock) | trainer |
| — | **Defect found:** the first run failed in the container with `getpwuid(): uid not found: 1000`. Fixed by creating the user in the trainer image (`Dockerfile`, `APP_UID` build arg). | trainer `/status` log tail |
| — | Alertmanager correctly did **not** re-notify after a restart (its notification log persisted, `repeat_interval: 1h`). This is the anti-storm behavior working as designed. | Alertmanager |
| 22:18:27 | Fresh Alertmanager re-delivers the firing alert, and the pipeline starts (reason `alert:InputBrightnessDrift`) | trainer `/status` |
| 22:27:19 | Pipeline **succeeded** in 532 s: ingest 10.4 s, validate 15.6 s, preprocess 18.8 s, train 478.4 s, evaluate 7.9 s | trainer `/status` |
| 22:27:19 | v4 registered: test F1 0.9662 ≥ production 0.9662 and ≥ gate 0.90, so it is **promoted** to `@production` | evaluate log |
| 22:27:39 | Backend hot-loads v4 (`reason=alias_changed`). The model server logs `alias moved v2 -> v4; restarting server` | container logs |
| 22:28 | `POST /predict` (apple scab) returns `Apple___Apple_scab`, 0.999, **v4**, `mlflow_serve` | curl |

**Rollback** was tested separately: `POST /model/rollback` moved `@production` from v3 back to v2 and tagged v3 `rolled_back=true`. The backend reloaded immediately.

**Reproducibility:** three independent trainings with identical data, params and code (v2 on the host, v3 on the host, v4 in Docker) gave the **identical** test macro-F1 of 0.9662.

## 3. Acceptance criterion AC10
The full stack starts with one command and all 7 E2E tests pass, so **AC10 is met ✅**.

## 4. Known limitations found during testing
- With no new feedback, retraining on the same versioned data reproduces the same model. This was the reason for adding the feedback-import step (`scripts/retrain.sh`, tested in `tests/test_import_feedback.py` and `test_feedback_queues_labelled_photo_for_retraining`). Labelled field photos now become a new DVC data version before each retrain.
- The first build needs about 6 GB of disk for images. On a nearly full disk, build services one at a time (`docker compose build backend`, then `trainer`, and so on).
