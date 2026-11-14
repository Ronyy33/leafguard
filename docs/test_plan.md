# LeafGuard: Test Plan and Test Report

## 1. Scope
**In scope:**
- The data pipeline: validation and transforms
- Drift statistics
- The model definition
- The backend REST API (every endpoint in [LLD.md](LLD.md))
- The trainer state machine
- The deployed stack end to end
- ML quality and system performance against the acceptance criteria

**Out of scope:** third-party services themselves (MLflow, Prometheus, Grafana internals) and browser compatibility beyond current Chrome and Firefox.

## 2. Strategy

| Level | Tooling | Where | Runs |
|---|---|---|---|
| Unit | pytest | `tests/test_drift.py`, `test_data_pipeline.py`, `test_model.py`, `test_trainer.py` | Every `dvc repro` (`test` stage) |
| Integration (API) | pytest + FastAPI `TestClient`, an untrained in-process model, no network | `tests/test_api.py` | Every `dvc repro` (`test` stage) |
| End-to-end | pytest + `requests` against the running compose stack | `tests/e2e/test_stack.py` | `LEAFGUARD_E2E=1 pytest tests/e2e` after `docker compose up` |
| ML quality | held-out test split, `src/train/evaluate.py` | DVC `evaluate` stage | Every training run (doubles as the promotion gate) |
| Performance / drift | `scripts/simulate_traffic.py` | Manual or demo | Before each release |
| Static | flake8 (PEP 8, max line length 120), black | repo | Before each commit |

**Test data:** three real leaf photos in `tests/samples/` (Git LFS), synthetic images generated in the fixtures, and the DVC-versioned test split of 428 images.

## 3. Acceptance criteria

| ID | Criterion | Threshold | Measured | Result |
|---|---|---|---|---|
| AC1 | Test-set macro-F1 | ≥ 0.90 | **0.966** (model v2) | ✅ |
| AC2 | Minimum per-class recall | ≥ 0.80 | **0.844** (Tomato Early blight) | ✅ |
| AC3 | p95 `/predict` latency, with heatmap, CPU, sequential | < 200 ms | **158 ms** (60 requests) | ✅ |
| AC4 | API 5xx error rate under normal traffic | < 1 % | **0 %** (0 of 160 requests) | ✅ |
| AC5 | Invalid inputs rejected with a clear 4xx message | 100 % | 415 / 400 / 413 / 422 verified | ✅ |
| AC6 | Brightness drift detected within one window (50 uploads) | \|z\| > 3 | **z = −25.7** after 40 dark uploads | ✅ |
| AC7 | A worse model is never promoted | gate holds | v1 (F1 0.881) rejected; v2 (0.966) promoted | ✅ |
| AC8 | Data validation rejects bad files and keeps good ones | bad ≤ 5 % | 1 duplicate dropped, 2,851 kept | ✅ |
| AC9 | All unit and integration tests pass | 100 % | **46 / 46** | ✅ |
| AC10 | Full stack starts with one command and E2E tests pass | 100 % | see §5 | see §5 |

## 4. Test cases
The full per-test table is generated on every run at `reports/tests/test_cases.md` (via `scripts/test_report.py` from the JUnit XML).

| ID | Area | Test | Input | Expected |
|---|---|---|---|---|
| TC01 | API | `test_health` | GET /health | 200 `{"status":"ok"}` |
| TC02 | API | `test_ready_with_model_but_no_model_server` | Model loaded, model server down | `ready=true`, `model_server_reachable=false` |
| TC03 | API | `test_model_info` | GET /model/info | version, classes, num_classes |
| TC04 | API | `test_classes` | GET /classes | all labels with display names |
| TC05 | API | `test_predict_happy_path` | Real JPG leaf | 200; label ∈ classes; confidence ∈ [0,1]; top3 sorted; 256×256 heatmap |
| TC06 | API | `test_predict_without_explanation` | `explain=false` | `heatmap_png_base64 = null` |
| TC07 | API | `test_predict_png` | Synthetic PNG | 200 |
| TC08 | API | `test_predict_rejects_wrong_type` | text/plain | 415 |
| TC09 | API | `test_predict_rejects_empty` | 0 bytes | 400 |
| TC10 | API | `test_predict_rejects_corrupt_image` | JPEG header + garbage | 400 |
| TC11 | API | `test_predict_rejects_too_large` | ~4 MB with a 1 MB limit | 413 |
| TC12 | API | `test_dark_photo_warns` | Photo at 10 % brightness | warning contains "dark" |
| TC13 | Security | `test_upload_is_stored_encrypted` | Any upload | file on disk ≠ plaintext; decrypts to the original |
| TC14 | API | `test_feedback_flow` | Correct label, then a repeat | 200 `correct=true`; repeat gives 404 |
| TC15 | API | `test_feedback_unknown_label` | `Banana___x` | 422 |
| TC16 | API | `test_feedback_unknown_prediction` | Random id | 404 |
| TC17 | API | `test_feedback_validation_error` | Missing `prediction_id` | 422 |
| TC18 | Monitoring | `test_drift_window_fills_and_resets` | 3 uploads, then reset | samples 3, then 0 |
| TC19 | Monitoring | `test_metrics_exposed` | GET /metrics/ | the 5 core metric families are present |
| TC20 | Pipeline | `test_retrain_trigger_when_trainer_down` | Trainer offline | `accepted=false`, "unreachable", no crash |
| TC21 | Pipeline | `test_alert_webhook_ignores_non_retrain_alerts` | HighPredictLatency firing | `retrain=false` |
| TC22 | Pipeline | `test_alert_webhook_triggers_retrain_on_drift` | InputBrightnessDrift firing | POST /retrain called with the alert reason |
| TC23 | Resilience | `test_model_not_loaded_returns_503` | No model | 503 `model_unavailable` |
| TC24–29 | Data | `test_data_pipeline.py` (6) | Valid, grayscale, too small, corrupt, missing; transform | accept / reject reasons; 64×64×3 uint8 + stats columns |
| TC30–36 | Drift | `test_drift.py` (7) | Synthetic leaf, darkened, blurred | brightness and blur move the expected way; z < −3 on dark inputs; baseline round-trips |
| TC37–42 | Model | `test_model.py` (6) | Both backbones; freeze flags | output `[N,10]`; trainable params as configured |
| TC43–46 | Trainer | `test_trainer.py` (4) | Fake pipelines | stage parsing; failure state; single-run lock; cooldown for alerts but not manual |
| TC47–53 | E2E | `tests/e2e/test_stack.py` (7) | Running stack | UI served with config; backend uses the model server; the 3 real samples classified correctly; all Prometheus targets up; registry has a production version |

## 5. Test report

**Run of 2026-11-14, `dvc repro test`, commit `be8b884`:**

| Suite | Total | Passed | Failed | Skipped |
|---|---|---|---|---|
| Unit + integration | 46 | **46** | 0 | 0 |
| E2E (needs the running stack) | 7 | — | — | 7 (run separately, see below) |

**Performance and drift run** (`scripts/simulate_traffic.py`, local stack, model v2 via `mlflow models serve`):

| Scenario | Requests | OK | Accuracy | p50 | p95 | Throughput |
|---|---|---|---|---|---|---|
| Normal, with heatmap, concurrency 1 | 60 | 60 | 1.000 | 113 ms | 158 ms | 8.3 req/s |
| Normal, no heatmap, concurrency 4 | 60 | 60 | 1.000 | 221 ms | 271 ms | 16.7 req/s |
| Dark photos (×0.35), no heatmap | 40 | 40 | **0.825** | 54 ms | 83 ms | 16.6 req/s |

These accuracy figures come from random dataset images, most of which were in the training split. They are smoke-test numbers. The held-out figure is AC1.

On the dark-photo run, the drift z-score for brightness reached **−25.7**, past the threshold of 3. The accuracy drop on dark photos shows why drift monitoring is needed.

**E2E results:** see [`test_report_e2e.md`](test_report_e2e.md) (filled after `docker compose up`).

**Conclusion:** acceptance criteria AC1–AC9 are met.
