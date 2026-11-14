# LeafGuard: High-Level Design

**Rohan Sirsewad (DA24B023)** · DA3408 AI Operations (Prof. Sudarsun) · IIT Madras

## 1. Goals and non-goals

**Goals:**
- Diagnose leaf diseases from a photo for non-technical users.
- Explain every prediction with a heatmap.
- Run a fully automated, reproducible lifecycle: versioned data, a tracked pipeline, a gated registry, monitored serving, and alert-driven retraining.

**Non-goals (v1):**
- Mobile app
- Multi-leaf or whole-plant images
- Cloud deployment (forbidden by the course)
- Multi-tenant authentication

## 2. Design principles
1. **Loose coupling.** The UI, API, model server, trainer and monitoring are separate processes and containers that talk only over HTTP. The UI knows a single configurable URL.
2. **One source of truth per concern:**
   - `params.yaml` for tunables
   - `dvc.yaml` for the pipeline
   - the MLflow Registry alias `@production` for "which model is live"
   - `src/monitoring/drift.py` for image statistics, used identically in preprocessing (the baseline) and serving (live)
3. **Reproducibility.** Every MLflow run is tagged with the Git commit and the DVC md5 of the data. `dvc.lock` pins the hash of every stage input and output.
4. **Safe automation.** Retraining can never make production worse. The evaluate stage promotes a model only if test macro-F1 is at least the 0.90 gate *and* at least the current production F1. A cooldown and single-run lock prevent retrain storms, and one click rolls back.
5. **Graceful degradation.** If the model server is down, the backend serves from its local copy of the same registry version and raises `ModelServerDown`. If no model is loaded, `/predict` returns 503 with a friendly message instead of crashing.

## 3. Key design decisions and rationale

| Decision | Alternatives considered | Rationale |
|---|---|---|
| **Ray Data** for ingest, validate and preprocess | Spark | The data is images handled in Python (PIL/OpenCV). Ray runs Python functions in parallel natively, with no JVM and no serialization to DataFrames of bytes, and the same code scales from a laptop to a cluster. Spark suits tabular ETL better. |
| **DVC pipeline as CI** | Shell scripts, Makefile | Content-hash caching means only changed stages rerun. `dvc dag` documents the pipeline, `dvc metrics/params diff` compares experiments, and `dvc exp run -S` runs parameter sweeps. |
| **MobileNetV2 transfer learning**, 128 px input | ResNet18, training from scratch | Fast on CPU (about 1 min/epoch fine-tuning) and a small model of 9 MB. Test macro-F1 0.966 already beats the 0.90 target. ResNet18 is supported through `params.yaml`. |
| **Frozen backbone first, then full fine-tune** | — | Experiment 1 (frozen) reached F1 0.88 and was rejected by the gate. Experiment 2 (unfrozen, lr 3e-4) reached 0.966 and was promoted. Both are tracked in MLflow, which demonstrates that the gate works. |
| **Normalization inside the model** | Client-side normalization | Any client (FastAPI, `mlflow models serve`, tests) sends plain [0,1] pixels, so there's no train/serve skew. |
| **`mlflow models serve` + FastAPI** | FastAPI only, Ray Serve | The rubric asks for MLflow APIfication of the model. FastAPI adds what a raw model server can't: validation, user-friendly errors, Grad-CAM, drift, feedback, metrics and pipeline control. |
| **Grad-CAM in the backend** | SHAP, LIME | One extra backward pass (~30 ms) gives a visual that non-technical users understand ("where the model looked"). |
| **Registry alias polling** for deployment | Restart containers from CI | No privileged Docker socket is needed. Both the model server and the backend converge to `@production` within 30 s, and rollback is just moving the alias. |
| **Drift = rolling z-score of brightness and blur** | KS test, PSI | Interpretable for operators ("photos are 25σ darker than training"), cheap per request, and maps directly onto the project statement's examples (brightness, blur). The window of 50 and the minimum of 20 samples avoid alerting on a single odd photo. |
| **Feedback-labelled uploads become training data** | Retrain on the static dataset only | When a user confirms or corrects a prediction, the encrypted upload is queued under `data/feedback/<true label>/`. Before every retrain, `src/data/import_feedback.py` decrypts these into `data/raw`, and `scripts/retrain.sh` runs `dvc add data/raw`, which creates a new data version. Retraining therefore adapts to real field photos (dark, blurry, cluttered) instead of only reproducing the lab-photo model. The validate stage still checks the new photos and drops duplicates. |
| **Field-condition augmentation + robustness metrics** | Augment with flips only | The drift demo showed the lab-trained model fails on field-like photos: on darkened and blurred copies of the test set, v4 scored F1 0.84 (dark) and **0.20 (blur)**. Training now optionally adds dark and blur augmentation (`train.field_augment`). Evaluation logs `test_dark_macro_f1` and `test_blur_macro_f1` to MLflow on every run, so robustness is tracked like accuracy. Augmentation runs in OpenCV on uint8 arrays, because torch ops inside forked DataLoader workers deadlock. |
| **"Not a leaf" check (colour heuristic)** | A trained "other" class | No non-leaf training data is available in the course setup. A leaf-colour pixel ratio (HSV) flags blank pages, sky and screenshots, and wrongly flags under 1% of real leaves, measured on 600 dataset photos. The photo is flagged with a warning, not rejected, so a false flag never blocks a real user. A learned "other" class is the robust fix. |
| **Dataset manifest + download script** | Only the DVC remote | The DVC remote is a local folder. `data/plantvillage_subset.txt` plus `scripts/download_data.py` rebuild the exact files from the public source, and `dvc status data/raw.dvc` proves they are byte-identical. That gives reproducibility from a fresh clone with no cloud storage. |
| **Accuracy from user feedback** | Only offline evaluation | Ground truth exists in production only when users confirm or correct a prediction. A rolling accuracy over the last 100 feedbacks drives the `AccuracyDrop` alert. |
| **Fernet encryption of stored uploads** | Plaintext storage | Meets the security best practice: user images are kept for future labelled retraining but never stored in plaintext. The key comes from the `ENCRYPTION_KEY` env var. |
| **Single multi-stage Dockerfile** for Python services | One Dockerfile per service | Torch, MLflow and Ray are about 1.5 GB of dependencies. Sharing the layer cuts build time and disk use by about 4×. The frontend still has its own Dockerfile. |
| **SQLite MLflow backend** | PostgreSQL | One-machine deployment, so there's no extra service to operate. The store lives in the bind-mounted `mlflow_store/`, so it survives container rebuilds. |

## 4. Data design
- **Source:** the PlantVillage *color* set. We use 10 classes with up to 300 images each, 2,852 files in total. `data/raw` is tracked by DVC (md5 `d53bdfd0…`).
- **Validation:** checks that each file decodes, its color mode, its minimum size, and that it isn't a sha256 duplicate. One exact duplicate was found and dropped.
- **Split:** stratified 70/15/15 with seed 42, giving 1,995 train, 428 val and 428 test images.
- **Baseline:** the mean and std of brightness, blur and RGB means over the **training split only**, saved as `baseline_stats.json`. It's logged with the model as an artifact, so each model version carries its own drift reference.

## 5. Model lifecycle
```
user feedback ─► data/feedback ─► import_feedback ─► dvc add data/raw (new data version)
params.yaml change ─► dvc exp run / dvc repro ─► MLflow run (params, metrics, artifacts, model)
      ─► evaluate: register vN ─► gate (F1 ≥ 0.90 and ≥ prod) ─► alias @production = vN
      ─► model-server + backend poll alias ─► serve vN ─► Prometheus/Grafana ─► alerts ─► retrain
```
**Rollback:** `POST /model/rollback` (or the console button) moves `@production` to the previous version and tags the bad version `rolled_back=true`.

## 6. Observability design
- **Metrics:** latency histograms, request and error counters by type, predictions per class, a confidence histogram, drift z-scores, rolling accuracy, the model version, model-server up, pipeline runs, stage durations, throughput and data quality.
- **Alerts:**
  - `InputBrightnessDrift`, `InputBlurDrift` and `AccuracyDrop` trigger a retrain.
  - `HighErrorRate`, `HighPredictLatency`, `ModelServerDown`, `ServiceDown` and `PipelineFailed` alert humans.
- **Logs:** structured `time level logger: message` lines from every Python component. Pipeline logs go to `logs/pipeline.log`, and the trainer streams the `dvc repro` output to the console.

## 7. Security considerations
- Uploads are validated by content type, size (5 MB) and decodability, and stored encrypted (Fernet: AES-128-CBC + HMAC).
- No secrets are in Git: the encryption key and Grafana password come from environment variables.
- CORS origins are configurable. Admin endpoints (retrain, rollback) are intended for the internal network. Adding authentication is listed as future work.

## 8. Future work
- Train with brightness and blur augmentation so the model is more robust to field conditions.
- Add authentication on the admin endpoints.
- Add Kubernetes manifests (Minikube) for horizontal scaling of the model server.
