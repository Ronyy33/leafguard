# LeafGuard: Demo Script (about 10 minutes)

Do the prep before the evaluator joins. Each step names the rubric item it demonstrates.

## Prep
```bash
export UID GID=$(id -g)
docker compose up -d && docker compose ps     # all services up
curl -s localhost:8000/ready                  # {"ready":true,...,"model_server_reachable":true}
curl -s -X POST localhost:8000/drift/reset >/dev/null
```

Open these tabs:
1. UI on :8080
2. Pipeline console on :8080/pipeline.html
3. Grafana on :3000
4. MLflow on :5000
5. Prometheus alerts on :9090/alerts
6. A terminal in the repo

## 1. The user experience (Web UI/UX, 3 points)
1. In the UI, show the green status pill ("Online · model v2").
2. Drag `tests/samples/potato_late_blight.jpg` onto the box and click **Check this leaf**. Point out:
   - the red verdict
   - the confidence bar
   - the top-3 possibilities
   - the Grad-CAM heatmap
   - the model version, latency and "via MLflow model server" line
3. Click **No, it's wrong**, pick a class and click **Send**. This is feedback, and it feeds the live accuracy metric.
4. Show the foolproofing: upload a `.txt` or PDF to get a friendly error, and upload a photo larger than 5 MB to get a size error.
5. Open **Help** (the user manual). Resize the window to phone width to show the layout is responsive.

## 2. Versioning and CI (2 points)
```bash
git log --oneline | head
cat data/raw.dvc                 # dataset version (md5) under DVC
dvc dag                          # the CI DAG
dvc status                       # everything up to date, so nothing reruns
dvc metrics show                 # metrics from every stage
dvc exp show --only-changed      # experiments: frozen vs fine-tuned
git lfs ls-files                 # LFS-tracked samples
```
Explain how caching works: change `train.lr` and only `train`, `evaluate` (and `test`) rerun.

## 3. Data engineering (2 points)
Open the pipeline console's DAG and its throughput line (ingest, validate and preprocess run at about 700 items/s with Ray). Show `reports/validation.json`: the exact duplicate it found and the per-class counts.

## 4. Experiment tracking (2 points)
1. In MLflow, open the `leafguard` experiment and compare the two runs:
   - frozen backbone: F1 0.88, **rejected by the gate**
   - fine-tuned: F1 0.966, **promoted**
2. Show what goes beyond autolog:
   - per-epoch curves
   - the tags `git_commit` and `dvc_data_md5`
   - the artifacts: confusion matrix, classification report, `labels.json`, `baseline_stats.json`, data reports
   - the model logged with a signature and input example
3. Open **Models**: `leafguard` with v1 and v2, where v2 carries the `@production` alias.

## 5. Monitoring, then drift, then automatic retraining (2 points; the climax of the demo)
1. In Grafana, walk through the rows: overview, traffic and latency, drift, model behaviour, pipeline, infrastructure.
2. In the terminal, send normal traffic:
   ```bash
   python scripts/simulate_traffic.py -n 40
   ```
   Grafana updates in near real time, and the drift z-scores stay near 0.
3. Simulate field photos taken in poor light:
   ```bash
   python scripts/simulate_traffic.py -n 40 --darken 0.35 --no-explain
   ```
   Watch the following happen:
   - The Grafana drift panel goes red, with z around −25.
   - The console shows "DRIFT".
   - Prometheus shows `InputBrightnessDrift` as *pending*, then *firing*, after about 30 s.
   - Alertmanager calls the backend webhook, and the backend starts the trainer.
   - The console shows **Pipeline: running**, the DAG stages light up, and the log streams.
4. While it runs (about 8 minutes), explain the gate and the hot-swap: the new version is promoted only if F1 ≥ the current one, and the model server and backend poll the alias every 30 s.
5. Show rollback: click **Roll back model** twice in the console. The registry table and the "Serving model" figure change.

## 6. Packaging (2 points)
```bash
docker compose ps
```
- Point out the separate frontend and backend services.
- Explain the single multi-stage `Dockerfile`, and that `frontend/Dockerfile` is separate.
- Show `config.js` being injected from `API_BASE_URL`, so the UI and backend stay loosely coupled.
- Show `MLproject`: `mlflow run . -e train -P epochs=3`.

## 7. Software engineering (5 points)
Walk through `docs/`:
- architecture
- HLD
- LLD with the OpenAPI spec
- test plan and report: `dvc repro test` gives 46/46
- user manual

Run `flake8` to show it's clean. Show the OO modules (`ModelService`, `RuntimeMonitor`, `Trainer`), the logging and the exception handlers.

## Likely viva questions (have answers ready)
- **Why Ray, not Spark?** Image work is Python-native (PIL/OpenCV). Ray runs it in parallel with no JVM, and the same code scales from a laptop to a cluster.
- **Why both MLflow serve and FastAPI?** MLflow provides standard model APIfication. FastAPI is the product API: validation, explanations, drift, feedback, metrics and pipeline control.
- **How do you avoid retrain storms?** Alerts need `for: 30s` and at least 20 samples, Alertmanager uses `repeat_interval: 1h`, and the trainer allows one run at a time with a 600 s cooldown for automatic triggers.
- **How do you prevent a bad model going live?** The gate in `evaluate.py` (F1 ≥ 0.90 and ≥ production), plus one-click rollback.
- **How is an experiment reproduced?** Use the MLflow run's `git_commit` tag, then `git checkout <sha> && dvc checkout && dvc repro`. The data is pinned by `dvc_data_md5`.
- **Limitations:** PlantVillage photos are lab images, so real field photos drift, which is why the system monitors for it. Retraining currently reruns on the versioned dataset. Adding feedback-labelled uploads to the training data is the next step.
