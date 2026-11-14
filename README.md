# LeafGuard: AIOps-driven plant disease detection

**Rohan Sirsewad (DA24B023)** · DA3408 AI Operations (Prof. Sudarsun) · IIT Madras

LeafGuard classifies photos of tomato, potato and apple leaves into 10 healthy or disease classes, and shows a Grad-CAM heatmap of what it looked at. Around the model it runs a full AIOps lifecycle built only with the DA3408 toolchain:
- versioned data
- a Ray data pipeline
- a DVC CI DAG
- MLflow tracking, registry and serving
- a FastAPI backend
- Prometheus, Grafana and Alertmanager monitoring with **drift-triggered automatic retraining**
- Docker Compose packaging

| Concern | Tool | Where |
|---|---|---|
| Code / data / model versioning | Git, Git LFS, DVC | `.gitattributes`, `data/raw.dvc`, `dvc.lock` |
| Data engineering (ingest → validate → preprocess) | **Ray Data** | `src/data/` |
| CI pipeline | **DVC DAG** (`dvc repro`) | `dvc.yaml`, `params.yaml` |
| Experiment tracking + registry | **MLflow** | `src/train/train.py`, `src/train/evaluate.py` |
| Reproducible env | **MLproject** | `MLproject`, `python_env.yaml` |
| Model APIfication | **`mlflow models serve`** | `scripts/serve_model.sh` |
| Web API | **FastAPI** | `backend/app/` |
| Monitoring / alerting | **Prometheus, Alertmanager, Grafana** | `monitoring/` |
| Packaging | **Docker, Docker Compose** | `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |

**Results:**
- Production model v5 (MobileNetV2, fine-tuned with field-condition augmentation) has test **macro-F1 0.962**. It is robust to field photos: F1 is **0.93 on dark** and **0.88 on blurry** test photos, up from 0.84 and 0.20 for the previous model.
- p95 `/predict` latency is **158 ms** on CPU, including the heatmap.
- 59/59 unit and integration tests pass.

## Documentation
| Doc | Purpose |
|---|---|
| [docs/problem_statement.md](docs/problem_statement.md) | Problem, data, success metrics |
| [docs/architecture.md](docs/architecture.md) | Architecture diagram and description of each block, plus the DVC DAG |
| [docs/HLD.md](docs/HLD.md) | High-level design: decisions and rationale |
| [docs/LLD.md](docs/LLD.md) | Low-level design: every endpoint with its I/O ([openapi.json](docs/openapi.json)) |
| [docs/test_plan.md](docs/test_plan.md) | Test plan, test cases, acceptance criteria, test report |
| [docs/user_manual.md](docs/user_manual.md) | Manual for non-technical users (also in-app under **Help**) |
| [docs/demo_script.md](docs/demo_script.md) | Step-by-step demo and viva walkthrough |

## Run everything (Docker)
```bash
dvc pull                              # data + model weights from the DVC remote
cp .env.example .env                  # then set ENCRYPTION_KEY (command is in the file)
export UID GID=$(id -g)               # containers write to the repo as you
docker compose up -d --build
```

| Service | URL |
|---|---|
| UI | http://localhost:8080 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Alertmanager | http://localhost:9093 |
| Trainer | http://localhost:8001/status |

The first start needs an MLflow registry with a `leafguard@production` model. Run the pipeline once (below, or click **Run pipeline now** in the console).

## Getting the data without the DVC remote
The DVC remote is a local folder, so a fresh clone can rebuild the exact dataset from the public PlantVillage repo instead:
```bash
python scripts/download_data.py   # downloads the 2,852 files listed in data/plantvillage_subset.txt
dvc status data/raw.dvc           # "up to date" proves the files are byte-identical to the versioned dataset
```

## Develop locally (no Docker)
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
source .venv/bin/activate
mlflow server --port 5000 --backend-store-uri sqlite:///mlflow_store/mlflow.db --artifacts-destination ./mlflow_store/artifacts &
dvc repro                             # eda, ingest, validate, preprocess, train, evaluate, test
dvc dag                               # the CI DAG
dvc metrics show                      # all pipeline metrics
dvc exp run -S train.lr=0.0005        # an experiment with an overridden param (also tracked in MLflow)
mlflow run . -e train -P epochs=3     # same, through the MLproject entry point
pytest                                # tests (E2E: LEAFGUARD_E2E=1 pytest tests/e2e)
```

## Repository layout
```
src/data/          Ray pipeline stages: eda, ingest, validate, preprocess
src/train/         model, training and evaluation/registration (MLflow)
src/monitoring/    drift statistics shared by the pipeline and the API
backend/app/       FastAPI service (config, schemas, model service, monitor, metrics, storage)
trainer/           pipeline-runner service (dvc repro on demand)
frontend/          web UI (nginx container, REST-only coupling)
monitoring/        Prometheus rules, Alertmanager routing, Grafana provisioning + dashboard
scripts/           model-server supervisor, traffic simulator, dashboard generator, test report
tests/             unit, integration and e2e tests; samples/ (Git LFS)
docs/              architecture, HLD, LLD, test plan, user manual, demo script
reports/           DVC metrics and plots
```
