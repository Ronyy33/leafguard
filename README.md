# LeafGuard: AIOps-driven plant disease detection

LeafGuard classifies photos of leaves into 10 disease and healthy classes for tomato, potato and apple. Around the model it runs a full AIOps lifecycle, built only with the DA3408 toolchain:

| Concern | Tool |
|---|---|
| Code / data / model versioning | Git, Git LFS, DVC |
| Data pipeline (ingest → validate → preprocess) | Ray Data |
| CI pipeline | DVC DAG (`dvc repro`) |
| Experiment tracking, registry, serving | MLflow |
| Backend API | FastAPI |
| Monitoring and alerting | Prometheus, Alertmanager, Grafana |
| Packaging | Docker, Docker Compose |

See [docs/problem_statement.md](docs/problem_statement.md) for the problem, the data and the success metrics.

## Quick start (development)
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
dvc pull          # fetch data/raw from the DVC remote
dvc repro         # run the whole pipeline
dvc dag           # show the pipeline DAG
```

## Repository layout
```
src/data/        Ray pipeline stages: eda, ingest, validate, preprocess
src/train/       training and evaluation (MLflow tracked)
src/monitoring/  drift statistics shared by the pipeline and the API
backend/         FastAPI inference service
frontend/        web UI (separate container, talks to the backend only over REST)
monitoring/      Prometheus, Alertmanager and Grafana config
docs/            architecture, HLD, LLD, test plan, user manual
reports/         metrics and plots produced by the pipeline (DVC metrics)
```
