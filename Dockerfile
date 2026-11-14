# Multi-stage build: one shared dependency layer, one target per Python service.
#   base          pinned runtime deps (torch CPU, mlflow, ray, dvc, fastapi, ...)
#   backend       FastAPI inference API
#   trainer       DVC pipeline runner (repo is bind-mounted at /repo)
#   mlflow        MLflow tracking server + model registry
#   model-server  `mlflow models serve` for models:/leafguard@production
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=1 HOME=/tmp
RUN apt-get update && apt-get install -y --no-install-recommends curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

# ---------------------------------------------------------------- backend
FROM base AS backend
WORKDIR /app
COPY src/ /app/src/
COPY backend/app/ /app/backend/app/
ENV PYTHONPATH=/app:/app/backend
WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ---------------------------------------------------------------- trainer
FROM base AS trainer
# the DVC `test` stage runs the test suite as part of the CI pipeline
RUN pip install pytest==8.3.5 httpx==0.28.1
# runs as the host user (compose `user:`) so files written into the bind-mounted repo stay
# owned by them; DVC needs a passwd entry for that uid
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} app 2>/dev/null || true; useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash app
WORKDIR /repo
ENV REPO_DIR=/repo PYTHONPATH=/repo
EXPOSE 8001
CMD ["uvicorn", "trainer.app:app", "--host", "0.0.0.0", "--port", "8001"]

# ---------------------------------------------------------------- mlflow
FROM base AS mlflow
WORKDIR /mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "2", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts"]

# ---------------------------------------------------------------- model-server
FROM base AS model-server
COPY scripts/serve_model.sh /usr/local/bin/serve_model.sh
RUN chmod +x /usr/local/bin/serve_model.sh
EXPOSE 5001
CMD ["serve_model.sh"]
