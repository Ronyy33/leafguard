#!/usr/bin/env bash
# MLflow model server with continuous deployment.
#
# Serves models:/$MODEL_NAME@$MODEL_ALIAS with `mlflow models serve` (REST: /ping, /invocations).
# Every $POLL_S seconds it re-resolves the alias; when retraining promotes a new
# version (or an operator rolls back), the server is restarted on the new version.
set -u
MODEL_NAME=${MODEL_NAME:-leafguard}
MODEL_ALIAS=${MODEL_ALIAS:-production}
PORT=${PORT:-5001}
POLL_S=${POLL_S:-30}

resolve_version() {
  python - <<EOF 2>/dev/null
from mlflow import MlflowClient
print(MlflowClient().get_model_version_by_alias("$MODEL_NAME", "$MODEL_ALIAS").version)
EOF
}

current=""
pid=""
while true; do
  v=$(resolve_version)
  if [ -z "$v" ]; then
    echo "[serve_model] no ${MODEL_NAME}@${MODEL_ALIAS} yet; waiting..."
  elif [ "$v" != "$current" ]; then
    if [ -n "$pid" ]; then
      echo "[serve_model] alias moved v${current} -> v${v}; restarting server"
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    fi
    echo "[serve_model] serving ${MODEL_NAME} v${v} on :${PORT}"
    mlflow models serve -m "models:/${MODEL_NAME}/${v}" -h 0.0.0.0 -p "$PORT" --env-manager local -w 1 &
    pid=$!
    current=$v
  elif ! kill -0 "$pid" 2>/dev/null; then
    echo "[serve_model] server process died; restarting"
    current=""
    continue
  fi
  sleep "$POLL_S"
done
