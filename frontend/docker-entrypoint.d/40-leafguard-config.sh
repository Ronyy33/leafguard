#!/bin/sh
# Write runtime configuration for the browser from environment variables.
cat > /usr/share/nginx/html/config.js <<CFG
window.LEAFGUARD_CONFIG = {
  API_BASE_URL: "${API_BASE_URL:-http://localhost:8000}",
  GRAFANA_URL: "${GRAFANA_URL:-http://localhost:3000}",
  MLFLOW_URL: "${MLFLOW_URL:-http://localhost:5000}",
  PROMETHEUS_URL: "${PROMETHEUS_URL:-http://localhost:9090}",
  ALERTMANAGER_URL: "${ALERTMANAGER_URL:-http://localhost:9093}"
};
CFG
echo "leafguard: API_BASE_URL=${API_BASE_URL:-http://localhost:8000}"
