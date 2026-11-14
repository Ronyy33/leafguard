"""Prometheus instrumentation for the backend (pattern from DA3408 Monitoring/ai_app.py).

Every metric is prefixed `leafguard_`. Scraped by Prometheus at GET /metrics.
"""

from prometheus_client import Counter, Gauge, Histogram, Info, Summary

REQUESTS = Counter("leafguard_requests_total", "HTTP requests", ["endpoint", "method", "status"])
LATENCY = Histogram(
    "leafguard_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5),
)
INFERENCE_LATENCY = Histogram(
    "leafguard_inference_latency_seconds",
    "Model inference latency only",
    ["backend"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2),
)
ERRORS = Counter("leafguard_errors_total", "Errors by type", ["type"])
INPUT_BYTES = Summary("leafguard_input_bytes", "Size of uploaded images in bytes")

PREDICTIONS = Counter("leafguard_predictions_total", "Predictions per class", ["label"])
CONFIDENCE = Histogram(
    "leafguard_prediction_confidence",
    "Top-1 confidence",
    buckets=(0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
LOW_CONFIDENCE = Counter("leafguard_low_confidence_total", "Predictions below the confidence threshold")
FALLBACKS = Counter("leafguard_inference_fallback_total", "Model-server failures served by the local model")

# data drift (rolling window vs training baseline)
INPUT_STAT = Gauge("leafguard_input_stat", "Last upload's image statistic", ["feature"])
DRIFT_Z = Gauge("leafguard_drift_zscore", "Rolling-window drift z-score vs training baseline", ["feature"])
DRIFT_WINDOW_FILL = Gauge("leafguard_drift_window_samples", "Samples currently in the drift window")

# model performance from user feedback
FEEDBACK = Counter("leafguard_feedback_total", "User feedback received", ["correct"])
ROLLING_ACC = Gauge("leafguard_rolling_accuracy", "Accuracy over the last N feedback items")

# model lifecycle
MODEL_INFO = Info("leafguard_model", "Currently served model")
MODEL_VERSION = Gauge("leafguard_model_version", "Registry version currently served")
MODEL_RELOADS = Counter("leafguard_model_reloads_total", "Model (re)loads", ["reason"])
MODEL_SERVER_UP = Gauge("leafguard_model_server_up", "1 if the MLflow model server is reachable")
RETRAIN_REQUESTS = Counter("leafguard_retrain_requests_total", "Retrain triggers", ["source", "accepted"])
LABELLED_SAMPLES = Counter("leafguard_labelled_samples_total", "User-labelled uploads queued for the next retrain")
MLFLOW_UP = Gauge("leafguard_mlflow_up", "1 if the MLflow tracking server / registry is reachable")
