"""Generate the Grafana dashboard JSON (monitoring/grafana/dashboards/leafguard.json).

Kept as code so the dashboard is reviewable and reproducible.
Usage: python scripts/build_dashboard.py
"""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
_id = 0


def panel(title, kind, targets, x, y, w, h, unit=None, thresholds=None, extra=None, desc=""):
    global _id
    _id += 1
    p = {
        "id": _id,
        "title": title,
        "type": kind,
        "datasource": DS,
        "description": desc,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {"datasource": DS, "expr": e, "legendFormat": lg, "refId": chr(65 + i)} for i, (e, lg) in enumerate(targets)
        ],
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {},
    }
    d = p["fieldConfig"]["defaults"]
    if unit:
        d["unit"] = unit
    if thresholds:
        d["thresholds"] = {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in thresholds]}
        d["color"] = {"mode": "thresholds"}
    if kind == "timeseries":
        d["custom"] = {"lineWidth": 2, "fillOpacity": 10, "showPoints": "never"}
        p["options"] = {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}
    if kind == "stat":
        p["options"] = {"colorMode": "background", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}}
    if extra:
        for k, v in extra.items():
            p.setdefault(k, {}).update(v) if isinstance(v, dict) else p.__setitem__(k, v)
    return p


def row(title, y):
    global _id
    _id += 1
    return {"id": _id, "type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}}


P = []
# --- overview -------------------------------------------------------------
P.append(row("Overview", 0))
P.append(
    panel("Served model version", "stat", [("leafguard_model_version", "v")], 0, 1, 4, 4, thresholds=[(None, "blue")])
)
P.append(
    panel(
        "Requests / s",
        "stat",
        [("sum(rate(leafguard_requests_total[1m]))", "rps")],
        4,
        1,
        4,
        4,
        unit="reqps",
        thresholds=[(None, "green")],
    )
)
P.append(
    panel(
        "Error rate (5xx)",
        "stat",
        [
            (
                'sum(rate(leafguard_requests_total{status=~"5.."}[5m])) / clamp_min(sum(rate(leafguard_requests_total[5m])),1e-9)',
                "",
            )
        ],
        8,
        1,
        4,
        4,
        unit="percentunit",
        thresholds=[(None, "green"), (0.01, "red")],
    )
)
P.append(
    panel(
        "p95 /predict latency",
        "stat",
        [
            (
                'histogram_quantile(0.95, sum by (le) (rate(leafguard_request_latency_seconds_bucket{endpoint="/predict"}[5m])))',
                "",
            )
        ],
        12,
        1,
        4,
        4,
        unit="s",
        thresholds=[(None, "green"), (0.2, "orange"), (0.5, "red")],
    )
)
P.append(
    panel(
        "Live accuracy (feedback)",
        "stat",
        [("leafguard_rolling_accuracy", "")],
        16,
        1,
        4,
        4,
        unit="percentunit",
        thresholds=[(None, "red"), (0.8, "orange"), (0.9, "green")],
    )
)
P.append(
    panel(
        "Model server up",
        "stat",
        [("leafguard_model_server_up", "")],
        20,
        1,
        4,
        4,
        thresholds=[(None, "red"), (1, "green")],
        extra={
            "fieldConfig": {
                "defaults": {
                    "mappings": [{"type": "value", "options": {"0": {"text": "DOWN (fallback)"}, "1": {"text": "UP"}}}]
                }
            }
        },
    )
)

# --- traffic & latency ----------------------------------------------------
P.append(row("Traffic & latency", 5))
P.append(
    panel(
        "Requests by endpoint & status",
        "timeseries",
        [("sum by (endpoint, status) (rate(leafguard_requests_total[1m]))", "{{endpoint}} {{status}}")],
        0,
        6,
        12,
        8,
        unit="reqps",
    )
)
P.append(
    panel(
        "/predict latency percentiles",
        "timeseries",
        [
            (
                f'histogram_quantile({q}, sum by (le) (rate(leafguard_request_latency_seconds_bucket{{endpoint="/predict"}}[2m])))',
                f"p{int(q*100)}",
            )
            for q in (0.5, 0.95, 0.99)
        ]
        + [
            (
                "histogram_quantile(0.95, sum by (le, backend) (rate(leafguard_inference_latency_seconds_bucket[2m])))",
                "model p95 ({{backend}})",
            )
        ],
        12,
        6,
        12,
        8,
        unit="s",
    )
)

# --- data drift -----------------------------------------------------------
P.append(row("Data drift (inputs vs training baseline)", 14))
P.append(
    panel(
        "Drift z-score (alert at |z| > 3)",
        "timeseries",
        [('leafguard_drift_zscore{feature=~"brightness|blur"}', "{{feature}}")],
        0,
        15,
        12,
        8,
        thresholds=[(None, "green"), (3, "red")],
        extra={
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 0,
                        "showPoints": "never",
                        "thresholdsStyle": {"mode": "line+area"},
                    }
                }
            }
        },
        desc="Rolling-window mean of each feature vs training mean, in standard errors.",
    )
)
P.append(
    panel(
        "Last upload: brightness & sharpness",
        "timeseries",
        [
            ('leafguard_input_stat{feature="brightness"}', "brightness"),
            ('leafguard_input_stat{feature="blur"}', "sharpness (Laplacian var)"),
        ],
        12,
        15,
        12,
        8,
    )
)

# --- model behaviour ------------------------------------------------------
P.append(row("Model behaviour", 23))
P.append(
    panel(
        "Predictions per class (last 15m)",
        "bargauge",
        [("sum by (label) (increase(leafguard_predictions_total[15m]))", "{{label}}")],
        0,
        24,
        12,
        9,
        extra={
            "options": {
                "orientation": "horizontal",
                "displayMode": "gradient",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            }
        },
    )
)
P.append(
    panel(
        "Confidence & feedback",
        "timeseries",
        [
            (
                "histogram_quantile(0.5, sum by (le) (rate(leafguard_prediction_confidence_bucket[5m])))",
                "median confidence",
            ),
            (
                "sum(rate(leafguard_low_confidence_total[5m])) / clamp_min(sum(rate(leafguard_predictions_total[5m])),1e-9)",
                "low-confidence share",
            ),
            ("leafguard_rolling_accuracy", "live accuracy"),
        ],
        12,
        24,
        12,
        9,
        unit="percentunit",
    )
)

# --- pipeline -------------------------------------------------------------
P.append(row("Pipeline (DVC / Ray / training)", 33))
P.append(
    panel(
        "Pipeline running",
        "stat",
        [("leafguard_pipeline_running", "")],
        0,
        34,
        4,
        4,
        thresholds=[(None, "green"), (1, "orange")],
        extra={
            "fieldConfig": {
                "defaults": {
                    "mappings": [{"type": "value", "options": {"0": {"text": "idle"}, "1": {"text": "RUNNING"}}}]
                }
            }
        },
    )
)
P.append(
    panel(
        "Pipeline runs",
        "stat",
        [("sum by (status) (leafguard_pipeline_runs_total)", "{{status}}")],
        4,
        34,
        4,
        4,
        thresholds=[(None, "blue")],
    )
)
P.append(
    panel(
        "Last model test macro-F1",
        "stat",
        [("leafguard_last_eval_macro_f1", "")],
        8,
        34,
        4,
        4,
        unit="percentunit",
        thresholds=[(None, "red"), (0.9, "green")],
    )
)
P.append(
    panel(
        "Last pipeline duration",
        "stat",
        [("leafguard_pipeline_last_duration_seconds", "")],
        12,
        34,
        4,
        4,
        unit="s",
        thresholds=[(None, "blue")],
    )
)
P.append(
    panel(
        "Throughput per stage (items/s)",
        "bargauge",
        [("leafguard_pipeline_throughput", "{{stage}}")],
        16,
        34,
        8,
        8,
        extra={
            "options": {
                "orientation": "horizontal",
                "displayMode": "gradient",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            }
        },
    )
)
P.append(
    panel(
        "Stage durations, last run (s)",
        "bargauge",
        [("leafguard_pipeline_stage_seconds", "{{stage}}")],
        0,
        38,
        16,
        4,
        unit="s",
        extra={
            "options": {
                "orientation": "vertical",
                "displayMode": "gradient",
                "reduceOptions": {"calcs": ["lastNotNull"]},
            }
        },
    )
)

# --- infrastructure -------------------------------------------------------
P.append(row("Infrastructure", 42))
P.append(
    panel(
        "Host CPU usage",
        "timeseries",
        [('1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))', "cpu")],
        0,
        43,
        8,
        7,
        unit="percentunit",
    )
)
P.append(
    panel(
        "Host memory usage",
        "timeseries",
        [("1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes", "memory")],
        8,
        43,
        8,
        7,
        unit="percentunit",
    )
)
P.append(
    panel("Scrape targets up", "stat", [("up", "{{job}}")], 16, 43, 8, 7, thresholds=[(None, "red"), (1, "green")])
)

dashboard = {
    "uid": "leafguard",
    "title": "LeafGuard – Operations",
    "tags": ["leafguard", "aiops"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "5s",
    "time": {"from": "now-30m", "to": "now"},
    "panels": P,
    "annotations": {
        "list": [
            {
                "name": "Model version changes",
                "datasource": DS,
                "enable": True,
                "iconColor": "purple",
                "expr": "changes(leafguard_model_version[1m]) > 0",
                "titleFormat": "Model version changed",
            }
        ]
    },
}
out = Path("monitoring/grafana/dashboards/leafguard.json")
out.write_text(json.dumps(dashboard, indent=2))
print(f"wrote {out} with {len(P)} panels")
