"""Stage 5 - Evaluate, register and (conditionally) promote.

  * evaluates models/model.pt on the held-out test split
  * logs test metrics, per-class report and confusion matrix into the SAME MLflow run
  * registers the model as `leafguard` in the MLflow Model Registry
  * promotes it to alias `@production` only if
        test macro-F1 >= evaluate.min_macro_f1   AND
        test macro-F1 >= current production model's test macro-F1
    otherwise production stays on the previous version (automatic rollback / gate)

Outputs: reports/eval_metrics.json, reports/confusion_matrix.png, reports/classification_report.json

Usage:  python -m src.train.evaluate
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,  # noqa: E402
    confusion_matrix,
    f1_score,
)

from src.train.model import LeafNet  # noqa: E402
from src.utils import get_logger, load_params, write_json  # noqa: E402

logger = get_logger("evaluate")
MODEL_NAME = "leafguard"
ALIAS = "production"


def predict(model: torch.nn.Module, X: np.ndarray, batch: int = 128) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i : i + batch]).permute(0, 3, 1, 2).float() / 255.0
            out.append(model(x).argmax(1).numpy())
    return np.concatenate(out)


def degrade(X: np.ndarray, kind: str) -> np.ndarray:
    """Field-condition copies of the test set (same idea as the drift demo): dark or blurry photos."""
    if kind == "dark":
        return (X.astype(np.float32) * 0.35).astype(np.uint8)
    if kind == "blur":
        return np.stack([cv2.GaussianBlur(x, (0, 0), sigmaX=3) for x in X])
    raise ValueError(kind)


def robustness(model: torch.nn.Module, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        f"test_{k}_macro_f1": float(f1_score(y, predict(model, degrade(X, k)), average="macro"))
        for k in ("dark", "blur")
    }


ROBUST_KEYS = ("test_macro_f1", "test_dark_macro_f1", "test_blur_macro_f1")


def overall_score(m: dict[str, float]) -> float:
    """Mean F1 over clean, dark and blurry test photos (a metric missing = never measured = 0)."""
    return float(np.mean([m.get(k, 0.0) for k in ROBUST_KEYS]))


def current_production_metrics(client: MlflowClient) -> dict[str, float]:
    """Test metrics of the model currently aliased @production ({} if none)."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        return dict(client.get_run(mv.run_id).data.metrics)
    except Exception:
        return {}


def should_promote(
    new: dict[str, float], prod: dict[str, float], min_f1: float, max_regression: float
) -> tuple[bool, str]:
    """Promotion gate.

    1. clean test macro-F1 >= min_f1                          (absolute quality bar)
    2. clean F1 >= production clean F1 - max_regression       (no meaningful drop on normal photos)
    3. overall score (clean + dark + blur) >= production's    (better or equal under field conditions)
    """
    f1 = new["test_macro_f1"]
    if f1 < min_f1:
        return False, f"clean F1 {f1:.4f} < gate {min_f1:.2f}"
    if not prod:
        return True, "no production model yet"
    if f1 < prod.get("test_macro_f1", 0.0) - max_regression:
        return False, f"clean F1 {f1:.4f} dropped more than {max_regression} below production"
    s_new, s_prod = overall_score(new), overall_score(prod)
    if s_new < s_prod:
        return False, f"overall score {s_new:.4f} < production {s_prod:.4f}"
    return True, f"overall score {s_new:.4f} >= production {s_prod:.4f}"


def register_once(client: MlflowClient, run_id: str):
    """Reuse the registry version of this run if it exists (re-running evaluate must not duplicate it)."""
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if v.run_id == run_id:
            return v
    return mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)


def main() -> None:
    params = load_params()
    labels = json.loads(Path("data/processed/labels.json").read_text())
    names = [labels[str(i)] for i in range(len(labels))]
    meta = json.loads(Path("models/model_meta.json").read_text())
    run_id = Path("models/run_id.txt").read_text().strip()

    model = LeafNet(meta["num_classes"], meta["backbone"], pretrained=False)
    model.load_state_dict(torch.load("models/model.pt"))
    model.eval()

    test = np.load("data/processed/test.npz")
    y_true, y_pred = test["y"], predict(model, test["X"])
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, target_names=names, output_dict=True, zero_division=0)
    min_recall = min(report[n]["recall"] for n in names)
    robust = robustness(model, test["X"], y_true)
    logger.info("robustness: %s", robust)

    Path("reports").mkdir(exist_ok=True)
    write_json(report, "reports/classification_report.json")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=names).plot(
        ax=ax, xticks_rotation=90, colorbar=False, cmap="Greens"
    )
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=110)
    plt.close(fig)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    prod = current_production_metrics(client)
    new = {"test_macro_f1": f1, **robust}

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({"test_macro_f1": f1, "test_accuracy": acc, "test_min_class_recall": min_recall, **robust})
        mlflow.log_artifact("reports/confusion_matrix.png", artifact_path="eval")
        mlflow.log_artifact("reports/classification_report.json", artifact_path="eval")
        mlflow.log_metric("test_overall_score", overall_score(new))
        mv = register_once(client, run_id)

    ev = params["evaluate"]
    promoted, reason = should_promote(new, prod, ev["min_macro_f1"], ev.get("max_clean_regression", 0.0))
    client.set_model_version_tag(MODEL_NAME, mv.version, "test_macro_f1", f"{f1:.4f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "gate", reason)
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
        logger.info("promoted v%s to @%s: %s", mv.version, ALIAS, reason)
    else:
        logger.warning("v%s NOT promoted: %s", mv.version, reason)

    write_json(
        {
            "test_macro_f1": round(f1, 4),
            "test_accuracy": round(acc, 4),
            "test_min_class_recall": round(min_recall, 4),
            **{k: round(v, 4) for k, v in robust.items()},
            "model_version": int(mv.version),
            "promoted": promoted,
            "test_overall_score": round(overall_score(new), 4),
            "gate_reason": reason,
            "previous_production_f1": round(prod.get("test_macro_f1", -1.0), 4),
            "previous_production_score": round(overall_score(prod), 4) if prod else -1.0,
        },
        "reports/eval_metrics.json",
    )


if __name__ == "__main__":
    main()
