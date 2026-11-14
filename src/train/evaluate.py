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


def current_production_f1(client: MlflowClient) -> float:
    """Test macro-F1 of the model currently aliased @production (-1 if none)."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        return float(client.get_run(mv.run_id).data.metrics.get("test_macro_f1", -1))
    except Exception:
        return -1.0


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
    prod_f1 = current_production_f1(client)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({"test_macro_f1": f1, "test_accuracy": acc, "test_min_class_recall": min_recall})
        mlflow.log_artifact("reports/confusion_matrix.png", artifact_path="eval")
        mlflow.log_artifact("reports/classification_report.json", artifact_path="eval")
        mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)

    gate = params["evaluate"]["min_macro_f1"]
    promoted = f1 >= gate and f1 >= prod_f1
    client.set_model_version_tag(MODEL_NAME, mv.version, "test_macro_f1", f"{f1:.4f}")
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
        logger.info("promoted v%s to @%s (F1 %.4f >= gate %.2f, prev %.4f)", mv.version, ALIAS, f1, gate, prod_f1)
    else:
        logger.warning("v%s NOT promoted (F1 %.4f, gate %.2f, prod %.4f)", mv.version, f1, gate, prod_f1)

    write_json(
        {
            "test_macro_f1": round(f1, 4),
            "test_accuracy": round(acc, 4),
            "test_min_class_recall": round(min_recall, 4),
            "model_version": int(mv.version),
            "promoted": promoted,
            "previous_production_f1": round(prod_f1, 4),
        },
        "reports/eval_metrics.json",
    )


if __name__ == "__main__":
    main()
