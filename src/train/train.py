"""Stage 4 - Train (tracked in MLflow).

Fine-tunes a pretrained CNN on data/processed/{train,val}.npz.

Logged to MLflow, in addition to autolog:
  params      every key under `train` and `data` in params.yaml
  tags        git commit, DVC hash of the data, backbone, host
  metrics     per-epoch train/val loss and accuracy, val macro-F1, epoch time, total time
  artifacts   labels.json, baseline_stats.json, params.yaml, data reports
  model       PyTorch model with signature + input example

Outputs (for DVC): models/model.pt, models/run_id.txt, reports/train_metrics.json

Usage:  python -m src.train.train
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.train.model import LeafNet
from src.utils import get_logger, git_commit, load_params, write_json

logger = get_logger("train")
PROCESSED = Path("data/processed")
MODELS = Path("models")
EXPERIMENT = "leafguard"


class NpzDataset(Dataset):
    """uint8 HWC images -> float CHW tensors in [0,1], with optional light augmentation."""

    def __init__(self, path: Path, augment: bool = False):
        d = np.load(path)
        self.X, self.y, self.augment = d["X"], d["y"], augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).permute(2, 0, 1).float() / 255.0
        if self.augment:
            if torch.rand(1) < 0.5:
                x = x.flip(2)  # horizontal flip
            if torch.rand(1) < 0.5:
                x = x.flip(1)  # vertical flip
            x = (x * (0.8 + 0.4 * torch.rand(1))).clamp(0, 1)  # brightness jitter
        return x, int(self.y[i])


def run_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over `loader`; trains if optimizer is given. Returns loss, acc, macro-F1."""
    train = optimizer is not None
    model.train(train)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds.append(logits.argmax(1))
            targets.append(y)
    p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return total_loss / len(t), float((p == t).mean()), float(f1_score(t, p, average="macro"))


def dvc_data_hash() -> str:
    """md5 of data/raw as recorded by DVC (ties the run to an exact dataset version)."""
    try:
        meta = yaml.safe_load(Path("data/raw.dvc").read_text())
        return meta["outs"][0]["md5"]
    except Exception:
        return "unknown"


def main() -> None:
    params = load_params()
    tp = params["train"]
    torch.manual_seed(params["data"]["seed"])
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    labels = json.loads((PROCESSED / "labels.json").read_text())
    train_ds = NpzDataset(PROCESSED / "train.npz", augment=tp["augment"])
    val_ds = NpzDataset(PROCESSED / "val.npz")
    train_dl = DataLoader(train_ds, batch_size=tp["batch_size"], shuffle=True, num_workers=tp["num_workers"])
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=tp["num_workers"])

    model = LeafNet(len(labels), tp["backbone"], pretrained=True, freeze_backbone=tp["freeze_backbone"])
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=tp["lr"])
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)
    mlflow.autolog(log_models=False, silent=True)

    with mlflow.start_run(run_name=f"{tp['backbone']}-lr{tp['lr']}") as run:
        mlflow.set_tags({
            "git_commit": git_commit(),
            "dvc_data_md5": dvc_data_hash(),
            "backbone": tp["backbone"],
            "host": platform.node(),
            "stage": "train",
        })
        mlflow.log_params({f"train.{k}": v for k, v in tp.items()})
        mlflow.log_params({f"data.{k}": v for k, v in params["data"].items()})
        mlflow.log_params({"num_classes": len(labels), "train_size": len(train_ds), "val_size": len(val_ds),
                           "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)})

        best_f1, t_start = -1.0, time.time()
        for epoch in range(1, tp["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc, _ = run_epoch(model, train_dl, loss_fn, optimizer)
            va_loss, va_acc, va_f1 = run_epoch(model, val_dl, loss_fn)
            dt = time.time() - t0
            mlflow.log_metrics({"train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
                                "val_acc": va_acc, "val_macro_f1": va_f1, "epoch_seconds": dt,
                                "train_images_per_sec": len(train_ds) / dt}, step=epoch)
            logger.info("epoch %d/%d loss=%.4f val_acc=%.4f val_f1=%.4f (%.1fs)",
                        epoch, tp["epochs"], tr_loss, va_acc, va_f1, dt)
            if va_f1 > best_f1:
                best_f1 = va_f1
                MODELS.mkdir(exist_ok=True)
                torch.save(model.state_dict(), MODELS / "model.pt")

        total = time.time() - t_start
        mlflow.log_metrics({"best_val_macro_f1": best_f1, "train_total_seconds": total})

        # reload best weights and log the model with a signature + input example
        model.load_state_dict(torch.load(MODELS / "model.pt"))
        model.eval()
        size = params["data"]["image_size"]
        example = val_ds[0][0].unsqueeze(0).numpy()
        with torch.no_grad():
            signature = infer_signature(example, model(torch.from_numpy(example)).numpy())
        # code_paths bundles src/ so `mlflow models serve` can unpickle LeafNet without the repo
        mlflow.pytorch.log_model(model, artifact_path="model", signature=signature, input_example=example,
                                 code_paths=["src"])

        for f in ("labels.json", "baseline_stats.json"):
            mlflow.log_artifact(str(PROCESSED / f), artifact_path="meta")
        mlflow.log_artifact("params.yaml", artifact_path="meta")
        for f in ("reports/validation.json", "reports/preprocess_metrics.json", "reports/ingest_metrics.json"):
            if Path(f).exists():
                mlflow.log_artifact(f, artifact_path="data_reports")

        (MODELS / "run_id.txt").write_text(run.info.run_id)
        model_meta = {"backbone": tp["backbone"], "num_classes": len(labels), "image_size": size}
        (MODELS / "model_meta.json").write_text(json.dumps(model_meta))
        write_json({"best_val_macro_f1": round(best_f1, 4), "train_total_seconds": round(total, 1),
                    "mlflow_run_id": run.info.run_id}, "reports/train_metrics.json")
        logger.info("training done: best val F1 %.4f, run %s", best_f1, run.info.run_id)


if __name__ == "__main__":
    main()
