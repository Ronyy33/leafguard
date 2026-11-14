"""Small shared helpers: params loading, logging setup, JSON writing."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to stderr and to logs/pipeline.log."""
    Path("logs").mkdir(exist_ok=True)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            handlers=[logging.StreamHandler(), logging.FileHandler("logs/pipeline.log")],
        )
    return logging.getLogger(name)


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def git_commit() -> str:
    """Current Git commit hash (or 'unknown' outside a repo / before first commit)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
