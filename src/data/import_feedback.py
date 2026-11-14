"""Import user-labelled uploads into the training data (runs before every retrain).

The backend stores each upload encrypted; when a user confirms or corrects a
prediction, the encrypted file is copied to FEEDBACK_DIR/<true_label>/<id>.bin.
This step decrypts those files and writes them into data/raw/<label>/fb_<id>.jpg.
scripts/retrain.sh then runs `dvc add data/raw`, so every retrain on new field
photos gets its own data version (and the validate stage still checks them).

Output:  reports/feedback_import.json
Usage:   ENCRYPTION_KEY=... python -m src.data.import_feedback
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from PIL import Image, UnidentifiedImageError

from src.utils import get_logger, load_params, write_json

logger = get_logger("import_feedback")


def import_feedback(feedback_dir: Path, raw_dir: Path, key: str) -> dict:
    stats = {"found": 0, "imported": 0, "already_imported": 0, "unknown_label": 0, "unreadable": 0}
    if not key:
        logger.warning("ENCRYPTION_KEY not set - skipping feedback import")
        return {**stats, "skipped": "no key"}
    if not feedback_dir.exists():
        return stats
    fernet = Fernet(key.encode())
    classes = {p.name for p in raw_dir.iterdir() if p.is_dir()}

    for f in sorted(feedback_dir.glob("*/*.bin")):
        stats["found"] += 1
        label = f.parent.name
        if label not in classes:  # never let a typo create a new class
            stats["unknown_label"] += 1
            continue
        dest = raw_dir / label / f"fb_{f.stem}.jpg"
        if dest.exists():
            stats["already_imported"] += 1
            continue
        try:
            with Image.open(io.BytesIO(fernet.decrypt(f.read_bytes()))) as im:
                im.convert("RGB").save(dest, format="JPEG", quality=95)
            stats["imported"] += 1
        except (InvalidToken, UnidentifiedImageError, OSError) as exc:
            logger.warning("cannot import %s: %s", f, exc)
            stats["unreadable"] += 1
    return stats


def main() -> None:
    params = load_params()["data"]
    stats = import_feedback(
        Path(os.getenv("FEEDBACK_DIR", "data/feedback")), Path(params["raw_dir"]), os.getenv("ENCRYPTION_KEY", "")
    )
    write_json(stats, "reports/feedback_import.json")
    logger.info("feedback import: %s", stats)


if __name__ == "__main__":
    main()
