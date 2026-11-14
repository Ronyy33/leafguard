"""Unit tests: importing user-labelled (encrypted) uploads into the training data."""

from cryptography.fernet import Fernet
from PIL import Image

from src.data.import_feedback import import_feedback
from tests.conftest import SAMPLES


def _setup(tmp_path):
    key = Fernet.generate_key().decode()
    raw, fb = tmp_path / "raw", tmp_path / "feedback"
    (raw / "Tomato___healthy").mkdir(parents=True)
    token = Fernet(key.encode()).encrypt((SAMPLES / "tomato_healthy.jpg").read_bytes())
    (fb / "Tomato___healthy").mkdir(parents=True)
    (fb / "Tomato___healthy" / "abc123.bin").write_bytes(token)
    return key, raw, fb


def test_imports_new_labelled_photo(tmp_path):
    key, raw, fb = _setup(tmp_path)
    stats = import_feedback(fb, raw, key)
    assert stats["imported"] == 1
    out = raw / "Tomato___healthy" / "fb_abc123.jpg"
    assert Image.open(out).size == (256, 256)


def test_second_run_is_idempotent(tmp_path):
    key, raw, fb = _setup(tmp_path)
    import_feedback(fb, raw, key)
    stats = import_feedback(fb, raw, key)
    assert stats["imported"] == 0 and stats["already_imported"] == 1


def test_unknown_label_is_not_imported(tmp_path):
    key, raw, fb = _setup(tmp_path)
    (fb / "Banana___rot").mkdir()
    (fb / "Banana___rot" / "x.bin").write_bytes(b"whatever")
    stats = import_feedback(fb, raw, key)
    assert stats["unknown_label"] == 1 and not (raw / "Banana___rot").exists()


def test_wrong_key_is_reported_not_crashing(tmp_path):
    _, raw, fb = _setup(tmp_path)
    stats = import_feedback(fb, raw, Fernet.generate_key().decode())
    assert stats["unreadable"] == 1 and stats["imported"] == 0


def test_no_key_skips(tmp_path):
    _, raw, fb = _setup(tmp_path)
    assert import_feedback(fb, raw, "")["skipped"] == "no key"
