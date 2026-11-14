"""Unit tests: trainer service state machine (trainer/app.py) with a fake pipeline command."""

import sys
import time

import pytest

from trainer import app as T


def _wait(tr, timeout=10):
    t0 = time.time()
    while tr.state == "running" and time.time() - t0 < timeout:
        time.sleep(0.05)


@pytest.fixture
def trainer(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "REPO", tmp_path)
    return T.Trainer()


def test_successful_run_parses_stages(trainer, monkeypatch):
    script = "print(\"Running stage 'ingest':\"); print(\"Running stage 'train':\")"
    monkeypatch.setattr(T, "COMMAND", [sys.executable, "-c", script])
    ok, _ = trainer.start("manual")
    assert ok
    _wait(trainer)
    st = trainer.status()
    assert st["state"] == "succeeded"
    assert set(st["history"][0]["stages"]) == {"ingest", "train"}


def test_failed_run(trainer, monkeypatch):
    monkeypatch.setattr(T, "COMMAND", [sys.executable, "-c", "import sys; sys.exit(3)"])
    trainer.start("manual")
    _wait(trainer)
    assert trainer.state == "failed"


def test_rejects_concurrent_runs(trainer, monkeypatch):
    monkeypatch.setattr(T, "COMMAND", [sys.executable, "-c", "import time; time.sleep(1)"])
    assert trainer.start("manual")[0]
    ok, detail = trainer.start("manual")
    assert not ok and "in progress" in detail
    _wait(trainer)


def test_alert_retrain_respects_cooldown(trainer, monkeypatch):
    monkeypatch.setattr(T, "COMMAND", [sys.executable, "-c", "pass"])
    trainer.start("manual")
    _wait(trainer)
    ok, detail = trainer.start("alert:InputBrightnessDrift")
    assert not ok and "cooldown" in detail
    assert trainer.start("manual")[0]  # operators can always force a run
    _wait(trainer)
