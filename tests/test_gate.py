"""Unit tests: model promotion gate (src/train/evaluate.py)."""

from src.train.evaluate import should_promote

PROD = {"test_macro_f1": 0.966, "test_dark_macro_f1": 0.84, "test_blur_macro_f1": 0.20}


def test_first_model_promoted_if_above_gate():
    assert should_promote({"test_macro_f1": 0.95}, {}, 0.90, 0.01)[0]


def test_below_absolute_gate_rejected():
    ok, why = should_promote({"test_macro_f1": 0.88}, {}, 0.90, 0.01)
    assert not ok and "gate" in why


def test_more_robust_model_with_small_clean_drop_promoted():
    new = {"test_macro_f1": 0.962, "test_dark_macro_f1": 0.93, "test_blur_macro_f1": 0.88}
    assert should_promote(new, PROD, 0.90, 0.01)[0]


def test_large_clean_drop_rejected_even_if_robust():
    new = {"test_macro_f1": 0.94, "test_dark_macro_f1": 0.95, "test_blur_macro_f1": 0.95}
    ok, why = should_promote(new, PROD, 0.90, 0.01)
    assert not ok and "dropped" in why


def test_less_robust_model_rejected():
    new = {"test_macro_f1": 0.966, "test_dark_macro_f1": 0.50, "test_blur_macro_f1": 0.10}
    assert not should_promote(new, PROD, 0.90, 0.01)[0]
