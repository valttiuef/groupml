from __future__ import annotations

import pytest

from groupml.utils import normalize_models, normalize_selectors


def test_normalize_models_regression_strategy_trees() -> None:
    models = normalize_models("trees", task="regression", random_state=42)
    assert set(models.keys()) == {"extra_trees", "random_forest"}


def test_normalize_models_regression_single_name() -> None:
    models = normalize_models("linear_regression", task="regression", random_state=42)
    assert list(models.keys()) == ["linear_regression"]


def test_normalize_models_classification_strategy_svm() -> None:
    models = normalize_models("svm", task="classification", random_state=42)
    assert set(models.keys()) == {"svc", "linear_svc"}


def test_normalize_models_classification_single_alias_name() -> None:
    models = normalize_models("logistic", task="classification", random_state=42)
    assert list(models.keys()) == ["logistic_regression"]


def test_normalize_selectors_regression_strategy() -> None:
    selectors = normalize_selectors("linear_default", task="regression")
    assert list(selectors.keys()) == ["none", "kbest_f", "lasso"]


def test_normalize_selectors_classification_single_name() -> None:
    selectors = normalize_selectors("kbest_mi", task="classification")
    assert list(selectors.keys()) == ["kbest_mi"]


def test_normalize_selectors_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown regression feature selector"):
        normalize_selectors("not_a_selector", task="regression")
