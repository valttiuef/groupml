from __future__ import annotations

import numpy as np
import pytest

from groupml.utils import SafeSelectKBest, build_selector, normalize_models, normalize_selectors


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


def test_build_selector_kbest_auto_uses_rows_x_columns_heuristic() -> None:
    selector = build_selector("kbest_f", task="regression", random_state=42, kbest_features="auto")
    assert isinstance(selector, SafeSelectKBest)

    X = np.random.default_rng(0).normal(size=(120, 60))
    y = np.random.default_rng(1).normal(size=120)
    selector.fit(X, y)

    expected = int(round(np.sqrt(60 * np.log1p(120.0))))
    assert selector.effective_k_ == expected


def test_build_selector_kbest_allows_static_count() -> None:
    selector = build_selector("kbest_f", task="regression", random_state=42, kbest_features=7)
    assert isinstance(selector, SafeSelectKBest)

    X = np.random.default_rng(0).normal(size=(80, 30))
    y = np.random.default_rng(1).normal(size=80)
    selector.fit(X, y)

    assert selector.effective_k_ == 7


def test_build_selector_kbest_selector_dict_can_override_global_default() -> None:
    selector = build_selector({"name": "kbest_mi", "k": 5}, task="classification", random_state=42, kbest_features=12)
    assert isinstance(selector, SafeSelectKBest)

    X = np.random.default_rng(0).normal(size=(90, 40))
    y = np.random.default_rng(1).integers(0, 2, size=90)
    selector.fit(X, y)

    assert selector.effective_k_ == 5


def test_build_selector_non_kbest_keeps_its_native_auto_behavior() -> None:
    selector = build_selector("lasso", task="regression", random_state=42, kbest_features=3)
    assert selector.__class__.__name__ == "SelectFromModel"
