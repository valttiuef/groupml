"""Built-in classification model presets and strategies."""

from __future__ import annotations

from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC

ModelFactory = Callable[[int], BaseEstimator]


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _logistic_regression(random_state: int) -> BaseEstimator:
    return LogisticRegression(max_iter=1000, random_state=random_state)


def _sgd_classifier(random_state: int) -> BaseEstimator:
    return SGDClassifier(loss="log_loss", penalty="elasticnet", l1_ratio=0.15, random_state=random_state)


def _extra_trees(random_state: int) -> BaseEstimator:
    return ExtraTreesClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)


def _random_forest(random_state: int) -> BaseEstimator:
    return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)


def _svc(random_state: int) -> BaseEstimator:
    return SVC(kernel="rbf", C=3.0, probability=True, random_state=random_state)


def _linear_svc(random_state: int) -> BaseEstimator:
    return LinearSVC(random_state=random_state)


_MODEL_FACTORIES: dict[str, ModelFactory] = {
    "logistic_regression": _logistic_regression,
    "sgd_classifier": _sgd_classifier,
    "extra_trees": _extra_trees,
    "random_forest": _random_forest,
    "svc": _svc,
    "linear_svc": _linear_svc,
}

_MODEL_ALIASES: dict[str, str] = {
    "logistic": "logistic_regression",
    "extra_trees_classifier": "extra_trees",
    "random_forest_classifier": "random_forest",
}

_MODEL_STRATEGIES: dict[str, tuple[str, ...]] = {
    "default_fast": ("logistic_regression", "sgd_classifier", "extra_trees"),
    "linear_default": ("logistic_regression", "sgd_classifier"),
    "trees": ("extra_trees", "random_forest"),
    "svm": ("svc", "linear_svc"),
    "all": (
        "logistic_regression",
        "sgd_classifier",
        "extra_trees",
        "random_forest",
        "svc",
        "linear_svc",
    ),
}


def available_classification_model_names() -> list[str]:
    return sorted(_MODEL_FACTORIES.keys())


def available_classification_model_strategies() -> list[str]:
    return sorted(_MODEL_STRATEGIES.keys())


def resolve_classification_model_name(name: str) -> str:
    key = _normalize_key(name)
    key = _MODEL_ALIASES.get(key, key)
    if key not in _MODEL_FACTORIES:
        names = ", ".join(available_classification_model_names())
        raise ValueError(f"Unknown classification model '{name}'. Available model names: {names}")
    return key


def resolve_classification_model_strategy(strategy: str) -> tuple[str, ...]:
    key = _normalize_key(strategy)
    if key not in _MODEL_STRATEGIES:
        strategies = ", ".join(available_classification_model_strategies())
        raise ValueError(
            f"Unknown classification model strategy '{strategy}'. Available strategies: {strategies}"
        )
    return _MODEL_STRATEGIES[key]


def get_classification_model(name: str, random_state: int) -> BaseEstimator:
    resolved = resolve_classification_model_name(name)
    return _MODEL_FACTORIES[resolved](random_state)


def get_classification_models_by_strategy(strategy: str, random_state: int) -> dict[str, BaseEstimator]:
    names = resolve_classification_model_strategy(strategy)
    return {name: _MODEL_FACTORIES[name](random_state) for name in names}
