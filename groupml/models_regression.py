"""Built-in regression model presets and strategies."""

from __future__ import annotations

from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.svm import LinearSVR, SVR

ModelFactory = Callable[[int], BaseEstimator]


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _linear_regression(_: int) -> BaseEstimator:
    return LinearRegression()


def _ridge(random_state: int) -> BaseEstimator:
    return Ridge(alpha=1.0, random_state=random_state)


def _lasso(random_state: int) -> BaseEstimator:
    return Lasso(alpha=0.01, random_state=random_state)


def _elastic_net(random_state: int) -> BaseEstimator:
    return ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state)


def _sgd_regressor(random_state: int) -> BaseEstimator:
    return SGDRegressor(penalty="elasticnet", l1_ratio=0.15, random_state=random_state)


def _extra_trees(random_state: int) -> BaseEstimator:
    return ExtraTreesRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)


def _random_forest(random_state: int) -> BaseEstimator:
    return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)


def _svr(_: int) -> BaseEstimator:
    return SVR(kernel="rbf", C=3.0, epsilon=0.1)


def _linear_svr(random_state: int) -> BaseEstimator:
    return LinearSVR(random_state=random_state)


_MODEL_FACTORIES: dict[str, ModelFactory] = {
    "linear_regression": _linear_regression,
    "ridge": _ridge,
    "lasso": _lasso,
    "elastic_net": _elastic_net,
    "sgd_regressor": _sgd_regressor,
    "extra_trees": _extra_trees,
    "random_forest": _random_forest,
    "svr": _svr,
    "linear_svr": _linear_svr,
}

_MODEL_ALIASES: dict[str, str] = {
    "linear": "linear_regression",
    "enet": "elastic_net",
    "extra_trees_regressor": "extra_trees",
    "random_forest_regressor": "random_forest",
    "rbf_svr": "svr",
}

_MODEL_STRATEGIES: dict[str, tuple[str, ...]] = {
    "default_fast": ("linear_regression", "elastic_net", "sgd_regressor", "extra_trees"),
    "linear_default": ("linear_regression", "ridge", "elastic_net", "sgd_regressor"),
    "trees": ("extra_trees", "random_forest"),
    "svr": ("svr", "linear_svr"),
    "all": (
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "sgd_regressor",
        "extra_trees",
        "random_forest",
        "svr",
        "linear_svr",
    ),
}


def available_regression_model_names() -> list[str]:
    return sorted(_MODEL_FACTORIES.keys())


def available_regression_model_strategies() -> list[str]:
    return sorted(_MODEL_STRATEGIES.keys())


def resolve_regression_model_name(name: str) -> str:
    key = _normalize_key(name)
    key = _MODEL_ALIASES.get(key, key)
    if key not in _MODEL_FACTORIES:
        names = ", ".join(available_regression_model_names())
        raise ValueError(f"Unknown regression model '{name}'. Available model names: {names}")
    return key


def resolve_regression_model_strategy(strategy: str) -> tuple[str, ...]:
    key = _normalize_key(strategy)
    if key not in _MODEL_STRATEGIES:
        strategies = ", ".join(available_regression_model_strategies())
        raise ValueError(f"Unknown regression model strategy '{strategy}'. Available strategies: {strategies}")
    return _MODEL_STRATEGIES[key]


def get_regression_model(name: str, random_state: int) -> BaseEstimator:
    resolved = resolve_regression_model_name(name)
    return _MODEL_FACTORIES[resolved](random_state)


def get_regression_models_by_strategy(strategy: str, random_state: int) -> dict[str, BaseEstimator]:
    names = resolve_regression_model_strategy(strategy)
    return {name: _MODEL_FACTORIES[name](random_state) for name in names}
