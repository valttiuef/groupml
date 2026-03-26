"""Internal utilities for groupml."""

from __future__ import annotations

import ast
import operator
import re
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import GroupMLConfig
from .models_classification import (
    get_classification_model,
    get_classification_models_by_strategy,
    resolve_classification_model_name,
)
from .models_regression import (
    get_regression_model,
    get_regression_models_by_strategy,
    resolve_regression_model_name,
)
from .selectors_classification import (
    get_classification_selector_names_by_strategy,
    resolve_classification_selector_name,
)
from .selectors_regression import (
    get_regression_selector_names_by_strategy,
    resolve_regression_selector_name,
)

OPS: dict[str, Callable[[Any, Any], Any]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(slots=True)
class ParsedRule:
    column: str
    op: str
    value: Any

    def mask(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            raise ValueError(f"Rule column '{self.column}' not found in DataFrame.")
        fn = OPS[self.op]
        return fn(df[self.column], self.value)

    def label(self) -> str:
        return f"{self.column} {self.op} {self.value!r}"


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """SelectKBest that safely caps k to available columns at fit time."""

    def __init__(self, score_func: Callable[..., Any], k: int | str = "auto") -> None:
        self.score_func = score_func
        self.k = k
        self._selector: SelectKBest | None = None
        self.effective_k_: int | None = None

    def fit(self, X: Any, y: Any = None) -> "SafeSelectKBest":
        n_samples = int(X.shape[0])
        n_features = X.shape[1]
        if isinstance(self.k, str):
            if self.k != "auto":
                raise ValueError("SafeSelectKBest.k must be a positive integer or 'auto'.")
            k = self._auto_k(n_samples=n_samples, n_features=n_features)
        else:
            k = int(self.k)
        k = min(max(1, k), n_features)
        self.effective_k_ = k
        self._selector = SelectKBest(score_func=self.score_func, k=k)
        self._selector.fit(X, y)
        return self

    def transform(self, X: Any) -> Any:
        if self._selector is None:
            raise RuntimeError("SafeSelectKBest is not fitted.")
        return self._selector.transform(X)

    @staticmethod
    def _auto_k(n_samples: int, n_features: int) -> int:
        if n_features <= 1:
            return 1
        # Mildly increases selected features with dataset information content.
        # Works as a pragmatic default across small and medium industrial tables.
        raw = int(round(np.sqrt(max(1.0, n_features) * np.log1p(max(2.0, float(n_samples))))))
        return min(n_features, max(1, raw))


def stable_f_regression(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Run `f_regression` with a numeric-stability fallback for sparse high-offset data."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        scores = f_regression(X, y, center=True, force_finite=True)

    saw_invalid_sqrt = any(
        isinstance(item.message, RuntimeWarning)
        and "invalid value encountered in sqrt" in str(item.message)
        for item in caught
    )
    if saw_invalid_sqrt:
        # Fallback avoids catastrophic cancellation in sparse norm computation.
        return f_regression(X, y, center=False, force_finite=True)
    return scores


def parse_rule(rule: str) -> ParsedRule:
    """Parse a rule expression such as `Temperature < 20`."""
    expr = rule.strip()
    pattern = r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|!=|<|>)\s*(.+)\s*$"
    match = re.match(pattern, expr)
    if not match:
        raise ValueError(
            f"Invalid rule '{rule}'. Expected format like 'Temperature < 20'."
        )
    column, op, raw_value = match.groups()
    try:
        value = ast.literal_eval(raw_value)
    except Exception:
        value = raw_value.strip().strip("'").strip('"')
    return ParsedRule(column=column, op=op, value=value)


def infer_task(y: pd.Series, requested: str) -> str:
    """Infer task type; return 'regression' or 'classification'."""
    if requested in {"regression", "classification"}:
        return requested
    if pd.api.types.is_bool_dtype(y) or isinstance(y.dtype, CategoricalDtype) or pd.api.types.is_object_dtype(y):
        return "classification"
    nunique = y.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(y) and nunique <= min(25, max(2, int(0.05 * len(y)))):
        return "classification"
    return "regression"


def normalize_models(models: Any, task: str, random_state: int) -> dict[str, BaseEstimator]:
    """Convert model config into a named dictionary of estimators."""
    if isinstance(models, str):
        if task == "classification":
            try:
                return get_classification_models_by_strategy(models, random_state)
            except ValueError:
                model_name = resolve_classification_model_name(models)
                model = get_classification_model(models, random_state)
                return {model_name: model}
        try:
            return get_regression_models_by_strategy(models, random_state)
        except ValueError:
            model_name = resolve_regression_model_name(models)
            model = get_regression_model(models, random_state)
            return {model_name: model}

    if isinstance(models, BaseEstimator):
        return {models.__class__.__name__: models}

    if callable(getattr(models, "fit", None)):
        return {models.__class__.__name__: models}

    if isinstance(models, dict):
        return {name: est for name, est in models.items()}

    if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        out: dict[str, BaseEstimator] = {}
        for idx, est in enumerate(models):
            out[f"{est.__class__.__name__}_{idx}"] = est
        return out

    raise ValueError("models must be a model/strategy string, estimator, sequence, or dict.")


def normalize_selectors(selectors: Any, task: str) -> dict[str, Any]:
    """Convert feature selector config into a named dictionary."""
    if isinstance(selectors, str):
        if task == "classification":
            try:
                names = get_classification_selector_names_by_strategy(selectors)
                return {name: name for name in names}
            except ValueError:
                name = resolve_classification_selector_name(selectors)
                return {name: name}
        try:
            names = get_regression_selector_names_by_strategy(selectors)
            return {name: name for name in names}
        except ValueError:
            name = resolve_regression_selector_name(selectors)
            return {name: name}
    if isinstance(selectors, dict):
        return selectors
    if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)):
        out: dict[str, Any] = {}
        for idx, sel in enumerate(selectors):
            if isinstance(sel, str):
                if task == "classification":
                    name = resolve_classification_selector_name(sel)
                else:
                    name = resolve_regression_selector_name(sel)
                out[name] = name
            else:
                out[f"{sel.__class__.__name__}_{idx}"] = sel
        return out
    raise ValueError("feature_selectors must be a selector/strategy string, sequence, or dict.")


def build_selector(selector: Any, task: str, random_state: int, kbest_features: int | str = "auto") -> Any:
    """Create a sklearn-compatible selector or passthrough marker."""
    if isinstance(kbest_features, str):
        token = kbest_features.strip().lower()
        if token == "auto":
            kbest_features = "auto"
        elif token.isdigit() and int(token) >= 1:
            kbest_features = int(token)
        else:
            raise ValueError("kbest_features must be a positive integer or 'auto'.")
    elif not isinstance(kbest_features, int) or kbest_features < 1:
        raise ValueError("kbest_features must be a positive integer or 'auto'.")

    if isinstance(selector, dict):
        selector_name = selector.get("name")
        if selector_name is None:
            raise ValueError("Selector dict must include a 'name' key.")
        kbest_override = selector.get("k", kbest_features)
        return build_selector(selector_name, task, random_state, kbest_features=kbest_override)
    if selector in {"none", None}:
        return "passthrough"
    if not isinstance(selector, str):
        return clone(selector)
    if selector == "kbest_f":
        fn = f_classif if task == "classification" else stable_f_regression
        return SafeSelectKBest(score_func=fn, k=kbest_features)
    if selector == "kbest_mi":
        fn = mutual_info_classif if task == "classification" else mutual_info_regression
        return SafeSelectKBest(score_func=fn, k=kbest_features)
    if selector == "lasso":
        if task == "classification":
            base = LogisticRegression(
                penalty="l1", solver="liblinear", max_iter=1000, random_state=random_state
            )
        else:
            base = Lasso(alpha=0.01, random_state=random_state)
        return SelectFromModel(base)
    if selector == "extra_trees":
        if task == "classification":
            base = ExtraTreesClassifier(
                n_estimators=200, random_state=random_state, n_jobs=-1
            )
        else:
            base = ExtraTreesRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        return SelectFromModel(base)
    raise ValueError(f"Unknown selector '{selector}'.")


def build_preprocessor(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build a compact sklearn preprocessing graph."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No usable feature columns found after preprocessing setup.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def group_column_permutations(group_columns: Sequence[str]) -> list[tuple[str, ...]]:
    """Return single-column and combination permutations of group columns."""
    cols = list(group_columns)
    out: list[tuple[str, ...]] = []
    for r in range(1, len(cols) + 1):
        out.extend(combinations(cols, r))
    return out


def ensure_columns_exist(df: pd.DataFrame, columns: Iterable[str], kind: str) -> None:
    """Raise clear error for missing columns."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {kind} columns: {missing}")


def default_experiment_names(config: GroupMLConfig) -> list[str]:
    """Return default mode list as plain list."""
    return list(config.experiment_modes)
