"""Internal utilities for groupml."""

from __future__ import annotations

import ast
import operator
import re
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
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import GroupMLConfig

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

    def __init__(self, score_func: Callable[..., Any], k: int = 20) -> None:
        self.score_func = score_func
        self.k = k
        self._selector: SelectKBest | None = None

    def fit(self, X: Any, y: Any = None) -> "SafeSelectKBest":
        n_features = X.shape[1]
        k = min(max(1, self.k), n_features)
        self._selector = SelectKBest(score_func=self.score_func, k=k)
        self._selector.fit(X, y)
        return self

    def transform(self, X: Any) -> Any:
        if self._selector is None:
            raise RuntimeError("SafeSelectKBest is not fitted.")
        return self._selector.transform(X)


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
        if models != "default_fast":
            raise ValueError("Only 'default_fast' string preset is supported for models.")
        if task == "classification":
            return {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "sgd_classifier": SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    l1_ratio=0.15,
                    random_state=random_state,
                ),
                "extra_trees": ExtraTreesClassifier(
                    n_estimators=200, random_state=random_state, n_jobs=-1
                ),
            }
        return {
            "linear_regression": LinearRegression(),
            "elastic_net": ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=5000,
                random_state=random_state,
            ),
            "sgd_regressor": SGDRegressor(
                penalty="elasticnet",
                l1_ratio=0.15,
                random_state=random_state,
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
        }
    if isinstance(models, dict):
        return {name: est for name, est in models.items()}
    if isinstance(models, Sequence):
        out: dict[str, BaseEstimator] = {}
        for idx, est in enumerate(models):
            out[f"{est.__class__.__name__}_{idx}"] = est
        return out
    raise ValueError("models must be 'default_fast', a sequence, or a dict.")


def normalize_selectors(selectors: Any) -> dict[str, Any]:
    """Convert feature selector config into a named dictionary."""
    if isinstance(selectors, str):
        if selectors == "default_fast":
            return {"kbest_f": "kbest_f", "extra_trees": "extra_trees"}
        return {selectors: selectors}
    if isinstance(selectors, dict):
        return selectors
    if isinstance(selectors, Sequence):
        out: dict[str, Any] = {}
        for idx, sel in enumerate(selectors):
            if isinstance(sel, str):
                out[sel] = sel
            else:
                out[f"{sel.__class__.__name__}_{idx}"] = sel
        return out
    raise ValueError("feature_selectors must be a preset string, sequence, or dict.")


def build_selector(selector: Any, task: str, random_state: int) -> Any:
    """Create a sklearn-compatible selector or passthrough marker."""
    if selector in {"none", None}:
        return "passthrough"
    if not isinstance(selector, str):
        return clone(selector)
    if selector == "kbest_f":
        fn = f_classif if task == "classification" else f_regression
        return SafeSelectKBest(score_func=fn, k=20)
    if selector == "kbest_mi":
        fn = mutual_info_classif if task == "classification" else mutual_info_regression
        return SafeSelectKBest(score_func=fn, k=20)
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
