"""Custom sklearn-compatible estimators for grouped strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

from .utils import ParsedRule


def _key_tuple(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(values)


def _groupby_view(df: pd.DataFrame, split_columns: Sequence[str]) -> Any:
    by: Any = split_columns[0] if len(split_columns) == 1 else list(split_columns)
    return df.groupby(by, dropna=False)


@dataclass(slots=True)
class GroupSplitEstimator(BaseEstimator):
    """Fit one model per group key, with a fallback global model."""

    base_estimator: Any
    split_columns: Sequence[str]
    min_group_size: int = 15
    task: str = "regression"
    warnings_: list[str] = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GroupSplitEstimator":
        self.models_: dict[tuple[Any, ...], Any] = {}
        self.fallback_model_ = clone(self.base_estimator).fit(X, y)
        group_sizes = _groupby_view(X, self.split_columns).size()
        for key, size in group_sizes.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            if size < self.min_group_size:
                self.warnings_.append(
                    f"Small group {dict(zip(self.split_columns, key_tuple))} with n={size}; using fallback."
                )
                continue
            mask = np.ones(len(X), dtype=bool)
            for col, val in zip(self.split_columns, key_tuple):
                mask &= X[col].to_numpy() == val
            y_subset = y[mask]
            if self.task == "classification" and y_subset.nunique(dropna=True) < 2:
                self.warnings_.append(
                    f"Group {dict(zip(self.split_columns, key_tuple))} has <2 classes; using fallback."
                )
                continue
            try:
                model = clone(self.base_estimator).fit(X.loc[mask], y_subset)
            except Exception:
                continue
            self.models_[key_tuple] = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "fallback_model_"):
            raise RuntimeError("Estimator is not fitted.")
        predictions = np.empty(len(X), dtype=object)
        fallback_mask = np.ones(len(X), dtype=bool)
        for key, idx in _groupby_view(X, self.split_columns).groups.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            idx_array = np.asarray(list(idx))
            model = self.models_.get(key_tuple)
            if model is not None:
                predictions[idx_array] = model.predict(X.iloc[idx_array])
                fallback_mask[idx_array] = False
        if fallback_mask.any():
            predictions[fallback_mask] = self.fallback_model_.predict(X.loc[fallback_mask])
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.fallback_model_, "predict_proba"):
            return self.fallback_model_.predict_proba(X)
        raise AttributeError("Underlying model does not support predict_proba.")


@dataclass(slots=True)
class RuleSplitEstimator(BaseEstimator):
    """Fit one model per parsed rule with first-match prediction priority."""

    base_estimator: Any
    rules: Sequence[ParsedRule]
    min_group_size: int = 15
    task: str = "regression"
    warnings_: list[str] = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RuleSplitEstimator":
        self.fallback_model_ = clone(self.base_estimator).fit(X, y)
        self.rule_models_: list[tuple[ParsedRule, Any]] = []
        for rule in self.rules:
            mask = rule.mask(X).fillna(False).to_numpy()
            n = int(mask.sum())
            if n < self.min_group_size:
                self.warnings_.append(f"Rule '{rule.label()}' has small subset n={n}; using fallback.")
                continue
            y_subset = y[mask]
            if self.task == "classification" and y_subset.nunique(dropna=True) < 2:
                self.warnings_.append(f"Rule '{rule.label()}' has <2 classes; using fallback.")
                continue
            try:
                model = clone(self.base_estimator).fit(X.loc[mask], y_subset)
            except Exception:
                continue
            self.rule_models_.append((rule, model))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "fallback_model_"):
            raise RuntimeError("Estimator is not fitted.")
        preds = np.empty(len(X), dtype=object)
        assigned = np.zeros(len(X), dtype=bool)
        for rule, model in self.rule_models_:
            mask = rule.mask(X).fillna(False).to_numpy() & ~assigned
            if mask.any():
                preds[mask] = model.predict(X.loc[mask])
                assigned |= mask
        if (~assigned).any():
            preds[~assigned] = self.fallback_model_.predict(X.loc[~assigned])
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.fallback_model_, "predict_proba"):
            return self.fallback_model_.predict_proba(X)
        raise AttributeError("Underlying model does not support predict_proba.")


class GroupSplitRegressor(GroupSplitEstimator, RegressorMixin):
    """Regressor variant for sklearn compatibility."""


class GroupSplitClassifier(GroupSplitEstimator, ClassifierMixin):
    """Classifier variant for sklearn compatibility."""


class RuleSplitRegressor(RuleSplitEstimator, RegressorMixin):
    """Regressor variant for sklearn compatibility."""


class RuleSplitClassifier(RuleSplitEstimator, ClassifierMixin):
    """Classifier variant for sklearn compatibility."""
