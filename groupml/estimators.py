"""Custom sklearn-compatible estimators for grouped strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold

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
    candidate_estimators: dict[str, Any] | None = None
    scorer: Any = None
    cv: int = 3
    random_state: int = 42
    prefers_lower: bool = True
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    progress_context: dict[str, Any] | None = None
    emit_progress: bool = True
    tune_candidates_with_cv: bool = True
    min_group_size: int = 15
    task: str = "regression"
    warnings_: list[str] = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GroupSplitEstimator":
        self.warnings_ = []
        self.models_: dict[tuple[Any, ...], Any] = {}
        self.selected_candidates_: dict[tuple[Any, ...], str] = {}
        candidates = self._resolve_candidates()
        self.candidate_group_scores_: dict[str, list[float]] = {str(name): [] for name in candidates}
        self.candidate_avg_scores_: dict[str, float] = {}
        fallback_scores = self._score_candidates(X, y, candidates)
        fallback_name = self._best_candidate_from_scores(fallback_scores, candidates)
        self.fallback_candidate_name_ = fallback_name
        self.fallback_model_ = clone(candidates[fallback_name]).fit(X, y)
        self._emit_progress(
            event="group_tuning_started",
            payload={
                "candidate_count": len(candidates),
            },
        )
        group_sizes = _groupby_view(X, self.split_columns).size()
        self._emit_progress(
            event="group_tuning_group_count",
            payload={
                "group_count": int(len(group_sizes)),
            },
        )
        total_groups = int(len(group_sizes))
        for group_index, (key, size) in enumerate(group_sizes.items(), start=1):
            key_tuple = key if isinstance(key, tuple) else (key,)
            group_key = self._group_key_string(key_tuple)
            self._emit_progress(
                event="group_tuning_group_started",
                payload={
                    "group_index": group_index,
                    "group_total": total_groups,
                    "group_key": group_key,
                    "group_size": int(size),
                    "candidate_count": len(candidates),
                },
            )
            if size < self.min_group_size:
                self.warnings_.append(
                    f"Small group {dict(zip(self.split_columns, key_tuple))} with n={size}; using fallback."
                )
                self.selected_candidates_[key_tuple] = fallback_name
                self._emit_progress(
                    event="group_tuning_group_finished",
                    payload={
                        "group_index": group_index,
                        "group_total": total_groups,
                        "group_key": group_key,
                        "group_size": int(size),
                        "best_candidate": fallback_name,
                        "best_score": np.nan,
                        "used_fallback": True,
                        "reason": "small_group",
                    },
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
                self.selected_candidates_[key_tuple] = fallback_name
                self._emit_progress(
                    event="group_tuning_group_finished",
                    payload={
                        "group_index": group_index,
                        "group_total": total_groups,
                        "group_key": group_key,
                        "group_size": int(size),
                        "best_candidate": fallback_name,
                        "best_score": np.nan,
                        "used_fallback": True,
                        "reason": "insufficient_classes",
                    },
                )
                continue
            X_subset = X.loc[mask]
            score_map = self._score_candidates(X_subset, y_subset, candidates)
            candidate_total = len(score_map)
            for candidate_index, (candidate_name, score) in enumerate(score_map.items(), start=1):
                if np.isfinite(score):
                    self.candidate_group_scores_.setdefault(str(candidate_name), []).append(float(score))
                self._emit_progress(
                    event="group_tuning_candidate_scored",
                    payload={
                        "group_index": group_index,
                        "group_total": total_groups,
                        "group_key": group_key,
                        "group_size": int(size),
                        "candidate_index": candidate_index,
                        "candidate_total": candidate_total,
                        "candidate": candidate_name,
                        "cv_mean": float(score) if np.isfinite(score) else np.nan,
                        "score_source": "cv" if self.tune_candidates_with_cv else "train",
                    },
                )
            candidate_name = self._best_candidate_from_scores(score_map, candidates)
            best_score = float(score_map.get(candidate_name, np.nan))
            try:
                model = clone(candidates[candidate_name]).fit(X_subset, y_subset)
            except Exception:
                self.warnings_.append(
                    f"Group {dict(zip(self.split_columns, key_tuple))} fit failed for "
                    f"candidate '{candidate_name}'; using fallback."
                )
                self.selected_candidates_[key_tuple] = fallback_name
                self._emit_progress(
                    event="group_tuning_group_finished",
                    payload={
                        "group_index": group_index,
                        "group_total": total_groups,
                        "group_key": group_key,
                        "group_size": int(size),
                        "best_candidate": fallback_name,
                        "best_score": np.nan,
                        "used_fallback": True,
                        "reason": "fit_failed",
                    },
                )
                continue
            self.models_[key_tuple] = model
            self.selected_candidates_[key_tuple] = candidate_name
            self._emit_progress(
                event="group_tuning_group_finished",
                    payload={
                        "group_index": group_index,
                        "group_total": total_groups,
                        "group_key": group_key,
                        "group_size": int(size),
                        "best_candidate": candidate_name,
                        "best_score": best_score if np.isfinite(best_score) else np.nan,
                        "score_source": "cv" if self.tune_candidates_with_cv else "train",
                        "used_fallback": False,
                        "reason": "",
                    },
                )
        unique_configs = sorted({str(v) for v in self.selected_candidates_.values() if str(v)})
        self.candidate_avg_scores_ = {}
        for candidate_name, values in self.candidate_group_scores_.items():
            if values:
                self.candidate_avg_scores_[str(candidate_name)] = float(np.mean(np.asarray(values, dtype=float)))
            else:
                self.candidate_avg_scores_[str(candidate_name)] = np.nan
        self._emit_progress(
            event="group_tuning_finished",
            payload={
                "group_count": int(len(group_sizes)),
                "unique_selected_config_count": len(unique_configs),
                "selected_configs": unique_configs,
                "fallback_candidate": fallback_name,
            },
        )
        return self

    def _resolve_candidates(self) -> dict[str, Any]:
        if isinstance(self.candidate_estimators, dict) and self.candidate_estimators:
            return {str(name): est for name, est in self.candidate_estimators.items()}
        return {"default": self.base_estimator}

    def _build_cv_splits(self, X: pd.DataFrame, y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
        n_rows = int(len(X))
        if n_rows < 3:
            return []
        max_splits = min(max(2, int(self.cv)), n_rows)
        y_series = pd.Series(y).reset_index(drop=True)
        if self.task == "classification":
            class_counts = y_series.value_counts(dropna=True)
            if class_counts.empty:
                return []
            max_splits = min(max_splits, int(class_counts.min()))
        for n_splits in range(max_splits, 1, -1):
            try:
                if self.task == "classification":
                    splitter = StratifiedKFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=self.random_state,
                    )
                    return list(splitter.split(X, y_series))
                splitter = KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                return list(splitter.split(X))
            except Exception:
                continue
        return []

    def _candidate_cv_score(self, estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
        if not callable(self.scorer):
            return np.nan
        if not self.tune_candidates_with_cv:
            try:
                fitted = clone(estimator).fit(X, y)
                score = float(self.scorer(fitted, X, y))
                return score if np.isfinite(score) else np.nan
            except Exception:
                return np.nan
        splits = self._build_cv_splits(X, y)
        if not splits:
            return np.nan
        scores: list[float] = []
        for train_idx, val_idx in splits:
            fold_estimator = clone(estimator)
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            try:
                fold_estimator.fit(X_train, y_train)
                score = float(self.scorer(fold_estimator, X_val, y_val))
            except Exception:
                continue
            if np.isfinite(score):
                scores.append(score)
        if not scores:
            return np.nan
        return float(np.mean(scores))

    def _score_candidates(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        candidates: dict[str, Any],
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, estimator in candidates.items():
            out[name] = self._candidate_cv_score(estimator, X, y)
        return out

    def _best_candidate_from_scores(
        self,
        scores: dict[str, float],
        candidates: dict[str, Any],
    ) -> str:
        best_name: str | None = None
        best_score = np.nan
        for name, score in scores.items():
            if not np.isfinite(score):
                continue
            if best_name is None:
                best_name = name
                best_score = score
                continue
            if self.prefers_lower and score < best_score:
                best_name = name
                best_score = score
            if (not self.prefers_lower) and score > best_score:
                best_name = name
                best_score = score
        if best_name is not None:
            return best_name
        return next(iter(candidates))

    def _select_best_candidate_name(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        candidates: dict[str, Any],
    ) -> str:
        score_map = self._score_candidates(X, y, candidates)
        return self._best_candidate_from_scores(score_map, candidates)

    def _group_key_string(self, key_tuple: tuple[Any, ...]) -> str:
        return "|".join(str(v) for v in key_tuple)

    def _emit_progress(self, event: str, payload: dict[str, Any]) -> None:
        if not self.emit_progress:
            return
        if self.progress_callback is None:
            return
        context = dict(self.progress_context or {})
        final_payload = {"event": event, **context, **payload}
        self.progress_callback(final_payload)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "fallback_model_"):
            raise RuntimeError("Estimator is not fitted.")
        predictions = np.empty(len(X), dtype=object)
        fallback_mask = np.ones(len(X), dtype=bool)
        for key, idx in _groupby_view(X, self.split_columns).groups.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            idx_labels = list(idx)
            idx_pos = X.index.get_indexer(idx_labels)
            idx_pos = idx_pos[idx_pos >= 0]
            if idx_pos.size == 0:
                continue
            model = self.models_.get(key_tuple)
            if model is not None:
                predictions[idx_pos] = model.predict(X.loc[idx_labels])
                fallback_mask[idx_pos] = False
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
