"""Shared training/evaluation utilities used by runner modes."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone


def evaluate_estimator(
    *,
    estimator: Any,
    mode: str,
    method_type: str,
    variant: str,
    model_name: str,
    selector_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
    scorer: Callable[[Any, pd.DataFrame, pd.Series], float],
    task: str,
    warnings: list[str],
    run_datetime: str,
    required_group_columns: list[str] | None,
    warning_filter: Callable[[], Any],
    is_unstable_score: Callable[[float, pd.Series, str], bool],
    extract_group_feature_usage: Callable[[Any, list[str]], dict[str, Any]],
    extract_group_config_usage: Callable[[Any], dict[str, Any]],
    on_fitted_estimator: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    scores: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        fold_est = clone(estimator)
        if hasattr(fold_est, "set_params"):
            try:
                fold_est.set_params(emit_progress=False)
            except Exception:
                pass
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        try:
            with warning_filter():
                fold_est.fit(X_tr, y_tr)
                score = float(scorer(fold_est, X_val, y_val))
            if is_unstable_score(score, y_val, task):
                warnings.append(
                    f"CV failure in {mode}/{variant} ({model_name}, {selector_name}) fold={fold_idx}: "
                    f"invalid or unstable score={score:.6g}; treated as unsuccessful."
                )
                continue
            scores.append(score)
        except Exception as exc:
            warnings.append(
                f"CV failure in {mode}/{variant} ({model_name}, {selector_name}) fold={fold_idx}: {exc}"
            )
    fit_est = clone(estimator)
    if hasattr(fit_est, "set_params"):
        try:
            fit_est.set_params(emit_progress=True)
        except Exception:
            pass
    test_score = np.nan
    try:
        with warning_filter():
            fit_est.fit(X_train, y_train)
            if on_fitted_estimator is not None:
                try:
                    on_fitted_estimator(fit_est)
                except Exception as exc:
                    warnings.append(
                        f"Post-fit callback failure in {mode}/{variant} ({model_name}, {selector_name}): {exc}"
                    )
            test_candidate = float(scorer(fit_est, X_test, y_test))
        if is_unstable_score(test_candidate, y_test, task):
            warnings.append(
                f"Test failure in {mode}/{variant} ({model_name}, {selector_name}): "
                f"invalid or unstable score={test_candidate:.6g}; treated as unsuccessful."
            )
        else:
            test_score = test_candidate
        if hasattr(fit_est, "warnings_"):
            for item in list(getattr(fit_est, "warnings_")):
                warnings.append(
                    f"Model warning in {mode}/{variant} ({model_name}, {selector_name}): {item}"
                )
    except Exception as exc:
        warnings.append(f"Test failure in {mode}/{variant} ({model_name}, {selector_name}): {exc}")
    cv_mean = float(np.mean(scores)) if scores else np.nan
    cv_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan
    group_feature_usage = extract_group_feature_usage(
        fit_est,
        required_group_columns or [],
    )
    group_config_usage = extract_group_config_usage(fit_est)
    return {
        "mode": mode,
        "method_type": method_type,
        "variant": variant,
        "experiment_name": f"{mode}:{variant}",
        "model": model_name,
        "selector": selector_name,
        "run_datetime": run_datetime,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_folds_ok": len(scores),
        "test_score": float(test_score) if not np.isnan(test_score) else np.nan,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        **group_feature_usage,
        **group_config_usage,
    }
