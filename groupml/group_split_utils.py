"""Helpers for group-split strategy execution."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .estimators import GroupSplitClassifier, GroupSplitRegressor
from .utils import build_preprocessor, build_selector


def build_group_split_candidate_estimators(
    task: str,
    X_train: pd.DataFrame,
    feature_cols: list[str],
    models: dict[str, Any],
    selectors: dict[str, Any],
    scale_numeric: bool,
    random_state: int,
    kbest_features: int | str = "auto",
) -> dict[str, Any]:
    preprocessor = build_preprocessor(X_train, feature_cols, scale_numeric)
    candidates: dict[str, Any] = {}
    for (model_name, model), (selector_name, selector_spec) in product(models.items(), selectors.items()):
        selector = build_selector(selector_spec, task, random_state, kbest_features=kbest_features)
        steps = [("preprocess", preprocessor)]
        if selector != "passthrough":
            steps.append(("select", selector))
        steps.append(("model", model))
        candidates[f"{model_name}__{selector_name}"] = Pipeline(steps=steps)
    return candidates


def build_group_split_tuned_estimator(
    task: str,
    split_columns: tuple[str, ...],
    scorer: Callable[[Any, pd.DataFrame, pd.Series], float],
    random_state: int,
    min_group_size: int,
    prefers_lower: bool,
    cv: int | Any,
    prebuilt_candidates: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_context: dict[str, Any] | None = None,
) -> Any:
    base_estimator = next(iter(prebuilt_candidates.values()))
    cv_folds = cv if isinstance(cv, int) else 3
    estimator_kwargs = {
        "base_estimator": base_estimator,
        "split_columns": split_columns,
        "candidate_estimators": prebuilt_candidates,
        "scorer": scorer,
        "cv": int(cv_folds),
        "random_state": random_state,
        "prefers_lower": prefers_lower,
        "progress_callback": progress_callback,
        "progress_context": dict(progress_context or {}),
        "emit_progress": True,
        "min_group_size": min_group_size,
        "task": task,
    }
    if task == "classification":
        return GroupSplitClassifier(**estimator_kwargs)
    return GroupSplitRegressor(**estimator_kwargs)


def build_group_split_progress_callback(
    emit_callback: Callable[[str, dict[str, Any]], None],
) -> Callable[[dict[str, Any]], None]:
    def _callback(payload: dict[str, Any]) -> None:
        event_name = str(payload.get("event", "group_tuning_event"))
        emit_callback(event_name, payload)

    return _callback


def parse_group_selected_configs(serialized: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    raw = str(serialized or "").strip()
    if not raw:
        return out
    for item in raw.split(";"):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            out.append((token, ""))
            continue
        key, value = token.split(":", 1)
        out.append((key.strip(), value.strip()))
    return out


def parse_group_candidate_scores(serialized: str) -> dict[str, float]:
    out: dict[str, float] = {}
    raw = str(serialized or "").strip()
    if not raw:
        return out
    for item in raw.split(";"):
        token = item.strip()
        if not token or ":" not in token:
            continue
        key, value = token.split(":", 1)
        key = key.strip()
        if not key:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = np.nan
        out[key] = score
    return out
