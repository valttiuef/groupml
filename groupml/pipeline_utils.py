"""Pipeline construction and estimator-introspection helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .utils import build_preprocessor


def build_group_as_features_pipeline(
    X_ref: pd.DataFrame,
    feature_cols: list[str],
    group_cols: list[str],
    selector: Any,
    model: Any,
    scale_numeric: bool,
) -> Pipeline:
    group_feature_cols = [c for c in feature_cols if c in group_cols]
    base_feature_cols = [c for c in feature_cols if c not in group_feature_cols]
    transformers: list[tuple[str, Any, list[str]]] = []
    if base_feature_cols:
        base_preprocessor = build_preprocessor(X_ref, base_feature_cols, scale_numeric)
        if selector != "passthrough":
            base_transformer: Any = Pipeline(steps=[("preprocess", base_preprocessor), ("select", selector)])
        else:
            base_transformer = base_preprocessor
        transformers.append(("selected_base", base_transformer, base_feature_cols))
    if group_feature_cols:
        group_preprocessor = build_preprocessor(X_ref, group_feature_cols, scale_numeric)
        transformers.append(("forced_group", group_preprocessor, group_feature_cols))
    if not transformers:
        raise ValueError("No feature columns available for group_as_features mode.")
    feature_union = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("features", feature_union), ("model", model)])


def extract_group_feature_usage(estimator: Any, required_group_columns: list[str]) -> dict[str, Any]:
    base = {
        "group_features_required_count": 0,
        "group_features_selected_count": 0,
        "group_features_forced_count": 0,
        "group_features_all_selected": np.nan,
        "group_features_selected": "",
        "group_features_forced": "",
    }
    if not required_group_columns or not isinstance(estimator, Pipeline):
        return base
    features_step = estimator.named_steps.get("features")
    if not isinstance(features_step, ColumnTransformer):
        return base
    forced_group_transformer = features_step.named_transformers_.get("forced_group")
    forced_group_columns: list[str] = []
    for name, _, cols in getattr(features_step, "transformers_", []):
        if name == "forced_group":
            forced_group_columns = [str(c) for c in cols]
            break
    required_names: list[str] = []
    if forced_group_transformer is not None:
        if hasattr(forced_group_transformer, "get_feature_names_out"):
            try:
                names = forced_group_transformer.get_feature_names_out(forced_group_columns)
                required_names = [f"forced_group__{str(n)}" for n in names]
            except Exception:
                required_names = [f"forced_group__{col}" for col in forced_group_columns]
        else:
            required_names = [f"forced_group__{col}" for col in forced_group_columns]
    selected_base = features_step.named_transformers_.get("selected_base")
    has_selector = isinstance(selected_base, Pipeline) and ("select" in selected_base.named_steps)
    if has_selector:
        selected_names: list[str] = []
        forced_names = required_names
    else:
        selected_names = required_names
        forced_names = []
    return {
        "group_features_required_count": len(required_names),
        "group_features_selected_count": len(selected_names),
        "group_features_forced_count": len(forced_names),
        "group_features_all_selected": len(forced_names) == 0,
        "group_features_selected": "|".join(selected_names),
        "group_features_forced": "|".join(forced_names),
    }


def extract_group_config_usage(estimator: Any) -> dict[str, Any]:
    base = {
        "group_selected_configs": "",
        "group_selected_config_count": 0,
        "group_fallback_config": "",
        "group_candidate_avg_scores": "",
    }
    if not hasattr(estimator, "selected_candidates_"):
        return base
    selections = getattr(estimator, "selected_candidates_", {})
    if not isinstance(selections, dict):
        return base
    normalized: list[tuple[str, str]] = []
    for key, value in selections.items():
        if isinstance(key, tuple):
            key_str = "|".join(str(v) for v in key)
        else:
            key_str = str(key)
        normalized.append((key_str, str(value)))
    normalized.sort(key=lambda item: item[0])
    serialized = ";".join(f"{key}:{value}" for key, value in normalized)
    unique_values = {value for _, value in normalized if value}
    candidate_avg = getattr(estimator, "candidate_avg_scores_", {})
    candidate_serialized = ""
    if isinstance(candidate_avg, dict) and candidate_avg:
        candidate_items = sorted((str(name), float(score)) for name, score in candidate_avg.items())
        candidate_serialized = ";".join(
            f"{name}:{score:.12g}" if np.isfinite(score) else f"{name}:nan"
            for name, score in candidate_items
        )
    return {
        "group_selected_configs": serialized,
        "group_selected_config_count": len(unique_values),
        "group_fallback_config": str(getattr(estimator, "fallback_candidate_name_", "")),
        "group_candidate_avg_scores": candidate_serialized,
    }
