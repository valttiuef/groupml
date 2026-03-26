"""Mode and experiment metadata helpers."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def method_type_for_mode(mode: str) -> str:
    lowered = str(mode).strip().lower()
    if lowered == "group_as_features":
        return "one_hot_group_features"
    if lowered == "group_split":
        return "per_group_models"
    if lowered == "group_permutations":
        return "per_group_models_group_combos"
    if lowered == "rule_split":
        return "rule_based_split"
    if lowered == "full":
        return "no_group_awareness"
    return lowered or "method"


def comparison_label(row: dict[str, Any]) -> str:
    return method_type_for_mode(str(row.get("mode", "")).strip().lower())


def method_token(row: dict[str, Any]) -> str:
    mode = str(row.get("mode", "unknown")).strip().lower()
    variant = str(row.get("variant", "default")).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", f"{mode}__{variant}").strip("_")
    return token[:80] if token else "method"


def pick_best_rows_by_method(leaderboard: pd.DataFrame, prefers_lower: bool) -> list[dict[str, Any]]:
    if leaderboard.empty:
        return []
    ordered = leaderboard.sort_values(by=["cv_mean"], ascending=prefers_lower).copy()
    best = ordered.drop_duplicates(subset=["mode"], keep="first")
    mode_order = ["full", "group_as_features", "group_split", "group_permutations", "rule_split"]
    order_map = {mode: idx for idx, mode in enumerate(mode_order)}
    best["__mode_order__"] = best["mode"].map(lambda mode: order_map.get(str(mode), 999))
    best = best.sort_values(by=["__mode_order__", "cv_mean"], ascending=[True, prefers_lower])
    best = best.drop(columns=["__mode_order__"])
    return best.to_dict(orient="records")
