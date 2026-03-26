"""Recommendation and warning-table helpers."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def build_warning_details(warnings: list[str], run_datetime: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for warning in warnings:
        run_experiment = ""
        match = re.search(r"(?:CV|Test) failure in ([^/]+)/(.+?) \(", str(warning))
        if match:
            run_experiment = f"{match.group(1)}:{match.group(2)}"
        else:
            model_match = re.search(r"Model warning in ([^/]+)/(.+?) \(", str(warning))
            if model_match:
                run_experiment = f"{model_match.group(1)}:{model_match.group(2)}"
            else:
                raw_match = re.search(r"Raw report .* failure .* for (.+?) \(", str(warning))
                if raw_match:
                    run_experiment = str(raw_match.group(1))
        rows.append(
            {
                "warning_datetime": run_datetime,
                "run_datetime": run_datetime,
                "run_experiment": run_experiment,
                "warning": str(warning),
            }
        )
    return pd.DataFrame(rows, columns=["warning_datetime", "run_datetime", "run_experiment", "warning"])


def pick_baseline(leaderboard: pd.DataFrame, prefers_lower: bool) -> dict[str, Any]:
    baseline = leaderboard[leaderboard["mode"] == "full"]
    if baseline.empty:
        baseline = leaderboard.head(1)
    return baseline.sort_values(by=["cv_mean"], ascending=prefers_lower).iloc[0].to_dict()


def recommend(
    best_row: dict[str, Any],
    baseline_row: dict[str, Any],
    prefers_lower: bool,
    min_improvement: float,
    warnings: list[str],
) -> str:
    best_name = best_row["experiment_name"]
    baseline_name = baseline_row["experiment_name"]
    if prefers_lower:
        improvement = float(baseline_row["cv_mean"] - best_row["cv_mean"])
    else:
        improvement = float(best_row["cv_mean"] - baseline_row["cv_mean"])
    stable = (
        np.isfinite(best_row.get("cv_std", np.nan))
        and np.isfinite(baseline_row.get("cv_std", np.nan))
        and best_row["cv_std"] <= baseline_row["cv_std"] * 1.25
    )
    if best_name == baseline_name:
        return f"Use baseline ({baseline_name}); no better strategy found."
    if improvement < min_improvement:
        return (
            f"Keep baseline ({baseline_name}); best alternative ({best_name}) "
            f"improves CV by only {improvement:.5f}, below threshold {min_improvement:.5f}."
        )
    if not stable:
        warnings.append(f"Best strategy {best_name} has higher CV variance; validate with more data/folds.")
    if prefers_lower:
        return (
            f"Use {best_name}. It reduces CV RMSE by {improvement:.5f} versus baseline "
            f"{baseline_name} with test RMSE {best_row['test_score']:.5f}."
        )
    return (
        f"Use {best_name}. It improves CV score by {improvement:.5f} versus baseline "
        f"{baseline_name} with test score {best_row['test_score']:.5f}."
    )
