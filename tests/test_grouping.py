from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from groupml import GroupMLConfig, GroupMLRunner
from groupml.utils import group_column_permutations


def test_group_permutations_builder() -> None:
    perms = group_column_permutations(["A", "B", "C"])
    assert len(perms) == 7
    assert ("A",) in perms
    assert ("A", "B", "C") in perms


def test_group_split_mode_produces_rows() -> None:
    rng = np.random.default_rng(7)
    n = 180
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["S1", "S2"], size=n),
            "Material": rng.choice(["M1", "M2"], size=n),
        }
    )
    df["Target"] = 2.0 * df["x1"] - 0.8 * df["x2"] + np.where(df["ActionGroup"] == "S2", 2.5, 0.0)

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup", "Material"],
        experiment_modes=["group_split", "group_permutations"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size=0.25,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    assert (result.leaderboard["mode"] == "group_split").any()
    assert (result.leaderboard["mode"] == "group_permutations").any()


def test_group_permutations_skipped_with_single_group_column() -> None:
    rng = np.random.default_rng(8)
    n = 160
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["S1", "S2"], size=n),
        }
    )
    df["Target"] = 1.8 * df["x1"] - 0.6 * df["x2"] + np.where(df["ActionGroup"] == "S2", 1.2, 0.0)

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["group_split", "group_permutations"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size=0.25,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert (result.leaderboard["mode"] == "group_split").any()
    assert not (result.leaderboard["mode"] == "group_permutations").any()
    assert any("Skipping group_permutations: requires at least two group_columns." in msg for msg in result.warnings)


def test_runner_auto_stratifies_by_group_columns_for_default_cv() -> None:
    rng = np.random.default_rng(17)
    n = 240
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["A", "B"], size=n),
            "Material": rng.choice(["M1", "M2"], size=n),
        }
    )
    df["Target"] = 1.5 * df["x1"] - 0.3 * df["x2"] + np.where(df["ActionGroup"] == "B", 0.8, 0.0)

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup", "Material"],
        experiment_modes=["full", "group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=4,
        test_size=0.2,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert result.split_info["test"]["splitter"] == "train_test_split"
    assert result.split_info["test"]["strategy"] == "random"
    assert result.split_info["cv"]["strategy_requested"] == "stratifycv"
    assert result.split_info["cv"]["stratify_column"] == "__groupml_auto_stratify__"
    assert any("Auto-enabled stratification using group columns" in msg for msg in result.warnings)


def test_runner_drops_small_group_combinations_for_comparability() -> None:
    rng = np.random.default_rng(23)
    regular_n = 90
    rare_n = 5
    regular_df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, regular_n),
            "x2": rng.normal(0, 1, regular_n),
            "Material": ["M1"] * regular_n,
            "ActionGroup": ["A"] * regular_n,
        }
    )
    rare_df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, rare_n),
            "x2": rng.normal(0, 1, rare_n),
            "Material": ["M_rare"] * rare_n,
            "ActionGroup": ["A"] * rare_n,
        }
    )
    df = pd.concat([regular_df, rare_df], ignore_index=True)
    df["Target"] = 2.0 * df["x1"] - 0.5 * df["x2"]

    config = GroupMLConfig(
        target="Target",
        group_columns=["Material", "ActionGroup"],
        experiment_modes=["full", "group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size=0.2,
        min_group_size=10,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    total_rows_used = result.split_info["test"]["train_size"] + result.split_info["test"]["test_size"]
    assert total_rows_used == regular_n
    assert any("Dropped 5 row(s) with group combination size < min_group_size=10" in msg for msg in result.warnings)


def test_raw_report_contains_best_predictions_per_method_columns() -> None:
    rng = np.random.default_rng(33)
    n = 180
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["A", "B"], size=n),
        }
    )
    df["Target"] = 1.2 * df["x1"] + np.where(df["ActionGroup"] == "B", 1.0, -1.0)

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full", "group_as_features", "group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size=0.2,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    predicted_columns = [c for c in result.raw_report.columns if c.startswith("predicted_")]
    assert "predicted_no_group_awareness" in predicted_columns
    assert "predicted_one_hot_group_features" in predicted_columns
    assert "predicted_per_group_models" in predicted_columns
    assert len(predicted_columns) == 3
    assert result.raw_report[predicted_columns].notna().any(axis=1).all()
