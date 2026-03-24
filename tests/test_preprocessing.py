from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from groupml import GroupMLConfig, GroupMLRunner


def test_base_preprocessing_applies_target_range_dropna_and_static_removal() -> None:
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x_static": [5.0] * 10,
            "Target": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0, np.nan],
        }
    )

    config = GroupMLConfig(
        target="Target",
        feature_columns=["x1", "x_static"],
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        min_target=1.0,
        max_target=99.0,
        random_state=42,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    total_rows_used = result.split_info["test"]["train_size"] + result.split_info["test"]["test_size"]
    assert total_rows_used == 7
    joined_warnings = " | ".join(result.warnings)
    assert "min_target=1.0" in joined_warnings
    assert "max_target=99.0" in joined_warnings
    assert "static feature columns" in joined_warnings


def test_base_preprocessing_can_be_disabled() -> None:
    n = 40
    x1 = np.linspace(1.0, 8.0, n)
    x1[0] = np.nan
    df = pd.DataFrame(
        {
            "x1": x1,
            "x_static": [2.0] * n,
            "Target": np.linspace(10.0, 50.0, n),
        }
    )

    config = GroupMLConfig(
        target="Target",
        feature_columns=["x1", "x_static"],
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        dropna_base_rows=False,
        drop_static_base_features=False,
        random_state=42,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    total_rows_used = result.split_info["test"]["train_size"] + result.split_info["test"]["test_size"]
    assert total_rows_used == n


def test_rmse_default_and_alias_work() -> None:
    n = 60
    x = np.linspace(0.0, 10.0, n)
    df = pd.DataFrame({"x": x, "Target": 3.0 * x + 1.0})

    default_cfg = GroupMLConfig(
        target="Target",
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
    )
    alias_cfg = GroupMLConfig(
        target="Target",
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        scorer="rmse",
        random_state=42,
    )

    default_result = GroupMLRunner(default_cfg).fit_evaluate(df)
    alias_result = GroupMLRunner(alias_cfg).fit_evaluate(df)

    assert default_cfg.scorer == "neg_root_mean_squared_error"
    assert np.isfinite(default_result.leaderboard.iloc[0]["cv_mean"])
    assert np.isfinite(alias_result.leaderboard.iloc[0]["cv_mean"])


def test_kbest_f_regression_avoids_sparse_invalid_sqrt_warning() -> None:
    rng = np.random.default_rng(0)
    n = 120
    base = 1e12
    df = pd.DataFrame(
        {
            "x_big_1": base + rng.normal(scale=1.0, size=n),
            "x_big_2": base + rng.normal(scale=1.0, size=n),
            "cat": [f"grp_{i}" for i in range(n)],
            "Target": rng.normal(size=n),
        }
    )

    config = GroupMLConfig(
        target="Target",
        feature_columns=["x_big_1", "x_big_2", "cat"],
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["kbest_f"],
        cv=3,
        random_state=42,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = GroupMLRunner(config).fit_evaluate(df)

    messages = [str(item.message) for item in caught]
    assert not any("invalid value encountered in sqrt" in msg for msg in messages)
    assert not result.leaderboard.empty
