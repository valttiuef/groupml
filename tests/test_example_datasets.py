from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

from groupml import GroupMLConfig, GroupMLRunner


def test_group_aware_example_dataset_prefers_group_strategy() -> None:
    df = pd.read_csv("examples/data/group_split_demo.csv")
    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full", "group_as_features"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=4,
        random_state=42,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    assert result.leaderboard.iloc[0]["mode"] == "group_as_features"
    baseline_cv = float(result.leaderboard[result.leaderboard["mode"] == "full"]["cv_mean"].iloc[0])
    best_cv = float(result.leaderboard.iloc[0]["cv_mean"])
    assert best_cv < baseline_cv


def test_rule_split_example_dataset_prefers_rule_split() -> None:
    df = pd.read_csv("examples/data/rule_split_demo.csv")
    config = GroupMLConfig(
        target="Target",
        rule_splits=["Temperature < 20", "Temperature >= 20"],
        experiment_modes=["full", "rule_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=4,
        random_state=42,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    assert result.leaderboard.iloc[0]["mode"] == "rule_split"
    baseline_cv = float(result.leaderboard[result.leaderboard["mode"] == "full"]["cv_mean"].iloc[0])
    best_cv = float(result.leaderboard.iloc[0]["cv_mean"])
    assert best_cv < baseline_cv
