from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from groupml import GroupMLConfig, GroupMLRunner, compare_group_strategies
from groupml.result import GroupMLResult
from groupml.summaries import build_summary_tables


def _sample_df(n: int = 150, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Temperature": rng.normal(21, 4, n),
            "Pressure": rng.normal(9, 2, n),
            "ActionGroup": rng.choice(["Idle", "Run"], size=n),
        }
    )
    df["Target"] = 1.2 * df["Temperature"] + 0.7 * df["Pressure"] + np.where(df["ActionGroup"] == "Run", 1.5, 0.0)
    return df


def test_class_api() -> None:
    df = _sample_df()
    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    assert hasattr(result, "leaderboard")
    assert hasattr(result, "summary_text")


def test_functional_api() -> None:
    df = _sample_df()
    result = compare_group_strategies(
        df=df,
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full", "group_as_features"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
    )
    assert isinstance(result.recommendation, str)
    assert not result.leaderboard.empty


def test_functional_api_callbacks() -> None:
    df = _sample_df()
    events: list[str] = []

    def _callback(event: dict[str, object]) -> None:
        events.append(str(event.get("event")))

    result = compare_group_strategies(
        df=df,
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        callbacks=[_callback],
    )
    assert not result.leaderboard.empty
    assert "run_started" in events
    assert "run_finished" in events


def test_result_contains_default_raw_report() -> None:
    df = _sample_df()
    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    assert not result.raw_report.empty
    assert "split_assignment" in result.raw_report.columns
    assert "actual" in result.raw_report.columns
    assert "predicted" in result.raw_report.columns
    assert "error" in result.raw_report.columns


def test_summary_tables_include_best_configs_and_group_performance() -> None:
    df = _sample_df(n=200, seed=19)
    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full", "group_as_features", "group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    tables = build_summary_tables(result)

    assert "best_method_configs" in tables
    assert not tables["best_method_configs"].empty
    assert "group_performance" in tables
    assert not tables["group_performance"].empty


def test_summary_tables_group_performance_handles_non_string_group_values() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "experiment_name": "full:all_features",
                "mode": "full",
                "model": "linear_regression",
                "selector": "none",
                "cv_mean": 0.1,
                "test_score": 0.2,
            }
        ]
    )
    raw_report = pd.DataFrame(
        [
            {
                "split_assignment": "cv_1",
                "actual": 1.0,
                "predicted_full": 1.1,
                "MachineId": 101.0,
            },
            {
                "split_assignment": "test",
                "actual": 2.0,
                "predicted_full": 1.9,
                "MachineId": np.nan,
            },
        ]
    )
    result = GroupMLResult(
        leaderboard=leaderboard,
        recommendation="ok",
        best_experiment=leaderboard.iloc[0].to_dict(),
        baseline_experiment=leaderboard.iloc[0].to_dict(),
        split_info={"cv": {"group_columns": ["MachineId"]}},
        raw_report=raw_report,
    )

    tables = build_summary_tables(result)

    assert "group_performance" in tables
    assert not tables["group_performance"].empty
