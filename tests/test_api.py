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


def test_runner_includes_method_type_label_for_full_mode() -> None:
    df = _sample_df(n=120, seed=23)
    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    assert "method_type" in result.leaderboard.columns
    assert set(result.leaderboard["method_type"]) == {"no_group_awareness"}


def test_summary_recommendations_focus_on_recommended_setup_and_top_n() -> None:
    df = _sample_df(n=180, seed=17)
    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full", "group_as_features", "group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    tables = build_summary_tables(result, top_n=3)
    recommendations = tables["recommendations"]

    assert not recommendations.empty
    assert set(recommendations["recommendation_scope"]).issubset({"recommended_setup", "top_n_overall"})
    assert int((recommendations["recommendation_scope"] == "recommended_setup").sum()) == 1
    assert int((recommendations["recommendation_scope"] == "top_n_overall").sum()) <= 3
    assert "recommendation" in recommendations.columns


def test_group_performance_avoids_row_level_high_cardinality_columns() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "experiment_name": "full:all_features",
                "mode": "full",
                "method_type": "no_group_awareness",
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
                "DeviceId": "A1",
                "Shift": "Day",
            },
            {
                "split_assignment": "cv_2",
                "actual": 2.0,
                "predicted_full": 1.9,
                "DeviceId": "B2",
                "Shift": "Day",
            },
            {
                "split_assignment": "test",
                "actual": 3.0,
                "predicted_full": 3.2,
                "DeviceId": "C3",
                "Shift": "Night",
            },
            {
                "split_assignment": "test",
                "actual": 4.0,
                "predicted_full": 3.8,
                "DeviceId": "D4",
                "Shift": "Night",
            },
        ]
    )
    result = GroupMLResult(
        leaderboard=leaderboard,
        recommendation="ok",
        best_experiment=leaderboard.iloc[0].to_dict(),
        baseline_experiment=leaderboard.iloc[0].to_dict(),
        split_info={},
        raw_report=raw_report,
    )
    tables = build_summary_tables(result)
    assert "group_performance" in tables
    group_perf = tables["group_performance"]
    assert "DeviceId" not in set(group_perf["group_column"])
    assert "Shift" in set(group_perf["group_column"])
    assert "overall_method_comparison" in set(tables["summary"]["section"])


def test_group_performance_prefers_configured_groups_over_fallback_guessing() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "experiment_name": "full:all_features",
                "mode": "full",
                "method_type": "no_group_awareness",
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
                "predicted_full": 1.2,
                "Supplier": "S1",
                "Remark + Supplier": "foo + S1",
            },
            {
                "split_assignment": "test",
                "actual": 2.0,
                "predicted_full": 1.9,
                "Supplier": "S2",
                "Remark + Supplier": "bar + S2",
            },
        ]
    )
    result = GroupMLResult(
        leaderboard=leaderboard,
        recommendation="ok",
        best_experiment=leaderboard.iloc[0].to_dict(),
        baseline_experiment=leaderboard.iloc[0].to_dict(),
        split_info={"configured_group_columns": ["Supplier"], "cv": {"group_columns": []}},
        raw_report=raw_report,
    )
    tables = build_summary_tables(result)
    group_perf = tables["group_performance"]
    assert set(group_perf["group_column"]) == {"Supplier"}
