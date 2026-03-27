from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

from groupml import GroupMLConfig, GroupMLRunner
from groupml.summaries import build_summary_tables
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
    group_split_rows = result.leaderboard[result.leaderboard["mode"] == "group_split"]
    assert len(group_split_rows) >= 1
    assert (group_split_rows["model"] == "per_group_best").any()
    assert not (group_split_rows["model"] != "per_group_best").any()


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


def test_runner_ignores_missing_group_columns_and_continues_with_non_group_modes() -> None:
    rng = np.random.default_rng(44)
    n = 140
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        }
    )
    df["Target"] = 1.7 * df["x1"] - 0.4 * df["x2"]

    config = GroupMLConfig(
        target="Target",
        group_columns=["Material"],
        experiment_modes=["full", "group_as_features", "group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size=0.2,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    assert set(result.leaderboard["mode"]) == {"full"}
    assert any("Configured group column(s) not found and will be ignored" in msg for msg in result.warnings)
    assert any("Skipping group_as_features: no group_columns provided." in msg for msg in result.warnings)
    assert any("Skipping group_split: no group_columns provided." in msg for msg in result.warnings)


def test_group_split_tunes_model_per_group() -> None:
    rng = np.random.default_rng(52)
    n_per_group = 150
    x_a = rng.uniform(-3, 3, n_per_group)
    x_b = rng.uniform(-3, 3, n_per_group)
    df = pd.DataFrame(
        {
            "x": np.concatenate([x_a, x_b]),
            "ActionGroup": ["A"] * n_per_group + ["B"] * n_per_group,
        }
    )
    # Group A is linear, Group B is threshold-like.
    target_a = 2.0 * x_a + rng.normal(0, 0.1, n_per_group)
    target_b = np.where(x_b > 0.0, 3.0, -3.0) + rng.normal(0, 0.1, n_per_group)
    df["Target"] = np.concatenate([target_a, target_b])

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["group_split"],
        models={
            "linear": LinearRegression(),
            "rf": RandomForestRegressor(n_estimators=200, random_state=123),
        },
        feature_selectors=["none"],
        cv=3,
        test_size=0.25,
        min_group_size=20,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    group_split_rows = result.leaderboard[result.leaderboard["mode"] == "group_split"]
    tuned_rows = group_split_rows[
        (group_split_rows["model"] == "per_group_best") & (group_split_rows["selector"] == "per_group_best")
    ]

    assert not tuned_rows.empty
    tuned_row = tuned_rows.iloc[0]
    assert int(tuned_row.get("group_selected_config_count", 0)) >= 1
    assert str(tuned_row.get("group_selected_configs", "")).strip() != ""


def test_raw_report_keeps_context_columns_after_row_filters() -> None:
    rng = np.random.default_rng(61)
    n = 220
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["A", "B"], size=n),
        }
    )
    df["Target"] = 1.5 * df["x1"] - 0.5 * df["x2"] + np.where(df["ActionGroup"] == "B", 1.0, 0.0)

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full", "group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size=0.2,
        min_target=0.0,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert "ActionGroup" in result.raw_report.columns
    assert result.raw_report["ActionGroup"].notna().all()


def test_group_split_emits_group_level_progress_events() -> None:
    rng = np.random.default_rng(73)
    n = 180
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["A", "B", "C"], size=n),
        }
    )
    df["Target"] = 1.1 * df["x1"] - 0.4 * df["x2"] + np.where(df["ActionGroup"] == "C", 1.5, 0.0)

    events: list[str] = []

    def _callback(event: dict[str, object]) -> None:
        events.append(str(event.get("event", "")))

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size=0.2,
    )
    GroupMLRunner(config).fit_evaluate(df, callbacks=[_callback])

    assert "group_split_variant_started" in events
    assert "group_model_selected" in events
    assert "group_split_shared_best" not in events
    assert "group_split_variant_finished" in events


def test_group_split_shared_candidates_include_test_score() -> None:
    rng = np.random.default_rng(79)
    n = 220
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["A", "B", "C"], size=n),
        }
    )
    df["Target"] = 1.3 * df["x1"] - 0.7 * df["x2"] + np.where(df["ActionGroup"] == "C", 1.0, 0.0)

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size=0.2,
        group_split_compare_shared_candidates=True,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    group_rows = result.leaderboard[result.leaderboard["mode"] == "group_split"]
    shared_rows = group_rows[group_rows["model"] != "per_group_best"]

    assert not shared_rows.empty
    assert shared_rows["test_score"].notna().all()


def test_group_model_selected_event_includes_per_group_test_score_when_available() -> None:
    rng = np.random.default_rng(89)
    n_a = 120
    n_b = 20
    df = pd.DataFrame(
        {
            "x1": np.concatenate([rng.normal(0, 1, n_a), rng.normal(0, 1, n_b)]),
            "ActionGroup": (["A"] * n_a) + (["B"] * n_b),
        }
    )
    df["Target"] = 1.7 * df["x1"] + np.where(df["ActionGroup"] == "B", 1.0, 0.0)

    group_events: list[dict[str, object]] = []

    def _callback(event: dict[str, object]) -> None:
        if str(event.get("event")) == "group_model_selected":
            group_events.append(event)

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size_rows=20,
        test_split_strategy="last_rows",
    )
    GroupMLRunner(config).fit_evaluate(df, callbacks=[_callback])

    assert group_events
    assert any(bool(item.get("split_comparable_with_global")) for item in group_events)
    assert all(str(item.get("split_consistency", "")) == "shared_holdout_and_cv_plan" for item in group_events)

    assert any(int(item.get("group_test_rows", 0) or 0) > 0 for item in group_events)
    for item in group_events:
        rows = int(item.get("group_test_rows", 0) or 0)
        score = float(item.get("group_test_score", np.nan))
        cv_score = float(item.get("group_cv_mean", np.nan))
        if rows > 0:
            assert np.isfinite(score)
        else:
            assert np.isnan(score)
        assert np.isfinite(cv_score) or np.isnan(cv_score)


def test_group_split_shared_cv_uses_outer_cv_not_training_score() -> None:
    rng = np.random.default_rng(101)
    n = 260
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "Material": rng.choice(["M1", "M2", "M3"], size=n),
        }
    )
    # Independent noise target: train-fit RMSE can approach zero for flexible models,
    # but proper CV RMSE should stay clearly above zero.
    df["Target"] = rng.normal(0, 1, n)

    config = GroupMLConfig(
        target="Target",
        group_columns=["Material"],
        experiment_modes=["group_split"],
        models={"extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)},
        feature_selectors=["none"],
        cv=4,
        test_size=0.2,
        group_split_compare_shared_candidates=True,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    group_rows = result.leaderboard[result.leaderboard["mode"] == "group_split"]
    shared_rows = group_rows[group_rows["model"] != "per_group_best"]

    assert not shared_rows.empty
    cv_value = float(shared_rows.iloc[0]["cv_mean"])
    assert np.isfinite(cv_value)
    assert cv_value > 0.2


def test_group_tuning_progress_reports_score_source_and_finite_scores() -> None:
    rng = np.random.default_rng(113)
    n = 180
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["A", "B", "C"], size=n),
        }
    )
    df["Target"] = rng.normal(0, 1, n)

    scored_events: list[dict[str, object]] = []
    finished_events: list[dict[str, object]] = []

    def _callback(event: dict[str, object]) -> None:
        name = str(event.get("event", ""))
        if name == "group_tuning_candidate_scored":
            scored_events.append(event)
        if name == "group_tuning_group_finished":
            finished_events.append(event)

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size=0.2,
    )
    GroupMLRunner(config).fit_evaluate(df, callbacks=[_callback])

    assert scored_events
    assert finished_events
    assert all(str(item.get("score_source", "")) in {"cv", "train"} for item in scored_events)
    assert all(str(item.get("score_source", "")) in {"cv", "train"} for item in finished_events)
    assert any(np.isfinite(float(item.get("cv_mean", np.nan))) for item in scored_events)
    assert any(np.isfinite(float(item.get("best_score", np.nan))) for item in finished_events)


def test_all_runs_includes_group_level_rows_for_group_split() -> None:
    rng = np.random.default_rng(131)
    n = 180
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["A", "B", "C"], size=n),
        }
    )
    df["Target"] = 1.3 * df["x1"] + np.where(df["ActionGroup"] == "C", 0.7, 0.0)

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["group_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        test_size=0.2,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.all_runs.empty
    assert "run_scope" in result.all_runs.columns
    group_rows = result.all_runs[result.all_runs["run_scope"] == "group"]
    assert not group_rows.empty
    assert {"group_key", "group_selected_config", "run_status"}.issubset(set(group_rows.columns))
    assert set(group_rows["run_status"].astype(str)) == {"group_info"}
    assert {"cv_mean", "test_score", "train_rows", "test_rows"}.issubset(set(group_rows.columns))
    assert pd.to_numeric(group_rows["test_score"], errors="coerce").notna().any()


def test_default_mode_counts_avoid_extra_group_split_shared_runs_and_summary_uses_result_rows() -> None:
    rng = np.random.default_rng(171)
    n = 220
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "ActionGroup": np.where(np.arange(n) % 2 == 0, "A", "B"),
        }
    )
    df["Target"] = 2.0 * df["x"] + np.where(df["ActionGroup"] == "B", 1.0, -1.0)

    events: list[dict[str, object]] = []

    def _callback(event: dict[str, object]) -> None:
        events.append(event)

    config = GroupMLConfig(
        target="Target",
        feature_columns=["x", "ActionGroup"],
        group_columns=["ActionGroup"],
        experiment_modes=["full", "group_as_features", "group_split"],
        models={
            "linear": LinearRegression(),
            "dummy_mean": DummyRegressor(strategy="mean"),
            "dummy_median": DummyRegressor(strategy="median"),
            "rf": RandomForestRegressor(n_estimators=40, random_state=42),
        },
        feature_selectors={"none": "none", "kbest_f": "kbest_f"},
        cv=3,
        test_size=0.2,
    )
    result = GroupMLRunner(config).fit_evaluate(df, callbacks=[_callback])

    full_rows = result.leaderboard[result.leaderboard["mode"] == "full"]
    onehot_rows = result.leaderboard[result.leaderboard["mode"] == "group_as_features"]
    group_rows = result.leaderboard[result.leaderboard["mode"] == "group_split"]

    assert len(full_rows) == 8
    assert len(onehot_rows) == 8
    assert len(group_rows[group_rows["model"] == "per_group_best"]) == 1
    assert group_rows[group_rows["model"] != "per_group_best"].empty

    run_started = [e for e in events if str(e.get("event")) == "run_started"]
    assert run_started
    assert int(run_started[0].get("total_experiments", -1) or -1) == 17
    assert not any(str(e.get("event")) == "group_split_shared_candidate_evaluated" for e in events)

    summary_tables = build_summary_tables(result, top_n=5)
    valid_experiment_names = set(result.leaderboard["experiment_name"].astype(str).tolist())
    for table_name in ["summary", "recommendations"]:
        table = summary_tables[table_name]
        if "experiment_name" not in table.columns:
            continue
        names = set(table["experiment_name"].astype(str))
        names.discard("")
        assert names.issubset(valid_experiment_names)
