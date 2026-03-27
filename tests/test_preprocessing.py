from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression

from groupml import GroupMLConfig, GroupMLRunner


class _ConvergenceWarningModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):  # type: ignore[no-untyped-def]
        del X
        warnings.warn("synthetic convergence warning", ConvergenceWarning)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=float)


class _SklearnParallelDelayedWarningModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):  # type: ignore[no-untyped-def]
        del X
        warnings.warn(
            "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` "
            "to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.",
            UserWarning,
        )
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=float)


class _ExplodingPredictionModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):  # type: ignore[no-untyped-def]
        del X, y
        self.constant_ = 1e20
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        return np.full(shape=(len(X),), fill_value=self.constant_, dtype=float)


class _KeepFirstFeatureSelector(BaseEstimator):
    def fit(self, X, y=None):  # type: ignore[no-untyped-def]
        del y
        n_features = int(X.shape[1])
        self.support_ = np.zeros(n_features, dtype=bool)
        if n_features > 0:
            self.support_[0] = True
        return self

    def transform(self, X):  # type: ignore[no-untyped-def]
        return X[:, self.support_]

    def get_support(self):  # type: ignore[no-untyped-def]
        return self.support_


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
    assert float(default_result.leaderboard.iloc[0]["cv_mean"]) >= 0.0
    assert float(alias_result.leaderboard.iloc[0]["cv_mean"]) >= 0.0


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


def test_default_warning_verbosity_suppresses_convergence_warning() -> None:
    n = 50
    x = np.linspace(0.0, 10.0, n)
    df = pd.DataFrame({"x": x, "Target": 2.0 * x + 1.0})

    config = GroupMLConfig(
        target="Target",
        experiment_modes=["full"],
        models=[_ConvergenceWarningModel()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    assert not any(isinstance(item.message, ConvergenceWarning) for item in caught)


def test_warning_verbosity_all_allows_convergence_warning() -> None:
    n = 50
    x = np.linspace(0.0, 10.0, n)
    df = pd.DataFrame({"x": x, "Target": 2.0 * x + 1.0})

    config = GroupMLConfig(
        target="Target",
        experiment_modes=["full"],
        models=[_ConvergenceWarningModel()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
        warning_verbosity="all",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    assert any(isinstance(item.message, ConvergenceWarning) for item in caught)


def test_warning_verbosity_default_suppresses_sklearn_parallel_delayed_warning() -> None:
    n = 50
    x = np.linspace(0.0, 10.0, n)
    df = pd.DataFrame({"x": x, "Target": 2.0 * x + 1.0})

    config = GroupMLConfig(
        target="Target",
        experiment_modes=["full"],
        models=[_SklearnParallelDelayedWarningModel()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
        warning_verbosity="default",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    assert not any(
        "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`" in str(item.message)
        for item in caught
    )


def test_warning_verbosity_all_allows_sklearn_parallel_delayed_warning() -> None:
    n = 50
    x = np.linspace(0.0, 10.0, n)
    df = pd.DataFrame({"x": x, "Target": 2.0 * x + 1.0})

    config = GroupMLConfig(
        target="Target",
        experiment_modes=["full"],
        models=[_SklearnParallelDelayedWarningModel()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
        warning_verbosity="all",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    assert any(
        "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`" in str(item.message)
        for item in caught
    )


def test_unstable_rmse_runs_are_ignored_from_leaderboard_and_averages() -> None:
    n = 80
    x = np.linspace(0.0, 10.0, n)
    df = pd.DataFrame({"x": x, "Target": 3.0 * x + 1.0})

    config = GroupMLConfig(
        target="Target",
        experiment_modes=["full"],
        models={
            "linear": LinearRegression(),
            "exploding": _ExplodingPredictionModel(),
        },
        feature_selectors=["none"],
        scorer="rmse",
        cv=4,
        random_state=42,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    assert "exploding" not in set(result.leaderboard["model"].tolist())
    assert not result.all_runs.empty
    assert "exploding" in set(result.all_runs["model"].tolist())
    exploding_row = result.all_runs[result.all_runs["model"] == "exploding"].iloc[0]
    assert exploding_row["run_status"] == "failed"
    joined_warnings = " | ".join(result.warnings)
    assert "invalid or unstable score" in joined_warnings
    assert "Ignored 1 unsuccessful experiment" in joined_warnings
    assert not result.warning_details.empty
    assert {"warning_datetime", "run_datetime", "run_experiment", "warning"}.issubset(
        set(result.warning_details.columns)
    )


def test_group_onehot_features_are_forced_back_after_selection() -> None:
    n = 80
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "ActionGroup": np.where(np.arange(n) % 2 == 0, "A", "B"),
        }
    )
    df["Target"] = 2.0 * df["x0"] + np.where(df["ActionGroup"] == "A", 0.5, -0.5)

    config = GroupMLConfig(
        target="Target",
        feature_columns=["x0", "ActionGroup"],
        group_columns=["ActionGroup"],
        experiment_modes=["group_as_features"],
        models=[LinearRegression()],
        feature_selectors=[_KeepFirstFeatureSelector()],
        cv=3,
        random_state=42,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    row = result.leaderboard.iloc[0]

    assert int(row["group_features_required_count"]) > 0
    assert int(row["group_features_selected_count"]) == 0
    assert int(row["group_features_forced_count"]) > 0
    assert bool(row["group_features_all_selected"]) is False


def test_static_all_nan_feature_is_removed_before_dropna() -> None:
    n = 80
    x = np.linspace(0.0, 10.0, n)
    df = pd.DataFrame(
        {
            "x_useful": x,
            "x_all_nan": [np.nan] * n,
            "Target": 2.0 * x + 1.0,
        }
    )

    config = GroupMLConfig(
        target="Target",
        feature_columns=["x_useful", "x_all_nan"],
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    prep = result.split_info.get("preprocessing", {})
    assert int(prep.get("rows_initial", 0)) == n
    assert int(prep.get("rows_dropped_required_na", -1)) == 0
    assert int(prep.get("columns_removed_static", 0)) >= 1


def test_split_info_contains_preprocess_and_group_profile() -> None:
    n = 90
    x = np.linspace(0.0, 9.0, n)
    material = np.where(np.arange(n) % 3 == 0, "A", np.where(np.arange(n) % 3 == 1, "B", "C"))
    df = pd.DataFrame(
        {
            "x": x,
            "Material": material,
            "Target": 3.0 * x + 1.0,
        }
    )

    config = GroupMLConfig(
        target="Target",
        feature_columns=["x", "Material"],
        group_columns=["Material"],
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    prep = result.split_info.get("preprocessing", {})
    group_profile = result.split_info.get("group_profile", {})
    assert prep
    assert int(prep.get("rows_initial", 0)) == n
    assert "rows_after_comparability" in prep
    assert group_profile
    assert group_profile.get("group_columns") == ["Material"]
    unique_per_column = group_profile.get("unique_groups_per_column", {})
    assert int(unique_per_column.get("Material", 0)) == 3
