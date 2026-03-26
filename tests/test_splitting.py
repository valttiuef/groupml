from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from groupml import GroupMLConfig, GroupMLRunner
from groupml.splitting import plan_splits


def _sample_df(n: int = 120, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["A", "B", "C"], size=n),
        }
    )
    df["Target"] = 2.0 * df["x1"] - 0.7 * df["x2"] + np.where(df["ActionGroup"] == "C", 1.0, 0.0)
    return df


def _sample_df_with_split_columns(n: int = 120, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "ActionGroup": rng.choice(["A", "B", "C", "D"], size=n),
            "BatchDate": pd.date_range("2024-01-01", periods=n, freq="D"),
            "Strata": rng.choice(["low", "mid", "high"], size=n),
        }
    )
    df["Target"] = 1.1 * df["x1"] - 0.4 * df["x2"] + np.where(df["Strata"] == "high", 0.7, 0.0)
    return df


def test_plan_splits_supports_groupkfold_by_name() -> None:
    df = _sample_df()
    X = df[["x1", "x2", "ActionGroup"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv="GroupKFold",
        cv_params={"n_splits": 3},
        cv_group_columns=["ActionGroup"],
        random_state=42,
        test_size=0.2,
        include_indices=True,
    )

    assert split_plan.cv_splitter_name == "GroupKFold"
    assert split_plan.split_info["cv"]["n_splits"] == 3
    assert split_plan.split_info["cv"]["uses_groups"] is True
    assert len(split_plan.split_info["test"]["test_indices"]) > 0


def test_runner_accepts_custom_cv_callable() -> None:
    df = _sample_df()

    def custom_cv(X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None):
        n = len(X)
        idx = np.arange(n)
        mid = n // 2
        yield idx[:mid], idx[mid:]
        yield idx[mid:], idx[:mid]

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=custom_cv,
        include_split_indices=True,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    assert result.split_info["cv"]["n_splits"] == 2
    assert result.split_info["cv"]["splitter"] == "CallableSplitter"
    assert result.split_info["test"]["splitter"] == "last_rows"


def test_plan_splits_infers_groupcv_from_group_column() -> None:
    df = _sample_df_with_split_columns()
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv=4,
        random_state=42,
        test_size=0.2,
        cv_group_columns=["ActionGroup"],
        cv_source_df=df,
    )

    assert split_plan.cv_splitter_name == "GroupKFold"
    assert split_plan.split_info["cv"]["strategy_requested"] == "groupcv"
    assert split_plan.split_info["cv"]["strategy_used"] == "groupcv"
    assert split_plan.split_info["cv"]["inferred_from_columns"] is True


def test_plan_splits_prefers_timecv_over_groupcv_when_both_columns_provided() -> None:
    df = _sample_df_with_split_columns()
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv=4,
        random_state=42,
        test_size=0.2,
        split_group_columns=["ActionGroup"],
        split_date_column="BatchDate",
        cv_source_df=df,
    )

    assert split_plan.cv_splitter_name == "StratifiedGroupTimeSeriesCV"
    assert split_plan.split_info["cv"]["strategy_requested"] == "stratifytimecv"
    assert split_plan.split_info["cv"]["strategy_used"] == "stratifytimecv"


def test_plan_splits_infers_stratifycv_from_stratify_column() -> None:
    df = _sample_df_with_split_columns()
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv=4,
        random_state=42,
        test_size=0.2,
        cv_stratify_column="Strata",
        cv_source_df=df,
    )

    assert split_plan.cv_splitter_name == "StratifiedKFold"
    assert split_plan.split_info["cv"]["strategy_requested"] == "stratifycv"
    assert split_plan.split_info["cv"]["strategy_used"] == "stratifycv"


def test_plan_splits_stratifygroupcv_falls_back_to_groupcv() -> None:
    df = _sample_df_with_split_columns()
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv="stratifygroupcv",
        cv_params={"n_splits": 4},
        random_state=42,
        test_size=0.2,
        cv_group_columns=["ActionGroup"],
        cv_stratify_column="Strata",
        cv_source_df=df,
    )

    assert split_plan.split_info["cv"]["strategy_requested"] == "stratifygroupcv"
    assert split_plan.split_info["cv"]["strategy_used"] == "groupcv"
    assert split_plan.split_info["cv"]["fallback_applied"] is True
    assert len(split_plan.warnings) >= 1


def test_plan_splits_stratifytimecv_falls_back_to_timecv() -> None:
    df = _sample_df_with_split_columns()
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv="stratifytimecv",
        cv_params={"n_splits": 4},
        random_state=42,
        test_size=0.2,
        split_date_column="BatchDate",
        split_stratify_column="Strata",
        cv_source_df=df,
    )

    assert split_plan.split_info["cv"]["strategy_requested"] == "stratifytimecv"
    assert split_plan.split_info["cv"]["strategy_used"] == "stratifytimecv"
    assert split_plan.split_info["cv"]["fallback_applied"] is False


def test_stratified_time_holdout_uses_group_based_split_when_no_stratify_column() -> None:
    df = pd.DataFrame(
        {
            "x1": np.arange(8, dtype=float),
            "x2": np.arange(8, dtype=float) * 0.1,
            "Group": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "BatchDate": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                ]
            ),
            "Target": np.arange(8, dtype=float),
        }
    )
    split_plan = plan_splits(
        X=df[["x1", "x2"]],
        y=df["Target"],
        task="regression",
        cv=2,
        random_state=42,
        test_size=0.25,
        test_size_rows=4,
        split_group_columns=["Group"],
        split_date_column="BatchDate",
        cv_source_df=df,
    )
    assert split_plan.test_splitter_name == "StratifiedTimeLastRowsSplit"
    assert set(split_plan.test_indices.tolist()) == {4, 5, 6, 7}
    assert split_plan.split_info["test"]["stratify_source"] == "groups:Group"
    assert split_plan.split_info["test"]["fallback_applied"] is False


def test_stratified_time_cv_has_expected_grouped_fold_indices() -> None:
    df = pd.DataFrame(
        {
            "x1": np.arange(12, dtype=float),
            "x2": np.arange(12, dtype=float) * 0.01,
            "Group": ["A", "B"] * 6,
            "BatchDate": pd.date_range("2024-01-01", periods=12, freq="D"),
            "Target": np.arange(12, dtype=float),
        }
    )
    split_plan = plan_splits(
        X=df[["x1", "x2"]],
        y=df["Target"],
        task="regression",
        cv=2,
        random_state=42,
        test_size=2 / 12,
        test_size_rows=2,
        split_group_columns=["Group"],
        split_date_column="BatchDate",
        cv_source_df=df,
    )
    assert split_plan.cv_splitter_name == "StratifiedGroupTimeSeriesCV"
    assert split_plan.split_info["cv"]["strategy_used"] == "stratifytimecv"
    fold_1_val = set(split_plan.train_indices[split_plan.cv_splits[0][1]].tolist())
    fold_2_val = set(split_plan.train_indices[split_plan.cv_splits[1][1]].tolist())
    assert fold_1_val == {6, 7}
    assert fold_2_val == {8, 9}


def test_stratified_time_cv_merges_sparse_groups_and_reports_ties() -> None:
    df = pd.DataFrame(
        {
            "x1": np.arange(14, dtype=float),
            "x2": np.arange(14, dtype=float) * 0.1,
            "Group": ["A"] * 6 + ["B"] * 6 + ["C"] * 2,
            "BatchDate": pd.date_range("2024-01-01", periods=14, freq="D"),
            "Target": np.arange(14, dtype=float),
        }
    )
    split_plan = plan_splits(
        X=df[["x1", "x2"]],
        y=df["Target"],
        task="regression",
        cv=2,
        random_state=42,
        test_size=0.2,
        split_group_columns=["Group"],
        split_date_column="BatchDate",
        cv_source_df=df,
    )
    assert split_plan.split_info["cv"]["strategy_used"] == "stratifytimecv"
    tied = split_plan.split_info["cv"]["tied_groups"]
    assert isinstance(tied, dict)
    assert tied


def test_stratified_time_cv_falls_back_to_timecv_when_not_enough_group_buckets() -> None:
    df = pd.DataFrame(
        {
            "x1": np.arange(8, dtype=float),
            "x2": np.arange(8, dtype=float) * 0.1,
            "Group": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "BatchDate": pd.date_range("2024-01-01", periods=8, freq="D"),
            "Target": np.arange(8, dtype=float),
        }
    )
    split_plan = plan_splits(
        X=df[["x1", "x2"]],
        y=df["Target"],
        task="regression",
        cv=3,
        random_state=42,
        test_size=0.25,
        split_group_columns=["Group"],
        split_date_column="BatchDate",
        cv_source_df=df,
    )
    assert split_plan.split_info["cv"]["strategy_requested"] == "stratifytimecv"
    assert split_plan.split_info["cv"]["strategy_used"] == "timecv"
    assert split_plan.split_info["cv"]["fallback_applied"] is True
    assert len(split_plan.warnings) >= 1


def test_plan_splits_uses_last_rows_for_test_holdout_by_default() -> None:
    df = _sample_df_with_split_columns(n=20)
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv=3,
        random_state=42,
        test_size=0.15,
        cv_source_df=df,
    )

    assert split_plan.split_info["test"]["splitter"] == "last_rows"
    assert split_plan.split_info["test"]["test_size"] == 3
    assert split_plan.test_indices.tolist() == [17, 18, 19]


def test_plan_splits_orders_test_holdout_by_split_date_column() -> None:
    df = _sample_df_with_split_columns(n=20).sample(frac=1.0, random_state=7).reset_index(drop=True)
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv=3,
        random_state=42,
        test_size=0.2,
        split_date_column="BatchDate",
        cv_source_df=df,
    )
    expected_test_idx = (
        df.sort_values("BatchDate")
        .tail(4)
        .index.to_numpy(dtype=int)
        .tolist()
    )
    assert split_plan.test_indices.tolist() == expected_test_idx


def test_plan_splits_supports_test_size_rows() -> None:
    df = _sample_df_with_split_columns(n=25)
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv=3,
        random_state=42,
        test_size=0.15,
        test_size_rows=5,
        cv_source_df=df,
    )
    assert split_plan.split_info["test"]["test_size"] == 5
    assert split_plan.test_indices.tolist() == [20, 21, 22, 23, 24]


def test_plan_splits_resolves_n_splits_from_cv_fold_size_rows_for_non_time_cv() -> None:
    df = _sample_df_with_split_columns(n=250)
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv=5,
        random_state=42,
        test_size=0.2,
        cv_fold_size_rows=36,
        cv_source_df=df,
    )
    train_rows = split_plan.split_info["test"]["train_size"]
    expected_n_splits = train_rows // 36
    assert split_plan.split_info["cv"]["n_splits"] == expected_n_splits
    assert split_plan.split_info["cv"]["n_splits_derived_from_fold_size"] is True
    assert split_plan.split_info["cv"]["fold_size_rows"] == 36


def test_plan_splits_timecv_uses_cv_fold_size_rows_as_validation_size() -> None:
    df = _sample_df_with_split_columns(n=300)
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv=5,
        random_state=42,
        test_size=0.2,
        split_date_column="BatchDate",
        cv_fold_size_rows=36,
        cv_source_df=df,
    )

    val_sizes = [len(val_idx) for _, val_idx in split_plan.cv_splits]
    assert len(val_sizes) == 5
    assert all(size == 36 for size in val_sizes)
    assert split_plan.split_info["cv"]["fold_size_rows"] == 36


def test_plan_splits_timecv_allows_single_split() -> None:
    df = _sample_df_with_split_columns(n=120)
    X = df[["x1", "x2"]]
    y = df["Target"]
    split_plan = plan_splits(
        X=X,
        y=y,
        task="regression",
        cv=1,
        random_state=42,
        test_size=0.2,
        split_date_column="BatchDate",
        cv_fold_size_rows=18,
        cv_source_df=df,
    )
    assert split_plan.split_info["cv"]["n_splits"] == 1
    assert len(split_plan.cv_splits) == 1
    assert len(split_plan.cv_splits[0][1]) == 18
