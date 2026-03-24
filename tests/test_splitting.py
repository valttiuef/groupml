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
    assert result.split_info["test"]["splitter"] == "train_test_split"
