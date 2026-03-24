from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from groupml import GroupMLConfig, GroupMLRunner, compare_group_strategies


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
