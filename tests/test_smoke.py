from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from groupml import GroupMLConfig, GroupMLRunner


def test_smoke_regression_runs() -> None:
    rng = np.random.default_rng(42)
    n = 220
    df = pd.DataFrame(
        {
            "Temperature": rng.normal(20, 5, n),
            "Pressure": rng.normal(10, 2, n),
            "ActionGroup": rng.choice(["A", "B", "C"], size=n),
        }
    )
    df["Target"] = 1.5 * df["Temperature"] + 0.6 * df["Pressure"] + rng.normal(0, 1, n)

    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        rule_splits=["Temperature < 20", "Temperature >= 20"],
        experiment_modes=["full", "group_as_features", "group_split", "group_permutations", "rule_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
    )
    result = GroupMLRunner(config).fit_evaluate(df)

    assert not result.leaderboard.empty
    assert "recommendation" not in result.leaderboard.columns
    assert isinstance(result.recommendation, str)
    assert "mode" in result.leaderboard.columns
