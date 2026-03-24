"""Custom CV splitter + custom scorer example."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold

from groupml import compare_group_strategies


def build_data(n: int = 500, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mode = rng.choice(["M1", "M2", "M3"], size=n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    target = 2.0 * x1 - 1.4 * x2 + np.where(mode == "M3", 2.0, 0.0) + rng.normal(0, 0.7, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "Mode": mode, "Target": target})


if __name__ == "__main__":
    df = build_data()
    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    rmse_scorer = make_scorer(lambda yt, yp: -mean_squared_error(yt, yp, squared=False))
    result = compare_group_strategies(
        df=df,
        target="Target",
        group_columns=["Mode"],
        experiment_modes=["full", "group_as_features", "group_split"],
        cv=cv,
        scorer=rmse_scorer,
    )
    print(result.recommendation)
    print(result.leaderboard.head(10))

