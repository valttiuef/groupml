from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from groupml import GroupMLConfig, GroupMLRunner
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
