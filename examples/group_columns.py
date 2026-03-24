"""Compare global vs group-aware models with process states/material groups."""

from __future__ import annotations

import numpy as np
import pandas as pd

from groupml import GroupMLConfig, GroupMLRunner


def build_data(n: int = 700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    action = rng.choice(["Idle", "Run", "Boost"], size=n, p=[0.25, 0.55, 0.20])
    material = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
    temperature = rng.normal(24, 6, n)
    pressure = rng.normal(9, 2.5, n)
    group_effect = np.where(action == "Boost", 4.0, np.where(action == "Run", 1.5, -1.0))
    material_effect = np.where(material == "B", 2.5, 0.0)
    target = 1.2 * temperature + 0.7 * pressure + group_effect + material_effect + rng.normal(0, 1.5, n)
    return pd.DataFrame(
        {
            "Temperature": temperature,
            "Pressure": pressure,
            "ActionGroup": action,
            "Material": material,
            "Target": target,
        }
    )


if __name__ == "__main__":
    df = build_data()
    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup", "Material"],
        experiment_modes=["full", "group_as_features", "group_split", "group_permutations"],
        scorer="neg_mean_absolute_error",
        random_state=42,
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    print(result.recommendation)
    print(result.leaderboard.head(10))

