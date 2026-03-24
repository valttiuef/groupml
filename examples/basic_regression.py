"""Basic regression example using synthetic process-like data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from groupml import GroupMLConfig, GroupMLRunner


def build_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temperature = rng.normal(25, 7, n)
    pressure = rng.normal(10, 3, n)
    flow = rng.normal(80, 10, n)
    noise = rng.normal(0, 1.2, n)
    target = 0.8 * temperature + 0.5 * pressure - 0.1 * flow + noise
    return pd.DataFrame(
        {
            "Temperature": temperature,
            "Pressure": pressure,
            "Flow": flow,
            "Target": target,
        }
    )


if __name__ == "__main__":
    df = build_data()
    config = GroupMLConfig(target="Target", experiment_modes=["full"], scorer="neg_root_mean_squared_error")
    result = GroupMLRunner(config).fit_evaluate(df)
    print(result.summary_text())
    print(result.leaderboard.head(5))
