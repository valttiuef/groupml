"""Rule-split example using temperature regime boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from groupml import GroupMLConfig, GroupMLRunner


def build_data(n: int = 650, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temperature = rng.normal(22, 8, n)
    pressure = rng.normal(11, 2, n)
    low_regime = temperature < 20
    target = np.where(
        low_regime,
        0.4 * temperature + 0.9 * pressure,
        1.6 * temperature + 0.3 * pressure,
    ) + rng.normal(0, 1.0, n)
    return pd.DataFrame(
        {
            "Temperature": temperature,
            "Pressure": pressure,
            "Target": target,
        }
    )


if __name__ == "__main__":
    df = build_data()
    config = GroupMLConfig(
        target="Target",
        rule_splits=["Temperature < 20", "Temperature >= 20"],
        experiment_modes=["full", "rule_split"],
        scorer="neg_root_mean_squared_error",
    )
    result = GroupMLRunner(config).fit_evaluate(df)
    print(result.summary_text())
