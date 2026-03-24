"""Run experiments directly from CSV/Excel files with groupml file utilities."""

from __future__ import annotations

from groupml import GroupMLConfig, fit_evaluate_file


if __name__ == "__main__":
    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full", "group_as_features"],
        cv=3,
        random_state=42,
    )
    result = fit_evaluate_file("examples/data/group_split_demo.csv", config)
    print(result.summary_text())
    print(result.leaderboard.head(5))
