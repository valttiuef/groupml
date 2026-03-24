"""Run groupml functional API directly from a file path."""

from __future__ import annotations

from groupml import compare_group_strategies_file


if __name__ == "__main__":
    result = compare_group_strategies_file(
        path="examples/data/rule_split_demo.csv",
        target="Target",
        rule_splits=["Temperature < 20", "Temperature >= 20"],
        experiment_modes=["full", "rule_split"],
        cv=3,
        random_state=42,
    )
    print(result.summary_text())
    print(result.leaderboard.head(5))
