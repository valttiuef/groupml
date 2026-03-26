"""Functional public APIs."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import pandas as pd

from .config import GroupMLConfig
from .result import GroupMLResult


def compare_group_strategies(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str] | None = None,
    group_columns: list[str] | None = None,
    rule_splits: list[str] | None = None,
    callbacks: Iterable[Callable[[dict[str, Any]], None]] | None = None,
    **kwargs: Any,
) -> GroupMLResult:
    """Functional API wrapper around `GroupMLRunner`."""
    from .runner import GroupMLRunner

    config = GroupMLConfig(
        target=target,
        feature_columns=feature_columns,
        group_columns=group_columns or [],
        rule_splits=rule_splits or [],
        **kwargs,
    )
    return GroupMLRunner(config).fit_evaluate(df, callbacks=callbacks)
