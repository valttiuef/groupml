"""Configuration models for groupml."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

ExperimentMode = Literal[
    "full",
    "group_as_features",
    "group_split",
    "group_permutations",
    "rule_split",
]


@dataclass(slots=True)
class GroupMLConfig:
    """Configuration for `GroupMLRunner`.

    Parameters are intentionally pragmatic and sklearn-native.
    """

    target: str
    feature_columns: Sequence[str] | None = None
    group_columns: Sequence[str] = field(default_factory=list)
    rule_splits: Sequence[str] = field(default_factory=list)
    experiment_modes: Sequence[ExperimentMode] = field(
        default_factory=lambda: [
            "full",
            "group_as_features",
            "group_split",
            "group_permutations",
            "rule_split",
        ]
    )
    models: str | Sequence[Any] | dict[str, Any] = "default_fast"
    feature_selectors: str | Sequence[Any] | dict[str, Any] = "default_fast"
    cv: int | str | dict[str, Any] | Any = 5
    cv_params: dict[str, Any] = field(default_factory=dict)
    cv_group_column: str | None = None
    cv_group_columns: Sequence[str] | None = None
    cv_date_column: str | None = None
    cv_stratify_column: str | None = None
    test_splitter: Any = None
    include_split_indices: bool = True
    scorer: str | Callable[..., float] = "neg_root_mean_squared_error"
    test_size: float = 0.2
    random_state: int = 42
    scale_numeric: bool = False
    dropna_base_rows: bool = True
    drop_static_base_features: bool = True
    min_target: float | None = None
    max_target: float | None = None
    min_group_size: int = 15
    min_improvement: float = 0.01
    task: Literal["auto", "regression", "classification"] = "auto"

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1).")
        if self.min_group_size < 1:
            raise ValueError("min_group_size must be >= 1.")
        if self.cv is None:
            raise ValueError("cv must be an int or a splitter with .split.")
        if not isinstance(self.cv_params, dict):
            raise ValueError("cv_params must be a dictionary.")
        if self.cv_group_column is not None and not isinstance(self.cv_group_column, str):
            raise ValueError("cv_group_column must be a string column name or None.")
        if self.cv_group_columns is not None:
            if isinstance(self.cv_group_columns, str):
                self.cv_group_columns = [self.cv_group_columns]
            else:
                self.cv_group_columns = list(self.cv_group_columns)
        if self.cv_date_column is not None and not isinstance(self.cv_date_column, str):
            raise ValueError("cv_date_column must be a string column name or None.")
        if self.cv_stratify_column is not None and not isinstance(self.cv_stratify_column, str):
            raise ValueError("cv_stratify_column must be a string column name or None.")
        if self.min_target is not None and self.max_target is not None:
            if self.min_target > self.max_target:
                raise ValueError("min_target must be <= max_target.")
