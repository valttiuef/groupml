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
    cv_fold_size_rows: int | None = None
    split_group_column: str | None = None
    split_group_columns: Sequence[str] | None = None
    split_date_column: str | None = None
    split_stratify_column: str | None = None
    # Backward-compatible aliases for prior config field names.
    cv_group_column: str | None = None
    cv_group_columns: Sequence[str] | None = None
    cv_date_column: str | None = None
    cv_stratify_column: str | None = None
    test_splitter: Any = None
    test_split_strategy: Literal["last_rows", "random"] = "last_rows"
    include_split_indices: bool = True
    scorer: str | Callable[..., float] = "neg_root_mean_squared_error"
    test_size: float = 0.15
    test_size_rows: int | None = None
    random_state: int = 42
    scale_numeric: bool = False
    dropna_base_rows: bool = True
    drop_static_base_features: bool = True
    raw_report_enabled: bool = True
    raw_report_max_columns: int = 10
    min_target: float | None = None
    max_target: float | None = None
    min_group_size: int = 15
    min_improvement: float = 0.01
    task: Literal["auto", "regression", "classification"] = "auto"
    warning_verbosity: Literal["quiet", "default", "all"] = "quiet"

    def __post_init__(self) -> None:
        if self.test_size_rows is not None:
            if not isinstance(self.test_size_rows, int) or self.test_size_rows < 1:
                raise ValueError("test_size_rows must be a positive integer or None.")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1).")
        if self.min_group_size < 1:
            raise ValueError("min_group_size must be >= 1.")
        if self.cv is None:
            raise ValueError("cv must be an int or a splitter with .split.")
        if not isinstance(self.cv_params, dict):
            raise ValueError("cv_params must be a dictionary.")
        if self.cv_fold_size_rows is not None:
            if not isinstance(self.cv_fold_size_rows, int) or self.cv_fold_size_rows < 1:
                raise ValueError("cv_fold_size_rows must be a positive integer or None.")
        # Promote legacy cv_* aliases into split_* fields when needed.
        if self.split_group_column is None and self.cv_group_column is not None:
            self.split_group_column = self.cv_group_column
        if self.split_group_columns is None and self.cv_group_columns is not None:
            self.split_group_columns = self.cv_group_columns
        if self.split_date_column is None and self.cv_date_column is not None:
            self.split_date_column = self.cv_date_column
        if self.split_stratify_column is None and self.cv_stratify_column is not None:
            self.split_stratify_column = self.cv_stratify_column

        if self.split_group_column is not None and not isinstance(self.split_group_column, str):
            raise ValueError("split_group_column must be a string column name or None.")
        if self.split_group_columns is not None:
            if isinstance(self.split_group_columns, str):
                self.split_group_columns = [self.split_group_columns]
            else:
                self.split_group_columns = list(self.split_group_columns)
        if self.split_date_column is not None and not isinstance(self.split_date_column, str):
            raise ValueError("split_date_column must be a string column name or None.")
        if self.split_stratify_column is not None and not isinstance(self.split_stratify_column, str):
            raise ValueError("split_stratify_column must be a string column name or None.")
        if self.test_split_strategy not in {"last_rows", "random"}:
            raise ValueError("test_split_strategy must be one of: 'last_rows', 'random'.")
        if self.min_target is not None and self.max_target is not None:
            if self.min_target > self.max_target:
                raise ValueError("min_target must be <= max_target.")
        if not isinstance(self.raw_report_enabled, bool):
            raise ValueError("raw_report_enabled must be a boolean.")
        if not isinstance(self.raw_report_max_columns, int) or self.raw_report_max_columns < 1:
            raise ValueError("raw_report_max_columns must be an integer >= 1.")
        if self.warning_verbosity not in {"quiet", "default", "all"}:
            raise ValueError("warning_verbosity must be one of: 'quiet', 'default', 'all'.")
