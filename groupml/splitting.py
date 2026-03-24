"""Dataset split planning utilities for holdout and CV."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    train_test_split,
)

CV_FACTORY: dict[str, type[Any]] = {
    "kfold": KFold,
    "stratifiedkfold": StratifiedKFold,
    "groupkfold": GroupKFold,
    "stratifiedgroupkfold": StratifiedGroupKFold,
    "timeseriessplit": TimeSeriesSplit,
    "repeatedkfold": RepeatedKFold,
    "repeatedstratifiedkfold": RepeatedStratifiedKFold,
    "shufflesplit": ShuffleSplit,
    "stratifiedshufflesplit": StratifiedShuffleSplit,
    "groupshufflesplit": GroupShuffleSplit,
    "leaveoneout": LeaveOneOut,
    "leavepout": LeavePOut,
    "leaveonegroupout": LeaveOneGroupOut,
    "leavepgroupsout": LeavePGroupsOut,
    "predefinedsplit": PredefinedSplit,
}

GROUP_REQUIRED_SPLITTERS = {
    "groupkfold",
    "stratifiedgroupkfold",
    "groupshufflesplit",
    "leaveonegroupout",
    "leavepgroupsout",
}


@dataclass(slots=True)
class SplitPlan:
    """Fully materialized split artifacts used by the runner."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    cv_splits: list[tuple[np.ndarray, np.ndarray]]
    cv_splitter_name: str
    test_splitter_name: str
    split_info: dict[str, Any]
    warnings: list[str]


class CallableSplitter:
    """Adapter for user-provided splitter callables."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Any:
        try:
            return self.fn(X, y, groups)
        except TypeError:
            try:
                return self.fn(X, y)
            except TypeError:
                return self.fn(X)


class PrecomputedSplitter:
    """Adapter for precomputed CV splits."""

    def __init__(self, splits: Iterable[tuple[np.ndarray, np.ndarray]]) -> None:
        self._splits = [
            (np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int))
            for train_idx, val_idx in splits
        ]

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Any:
        return iter(self._splits)


class DateTimeSeriesSplitter:
    """Time-series splitter that first sorts rows by a provided datetime-like series."""

    def __init__(self, date_values: pd.Series, n_splits: int) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 for time-series CV.")
        datetimes = pd.to_datetime(date_values, errors="coerce")
        if datetimes.isna().any():
            raise ValueError("cv_date_column contains invalid datetime values.")
        self._ordered_idx = np.argsort(datetimes.to_numpy(), kind="mergesort")
        self._base = TimeSeriesSplit(n_splits=n_splits)

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Any:
        del X, y, groups
        n_samples = len(self._ordered_idx)
        ordered_positions = np.arange(n_samples)
        for train_sorted, val_sorted in self._base.split(ordered_positions):
            train_idx = self._ordered_idx[train_sorted]
            val_idx = self._ordered_idx[val_sorted]
            yield np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _canonical_name(name: str) -> str:
    return name.replace("_", "").replace("-", "").lower()


def _with_random_state_if_supported(
    splitter_cls: type[Any],
    params: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    sig = inspect.signature(splitter_cls.__init__)
    out = dict(params)
    if "random_state" in sig.parameters and "random_state" not in out:
        if "shuffle" in sig.parameters:
            if bool(out.get("shuffle", False)):
                out["random_state"] = random_state
        else:
            out["random_state"] = random_state
    if "shuffle" in sig.parameters and "shuffle" not in out:
        if splitter_cls in {KFold, StratifiedKFold}:
            out["shuffle"] = True
    return out


def build_groups_array(df: pd.DataFrame, group_columns: Sequence[str]) -> np.ndarray | None:
    """Return hashable row-wise groups, or None if no group columns are provided."""
    cols = list(group_columns)
    if not cols:
        return None
    if len(cols) == 1:
        return df[cols[0]].astype(str).to_numpy()
    return df[cols].astype(str).agg("||".join, axis=1).to_numpy()


def resolve_cv_splitter(
    cv: Any,
    task: str,
    random_state: int,
    cv_params: dict[str, Any] | None = None,
) -> Any:
    """Resolve int/string/custom CV input into a splitter with `.split`."""
    params = dict(cv_params or {})

    if isinstance(cv, int):
        if cv < 2:
            raise ValueError("cv must be >= 2.")
        if task == "classification":
            return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        return KFold(n_splits=cv, shuffle=True, random_state=random_state)

    if isinstance(cv, str):
        key = _canonical_name(cv)
        alias_map = {
            "groupcv": "groupkfold",
            "timecv": "timeseriessplit",
            "stratifycv": "stratifiedkfold",
        }
        key = alias_map.get(key, key)
        if key not in CV_FACTORY:
            raise ValueError(
                f"Unknown cv splitter '{cv}'. Supported: {sorted(CV_FACTORY.keys())}"
            )
        splitter_cls = CV_FACTORY[key]
        final_params = _with_random_state_if_supported(splitter_cls, params, random_state)
        return splitter_cls(**final_params)

    if isinstance(cv, dict):
        name = cv.get("name")
        if not isinstance(name, str):
            raise ValueError("cv dict must include string field 'name'.")
        merged_params = dict(cv.get("params", {}))
        merged_params.update(params)
        return resolve_cv_splitter(name, task, random_state, cv_params=merged_params)

    if callable(cv):
        return CallableSplitter(cv)

    if hasattr(cv, "split"):
        return cv

    try:
        return PrecomputedSplitter(cv)
    except Exception as exc:  # pragma: no cover
        raise ValueError(
            "cv must be an int, splitter name, splitter object, callable, or split iterable."
        ) from exc


def _run_splitter(
    splitter: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        raw = splitter.split(X, y, groups)
    except TypeError:
        try:
            raw = splitter.split(X, y)
        except TypeError:
            raw = splitter.split(X)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in raw:
        splits.append((np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)))
    if not splits:
        raise ValueError("Resolved CV splitter yielded zero folds.")
    return splits


def _splitter_display_name(splitter: Any) -> str:
    return splitter.__class__.__name__


def _needs_groups(splitter: Any) -> bool:
    key = _canonical_name(_splitter_display_name(splitter))
    return key in GROUP_REQUIRED_SPLITTERS


def _materialize_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    test_size: float,
    random_state: int,
    test_splitter: Any = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if test_splitter is None:
        stratify = y if task == "classification" and y.nunique(dropna=True) > 1 else None
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        return (
            np.asarray(train_idx, dtype=int),
            np.asarray(test_idx, dtype=int),
            "train_test_split",
        )

    splitter = resolve_cv_splitter(
        cv=test_splitter,
        task=task,
        random_state=random_state,
    )
    splits = _run_splitter(splitter, X, y, groups=None)
    train_idx, test_idx = splits[0]
    return train_idx, test_idx, _splitter_display_name(splitter)


def _infer_column_driven_cv(
    cv: Any,
    cv_date_column: str | None,
    cv_group_columns: Sequence[str] | None,
    cv_stratify_column: str | None,
) -> Any:
    if not isinstance(cv, int):
        return cv
    has_date = bool(cv_date_column)
    has_groups = bool(list(cv_group_columns or []))
    has_stratify = bool(cv_stratify_column)

    if has_stratify and has_date:
        return "stratifytimecv"
    if has_stratify and has_groups:
        return "stratifygroupcv"
    if has_date:
        return "timecv"
    if has_groups:
        return "groupcv"
    if has_stratify:
        return "stratifycv"
    return cv


def _resolve_n_splits(cv: Any, cv_params: dict[str, Any], default: int = 5) -> int:
    if isinstance(cv, int):
        return int(cv)
    value = cv_params.get("n_splits", default)
    return int(value)


def _is_valid_group_split(train_idx: np.ndarray, val_idx: np.ndarray, groups: np.ndarray) -> bool:
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    return len(train_groups.intersection(val_groups)) == 0


def _is_valid_time_split(train_idx: np.ndarray, val_idx: np.ndarray, date_values: pd.Series) -> bool:
    train_dates = pd.to_datetime(date_values.iloc[train_idx], errors="coerce")
    val_dates = pd.to_datetime(date_values.iloc[val_idx], errors="coerce")
    if train_dates.isna().any() or val_dates.isna().any():
        return False
    return bool(train_dates.max() <= val_dates.min())


def _build_hybrid_stratified_splits(
    n_splits: int,
    y_stratify: pd.Series,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    idx = np.arange(len(y_stratify))
    return [
        (np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int))
        for train_idx, val_idx in splitter.split(idx, y_stratify)
    ]


def _build_cv_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str,
    cv: Any,
    random_state: int,
    cv_params: dict[str, Any],
    cv_group_columns: list[str],
    fallback_cv_group_columns: list[str],
    cv_date_column: str | None,
    cv_stratify_column: str | None,
    cv_source_train: pd.DataFrame,
    warnings: list[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, dict[str, Any]]:
    selected_cv = _infer_column_driven_cv(
        cv=cv,
        cv_date_column=cv_date_column,
        cv_group_columns=cv_group_columns,
        cv_stratify_column=cv_stratify_column,
    )
    inferred_from_columns = selected_cv != cv and isinstance(cv, int)

    key = _canonical_name(selected_cv) if isinstance(selected_cv, str) else ""
    n_splits = _resolve_n_splits(cv, cv_params, default=5)
    if "n_splits" not in cv_params and isinstance(cv, int):
        cv_params = {**cv_params, "n_splits": int(cv)}

    if key == "groupcv":
        group_columns_for_split = list(cv_group_columns or fallback_cv_group_columns)
        if not group_columns_for_split:
            raise ValueError("groupcv requires cv_group_column/cv_group_columns.")
        groups = build_groups_array(cv_source_train, group_columns_for_split)
        if groups is None:
            raise ValueError("groupcv requires cv_group_column/cv_group_columns.")
        splitter = GroupKFold(n_splits=n_splits)
        splits = _run_splitter(splitter, X_train, y_train, groups=groups)
        meta = {
            "strategy_requested": "groupcv",
            "strategy_used": "groupcv",
            "uses_groups": True,
            "group_columns": group_columns_for_split,
            "date_column": cv_date_column,
            "stratify_column": cv_stratify_column,
            "fallback_applied": False,
            "inferred_from_columns": inferred_from_columns,
        }
        return splits, "GroupKFold", meta

    if key == "timecv":
        if not cv_date_column:
            raise ValueError("timecv requires cv_date_column.")
        splitter = DateTimeSeriesSplitter(cv_source_train[cv_date_column], n_splits=n_splits)
        splits = _run_splitter(splitter, X_train, y_train, groups=None)
        meta = {
            "strategy_requested": "timecv",
            "strategy_used": "timecv",
            "uses_groups": False,
            "group_columns": cv_group_columns,
            "date_column": cv_date_column,
            "stratify_column": cv_stratify_column,
            "fallback_applied": False,
            "inferred_from_columns": inferred_from_columns,
        }
        return splits, "DateTimeSeriesSplitter", meta

    if key == "stratifycv":
        if not cv_stratify_column:
            raise ValueError("stratifycv requires cv_stratify_column.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = _run_splitter(splitter, X_train, cv_source_train[cv_stratify_column], groups=None)
        meta = {
            "strategy_requested": "stratifycv",
            "strategy_used": "stratifycv",
            "uses_groups": False,
            "group_columns": cv_group_columns,
            "date_column": cv_date_column,
            "stratify_column": cv_stratify_column,
            "fallback_applied": False,
            "inferred_from_columns": inferred_from_columns,
        }
        return splits, "StratifiedKFold", meta

    if key in {"stratifygroupcv", "stratifytimecv"}:
        if not cv_stratify_column:
            raise ValueError(f"{selected_cv} requires cv_stratify_column.")
        stratify_values = cv_source_train[cv_stratify_column]
        candidate_splits = _build_hybrid_stratified_splits(
            n_splits=n_splits,
            y_stratify=stratify_values,
            random_state=random_state,
        )

        if key == "stratifygroupcv":
            group_columns_for_split = list(cv_group_columns or fallback_cv_group_columns)
            if not group_columns_for_split:
                raise ValueError("stratifygroupcv requires cv_group_column/cv_group_columns.")
            groups = build_groups_array(cv_source_train, group_columns_for_split)
            if groups is None:
                raise ValueError("stratifygroupcv requires cv_group_column/cv_group_columns.")
            valid = all(_is_valid_group_split(train_idx, val_idx, groups) for train_idx, val_idx in candidate_splits)
            if valid:
                meta = {
                    "strategy_requested": "stratifygroupcv",
                    "strategy_used": "stratifygroupcv",
                    "uses_groups": True,
                    "group_columns": group_columns_for_split,
                    "date_column": cv_date_column,
                    "stratify_column": cv_stratify_column,
                    "fallback_applied": False,
                    "inferred_from_columns": inferred_from_columns,
                }
                return candidate_splits, "StratifyGroupCV", meta

            warnings.append(
                "stratifygroupcv could not satisfy strict group isolation with stratified folds; falling back to groupcv."
            )
            splitter = GroupKFold(n_splits=n_splits)
            fallback_splits = _run_splitter(splitter, X_train, y_train, groups=groups)
            meta = {
                "strategy_requested": "stratifygroupcv",
                "strategy_used": "groupcv",
                "uses_groups": True,
                "group_columns": group_columns_for_split,
                "date_column": cv_date_column,
                "stratify_column": cv_stratify_column,
                "fallback_applied": True,
                "fallback_reason": "group constraints incompatible with stratified candidate folds",
                "inferred_from_columns": inferred_from_columns,
            }
            return fallback_splits, "GroupKFold", meta

        if not cv_date_column:
            raise ValueError("stratifytimecv requires cv_date_column.")
        valid = all(
            _is_valid_time_split(train_idx, val_idx, cv_source_train[cv_date_column])
            for train_idx, val_idx in candidate_splits
        )
        if valid:
            meta = {
                "strategy_requested": "stratifytimecv",
                "strategy_used": "stratifytimecv",
                "uses_groups": False,
                "group_columns": cv_group_columns,
                "date_column": cv_date_column,
                "stratify_column": cv_stratify_column,
                "fallback_applied": False,
                "inferred_from_columns": inferred_from_columns,
            }
            return candidate_splits, "StratifyTimeCV", meta

        warnings.append(
            "stratifytimecv could not satisfy strict time-order constraints with stratified folds; falling back to timecv."
        )
        splitter = DateTimeSeriesSplitter(cv_source_train[cv_date_column], n_splits=n_splits)
        fallback_splits = _run_splitter(splitter, X_train, y_train, groups=None)
        meta = {
            "strategy_requested": "stratifytimecv",
            "strategy_used": "timecv",
            "uses_groups": False,
            "group_columns": cv_group_columns,
            "date_column": cv_date_column,
            "stratify_column": cv_stratify_column,
            "fallback_applied": True,
            "fallback_reason": "time constraints incompatible with stratified candidate folds",
            "inferred_from_columns": inferred_from_columns,
        }
        return fallback_splits, "DateTimeSeriesSplitter", meta

    cv_splitter = resolve_cv_splitter(
        cv=selected_cv,
        task=task,
        random_state=random_state,
        cv_params=cv_params,
    )
    selected_group_columns = list(cv_group_columns or fallback_cv_group_columns)
    if not selected_group_columns and _needs_groups(cv_splitter):
        raise ValueError(
            f"{_splitter_display_name(cv_splitter)} requires groups. "
            "Set cv_group_column/cv_group_columns in GroupMLConfig."
        )
    groups = build_groups_array(cv_source_train, selected_group_columns)
    if _needs_groups(cv_splitter) and groups is None:
        raise ValueError(
            f"{_splitter_display_name(cv_splitter)} requires groups. "
            "Set cv_group_column/cv_group_columns in GroupMLConfig."
        )

    cv_splits = _run_splitter(cv_splitter, X_train, y_train, groups=groups)
    meta = {
        "strategy_requested": selected_cv if isinstance(selected_cv, str) else _splitter_display_name(cv_splitter),
        "strategy_used": _canonical_name(_splitter_display_name(cv_splitter)),
        "uses_groups": groups is not None,
        "group_columns": selected_group_columns,
        "date_column": cv_date_column,
        "stratify_column": cv_stratify_column,
        "fallback_applied": False,
        "inferred_from_columns": inferred_from_columns,
    }
    return cv_splits, _splitter_display_name(cv_splitter), meta


def plan_splits(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cv: Any,
    random_state: int,
    test_size: float,
    cv_params: dict[str, Any] | None = None,
    cv_group_columns: Sequence[str] | None = None,
    cv_date_column: str | None = None,
    cv_stratify_column: str | None = None,
    fallback_cv_group_columns: Sequence[str] | None = None,
    test_splitter: Any = None,
    include_indices: bool = True,
    cv_source_df: pd.DataFrame | None = None,
) -> SplitPlan:
    """Create holdout and CV splits with inspection metadata."""
    split_warnings: list[str] = []

    train_idx, test_idx, test_splitter_name = _materialize_holdout(
        X=X,
        y=y,
        task=task,
        test_size=test_size,
        random_state=random_state,
        test_splitter=test_splitter,
    )
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    source_df = cv_source_df if cv_source_df is not None else X
    if len(source_df) != len(X):
        raise ValueError("cv_source_df must have the same row count as X.")
    cv_source_train = source_df.iloc[train_idx]

    explicit_cv_group_columns = list(cv_group_columns or [])
    fallback_group_columns = list(fallback_cv_group_columns or [])

    if cv_date_column and cv_date_column not in cv_source_train.columns:
        raise ValueError(f"cv_date column '{cv_date_column}' is missing from split source dataframe.")
    if cv_stratify_column and cv_stratify_column not in cv_source_train.columns:
        raise ValueError(f"cv_stratify column '{cv_stratify_column}' is missing from split source dataframe.")
    columns_for_presence_check = list(dict.fromkeys(explicit_cv_group_columns + fallback_group_columns))
    if columns_for_presence_check:
        missing = [c for c in columns_for_presence_check if c not in cv_source_train.columns]
        if missing:
            raise ValueError(f"cv_group columns missing from split source dataframe: {missing}")

    cv_splits, cv_splitter_name, cv_meta = _build_cv_splits(
        X_train=X_train,
        y_train=y_train,
        task=task,
        cv=cv,
        random_state=random_state,
        cv_params=dict(cv_params or {}),
        cv_group_columns=explicit_cv_group_columns,
        fallback_cv_group_columns=fallback_group_columns,
        cv_date_column=cv_date_column,
        cv_stratify_column=cv_stratify_column,
        cv_source_train=cv_source_train,
        warnings=split_warnings,
    )

    fold_rows: list[dict[str, Any]] = []
    for fold, (fold_train_idx, fold_val_idx) in enumerate(cv_splits):
        row: dict[str, Any] = {
            "fold": fold,
            "train_size": int(len(fold_train_idx)),
            "val_size": int(len(fold_val_idx)),
        }
        if include_indices:
            row["train_indices"] = fold_train_idx.tolist()
            row["val_indices"] = fold_val_idx.tolist()
        fold_rows.append(row)

    split_info: dict[str, Any] = {
        "test": {
            "splitter": test_splitter_name,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
        },
        "cv": {
            "splitter": cv_splitter_name,
            "n_splits": int(len(cv_splits)),
            "uses_groups": bool(cv_meta.get("uses_groups", False)),
            "group_columns": list(cv_meta.get("group_columns", explicit_cv_group_columns + fallback_group_columns)),
            "date_column": cv_meta.get("date_column", cv_date_column),
            "stratify_column": cv_meta.get("stratify_column", cv_stratify_column),
            "strategy_requested": cv_meta.get("strategy_requested"),
            "strategy_used": cv_meta.get("strategy_used"),
            "fallback_applied": bool(cv_meta.get("fallback_applied", False)),
            "fallback_reason": cv_meta.get("fallback_reason"),
            "inferred_from_columns": bool(cv_meta.get("inferred_from_columns", False)),
            "folds": fold_rows,
        },
    }
    if include_indices:
        split_info["test"]["train_indices"] = train_idx.tolist()
        split_info["test"]["test_indices"] = test_idx.tolist()

    return SplitPlan(
        train_indices=train_idx,
        test_indices=test_idx,
        cv_splits=cv_splits,
        cv_splitter_name=cv_splitter_name,
        test_splitter_name=test_splitter_name,
        split_info=split_info,
        warnings=split_warnings,
    )
