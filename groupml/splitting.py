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


def plan_splits(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cv: Any,
    random_state: int,
    test_size: float,
    cv_params: dict[str, Any] | None = None,
    cv_group_columns: Sequence[str] | None = None,
    fallback_cv_group_columns: Sequence[str] | None = None,
    test_splitter: Any = None,
    include_indices: bool = True,
) -> SplitPlan:
    """Create holdout and CV splits with inspection metadata."""
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

    cv_splitter = resolve_cv_splitter(
        cv=cv,
        task=task,
        random_state=random_state,
        cv_params=cv_params,
    )
    selected_group_columns = list(cv_group_columns or [])
    if not selected_group_columns and _needs_groups(cv_splitter):
        selected_group_columns = list(fallback_cv_group_columns or [])
    groups = build_groups_array(X_train, selected_group_columns)
    if _needs_groups(cv_splitter) and groups is None:
        raise ValueError(
            f"{_splitter_display_name(cv_splitter)} requires groups. "
            "Set cv_group_columns in GroupMLConfig."
        )

    cv_splits = _run_splitter(cv_splitter, X_train, y_train, groups=groups)

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
            "splitter": _splitter_display_name(cv_splitter),
            "n_splits": int(len(cv_splits)),
            "uses_groups": groups is not None,
            "group_columns": selected_group_columns,
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
        cv_splitter_name=_splitter_display_name(cv_splitter),
        test_splitter_name=test_splitter_name,
        split_info=split_info,
    )
