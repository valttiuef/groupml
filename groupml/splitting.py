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

NSPLITS_PARAM_SPLITTER_KEYS = {
    "kfold",
    "stratifiedkfold",
    "groupkfold",
    "stratifiedgroupkfold",
    "timeseriessplit",
    "repeatedkfold",
    "repeatedstratifiedkfold",
    "shufflesplit",
    "stratifiedshufflesplit",
    "groupshufflesplit",
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

    def __init__(self, date_values: pd.Series, n_splits: int, test_size: int | None = None) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1 for time-series CV.")
        if test_size is not None and test_size < 1:
            raise ValueError("test_size must be >= 1 when provided for time-series CV.")
        datetimes = pd.to_datetime(date_values, errors="coerce")
        if datetimes.isna().any():
            raise ValueError("split_date_column contains invalid datetime values.")
        self._ordered_idx = np.argsort(datetimes.to_numpy(), kind="mergesort")
        self._n_splits = int(n_splits)
        self._test_size = int(test_size) if test_size is not None else None
        self._base: TimeSeriesSplit | None = None
        if self._n_splits > 1:
            kwargs: dict[str, Any] = {"n_splits": self._n_splits}
            if self._test_size is not None:
                kwargs["test_size"] = self._test_size
            self._base = TimeSeriesSplit(**kwargs)

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Any:
        del X, y, groups
        n_samples = len(self._ordered_idx)
        if self._n_splits == 1:
            val_size = self._test_size if self._test_size is not None else max(1, n_samples // 2)
            if val_size >= n_samples:
                raise ValueError(
                    f"Single-fold time CV needs train rows before validation. Got n_samples={n_samples}, "
                    f"val_size={val_size}."
                )
            train_sorted = np.arange(0, n_samples - val_size, dtype=int)
            val_sorted = np.arange(n_samples - val_size, n_samples, dtype=int)
            train_idx = self._ordered_idx[train_sorted]
            val_idx = self._ordered_idx[val_sorted]
            yield np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)
            return

        assert self._base is not None
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


def _build_stratify_values(
    source_df: pd.DataFrame,
    split_stratify_column: str | None,
    group_columns: Sequence[str],
) -> tuple[pd.Series | None, str | None]:
    if split_stratify_column:
        return source_df[split_stratify_column].astype(str), f"column:{split_stratify_column}"
    group_cols = list(group_columns or [])
    if not group_cols:
        return None, None
    groups = build_groups_array(source_df, group_cols)
    if groups is None:
        return None, None
    return pd.Series(groups, index=source_df.index, dtype=str), f"groups:{'+'.join(group_cols)}"


def _merge_sparse_strata(
    stratify_values: pd.Series,
    min_bucket_size: int,
) -> tuple[pd.Series, dict[str, list[str]]]:
    labels = stratify_values.astype(str).fillna("nan")
    counts = labels.value_counts(dropna=False)
    if counts.empty:
        return labels, {}
    if min_bucket_size <= 1 or int(counts.min()) >= int(min_bucket_size):
        return labels, {}

    items = sorted([(str(k), int(v)) for k, v in counts.items()], key=lambda x: (x[1], x[0]))
    buckets: list[list[str]] = []
    bucket_sizes: list[int] = []
    current: list[str] = []
    current_size = 0
    for key, size in items:
        current.append(key)
        current_size += int(size)
        if current_size >= min_bucket_size:
            buckets.append(list(current))
            bucket_sizes.append(int(current_size))
            current = []
            current_size = 0
    if current:
        if buckets:
            buckets[-1].extend(current)
            bucket_sizes[-1] += current_size
        else:
            buckets.append(list(current))
            bucket_sizes.append(int(current_size))

    key_to_bucket: dict[str, str] = {}
    tie_map: dict[str, list[str]] = {}
    for idx, keys in enumerate(buckets, start=1):
        bucket_label = keys[0] if len(keys) == 1 else f"tied_bucket_{idx}"
        if len(keys) > 1:
            tie_map[bucket_label] = list(keys)
        for key in keys:
            key_to_bucket[key] = bucket_label
    merged = labels.map(key_to_bucket).astype(str)
    return merged, tie_map


def _min_rows_needed_for_timecv(n_splits: int, test_size_rows: int | None) -> int:
    if n_splits < 1:
        return 2
    if test_size_rows is None:
        return int(n_splits + 1)
    return int((n_splits * int(test_size_rows)) + 1)


def _allocate_holdout_by_strata(
    strata_counts: pd.Series,
    n_test: int,
) -> dict[str, int] | None:
    counts = {str(k): int(v) for k, v in strata_counts.items()}
    if not counts:
        return None
    if n_test < len(counts):
        return None
    capacities = {k: max(0, v - 1) for k, v in counts.items()}
    if sum(capacities.values()) < n_test:
        return None

    assigned = {k: 1 for k in counts}
    remaining = int(n_test - len(counts))
    if remaining < 0:
        return None

    total_rows = float(sum(counts.values()))
    ideal_minus_base = {
        k: (float(n_test) * (float(v) / total_rows)) - 1.0 for k, v in counts.items()
    }
    while remaining > 0:
        candidates = [k for k in counts if assigned[k] < capacities[k]]
        if not candidates:
            break
        candidates.sort(key=lambda k: (ideal_minus_base[k] - (assigned[k] - 1), counts[k], k), reverse=True)
        chosen = candidates[0]
        assigned[chosen] += 1
        remaining -= 1
    if remaining != 0:
        return None
    if sum(assigned.values()) != n_test:
        return None
    return assigned


def _build_stratified_time_holdout(
    *,
    source_df: pd.DataFrame,
    split_date_column: str,
    stratify_values: pd.Series,
    n_test: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    datetimes = pd.to_datetime(source_df[split_date_column], errors="coerce")
    if datetimes.isna().any():
        raise ValueError("split_date_column contains invalid datetime values.")

    strata_counts = stratify_values.astype(str).value_counts(dropna=False)
    allocated = _allocate_holdout_by_strata(strata_counts, n_test=n_test)
    if allocated is None:
        raise ValueError("unable to allocate test rows across stratified groups.")

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for stratum, n_stratum_test in allocated.items():
        idx = np.flatnonzero(stratify_values.astype(str).to_numpy() == str(stratum))
        if len(idx) <= n_stratum_test:
            raise ValueError("stratum too small to keep both train and test rows.")
        stratum_dates = datetimes.iloc[idx]
        ordered_local = idx[np.argsort(stratum_dates.to_numpy(), kind="mergesort")]
        train_parts.append(np.asarray(ordered_local[:-n_stratum_test], dtype=int))
        test_parts.append(np.asarray(ordered_local[-n_stratum_test:], dtype=int))

    train_idx = np.concatenate(train_parts, axis=0) if train_parts else np.array([], dtype=int)
    test_idx = np.concatenate(test_parts, axis=0) if test_parts else np.array([], dtype=int)
    if len(train_idx) + len(test_idx) != len(source_df):
        raise ValueError("stratified time holdout did not partition all rows.")

    meta = {
        "strategy_used": "stratified_time_last_rows",
        "stratify_groups_count": int(strata_counts.shape[0]),
    }
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int), meta


def _build_stratified_time_cv_splits(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_source_train: pd.DataFrame,
    cv_date_column: str,
    stratify_values: pd.Series,
    n_splits: int,
    time_test_size_rows: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    all_labels = stratify_values.astype(str).to_numpy()
    unique_labels = sorted(set(all_labels.tolist()))
    if len(unique_labels) < 2:
        raise ValueError("not enough stratified groups for stratified time CV.")

    per_label_splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for label in unique_labels:
        label_idx = np.flatnonzero(all_labels == label).astype(int)
        if label_idx.size == 0:
            continue
        label_source = cv_source_train.iloc[label_idx]
        label_X = X_train.iloc[label_idx]
        label_y = y_train.iloc[label_idx]
        splitter = DateTimeSeriesSplitter(
            label_source[cv_date_column],
            n_splits=n_splits,
            test_size=time_test_size_rows,
        )
        local_splits = _run_splitter(splitter, label_X, label_y, groups=None)
        mapped: list[tuple[np.ndarray, np.ndarray]] = []
        for local_train, local_val in local_splits:
            mapped.append(
                (
                    np.asarray(label_idx[local_train], dtype=int),
                    np.asarray(label_idx[local_val], dtype=int),
                )
            )
        per_label_splits[label] = mapped

    if not per_label_splits:
        raise ValueError("stratified time CV yielded no per-group splits.")

    fold_count = len(next(iter(per_label_splits.values())))
    combined: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(fold_count):
        fold_train: list[np.ndarray] = []
        fold_val: list[np.ndarray] = []
        for label in unique_labels:
            label_splits = per_label_splits.get(label, [])
            if fold_idx >= len(label_splits):
                raise ValueError("inconsistent fold count across stratified groups.")
            local_train, local_val = label_splits[fold_idx]
            fold_train.append(local_train)
            fold_val.append(local_val)
        train_idx = np.concatenate(fold_train, axis=0) if fold_train else np.array([], dtype=int)
        val_idx = np.concatenate(fold_val, axis=0) if fold_val else np.array([], dtype=int)
        combined.append((np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)))
    return combined


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
    test_size_rows: int | None,
    random_state: int,
    split_source_df: pd.DataFrame,
    split_date_column: str | None,
    split_stratify_column: str | None,
    split_group_columns_for_stratify: Sequence[str] | None,
    test_split_strategy: str,
    warnings: list[str] | None = None,
    test_splitter: Any = None,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    n_samples = len(X)
    if n_samples < 2:
        raise ValueError("Need at least 2 rows to create train/test holdout splits.")

    if test_size_rows is not None:
        n_test = int(test_size_rows)
    else:
        n_test = int(np.ceil(float(test_size) * float(n_samples)))
    if n_test < 1 or n_test >= n_samples:
        raise ValueError(
            f"Resolved test set size ({n_test}) must be in [1, {n_samples - 1}] for {n_samples} rows."
        )

    if test_splitter is None:
        if test_split_strategy == "last_rows":
            if split_date_column:
                stratify_values, stratify_source = _build_stratify_values(
                    source_df=split_source_df,
                    split_stratify_column=split_stratify_column,
                    group_columns=split_group_columns_for_stratify or [],
                )
                if stratify_values is not None:
                    merged_strata, tie_map = _merge_sparse_strata(stratify_values, min_bucket_size=2)
                    try:
                        train_idx, test_idx, holdout_meta = _build_stratified_time_holdout(
                            source_df=split_source_df,
                            split_date_column=split_date_column,
                            stratify_values=merged_strata,
                            n_test=n_test,
                        )
                        holdout_meta["stratify_source"] = stratify_source
                        holdout_meta["tied_groups"] = tie_map
                        holdout_meta["fallback_applied"] = False
                        return train_idx, test_idx, "StratifiedTimeLastRowsSplit", holdout_meta
                    except Exception as exc:
                        if warnings is not None:
                            warnings.append(
                                "Stratified time holdout could not be built; falling back to plain last_rows/time split. "
                                f"Reason: {exc}"
                            )
                        holdout_meta = {
                            "strategy_used": "last_rows",
                            "stratify_source": stratify_source,
                            "tied_groups": tie_map,
                            "fallback_applied": True,
                            "fallback_reason": "insufficient/imbalanced groups for stratified time holdout",
                        }
                else:
                    holdout_meta = {}
                datetimes = pd.to_datetime(split_source_df[split_date_column], errors="coerce")
                if datetimes.isna().any():
                    raise ValueError("split_date_column contains invalid datetime values.")
                ordered_idx = np.argsort(datetimes.to_numpy(), kind="mergesort")
            else:
                ordered_idx = np.arange(n_samples, dtype=int)
                holdout_meta = {}
            train_idx = ordered_idx[:-n_test]
            test_idx = ordered_idx[-n_test:]
            return (
                np.asarray(train_idx, dtype=int),
                np.asarray(test_idx, dtype=int),
                "last_rows",
                holdout_meta,
            )
        if test_split_strategy == "random":
            stratify = None
            if split_stratify_column:
                stratify = split_source_df[split_stratify_column]
            elif task == "classification" and y.nunique(dropna=True) > 1:
                stratify = y
            try:
                train_idx, test_idx = train_test_split(
                    np.arange(n_samples),
                    test_size=n_test,
                    random_state=random_state,
                    stratify=stratify,
                )
            except ValueError as exc:
                if stratify is None:
                    raise
                if warnings is not None:
                    warnings.append(
                        "Holdout stratification failed; falling back to non-stratified random split. "
                        f"Reason: {exc}"
                    )
                train_idx, test_idx = train_test_split(
                    np.arange(n_samples),
                    test_size=n_test,
                    random_state=random_state,
                    stratify=None,
                )
            return (
                np.asarray(train_idx, dtype=int),
                np.asarray(test_idx, dtype=int),
                "train_test_split",
                {},
            )
        raise ValueError("Unknown test_split_strategy. Supported: 'last_rows', 'random'.")

    splitter = resolve_cv_splitter(
        cv=test_splitter,
        task=task,
        random_state=random_state,
    )
    splits = _run_splitter(splitter, X, y, groups=None)
    train_idx, test_idx = splits[0]
    return train_idx, test_idx, _splitter_display_name(splitter), {}


def _infer_column_driven_cv(
    cv: Any,
    split_date_column: str | None,
    split_group_columns: Sequence[str] | None,
    split_stratify_column: str | None,
) -> Any:
    if not isinstance(cv, int):
        return cv
    has_date = bool(split_date_column)
    has_groups = bool(list(split_group_columns or []))
    has_stratify = bool(split_stratify_column)

    if has_stratify and has_date:
        return "stratifytimecv"
    if has_date:
        if has_groups:
            return "stratifytimecv"
        return "timecv"
    if has_stratify:
        return "stratifycv"
    if has_groups:
        return "groupcv"
    return cv


def _resolve_n_splits(cv: Any, cv_params: dict[str, Any], default: int = 5) -> int:
    if isinstance(cv, int):
        return int(cv)
    value = cv_params.get("n_splits", default)
    return int(value)


def _is_time_strategy_key(key: str) -> bool:
    return key in {"timecv", "stratifytimecv", "timeseriessplit"}


def _resolve_cv_fold_parameters(
    *,
    cv: Any,
    cv_params: dict[str, Any],
    selected_key: str,
    n_train_rows: int,
    cv_fold_size_rows: int | None,
) -> tuple[int, int | None, bool]:
    n_splits = _resolve_n_splits(cv, cv_params, default=5)
    time_test_size_rows: int | None = None
    derived_from_fold_size = False

    if cv_fold_size_rows is None:
        return n_splits, time_test_size_rows, derived_from_fold_size
    if not isinstance(cv, (int, str, dict)):
        return n_splits, time_test_size_rows, derived_from_fold_size

    fold_size = int(cv_fold_size_rows)
    if _is_time_strategy_key(selected_key):
        time_test_size_rows = fold_size
        required_rows = n_splits * fold_size
        if required_rows >= n_train_rows:
            raise ValueError(
                "cv_fold_size_rows is too large for time CV. "
                f"Need n_train_rows > n_splits * cv_fold_size_rows ({n_train_rows} <= {required_rows})."
            )
        return n_splits, time_test_size_rows, derived_from_fold_size

    resolved = int(n_train_rows // fold_size)
    if resolved < 2:
        raise ValueError(
            "cv_fold_size_rows is too large for non-time CV. "
            f"Need at least 2 folds, but only {resolved} fold(s) fit into {n_train_rows} rows."
        )
    n_splits = min(resolved, max(2, n_train_rows - 1))
    derived_from_fold_size = True
    return n_splits, time_test_size_rows, derived_from_fold_size


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
    cv_fold_size_rows: int | None,
    warnings: list[str],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, dict[str, Any]]:
    inference_group_columns = list(cv_group_columns or fallback_cv_group_columns)
    selected_cv = _infer_column_driven_cv(
        cv=cv,
        split_date_column=cv_date_column,
        split_group_columns=inference_group_columns,
        split_stratify_column=cv_stratify_column,
    )
    inferred_from_columns = selected_cv != cv and isinstance(cv, int)

    key = _canonical_name(selected_cv) if isinstance(selected_cv, str) else ""
    n_splits, time_test_size_rows, derived_from_fold_size = _resolve_cv_fold_parameters(
        cv=cv,
        cv_params=cv_params,
        selected_key=key,
        n_train_rows=len(X_train),
        cv_fold_size_rows=cv_fold_size_rows,
    )
    if "n_splits" not in cv_params and isinstance(cv, int):
        cv_params = {**cv_params, "n_splits": int(cv)}
    if derived_from_fold_size:
        warnings.append(
            f"Resolved non-time CV n_splits={n_splits} from cv_fold_size_rows={cv_fold_size_rows} "
            f"and train_rows={len(X_train)}."
        )

    if key == "groupcv":
        group_columns_for_split = list(cv_group_columns or fallback_cv_group_columns)
        if not group_columns_for_split:
            raise ValueError("groupcv requires split_group_column/split_group_columns.")
        groups = build_groups_array(cv_source_train, group_columns_for_split)
        if groups is None:
            raise ValueError("groupcv requires split_group_column/split_group_columns.")
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
            "fold_size_rows": cv_fold_size_rows,
            "n_splits_derived_from_fold_size": derived_from_fold_size,
        }
        return splits, "GroupKFold", meta

    if key == "timecv":
        if not cv_date_column:
            raise ValueError("timecv requires split_date_column.")
        splitter = DateTimeSeriesSplitter(
            cv_source_train[cv_date_column],
            n_splits=n_splits,
            test_size=time_test_size_rows,
        )
        splits = _run_splitter(splitter, X_train, y_train, groups=None)
        meta = {
            "strategy_requested": "timecv",
            "strategy_used": "timecv",
            "uses_groups": False,
            "group_columns": inference_group_columns,
            "date_column": cv_date_column,
            "stratify_column": cv_stratify_column,
            "fallback_applied": False,
            "inferred_from_columns": inferred_from_columns,
            "fold_size_rows": cv_fold_size_rows,
            "n_splits_derived_from_fold_size": derived_from_fold_size,
        }
        return splits, "DateTimeSeriesSplitter", meta

    if key == "stratifycv":
        if not cv_stratify_column:
            raise ValueError("stratifycv requires split_stratify_column.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        try:
            splits = _run_splitter(splitter, X_train, cv_source_train[cv_stratify_column], groups=None)
            strategy_used = "stratifycv"
            fallback_applied = False
            fallback_reason = None
            splitter_name = "StratifiedKFold"
        except ValueError as exc:
            warnings.append(
                "stratifycv could not satisfy class/group frequency constraints; falling back to kfold. "
                f"Reason: {exc}"
            )
            fallback_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = _run_splitter(fallback_splitter, X_train, y_train, groups=None)
            strategy_used = "kfold"
            fallback_applied = True
            fallback_reason = "stratification constraints incompatible with requested folds"
            splitter_name = "KFold"
        meta = {
            "strategy_requested": "stratifycv",
            "strategy_used": strategy_used,
            "uses_groups": False,
            "group_columns": cv_group_columns,
            "date_column": cv_date_column,
            "stratify_column": cv_stratify_column,
            "fallback_applied": fallback_applied,
            "fallback_reason": fallback_reason,
            "inferred_from_columns": inferred_from_columns,
            "fold_size_rows": cv_fold_size_rows,
            "n_splits_derived_from_fold_size": derived_from_fold_size,
        }
        return splits, splitter_name, meta

    if key in {"stratifygroupcv", "stratifytimecv"}:
        if key == "stratifygroupcv" and not cv_stratify_column:
            raise ValueError(f"{selected_cv} requires split_stratify_column.")
        stratify_values: pd.Series
        stratify_source: str | None
        if key == "stratifytimecv":
            stratify_values, stratify_source = _build_stratify_values(
                source_df=cv_source_train,
                split_stratify_column=cv_stratify_column,
                group_columns=inference_group_columns,
            )
            if stratify_values is None:
                warnings.append(
                    "stratifytimecv requested but no stratify/group columns available; falling back to timecv."
                )
                if not cv_date_column:
                    raise ValueError("stratifytimecv requires split_date_column.")
                splitter = DateTimeSeriesSplitter(
                    cv_source_train[cv_date_column],
                    n_splits=n_splits,
                    test_size=time_test_size_rows,
                )
                fallback_splits = _run_splitter(splitter, X_train, y_train, groups=None)
                meta = {
                    "strategy_requested": "stratifytimecv",
                    "strategy_used": "timecv",
                    "uses_groups": False,
                    "group_columns": inference_group_columns,
                    "date_column": cv_date_column,
                    "stratify_column": cv_stratify_column,
                    "stratify_source": None,
                    "tied_groups": {},
                    "fallback_applied": True,
                    "fallback_reason": "no stratify/group columns available",
                    "inferred_from_columns": inferred_from_columns,
                    "fold_size_rows": cv_fold_size_rows,
                    "n_splits_derived_from_fold_size": derived_from_fold_size,
                }
                return fallback_splits, "DateTimeSeriesSplitter", meta
        else:
            stratify_values = cv_source_train[cv_stratify_column]
            stratify_source = f"column:{cv_stratify_column}"
        if key == "stratifygroupcv":
            try:
                candidate_splits = _build_hybrid_stratified_splits(
                    n_splits=n_splits,
                    y_stratify=stratify_values,
                    random_state=random_state,
                )
            except ValueError as exc:
                warnings.append(
                    "stratifygroupcv could not build stratified candidate folds; falling back to groupcv. "
                    f"Reason: {exc}"
                )
                key = "groupcv"
                candidate_splits = []
            group_columns_for_split = list(cv_group_columns or fallback_cv_group_columns)
            if not group_columns_for_split:
                raise ValueError("stratifygroupcv requires split_group_column/split_group_columns.")
            groups = build_groups_array(cv_source_train, group_columns_for_split)
            if groups is None:
                raise ValueError("stratifygroupcv requires split_group_column/split_group_columns.")
            valid = bool(candidate_splits) and all(
                _is_valid_group_split(train_idx, val_idx, groups) for train_idx, val_idx in candidate_splits
            )
            if valid and key == "stratifygroupcv":
                meta = {
                    "strategy_requested": "stratifygroupcv",
                    "strategy_used": "stratifygroupcv",
                    "uses_groups": True,
                    "group_columns": group_columns_for_split,
                    "date_column": cv_date_column,
                    "stratify_column": cv_stratify_column,
                    "fallback_applied": False,
                    "inferred_from_columns": inferred_from_columns,
                    "fold_size_rows": cv_fold_size_rows,
                    "n_splits_derived_from_fold_size": derived_from_fold_size,
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
                "fold_size_rows": cv_fold_size_rows,
                "n_splits_derived_from_fold_size": derived_from_fold_size,
            }
            return fallback_splits, "GroupKFold", meta

        if not cv_date_column:
            raise ValueError("stratifytimecv requires split_date_column.")
        min_rows_needed = _min_rows_needed_for_timecv(n_splits=n_splits, test_size_rows=time_test_size_rows)
        merged_strata, tie_map = _merge_sparse_strata(stratify_values.astype(str), min_bucket_size=min_rows_needed)
        try:
            grouped_splits = _build_stratified_time_cv_splits(
                X_train=X_train,
                y_train=y_train,
                cv_source_train=cv_source_train,
                cv_date_column=cv_date_column,
                stratify_values=merged_strata,
                n_splits=n_splits,
                time_test_size_rows=time_test_size_rows,
            )
            meta = {
                "strategy_requested": "stratifytimecv",
                "strategy_used": "stratifytimecv",
                "uses_groups": bool(stratify_source and stratify_source.startswith("groups:")),
                "group_columns": inference_group_columns,
                "date_column": cv_date_column,
                "stratify_column": cv_stratify_column,
                "stratify_source": stratify_source,
                "tied_groups": tie_map,
                "fallback_applied": False,
                "inferred_from_columns": inferred_from_columns,
                "fold_size_rows": cv_fold_size_rows,
                "n_splits_derived_from_fold_size": derived_from_fold_size,
            }
            return grouped_splits, "StratifiedGroupTimeSeriesCV", meta
        except Exception as exc:
            warnings.append(
                "stratifytimecv could not build grouped time splits; falling back to timecv. "
                f"Reason: {exc}"
            )

        splitter = DateTimeSeriesSplitter(
            cv_source_train[cv_date_column],
            n_splits=n_splits,
            test_size=time_test_size_rows,
        )
        fallback_splits = _run_splitter(splitter, X_train, y_train, groups=None)
        meta = {
            "strategy_requested": "stratifytimecv",
            "strategy_used": "timecv",
            "uses_groups": False,
            "group_columns": inference_group_columns,
            "date_column": cv_date_column,
            "stratify_column": cv_stratify_column,
            "stratify_source": stratify_source,
            "tied_groups": tie_map,
            "fallback_applied": True,
            "fallback_reason": "insufficient/imbalanced groups for grouped time folds",
            "inferred_from_columns": inferred_from_columns,
            "fold_size_rows": cv_fold_size_rows,
            "n_splits_derived_from_fold_size": derived_from_fold_size,
        }
        return fallback_splits, "DateTimeSeriesSplitter", meta

    if derived_from_fold_size:
        if isinstance(selected_cv, int):
            cv_params = {**cv_params, "n_splits": int(n_splits)}
        elif isinstance(selected_cv, str):
            selected_key_for_params = _canonical_name(selected_cv)
            if selected_key_for_params in NSPLITS_PARAM_SPLITTER_KEYS:
                cv_params = {**cv_params, "n_splits": int(n_splits)}
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
            "Set split_group_column/split_group_columns in GroupMLConfig."
        )
    groups = build_groups_array(cv_source_train, selected_group_columns)
    if _needs_groups(cv_splitter) and groups is None:
        raise ValueError(
            f"{_splitter_display_name(cv_splitter)} requires groups. "
            "Set split_group_column/split_group_columns in GroupMLConfig."
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
        "fold_size_rows": cv_fold_size_rows,
        "n_splits_derived_from_fold_size": derived_from_fold_size,
    }
    return cv_splits, _splitter_display_name(cv_splitter), meta


def plan_splits(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cv: Any,
    random_state: int,
    test_size: float,
    test_size_rows: int | None = None,
    cv_params: dict[str, Any] | None = None,
    cv_fold_size_rows: int | None = None,
    split_group_columns: Sequence[str] | None = None,
    split_date_column: str | None = None,
    split_stratify_column: str | None = None,
    fallback_cv_group_columns: Sequence[str] | None = None,
    test_splitter: Any = None,
    test_split_strategy: str = "last_rows",
    include_indices: bool = True,
    cv_source_df: pd.DataFrame | None = None,
    # Backward-compatible aliases.
    cv_group_columns: Sequence[str] | None = None,
    cv_date_column: str | None = None,
    cv_stratify_column: str | None = None,
) -> SplitPlan:
    """Create holdout and CV splits with inspection metadata."""
    split_warnings: list[str] = []

    resolved_split_group_columns = list(split_group_columns or cv_group_columns or [])
    resolved_split_date_column = split_date_column or cv_date_column
    resolved_split_stratify_column = split_stratify_column or cv_stratify_column

    source_df = cv_source_df if cv_source_df is not None else X
    if len(source_df) != len(X):
        raise ValueError("cv_source_df must have the same row count as X.")
    if resolved_split_date_column and resolved_split_date_column not in source_df.columns:
        raise ValueError(f"split_date column '{resolved_split_date_column}' is missing from split source dataframe.")
    if resolved_split_stratify_column and resolved_split_stratify_column not in source_df.columns:
        raise ValueError(
            f"split_stratify column '{resolved_split_stratify_column}' is missing from split source dataframe."
        )

    holdout_group_columns = list(dict.fromkeys(resolved_split_group_columns + list(fallback_cv_group_columns or [])))
    train_idx, test_idx, test_splitter_name, test_meta = _materialize_holdout(
        X=X,
        y=y,
        task=task,
        test_size=test_size,
        test_size_rows=test_size_rows,
        random_state=random_state,
        split_source_df=source_df,
        split_date_column=resolved_split_date_column,
        split_stratify_column=resolved_split_stratify_column,
        split_group_columns_for_stratify=holdout_group_columns,
        test_split_strategy=test_split_strategy,
        warnings=split_warnings,
        test_splitter=test_splitter,
    )
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    cv_source_train = source_df.iloc[train_idx]

    explicit_cv_group_columns = list(resolved_split_group_columns)
    fallback_group_columns = list(fallback_cv_group_columns or [])

    if resolved_split_date_column and resolved_split_date_column not in cv_source_train.columns:
        raise ValueError(f"split_date column '{resolved_split_date_column}' is missing from split source dataframe.")
    if resolved_split_stratify_column and resolved_split_stratify_column not in cv_source_train.columns:
        raise ValueError(
            f"split_stratify column '{resolved_split_stratify_column}' is missing from split source dataframe."
        )
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
        cv_date_column=resolved_split_date_column,
        cv_stratify_column=resolved_split_stratify_column,
        cv_source_train=cv_source_train,
        cv_fold_size_rows=cv_fold_size_rows,
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
            "strategy": "custom" if test_splitter is not None else test_split_strategy,
            "date_column": resolved_split_date_column,
            "stratify_column": resolved_split_stratify_column,
            "stratify_source": test_meta.get("stratify_source"),
            "tied_groups": dict(test_meta.get("tied_groups", {})),
            "fallback_applied": bool(test_meta.get("fallback_applied", False)),
            "fallback_reason": test_meta.get("fallback_reason"),
        },
        "cv": {
            "splitter": cv_splitter_name,
            "n_splits": int(len(cv_splits)),
            "uses_groups": bool(cv_meta.get("uses_groups", False)),
            "group_columns": list(cv_meta.get("group_columns", explicit_cv_group_columns + fallback_group_columns)),
            "date_column": cv_meta.get("date_column", resolved_split_date_column),
            "stratify_column": cv_meta.get("stratify_column", resolved_split_stratify_column),
            "strategy_requested": cv_meta.get("strategy_requested"),
            "strategy_used": cv_meta.get("strategy_used"),
            "fallback_applied": bool(cv_meta.get("fallback_applied", False)),
            "fallback_reason": cv_meta.get("fallback_reason"),
            "stratify_source": cv_meta.get("stratify_source"),
            "tied_groups": dict(cv_meta.get("tied_groups", {})),
            "inferred_from_columns": bool(cv_meta.get("inferred_from_columns", False)),
            "fold_size_rows": cv_meta.get("fold_size_rows"),
            "n_splits_derived_from_fold_size": bool(cv_meta.get("n_splits_derived_from_fold_size", False)),
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
