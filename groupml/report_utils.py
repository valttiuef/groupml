"""Raw-report building helpers for GroupML results."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone


def build_single_experiment_prediction_report(
    *,
    data: pd.DataFrame,
    split_plan: Any,
    task: str,
    models: dict[str, Any],
    selectors: dict[str, Any],
    feature_cols: list[str],
    group_cols: list[str],
    parsed_rules: list[Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_full: pd.Series,
    experiment_row: dict[str, Any],
    warnings: list[str],
    build_best_estimator: Callable[..., Any],
    warning_filter: Callable[[], Any],
) -> pd.DataFrame:
    estimator = build_best_estimator(
        best_row=experiment_row,
        task=task,
        models=models,
        selectors=selectors,
        feature_cols=feature_cols,
        group_cols=group_cols,
        parsed_rules=parsed_rules,
        X_train=X_train,
    )
    n_rows = len(data)
    per_row_predictions: list[list[Any]] = [[] for _ in range(n_rows)]
    split_labels: list[set[str]] = [set() for _ in range(n_rows)]

    for fold_idx, (train_idx, val_idx) in enumerate(split_plan.cv_splits, start=1):
        fold_estimator = clone(estimator)
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        try:
            with warning_filter():
                fold_estimator.fit(X_tr, y_tr)
                val_pred = fold_estimator.predict(X_val)
        except Exception as exc:
            warnings.append(
                f"Raw report CV prediction failure on fold {fold_idx} for "
                f"{experiment_row.get('experiment_name')} ({experiment_row.get('model')}, "
                f"{experiment_row.get('selector')}): {exc}"
            )
            continue
        global_val_idx = split_plan.train_indices[val_idx]
        for idx, pred in zip(global_val_idx, val_pred):
            per_row_predictions[int(idx)].append(pred)
            split_labels[int(idx)].add(f"cv_{fold_idx}")

    fitted = clone(estimator)
    X_test = data[feature_cols].iloc[split_plan.test_indices]
    try:
        with warning_filter():
            fitted.fit(X_train, y_train)
            test_pred = fitted.predict(X_test)
        for idx, pred in zip(split_plan.test_indices, test_pred):
            per_row_predictions[int(idx)].append(pred)
            split_labels[int(idx)].add("test")
    except Exception as exc:
        warnings.append(
            "Raw report test prediction failure for "
            f"{experiment_row.get('experiment_name')} ({experiment_row.get('model')}, "
            f"{experiment_row.get('selector')}): {exc}"
        )

    prediction: list[Any] = [np.nan] * n_rows
    for row_idx, values in enumerate(per_row_predictions):
        if not values:
            continue
        if task == "regression":
            prediction[row_idx] = float(np.mean(np.asarray(values, dtype=float)))
        else:
            counts = pd.Series(values).value_counts(dropna=False)
            prediction[row_idx] = counts.index[0]

    split_assignment: list[str] = []
    for row_idx in range(n_rows):
        labels = split_labels[row_idx]
        if labels:
            split_assignment.append("|".join(sorted(labels)))
        else:
            split_assignment.append("train")

    report = pd.DataFrame(
        {
            "row_index": np.arange(n_rows, dtype=int),
            "split_assignment": split_assignment,
            "actual": y_full.to_numpy(),
            "predicted": prediction,
        }
    )
    if task == "regression":
        report["error"] = report["predicted"] - report["actual"]
    else:
        report["error"] = np.where(
            report["predicted"].isna(),
            np.nan,
            (report["predicted"] != report["actual"]).astype(float),
        )
    return report


def build_raw_report(
    *,
    data: pd.DataFrame,
    split_plan: Any,
    task: str,
    models: dict[str, Any],
    selectors: dict[str, Any],
    feature_cols: list[str],
    group_cols: list[str],
    parsed_rules: list[Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_full: pd.Series,
    best_row: dict[str, Any],
    best_rows_by_method: list[dict[str, Any]] | None,
    warnings: list[str],
    split_group_columns: list[str] | None,
    split_group_column: str | None,
    split_stratify_column: str | None,
    split_date_column: str | None,
    target: str,
    raw_report_max_columns: int,
    method_token: Callable[[dict[str, Any]], str],
    comparison_label: Callable[[dict[str, Any]], str],
    build_single_report: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    rows_to_render = best_rows_by_method or [best_row]
    reports_by_method: list[tuple[str, dict[str, Any], pd.DataFrame]] = []
    for row in rows_to_render:
        token = method_token(row)
        method_report = build_single_report(
            data=data,
            split_plan=split_plan,
            task=task,
            models=models,
            selectors=selectors,
            feature_cols=feature_cols,
            group_cols=group_cols,
            parsed_rules=parsed_rules,
            X_train=X_train,
            y_train=y_train,
            y_full=y_full,
            experiment_row=row,
            warnings=warnings,
        )
        reports_by_method.append((token, row, method_report))

    if not reports_by_method:
        return pd.DataFrame()
    primary_token = method_token(best_row)
    primary_report = None
    for token, _, rep in reports_by_method:
        if token == primary_token:
            primary_report = rep
            break
    if primary_report is None:
        primary_report = reports_by_method[0][2]
        primary_token = reports_by_method[0][0]

    report = primary_report.copy()
    report = report.rename(
        columns={
            "predicted": f"predicted_{primary_token}",
            "error": f"error_{primary_token}",
        }
    )
    for token, _row, method_report in reports_by_method:
        if token == primary_token:
            continue
        report = report.join(
            method_report.set_index("row_index")[["predicted", "error"]].rename(
                columns={
                    "predicted": f"predicted_{token}",
                    "error": f"error_{token}",
                }
            ),
            on="row_index",
        )

    prediction_columns = [c for c in report.columns if c.startswith("predicted_")]
    error_columns = [c for c in report.columns if c.startswith("error_")]
    if prediction_columns:
        eval_mask = report["split_assignment"] != "train"
        invalid_mask = eval_mask & report[prediction_columns].isna().any(axis=1)
        dropped_for_comparability = int(invalid_mask.sum())
        if dropped_for_comparability:
            report.loc[invalid_mask, prediction_columns + error_columns + ["predicted", "error"]] = np.nan
            warnings.append(
                "Raw report omitted predictions for "
                f"{dropped_for_comparability} eval row(s) to keep method comparisons aligned."
            )

    label_counts: dict[str, int] = {}
    detailed_prediction_cols = list(prediction_columns)
    detailed_error_cols = list(error_columns)
    for token, row, _ in reports_by_method:
        base_label = comparison_label(row)
        count = label_counts.get(base_label, 0)
        label_counts[base_label] = count + 1
        alias = base_label if count == 0 else f"{base_label}_{count + 1}"
        pred_col = f"predicted_{token}"
        err_col = f"error_{token}"
        if pred_col in report.columns:
            report[f"predicted_{alias}"] = report[pred_col]
        if err_col in report.columns:
            report[f"error_{alias}"] = report[err_col]
    report = report.drop(columns=detailed_prediction_cols + detailed_error_cols, errors="ignore")

    aliased_prediction_columns = [c for c in report.columns if c.startswith("predicted_")]
    if aliased_prediction_columns:
        has_any_prediction = report[aliased_prediction_columns].notna().any(axis=1)
        dropped_empty = int((~has_any_prediction).sum())
        if dropped_empty:
            report = report.loc[has_any_prediction].copy()
            warnings.append(
                f"Raw report removed {dropped_empty} row(s) without any method predictions."
            )

    identity_columns = list(
        dict.fromkeys(
            list(split_group_columns or [])
            + ([split_group_column] if split_group_column else [])
            + ([split_stratify_column] if split_stratify_column else [])
            + ([split_date_column] if split_date_column else [])
            + group_cols
        )
    )
    identity_columns = [c for c in identity_columns if c in data.columns]
    sample_columns = [c for c in data.columns if c != target][: raw_report_max_columns]
    sample_columns = [c for c in sample_columns if c not in identity_columns]

    ordered_columns = (
        ["row_index"]
        + identity_columns
        + sample_columns
        + ["split_assignment", "actual"]
    )
    extra_columns = [c for c in report.columns if c not in ordered_columns]
    context_df = data[identity_columns + sample_columns].reset_index(drop=True).copy()
    context_df.insert(0, "row_index", np.arange(len(context_df), dtype=int))
    merged = report.merge(context_df, on="row_index", how="left")
    return merged.loc[:, ordered_columns + extra_columns]
