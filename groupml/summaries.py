"""Summary builders for GroupML results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .result import GroupMLResult


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: Any, digits: int = 5) -> str:
    number = _safe_float(value)
    if np.isnan(number):
        return "nan"
    return f"{number:.{digits}f}"


def _cv_prefers_lower(result: "GroupMLResult") -> bool:
    scorer = str((result.split_info or {}).get("scorer", "")).strip().lower()
    return scorer in {"rmse", "neg_root_mean_squared_error", "root_mean_squared_error"}


def _best_by_average(leaderboard: pd.DataFrame, column: str, prefers_lower: bool) -> dict[str, Any]:
    if leaderboard.empty or column not in leaderboard.columns:
        return {}
    best_agg = "min" if prefers_lower else "max"
    stats = (
        leaderboard.groupby(column, dropna=False)
        .agg(
            avg_cv_mean=("cv_mean", "mean"),
            avg_test_score=("test_score", "mean"),
            best_cv_mean=("cv_mean", best_agg),
            runs=(column, "size"),
        )
        .reset_index()
        .sort_values(by=["avg_cv_mean", "best_cv_mean"], ascending=prefers_lower)
    )
    if stats.empty:
        return {}
    row = stats.iloc[0].to_dict()
    row["value"] = row.get(column)
    return row


SUMMARY_COLUMNS = [
    "section",
    "view_order",
    "scope",
    "group_column",
    "group_value",
    "method_type",
    "mode",
    "variant",
    "experiment_name",
    "model",
    "selector",
    "cv_mean",
    "test_score",
    "n_eval_rows",
    "metric_name",
    "metric_value",
    "notes",
]

MODE_TO_METHOD_TYPE = {
    "full": "no_group_awareness",
    "group_as_features": "one_hot_group_features",
    "group_split": "per_group_models",
    "group_permutations": "per_group_models_group_combos",
    "rule_split": "rule_based_split",
}

METHOD_TYPE_TO_MODE = {value: key for key, value in MODE_TO_METHOD_TYPE.items()}
METHOD_ORDER = [
    "no_group_awareness",
    "one_hot_group_features",
    "per_group_models",
    "per_group_models_group_combos",
    "rule_based_split",
]

RECOMMENDATION_COLUMNS = [
    "recommend_rank",
    "method",
    "experiment_name",
    "model",
    "selector",
    "cv_mean",
    "test_score",
    "notes",
]


def _empty_summary_table() -> pd.DataFrame:
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


def _best_by_mode(leaderboard: pd.DataFrame, prefers_lower: bool) -> dict[str, dict[str, Any]]:
    if leaderboard.empty:
        return {}
    if "mode" not in leaderboard.columns:
        return {}
    ordered = leaderboard.sort_values(by=["cv_mean"], ascending=prefers_lower)
    rows = ordered.drop_duplicates(subset=["mode"], keep="first")
    return {str(row["mode"]): row for row in rows.to_dict(orient="records")}


def _resolve_group_columns(
    result: "GroupMLResult",
    raw_report: pd.DataFrame,
    filtered_eval_rows: pd.DataFrame,
) -> list[str]:
    del raw_report
    split_info = result.split_info if isinstance(result.split_info, dict) else {}
    configured_cols = list(split_info.get("configured_group_columns", []) or [])
    profile_group_cols = list((split_info.get("group_profile", {}) or {}).get("group_columns", []) or [])
    explicit_group_cols = list(dict.fromkeys([str(c) for c in configured_cols + profile_group_cols if str(c)]))
    if explicit_group_cols:
        present_cols = [c for c in explicit_group_cols if c in filtered_eval_rows.columns]
        # Per-group summary is only valid for explicitly configured group columns.
        return [c for c in present_cols if filtered_eval_rows[c].notna().any()]
    return []


def _build_full_dataset_rows(
    leaderboard: pd.DataFrame,
    best_mode_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_type in METHOD_ORDER:
        mode = METHOD_TYPE_TO_MODE[method_type]
        best = best_mode_map.get(mode, {})
        if not best:
            continue
        rows.append(
            {
                "section": "full_dataset_best",
                "view_order": 1,
                "scope": "dataset",
                "group_column": "",
                "group_value": "ALL",
                "method_type": method_type,
                "mode": mode,
                "variant": best.get("variant", ""),
                "experiment_name": best.get("experiment_name", ""),
                "model": best.get("model", ""),
                "selector": best.get("selector", ""),
                "cv_mean": _safe_float(best.get("cv_mean")),
                "test_score": _safe_float(best.get("test_score")),
                "n_eval_rows": np.nan,
                "metric_name": "best_config",
                "metric_value": _safe_float(best.get("cv_mean")),
                "notes": "",
            }
        )
    return rows


def _build_per_group_rows(
    result: "GroupMLResult",
    raw_report: pd.DataFrame,
    best_mode_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    def _compute_metric(frame: pd.DataFrame, pred_col: str, is_numeric: bool) -> float:
        if frame.empty:
            return np.nan
        if is_numeric:
            actual_values = pd.to_numeric(frame["actual"], errors="coerce")
            pred_values = pd.to_numeric(frame[pred_col], errors="coerce")
            mask = actual_values.notna() & pred_values.notna()
            if not mask.any():
                return np.nan
            err = pred_values[mask] - actual_values[mask]
            return float(np.sqrt(np.mean(np.square(err))))
        actual = frame["actual"]
        predicted = frame[pred_col]
        mask = actual.notna() & predicted.notna()
        if not mask.any():
            return np.nan
        return float((predicted[mask] == actual[mask]).mean())

    if raw_report.empty:
        return []
    predicted_cols = [
        c for c in raw_report.columns if c.startswith("predicted_") and c != "predicted"
    ]
    if not predicted_cols:
        return []
    eval_mask = raw_report.get("split_assignment", pd.Series([""] * len(raw_report))).astype(str) != "train"
    eval_rows = raw_report.loc[eval_mask].copy()
    eval_rows = eval_rows[eval_rows["actual"].notna()]
    eval_rows = eval_rows[eval_rows[predicted_cols].notna().any(axis=1)]

    group_cols = _resolve_group_columns(result, raw_report, eval_rows)
    if not group_cols:
        return []
    output: list[dict[str, Any]] = []
    for pred_col in sorted(predicted_cols):
        method_type = pred_col.replace("predicted_", "", 1)
        mapped_mode = METHOD_TYPE_TO_MODE.get(method_type, "")
        best_row = best_mode_map.get(mapped_mode, {})

        subset = eval_rows.loc[:, ["split_assignment", "actual", pred_col] + group_cols].copy()
        subset = subset.dropna(subset=[pred_col, "actual"])
        if subset.empty:
            continue

        actual_num = pd.to_numeric(subset["actual"], errors="coerce")
        pred_num = pd.to_numeric(subset[pred_col], errors="coerce")
        is_numeric = actual_num.notna().all() and pred_num.notna().all()
        metric_name = "rmse" if is_numeric else "accuracy"
        overall_n_eval = int(len(subset))
        if overall_n_eval > 0:
            if is_numeric:
                overall_actual = pd.to_numeric(subset["actual"], errors="coerce")
                overall_pred = pd.to_numeric(subset[pred_col], errors="coerce")
                overall_err = overall_pred - overall_actual
                overall_metric = float(np.sqrt(np.mean(np.square(overall_err))))
            else:
                overall_metric = float((subset[pred_col] == subset["actual"]).mean())
            output.append(
                {
                    "section": "overall_method_comparison",
                    "view_order": 2,
                    "scope": "dataset",
                    "group_column": "",
                    "group_value": "ALL",
                    "method_type": method_type,
                    "mode": mapped_mode,
                    "variant": best_row.get("variant", ""),
                    "experiment_name": best_row.get("experiment_name", ""),
                    "model": best_row.get("model", ""),
                    "selector": best_row.get("selector", ""),
                    "cv_mean": _safe_float(best_row.get("cv_mean")),
                    "test_score": _safe_float(best_row.get("test_score")),
                    "n_eval_rows": overall_n_eval,
                    "metric_name": metric_name,
                    "metric_value": overall_metric,
                    "notes": "",
                }
            )

        for group_col in group_cols:
            grouped = subset.dropna(subset=[group_col])
            for group_value, group_df in grouped.groupby(group_col, dropna=True):
                n_eval = int(len(group_df))
                if n_eval == 0:
                    continue
                if is_numeric:
                    actual_values = pd.to_numeric(group_df["actual"], errors="coerce")
                    pred_values = pd.to_numeric(group_df[pred_col], errors="coerce")
                    err = pred_values - actual_values
                    metric_value = float(np.sqrt(np.mean(np.square(err))))
                else:
                    metric_value = float((group_df[pred_col] == group_df["actual"]).mean())
                group_value_str = str(group_value)
                cv_rows = group_df[group_df["split_assignment"].astype(str).str.startswith("cv_")]
                test_rows = group_df[group_df["split_assignment"].astype(str).str.contains("test")]
                output.append(
                    {
                        "section": "per_group_comparison",
                        "view_order": 2,
                        "scope": "group",
                        "group_column": group_col,
                        "group_value": group_value_str,
                        "method_type": method_type,
                        "mode": mapped_mode,
                        "variant": best_row.get("variant", ""),
                        "experiment_name": best_row.get("experiment_name", ""),
                        "model": best_row.get("model", ""),
                        "selector": best_row.get("selector", ""),
                        "cv_mean": _compute_metric(cv_rows, pred_col, is_numeric),
                        "test_score": _compute_metric(test_rows, pred_col, is_numeric),
                        "n_eval_rows": n_eval,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "notes": "Per-group metric_value uses all eval rows; cv_mean uses CV rows and test_score uses holdout rows for this group.",
                    }
                )
    return output


def _build_group_split_comparison_rows(
    leaderboard: pd.DataFrame,
    prefers_lower: bool,
) -> list[dict[str, Any]]:
    if leaderboard.empty:
        return []
    if "mode" not in leaderboard.columns or "variant" not in leaderboard.columns:
        return []
    split_rows = leaderboard[leaderboard["mode"].isin(["group_split", "group_permutations"])].copy()
    if split_rows.empty:
        return []

    output: list[dict[str, Any]] = []
    for (mode, variant), variant_rows in split_rows.groupby(["mode", "variant"], dropna=False):
        variant_df = variant_rows.copy()
        optimized_mask = (variant_df["model"] == "per_group_best") & (
            variant_df["selector"] == "per_group_best"
        )
        optimized_rows = variant_df.loc[optimized_mask]
        shared_rows = variant_df.loc[~optimized_mask]
        if optimized_rows.empty or shared_rows.empty:
            continue

        optimized_row = optimized_rows.sort_values(by=["cv_mean"], ascending=prefers_lower).iloc[0]
        shared_row = shared_rows.sort_values(by=["cv_mean"], ascending=prefers_lower).iloc[0]
        method_type = MODE_TO_METHOD_TYPE.get(str(mode), str(mode))
        variant_value = str(variant)

        output.append(
            {
                "section": "group_split_combined_comparison",
                "view_order": 2,
                "scope": "dataset",
                "group_column": "",
                "group_value": "ALL",
                "method_type": method_type,
                "mode": str(mode),
                "variant": variant_value,
                "experiment_name": str(optimized_row.get("experiment_name", "")),
                "model": str(optimized_row.get("model", "")),
                "selector": str(optimized_row.get("selector", "")),
                "cv_mean": _safe_float(optimized_row.get("cv_mean")),
                "test_score": _safe_float(optimized_row.get("test_score")),
                "n_eval_rows": np.nan,
                "metric_name": "optimized_per_group",
                "metric_value": _safe_float(optimized_row.get("cv_mean")),
                "notes": "Optimized model+selector per group.",
            }
        )
        output.append(
            {
                "section": "group_split_combined_comparison",
                "view_order": 2,
                "scope": "dataset",
                "group_column": "",
                "group_value": "ALL",
                "method_type": method_type,
                "mode": str(mode),
                "variant": variant_value,
                "experiment_name": str(shared_row.get("experiment_name", "")),
                "model": str(shared_row.get("model", "")),
                "selector": str(shared_row.get("selector", "")),
                "cv_mean": _safe_float(shared_row.get("cv_mean")),
                "test_score": _safe_float(shared_row.get("test_score")),
                "n_eval_rows": np.nan,
                "metric_name": "best_shared_single_config",
                "metric_value": _safe_float(shared_row.get("cv_mean")),
                "notes": "Single shared model+selector for all groups (best by CV).",
            }
        )
    return output


def _build_recommendation_table(
    leaderboard: pd.DataFrame,
    result: "GroupMLResult",
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not leaderboard.empty:
        top_rows = leaderboard.head(max(1, int(top_n)))
        for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
            mode = str(row.get("mode", ""))
            rows.append(
                {
                    "recommend_rank": rank,
                    "method": MODE_TO_METHOD_TYPE.get(mode, mode),
                    "experiment_name": row.get("experiment_name", ""),
                    "model": row.get("model", ""),
                    "selector": row.get("selector", ""),
                    "cv_mean": _safe_float(row.get("cv_mean")),
                    "test_score": _safe_float(row.get("test_score")),
                    "notes": str(result.recommendation) if rank == 1 else "",
                }
            )
    if not rows:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def build_summary_payload(result: "GroupMLResult", top_n: int = 5) -> dict[str, Any]:
    """Build a structured summary payload from a GroupML result."""
    leaderboard = result.leaderboard.copy()
    if leaderboard.empty:
        return {
            "best_experiment": {},
            "baseline_experiment": {},
            "cv_improvement_vs_baseline": float("nan"),
            "best_grouping_method": {},
            "best_model": {},
            "best_feature_selector": {},
            "top_experiments": [],
            "recommendation": result.recommendation,
            "warnings": list(result.warnings),
            "split_info": dict(result.split_info),
        }

    best_experiment = dict(result.best_experiment or leaderboard.iloc[0].to_dict())
    prefers_lower = _cv_prefers_lower(result)

    if result.baseline_experiment:
        baseline_experiment = dict(result.baseline_experiment)
    else:
        baseline_rows = leaderboard[leaderboard["mode"] == "full"]
        if baseline_rows.empty:
            baseline_experiment = leaderboard.iloc[0].to_dict()
        else:
            baseline_experiment = baseline_rows.sort_values(
                by=["cv_mean"], ascending=prefers_lower
            ).iloc[0].to_dict()

    best_cv = _safe_float(best_experiment.get("cv_mean"))
    baseline_cv = _safe_float(baseline_experiment.get("cv_mean"))
    cv_improvement_vs_baseline = (baseline_cv - best_cv) if prefers_lower else (best_cv - baseline_cv)

    top_rows = leaderboard.head(max(1, top_n))
    top_experiments = top_rows.to_dict(orient="records")
    run_datetime = str(best_experiment.get("run_datetime") or result.split_info.get("run_datetime", ""))

    return {
        "best_experiment": best_experiment,
        "baseline_experiment": baseline_experiment,
        "run_datetime": run_datetime,
        "cv_prefers_lower": prefers_lower,
        "cv_improvement_vs_baseline": cv_improvement_vs_baseline,
        "best_grouping_method": _best_by_average(leaderboard, "mode", prefers_lower=prefers_lower),
        "best_model": _best_by_average(leaderboard, "model", prefers_lower=prefers_lower),
        "best_feature_selector": _best_by_average(leaderboard, "selector", prefers_lower=prefers_lower),
        "top_experiments": top_experiments,
        "recommendation": result.recommendation,
        "warnings": list(result.warnings),
        "split_info": dict(result.split_info),
    }


def build_summary_tables(result: "GroupMLResult", top_n: int = 10) -> dict[str, pd.DataFrame]:
    """Build export-friendly summary tables."""
    payload = build_summary_payload(result, top_n=top_n)
    best = payload["best_experiment"]
    baseline = payload["baseline_experiment"]
    grouping = payload["best_grouping_method"]
    model = payload["best_model"]
    selector = payload["best_feature_selector"]

    overview_rows = [
        {"metric": "run_datetime", "value": payload.get("run_datetime", "")},
        {"metric": "best_experiment", "value": best.get("experiment_name", "")},
        {"metric": "best_mode", "value": best.get("mode", "")},
        {"metric": "best_model", "value": best.get("model", "")},
        {"metric": "best_selector", "value": best.get("selector", "")},
        {"metric": "best_cv_mean", "value": _safe_float(best.get("cv_mean"))},
        {"metric": "best_test_score", "value": _safe_float(best.get("test_score"))},
        {"metric": "baseline_experiment", "value": baseline.get("experiment_name", "")},
        {
            "metric": "cv_improvement_vs_baseline",
            "value": _safe_float(payload.get("cv_improvement_vs_baseline")),
        },
        {"metric": "best_grouping_method", "value": grouping.get("value", "")},
        {"metric": "best_grouping_avg_cv", "value": _safe_float(grouping.get("avg_cv_mean"))},
        {"metric": "best_model_overall", "value": model.get("value", "")},
        {"metric": "best_model_avg_cv", "value": _safe_float(model.get("avg_cv_mean"))},
        {"metric": "best_selector_overall", "value": selector.get("value", "")},
        {"metric": "best_selector_avg_cv", "value": _safe_float(selector.get("avg_cv_mean"))},
        {"metric": "recommendation", "value": payload.get("recommendation", "")},
        {"metric": "warning_count", "value": len(payload.get("warnings", []))},
    ]

    tables = {
        "overview": pd.DataFrame(overview_rows),
        "top_experiments": pd.DataFrame(payload["top_experiments"]),
    }

    leaderboard = result.leaderboard
    prefers_lower = bool(payload.get("cv_prefers_lower", False))
    best_mode_map = _best_by_mode(leaderboard, prefers_lower=prefers_lower)
    if not leaderboard.empty and "mode" in leaderboard.columns:
        tables["by_mode"] = (
            leaderboard.groupby("mode", dropna=False)
            .agg(avg_cv_mean=("cv_mean", "mean"), avg_test_score=("test_score", "mean"), runs=("mode", "size"))
            .sort_values(by=["avg_cv_mean"], ascending=prefers_lower)
            .reset_index()
        )
        if "model" in leaderboard.columns:
            tables["by_model"] = (
                leaderboard.groupby("model", dropna=False)
                .agg(avg_cv_mean=("cv_mean", "mean"), avg_test_score=("test_score", "mean"), runs=("model", "size"))
                .sort_values(by=["avg_cv_mean"], ascending=prefers_lower)
                .reset_index()
            )
        if "selector" in leaderboard.columns:
            tables["by_selector"] = (
                leaderboard.groupby("selector", dropna=False)
                .agg(
                    avg_cv_mean=("cv_mean", "mean"),
                    avg_test_score=("test_score", "mean"),
                    runs=("selector", "size"),
                )
                .sort_values(by=["avg_cv_mean"], ascending=prefers_lower)
                .reset_index()
            )
        best_by_mode = (
            leaderboard.sort_values(by=["cv_mean"], ascending=prefers_lower)
            .drop_duplicates(subset=["mode"], keep="first")
            .loc[
                :,
                [
                    c
                    for c in ["mode", "experiment_name", "variant", "model", "selector", "cv_mean", "test_score"]
                    if c in leaderboard.columns
                ],
            ]
            .reset_index(drop=True)
        )
        tables["best_method_configs"] = best_by_mode

    raw_report = result.raw_report if isinstance(result.raw_report, pd.DataFrame) else pd.DataFrame()
    summary_rows = _build_full_dataset_rows(leaderboard=leaderboard, best_mode_map=best_mode_map)
    summary_rows.extend(
        _build_per_group_rows(result=result, raw_report=raw_report, best_mode_map=best_mode_map)
    )
    summary_rows.extend(
        _build_group_split_comparison_rows(leaderboard=leaderboard, prefers_lower=prefers_lower)
    )
    if summary_rows:
        summary_table = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
        summary_table = summary_table.sort_values(
            by=["view_order", "section", "group_column", "group_value", "method_type"]
        ).reset_index(drop=True)
    else:
        summary_table = _empty_summary_table()
    tables["summary"] = summary_table
    tables["recommendations"] = _build_recommendation_table(leaderboard=leaderboard, result=result, top_n=top_n)

    if not raw_report.empty:
        per_group = summary_table[summary_table["section"] == "per_group_comparison"].copy()
        if not per_group.empty:
            tables["group_performance"] = (
                per_group.loc[
                    :,
                    [
                        "group_column",
                        "group_value",
                        "method_type",
                        "mode",
                        "experiment_name",
                        "model",
                        "selector",
                        "cv_mean",
                        "test_score",
                        "n_eval_rows",
                        "metric_name",
                        "metric_value",
                    ],
                ]
                .sort_values(by=["group_column", "group_value", "method_type"])
                .reset_index(drop=True)
            )

    warning_details = result.warning_details if isinstance(result.warning_details, pd.DataFrame) else pd.DataFrame()
    if not warning_details.empty:
        tables["warnings"] = warning_details.copy()
    elif payload["warnings"]:
        tables["warnings"] = pd.DataFrame(
            {
                "warning_datetime": [str(result.split_info.get("run_datetime", ""))] * len(payload["warnings"]),
                "run_datetime": [str(result.split_info.get("run_datetime", ""))] * len(payload["warnings"]),
                "run_experiment": [""] * len(payload["warnings"]),
                "warning": payload["warnings"],
            }
        )

    return tables


def summary_text(result: "GroupMLResult", top_n: int = 5) -> str:
    """Render a practical text summary for humans/CLI."""
    payload = build_summary_payload(result, top_n=top_n)
    best = payload["best_experiment"]
    baseline = payload["baseline_experiment"]
    grouping = payload["best_grouping_method"]
    model = payload["best_model"]
    selector = payload["best_feature_selector"]

    lines: list[str] = []
    lines.append("Result summary")

    if best:
        lines.append(
            "Best experiment (CV-first): "
            f"{best.get('experiment_name')} | mode={best.get('mode')} | "
            f"model={best.get('model')} | selector={best.get('selector')} | "
            f"cv_mean={_fmt(best.get('cv_mean'))} | test_score={_fmt(best.get('test_score'))}"
        )

    if baseline:
        lines.append(
            "Baseline reference: "
            f"{baseline.get('experiment_name')} | cv_mean={_fmt(baseline.get('cv_mean'))} | "
            f"test_score={_fmt(baseline.get('test_score'))}"
        )

    lines.append(
        "Test score is holdout comparison only; ranking and recommendation are based on CV mean."
    )

    improvement = payload.get("cv_improvement_vs_baseline")
    if improvement is not None:
        lines.append(f"CV improvement vs baseline: {_fmt(improvement)}")

    if grouping:
        lines.append(
            "Best overall grouping method (average CV): "
            f"{grouping.get('value')} | avg_cv_mean={_fmt(grouping.get('avg_cv_mean'))} | "
            f"avg_test_score={_fmt(grouping.get('avg_test_score'))}"
        )
    if model:
        lines.append(
            "Best overall model (average CV): "
            f"{model.get('value')} | avg_cv_mean={_fmt(model.get('avg_cv_mean'))} | "
            f"avg_test_score={_fmt(model.get('avg_test_score'))}"
        )
    if selector:
        lines.append(
            "Best overall feature selector (average CV): "
            f"{selector.get('value')} | avg_cv_mean={_fmt(selector.get('avg_cv_mean'))} | "
            f"avg_test_score={_fmt(selector.get('avg_test_score'))}"
        )

    top_experiments = payload.get("top_experiments", [])
    if top_experiments:
        lines.append(f"Top {len(top_experiments)} experiments:")
        for idx, row in enumerate(top_experiments, start=1):
            lines.append(
                f"{idx}. {row.get('experiment_name')} | model={row.get('model')} | "
                f"selector={row.get('selector')} | cv_mean={_fmt(row.get('cv_mean'))} | "
                f"test_score={_fmt(row.get('test_score'))}"
            )

    leaderboard = result.leaderboard if isinstance(result.leaderboard, pd.DataFrame) else pd.DataFrame()
    if not leaderboard.empty and "mode" in leaderboard.columns and "variant" in leaderboard.columns:
        split_rows = leaderboard[leaderboard["mode"].isin(["group_split", "group_permutations"])].copy()
        if not split_rows.empty:
            lines.append("Combined per-group comparison:")
            prefers_lower = bool(payload.get("cv_prefers_lower", False))
            for (mode, variant), variant_rows in split_rows.groupby(["mode", "variant"], dropna=False):
                optimized_mask = (variant_rows["model"] == "per_group_best") & (
                    variant_rows["selector"] == "per_group_best"
                )
                optimized_rows = variant_rows.loc[optimized_mask]
                shared_rows = variant_rows.loc[~optimized_mask]
                if optimized_rows.empty or shared_rows.empty:
                    continue
                optimized_row = optimized_rows.sort_values(by=["cv_mean"], ascending=prefers_lower).iloc[0]
                shared_row = shared_rows.sort_values(by=["cv_mean"], ascending=prefers_lower).iloc[0]
                lines.append(
                    f"- {mode}:{variant} | optimized_per_group cv_mean={_fmt(optimized_row.get('cv_mean'))} "
                    f"test_score={_fmt(optimized_row.get('test_score'))}"
                )
                lines.append(
                    f"- {mode}:{variant} | best_shared_single_config "
                    f"{shared_row.get('model')}/{shared_row.get('selector')} "
                    f"cv_mean={_fmt(shared_row.get('cv_mean'))} test_score={_fmt(shared_row.get('test_score'))}"
                )

    lines.append(f"Recommendation: {payload.get('recommendation', '')}")
    lines.append(
        f"Recommendation report exports the recommended setup plus top {max(1, int(top_n))} CV-ranked options overall."
    )

    warnings = payload.get("warnings", [])
    if warnings:
        lines.append(f"Warnings ({len(warnings)}):")
        for item in warnings:
            lines.append(f"- {item}")

    split_info = payload.get("split_info", {})
    if split_info:
        test_info = split_info.get("test", {})
        cv_info = split_info.get("cv", {})
        cv_extra: list[str] = []
        if cv_info.get("fold_size_rows") is not None:
            cv_extra.append(f"fold_size_rows={cv_info.get('fold_size_rows')}")
        if cv_info.get("n_splits_derived_from_fold_size"):
            cv_extra.append("n_splits_derived_from_fold_size=True")
        extra_text = f", {', '.join(cv_extra)}" if cv_extra else ""
        lines.append(
            "Splits: "
            f"test={test_info.get('splitter')} (train={test_info.get('train_size')}, test={test_info.get('test_size')}), "
            f"cv={cv_info.get('splitter')} (folds={cv_info.get('n_splits')}{extra_text})"
        )

    return "\n".join(lines)
