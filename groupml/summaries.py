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


def _best_by_average(leaderboard: pd.DataFrame, column: str) -> dict[str, Any]:
    if leaderboard.empty or column not in leaderboard.columns:
        return {}
    stats = (
        leaderboard.groupby(column, dropna=False)
        .agg(
            avg_cv_mean=("cv_mean", "mean"),
            avg_test_score=("test_score", "mean"),
            best_cv_mean=("cv_mean", "max"),
            runs=(column, "size"),
        )
        .reset_index()
        .sort_values(by=["avg_cv_mean", "best_cv_mean", "avg_test_score"], ascending=False)
    )
    if stats.empty:
        return {}
    row = stats.iloc[0].to_dict()
    row["value"] = row.get(column)
    return row


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

    if result.baseline_experiment:
        baseline_experiment = dict(result.baseline_experiment)
    else:
        baseline_rows = leaderboard[leaderboard["mode"] == "full"]
        if baseline_rows.empty:
            baseline_experiment = leaderboard.iloc[0].to_dict()
        else:
            baseline_experiment = baseline_rows.sort_values(
                by=["cv_mean", "test_score"], ascending=False
            ).iloc[0].to_dict()

    cv_improvement_vs_baseline = _safe_float(best_experiment.get("cv_mean")) - _safe_float(
        baseline_experiment.get("cv_mean")
    )

    top_rows = leaderboard.head(max(1, top_n))
    top_experiments = top_rows.to_dict(orient="records")

    return {
        "best_experiment": best_experiment,
        "baseline_experiment": baseline_experiment,
        "cv_improvement_vs_baseline": cv_improvement_vs_baseline,
        "best_grouping_method": _best_by_average(leaderboard, "mode"),
        "best_model": _best_by_average(leaderboard, "model"),
        "best_feature_selector": _best_by_average(leaderboard, "selector"),
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
    if not leaderboard.empty:
        tables["by_mode"] = (
            leaderboard.groupby("mode", dropna=False)
            .agg(avg_cv_mean=("cv_mean", "mean"), avg_test_score=("test_score", "mean"), runs=("mode", "size"))
            .sort_values(by=["avg_cv_mean", "avg_test_score"], ascending=False)
            .reset_index()
        )
        tables["by_model"] = (
            leaderboard.groupby("model", dropna=False)
            .agg(avg_cv_mean=("cv_mean", "mean"), avg_test_score=("test_score", "mean"), runs=("model", "size"))
            .sort_values(by=["avg_cv_mean", "avg_test_score"], ascending=False)
            .reset_index()
        )
        tables["by_selector"] = (
            leaderboard.groupby("selector", dropna=False)
            .agg(
                avg_cv_mean=("cv_mean", "mean"),
                avg_test_score=("test_score", "mean"),
                runs=("selector", "size"),
            )
            .sort_values(by=["avg_cv_mean", "avg_test_score"], ascending=False)
            .reset_index()
        )
        best_by_mode = (
            leaderboard.sort_values(by=["cv_mean", "test_score"], ascending=False)
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
        best_mode_map = {str(row["mode"]): row for row in best_by_mode.to_dict(orient="records")}
        best_group_aware = (
            leaderboard[leaderboard["mode"].isin(["group_split", "group_permutations", "rule_split"])]
            .sort_values(by=["cv_mean", "test_score"], ascending=False)
            .head(1)
        )
        method_to_config: dict[str, dict[str, Any]] = {}
        if "group_as_features" in best_mode_map:
            method_to_config["onehot"] = best_mode_map["group_as_features"]
        if "full" in best_mode_map:
            method_to_config["full"] = best_mode_map["full"]
        if not best_group_aware.empty:
            method_to_config["group_aware"] = best_group_aware.iloc[0].to_dict()
    else:
        method_to_config = {}

    raw_report = result.raw_report if isinstance(result.raw_report, pd.DataFrame) else pd.DataFrame()
    if not raw_report.empty:
        predicted_cols = [c for c in raw_report.columns if c.startswith("predicted_")]
        group_cols = list((result.split_info or {}).get("cv", {}).get("group_columns", []) or [])
        group_cols = [c for c in group_cols if c in raw_report.columns]
        if not group_cols:
            excluded = {
                "row_index",
                "split_assignment",
                "actual",
                "predicted",
                "error",
                "prediction_method",
            }
            excluded.update({c for c in raw_report.columns if c.startswith("predicted_") or c.startswith("error_")})
            fallback_candidates = [
                c
                for c in raw_report.columns
                if c not in excluded and not str(c).startswith("__groupml_auto_stratify__")
            ]
            group_cols = [
                c
                for c in fallback_candidates
                if not pd.api.types.is_numeric_dtype(raw_report[c]) and raw_report[c].nunique(dropna=False) > 1
            ][:1]
        if predicted_cols and group_cols:
            eval_mask = raw_report.get("split_assignment", pd.Series([""] * len(raw_report))).astype(str) != "train"
            group_key = raw_report[group_cols].apply(
                lambda row: "||".join("" if pd.isna(value) else str(value) for value in row.tolist()),
                axis=1,
            )
            rows: list[dict[str, Any]] = []
            for pred_col in predicted_cols:
                method = pred_col.replace("predicted_", "", 1)
                subset = raw_report.loc[eval_mask, ["actual", pred_col]].copy()
                subset = subset.dropna(subset=[pred_col])
                if subset.empty:
                    continue
                for key in sorted(group_key[subset.index].unique().tolist()):
                    idx = group_key[subset.index] == key
                    chunk = subset.loc[idx]
                    n = len(chunk)
                    actual = chunk["actual"]
                    pred = chunk[pred_col]
                    actual_num = pd.to_numeric(actual, errors="coerce")
                    pred_num = pd.to_numeric(pred, errors="coerce")
                    numeric = actual_num.notna().all() and pred_num.notna().all()
                    row: dict[str, Any] = {
                        "method": method,
                        "group_key": key,
                        "n_eval_rows": int(n),
                    }
                    config_row = method_to_config.get(method)
                    if config_row:
                        row["experiment_name"] = config_row.get("experiment_name")
                        row["model"] = config_row.get("model")
                        row["selector"] = config_row.get("selector")
                    if numeric:
                        err = pred_num - actual_num
                        row["rmse"] = float(np.sqrt(np.mean(np.square(err))))
                        row["mae"] = float(np.mean(np.abs(err)))
                    else:
                        row["accuracy"] = float((pred == actual).mean())
                    rows.append(row)
            if rows:
                tables["group_performance"] = pd.DataFrame(rows).sort_values(
                    by=["method", "group_key"]
                ).reset_index(drop=True)

    if payload["warnings"]:
        tables["warnings"] = pd.DataFrame({"warning": payload["warnings"]})

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

    lines.append(f"Recommendation: {payload.get('recommendation', '')}")

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
