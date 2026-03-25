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
