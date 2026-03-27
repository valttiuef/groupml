"""Command line interface for running groupml from terminal."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import GroupMLConfig
from .file_utils import (
    default_report_filename,
    default_summary_filename,
    export_reporting_bundle,
    export_raw_report,
    export_report,
    export_summary,
    preferred_tabular_extension,
    fit_evaluate_file,
)
from .result import GroupMLResult


MODE_LABELS = {
    "full": "no_group_awareness",
    "group_as_features": "one_hot_group_features",
    "group_split": "per_group_models",
    "group_permutations": "per_group_models_group_combos",
    "rule_split": "rule_based_split",
}


def _parse_cv_value(raw: str) -> int | str:
    value = str(raw).strip()
    if value.isdigit():
        return int(value)
    return value


def _parse_kbest_features(raw: str) -> int | str:
    value = str(raw).strip().lower()
    if value == "auto":
        return "auto"
    if value.isdigit() and int(value) >= 1:
        return int(value)
    raise ValueError("--kbest-features must be a positive integer or 'auto'.")


def _resolve_test_size(
    *,
    test_size: float,
    test_size_strategy: str,
) -> tuple[float, int | None]:
    size = float(test_size)
    strategy = str(test_size_strategy).strip().lower()

    if strategy == "rows":
        if size < 1.0 or not size.is_integer():
            raise ValueError("--test-size with strategy 'rows' must be an integer >= 1.")
        return 0.15, int(size)

    if strategy == "pct":
        if size <= 0.0:
            raise ValueError("--test-size with strategy 'pct' must be > 0.")
        if size >= 100.0:
            raise ValueError("--test-size with strategy 'pct' must be < 100.")
        if size < 1.0:
            return size, None
        return size / 100.0, None

    if strategy == "auto":
        if size <= 0.0:
            raise ValueError("--test-size with strategy 'auto' must be > 0.")
        if size < 1.0:
            return size, None
        if size.is_integer():
            return 0.15, int(size)
        if size >= 100.0:
            raise ValueError(
                "--test-size with strategy 'auto' as percentage must be < 100 when using decimals >= 1."
            )
        return size / 100.0, None

    raise ValueError("--test-size-strategy must be one of: auto, pct, rows.")


def build_parser() -> argparse.ArgumentParser:
    """Build the groupml CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="groupml",
        description="Run groupml experiments directly from a CSV/Excel dataset.",
    )
    parser.add_argument("--path", required=True, help="Path to dataset (.csv, .xls, .xlsx).")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Group column names (space-separated).",
    )
    parser.add_argument(
        "--rules",
        nargs="*",
        default=None,
        help='Rule splits, for example: --rules "Temperature < 20" "Temperature >= 20".',
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional explicit feature columns. Defaults to all columns except target.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=["full", "group_as_features", "group_split", "group_permutations", "rule_split"],
        default=None,
        help="Experiment modes to run. Defaults to all modes.",
    )
    parser.add_argument(
        "--models",
        default="default_fast",
        help="Model name or model strategy (for example: default_fast, trees, all, ridge, extra_trees).",
    )
    parser.add_argument(
        "--feature-selectors",
        "--selectors",
        dest="feature_selectors",
        default="default_fast",
        help="Feature selector name or strategy (for example: default_fast, linear_default, none, kbest_f).",
    )
    parser.add_argument(
        "--kbest-features",
        default="auto",
        help="Default feature count for kbest selectors (positive integer or 'auto').",
    )
    parser.add_argument(
        "--cv",
        type=str,
        default="5",
        help=(
            "CV setting: fold count (for example 5) or splitter name "
            "(for example groupcv, timecv, stratifycv, stratifygroupcv, stratifytimecv)."
        ),
    )
    parser.add_argument(
        "--cv-fold-size-rows",
        type=int,
        default=None,
        help=(
            "Optional CV fold size in rows. "
            "For non-time CV this resolves fold count from train_rows // size. "
            "For time CV this is used as validation block size per split."
        ),
    )
    parser.add_argument(
        "--split-group-column",
        "--cv-group-column",
        dest="split_group_column",
        default=None,
        help="Single group column used for split-aware CV behavior.",
    )
    parser.add_argument(
        "--split-date-column",
        "--cv-date-column",
        dest="split_date_column",
        default=None,
        help="Datetime column used for ordering test split and time-based CV.",
    )
    parser.add_argument(
        "--split-stratify-column",
        "--cv-stratify-column",
        dest="split_stratify_column",
        default=None,
        help="Column used for stratified split behavior.",
    )
    parser.add_argument(
        "--scorer",
        default="neg_root_mean_squared_error",
        help="sklearn scorer name (or 'rmse' alias).",
    )
    parser.add_argument("--task", choices=["auto", "regression", "classification"], default="auto")
    parser.add_argument(
        "--test-split",
        choices=["last_rows", "random"],
        default="last_rows",
        help="Holdout strategy. 'last_rows' uses tail rows (date-ordered if split_date_column is set).",
    )
    parser.add_argument(
        "--test-size-strategy",
        choices=["auto", "pct", "rows"],
        default="auto",
        help=(
            "How to interpret --test-size. "
            "'auto' (default): <1 -> fraction, integer >=1 -> rows, decimal >=1 -> percent."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Holdout size value interpreted by --test-size-strategy.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--warning-verbosity",
        choices=["quiet", "default", "all"],
        default="quiet",
        help=(
            "Library warning output level. "
            "'quiet' hides common sklearn/joblib noise, "
            "'default' keeps standard warnings but hides known sklearn/joblib noise, "
            "'all' shows all warnings."
        ),
    )
    parser.add_argument("--min-group-size", type=int, default=15)
    parser.add_argument(
        "--compare-shared-group-split",
        action="store_true",
        help=(
            "Also evaluate shared model+selector candidates in group_split/group_permutations. "
            "Disabled by default to avoid retraining all shared candidates."
        ),
    )
    parser.add_argument(
        "--group-split-tune-with-cv",
        action="store_true",
        help=(
            "Use nested CV inside per-group candidate tuning. "
            "Disabled by default for speed; outer CV still evaluates final strategy."
        ),
    )
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--scale-numeric", action="store_true", help="Scale numeric features.")
    parser.add_argument("--keep-nans", action="store_true", help="Disable base-row NaN dropping.")
    parser.add_argument(
        "--keep-static-features",
        action="store_true",
        help="Disable static feature-column removal in base preprocessing.",
    )
    parser.add_argument("--min-target", type=float, default=None, help="Optional min regression target filter.")
    parser.add_argument("--max-target", type=float, default=None, help="Optional max regression target filter.")
    parser.add_argument("--sheet-name", default=None, help="Excel sheet name when using xls/xlsx.")
    parser.add_argument("--top", type=int, default=10, help="Leaderboard rows to print.")
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Optional report output path. For tabular exports defaults to a timestamped .xlsx "
            "(or .csv fallback when openpyxl is unavailable). "
            "Supports .csv/.xls/.xlsx/.txt/.md/.json."
        ),
    )
    parser.add_argument(
        "--report-format",
        choices=["auto", "excel", "csv"],
        default="auto",
        help=(
            "Tabular report format. 'auto' prefers Excel workbook with separate sheets "
            "and falls back to CSV files if openpyxl is unavailable."
        ),
    )
    parser.add_argument(
        "--leaderboard-out",
        default=None,
        help="Optional raw leaderboard output path (.csv/.xls/.xlsx).",
    )
    parser.add_argument(
        "--raw-report-out",
        default=None,
        help="Optional per-row raw report output path (.csv/.xls/.xlsx).",
    )
    parser.add_argument(
        "--no-raw-report",
        action="store_true",
        help="Disable per-row raw report export.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI command and return process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        test_size, test_size_rows = _resolve_test_size(
            test_size=args.test_size,
            test_size_strategy=args.test_size_strategy,
        )
        kbest_features = _parse_kbest_features(args.kbest_features)
    except ValueError as exc:
        parser.error(str(exc))

    config = GroupMLConfig(
        target=args.target,
        feature_columns=args.features,
        group_columns=args.groups or [],
        rule_splits=args.rules or [],
        experiment_modes=args.modes or ["full", "group_as_features", "group_split", "group_permutations", "rule_split"],
        models=args.models,
        feature_selectors=args.feature_selectors,
        kbest_features=kbest_features,
        cv=_parse_cv_value(args.cv),
        cv_fold_size_rows=args.cv_fold_size_rows,
        split_group_column=args.split_group_column,
        split_date_column=args.split_date_column,
        split_stratify_column=args.split_stratify_column,
        scorer=args.scorer,
        task=args.task,
        test_split_strategy=args.test_split,
        test_size=test_size,
        test_size_rows=test_size_rows,
        random_state=args.random_state,
        warning_verbosity=args.warning_verbosity,
        min_group_size=args.min_group_size,
        group_split_compare_shared_candidates=args.compare_shared_group_split,
        group_split_tune_candidates_with_cv=args.group_split_tune_with_cv,
        min_improvement=args.min_improvement,
        scale_numeric=args.scale_numeric,
        dropna_base_rows=not args.keep_nans,
        drop_static_base_features=not args.keep_static_features,
        raw_report_enabled=not args.no_raw_report,
        min_target=args.min_target,
        max_target=args.max_target,
    )

    read_kwargs: dict[str, str] = {}
    if args.sheet_name is not None:
        read_kwargs["sheet_name"] = args.sheet_name

    scorer_name = str(args.scorer).strip().lower()
    rmse_display = scorer_name in {"rmse", "neg_root_mean_squared_error"}

    def _format_score(value: object, positive_rmse: bool = False) -> str:
        if value is None:
            return "nan"
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return "nan"
        if not math.isfinite(parsed):
            return "nan"
        if positive_rmse:
            parsed = abs(parsed)
        return f"{parsed:.5f}"

    def _leaderboard_for_display(leaderboard: pd.DataFrame) -> pd.DataFrame:
        display = leaderboard.copy()
        if display.empty:
            return display
        if "mode" in display.columns and "method_type" not in display.columns:
            display.insert(1, "method_type", display["mode"].map(lambda m: MODE_LABELS.get(str(m), str(m))))
        if not rmse_display:
            return display
        for col in ("cv_mean", "test_score"):
            if col not in display.columns:
                continue
            numeric = pd.to_numeric(display[col], errors="coerce")
            display[col] = numeric.where(~numeric.notna(), numeric.abs())
        return display

    def _split_candidate_label(label: str) -> tuple[str, str]:
        token = str(label or "")
        if "__" in token:
            model, selector = token.split("__", 1)
            return model, selector
        return token, ""

    partial_rows: list[dict[str, object]] = []
    partial_warnings: list[str] = []
    partial_split_info: dict[str, object] = {}
    partial_total_experiments = 0
    partial_completed_experiments = 0
    partial_best_raw_report = pd.DataFrame()
    partial_run_datetime = ""

    def _capture_progress_callback(event: dict[str, object]) -> None:
        nonlocal partial_total_experiments, partial_completed_experiments, partial_split_info, partial_best_raw_report, partial_run_datetime
        event_name = event.get("event")
        if event_name == "run_started":
            partial_run_datetime = str(event.get("run_datetime", "") or "")
            try:
                partial_total_experiments = int(event.get("total_experiments", 0))
            except (TypeError, ValueError):
                partial_total_experiments = 0
            partial_split_info = {
                "test": {
                    "splitter": event.get("test_splitter"),
                    "strategy": event.get("test_strategy"),
                    "train_size": event.get("test_train_size"),
                    "test_size": event.get("test_test_size"),
                },
                "cv": {
                    "splitter": event.get("cv_splitter"),
                    "n_splits": event.get("cv_n_splits"),
                    "strategy_used": event.get("cv_strategy_used"),
                    "strategy_requested": event.get("cv_strategy_requested"),
                    "fallback_applied": event.get("cv_fallback_applied"),
                    "fallback_reason": event.get("cv_fallback_reason"),
                    "group_columns": event.get("split_group_columns"),
                    "date_column": event.get("split_date_column"),
                    "stratify_column": event.get("split_stratify_column"),
                    "inferred_from_columns": event.get("cv_inferred_from_columns"),
                    "fold_size_rows": event.get("cv_fold_size_rows"),
                    "n_splits_derived_from_fold_size": event.get("cv_n_splits_derived_from_fold_size"),
                    "scorer": event.get("scorer"),
                },
                "preprocessing": event.get("preprocessing", {}),
                "group_profile": event.get("group_profile", {}),
            }
            if partial_run_datetime:
                partial_split_info["run_datetime"] = partial_run_datetime
            return
        if event_name == "experiment_completed":
            try:
                partial_completed_experiments = int(event.get("completed_experiments", partial_completed_experiments))
            except (TypeError, ValueError):
                pass
            mode = str(event.get("mode", "unknown"))
            variant = str(event.get("variant", ""))
            experiment_name = f"{mode}:{variant}" if variant else mode
            partial_rows.append(
                {
                    "mode": mode,
                    "method_type": str(event.get("method_type", MODE_LABELS.get(mode, mode))),
                    "variant": variant,
                    "experiment_name": experiment_name,
                    "model": event.get("model", "unknown"),
                    "selector": event.get("selector", "unknown"),
                    "run_datetime": event.get("run_datetime", partial_run_datetime),
                    "cv_mean": event.get("cv_mean", float("nan")),
                    "cv_std": event.get("cv_std", float("nan")),
                    "cv_folds_ok": event.get("cv_folds_ok", float("nan")),
                    "test_score": event.get("test_score", float("nan")),
                }
            )
            if bool(event.get("best_so_far_updated")):
                candidate_raw = event.get("best_raw_report")
                if isinstance(candidate_raw, pd.DataFrame):
                    partial_best_raw_report = candidate_raw.copy()
            return
        if event_name == "group_model_selected":
            mode = str(event.get("mode", "group_split"))
            variant = str(event.get("variant", ""))
            experiment_name = f"{mode}:{variant}" if variant else mode
            selected_config = str(event.get("selected_config", ""))
            selected_model, selected_selector = _split_candidate_label(selected_config)
            partial_rows.append(
                {
                    "mode": mode,
                    "method_type": str(event.get("method_type", MODE_LABELS.get(mode, mode))),
                    "variant": variant,
                    "experiment_name": experiment_name,
                    "model": selected_model,
                    "selector": selected_selector,
                    "run_scope": "group",
                    "group_key": str(event.get("group_key", "")),
                    "group_index": event.get("group_index", float("nan")),
                    "group_total": event.get("group_total", float("nan")),
                    "group_selected_config": selected_config,
                    "group_fallback_config": str(event.get("fallback_config", "")),
                    "group_train_rows": event.get("group_train_rows", float("nan")),
                    "group_test_rows": event.get("group_test_rows", float("nan")),
                    "group_test_score": event.get("group_test_score", float("nan")),
                    "group_test_metric": event.get("group_test_metric", ""),
                    "run_datetime": event.get("run_datetime", partial_run_datetime),
                    "cv_mean": event.get("group_cv_mean", float("nan")),
                    "cv_std": float("nan"),
                    "cv_folds_ok": float("nan"),
                    "test_score": event.get("group_test_score", float("nan")),
                    "train_rows": event.get("group_train_rows", float("nan")),
                    "test_rows": event.get("group_test_rows", float("nan")),
                    "run_status": "group_info",
                }
            )
            return

    def _cli_progress_callback(event: dict[str, object]) -> None:
        event_name = event.get("event")
        if event_name == "run_started":
            total = int(event.get("total_experiments", 0))
            splitter = event.get("cv_splitter", "unknown")
            strategy = event.get("cv_strategy_used", "unknown")
            folds = int(event.get("cv_n_splits", 0))
            inferred = " [inferred]" if bool(event.get("cv_inferred_from_columns")) else ""
            fold_size_rows = event.get("cv_fold_size_rows")
            is_time_strategy = str(strategy).lower() in {"timecv", "stratifytimecv", "timeseriessplit"}
            fold_size_suffix = ""
            if is_time_strategy and fold_size_rows is not None:
                fold_size_suffix = f", fold_size_rows={fold_size_rows}"
            print(
                f"[groupml] Running {total} experiment(s) | cv={splitter} "
                f"(strategy={strategy}, folds={folds}{fold_size_suffix}){inferred}"
            )
            preprocessing = event.get("preprocessing", {})
            if isinstance(preprocessing, dict) and preprocessing:
                print(
                    "[groupml] Preprocess rows: "
                    f"initial={preprocessing.get('rows_initial')} | "
                    f"after_target={preprocessing.get('rows_after_target_filters')} | "
                    f"after_dropna={preprocessing.get('rows_after_dropna')} | "
                    f"after_comparability={preprocessing.get('rows_after_comparability')}"
                )
                print(
                    "[groupml] Preprocess drops: "
                    f"min_target={preprocessing.get('rows_dropped_min_target', 0)} | "
                    f"max_target={preprocessing.get('rows_dropped_max_target', 0)} | "
                    f"required_na={preprocessing.get('rows_dropped_required_na', 0)} | "
                    f"group_comparability={preprocessing.get('rows_dropped_group_comparability', 0)}"
                )
                print(
                    "[groupml] Preprocess features: "
                    f"initial={preprocessing.get('columns_initial_features')} | "
                    f"removed_static={preprocessing.get('columns_removed_static')} | "
                    f"final={preprocessing.get('columns_final_features')}"
                )
            group_profile = event.get("group_profile", {})
            if isinstance(group_profile, dict) and group_profile.get("group_columns"):
                print(
                    "[groupml] Groups: "
                    f"columns={group_profile.get('group_columns')} | "
                    f"unique_per_column={group_profile.get('unique_groups_per_column', {})} | "
                    f"unique_combinations={group_profile.get('unique_group_combinations')} | "
                    f"min_group_size={group_profile.get('min_group_size')}"
                )
            if bool(event.get("cv_n_splits_derived_from_fold_size")):
                fold_size = event.get("cv_fold_size_rows")
                print(f"[groupml] CV folds derived from cv_fold_size_rows={fold_size}: using n_splits={folds}")
            if bool(event.get("cv_fallback_applied")):
                reason = event.get("cv_fallback_reason", "constraint mismatch")
                print(f"[groupml] CV hybrid fallback applied ({reason}); check warnings in final summary.")
            return
        if event_name == "mode_started":
            mode = event.get("mode", "unknown")
            mode_label = MODE_LABELS.get(str(mode), str(mode))
            planned = int(event.get("planned_experiments", 0))
            print(f"[groupml] Mode {mode_label}: {planned} experiment(s)")
            return
        if event_name == "group_split_variant_started":
            method = str(event.get("method_type", "per_group_models"))
            variant = str(event.get("variant", ""))
            group_count = int(event.get("group_count", 0) or 0)
            candidate_count = int(event.get("candidate_count", 0) or 0)
            print(
                f"[groupml] {method} ({variant}) training per-group models: "
                f"{group_count} groups x {candidate_count} candidates"
            )
            return
        if event_name == "group_split_shared_search_started":
            method = str(event.get("method_type", "per_group_models"))
            variant = str(event.get("variant", ""))
            total = int(event.get("shared_total", 0) or 0)
            print(f"[groupml] {method} ({variant}) comparing one shared model+selector across all groups ({total} candidates)")
            return
        if event_name == "group_split_shared_candidate_evaluated":
            method = str(event.get("method_type", "per_group_models"))
            variant = str(event.get("variant", ""))
            idx = int(event.get("shared_index", 0) or 0)
            total = int(event.get("shared_total", 0) or 0)
            model = str(event.get("model", ""))
            selector = str(event.get("selector", ""))
            cv_mean = _format_score(event.get("cv_mean"), positive_rmse=rmse_display)
            test_score = _format_score(event.get("test_score"), positive_rmse=rmse_display)
            if rmse_display:
                print(
                    f"[groupml]   shared[{idx}/{total}] {model}/{selector} | "
                    f"cv_rmse={cv_mean} | test_rmse={test_score}"
                )
            else:
                print(
                    f"[groupml]   shared[{idx}/{total}] {model}/{selector} | "
                    f"cv_score={cv_mean} | test_score={test_score}"
                )
            return
        if event_name == "group_split_optimized_search_started":
            method = str(event.get("method_type", "per_group_models"))
            variant = str(event.get("variant", ""))
            groups = int(event.get("group_count", 0) or 0)
            candidates = int(event.get("candidate_count", 0) or 0)
            print(
                f"[groupml] {method} ({variant}) training groups first "
                f"({groups} groups x {candidates} candidates)"
            )
            return
        if event_name == "group_tuning_group_started":
            group_index = int(event.get("group_index", 0) or 0)
            group_total = int(event.get("group_total", 0) or 0)
            group_key = str(event.get("group_key", ""))
            group_size = int(event.get("group_size", 0) or 0)
            print(f"[groupml]   group {group_index}/{group_total}: {group_key} (n={group_size})")
            return
        if event_name == "group_tuning_candidate_scored":
            candidate_index = int(event.get("candidate_index", 0) or 0)
            candidate_total = int(event.get("candidate_total", 0) or 0)
            candidate = str(event.get("candidate", ""))
            model, selector = _split_candidate_label(candidate)
            score = _format_score(event.get("cv_mean"), positive_rmse=rmse_display)
            score_source = str(event.get("score_source", "cv")).strip().lower()
            prefix = "cv" if score_source == "cv" else "train"
            if rmse_display:
                print(
                    f"[groupml]     {candidate_index}/{candidate_total}: "
                    f"model={model} | selector={selector} | {prefix}_rmse={score}"
                )
            else:
                print(
                    f"[groupml]     {candidate_index}/{candidate_total}: "
                    f"model={model} | selector={selector} | {prefix}_score={score}"
                )
            return
        if event_name == "group_tuning_group_finished":
            group_key = str(event.get("group_key", ""))
            best = str(event.get("best_candidate", ""))
            best_model, best_selector = _split_candidate_label(best)
            best_score = _format_score(event.get("best_score"), positive_rmse=rmse_display)
            score_source = str(event.get("score_source", "cv")).strip().lower()
            prefix = "cv" if score_source == "cv" else "train"
            used_fallback = bool(event.get("used_fallback", False))
            reason = str(event.get("reason", ""))
            suffix = f" | fallback ({reason})" if used_fallback else ""
            if rmse_display:
                print(
                    f"[groupml]   best for {group_key}: model={best_model} | "
                    f"selector={best_selector} | {prefix}_rmse={best_score}{suffix}"
                )
            else:
                print(
                    f"[groupml]   best for {group_key}: model={best_model} | "
                    f"selector={best_selector} | {prefix}_score={best_score}{suffix}"
                )
            return
        if event_name == "group_model_selected":
            group_key = str(event.get("group_key", ""))
            selected = str(event.get("selected_config", ""))
            model, selector = _split_candidate_label(selected)
            group_cv = _format_score(event.get("group_cv_mean"), positive_rmse=rmse_display)
            test_rows = int(event.get("group_test_rows", 0) or 0)
            if test_rows > 0:
                test_score = _format_score(event.get("group_test_score"), positive_rmse=rmse_display)
                metric_name = str(event.get("group_test_metric", "test_score")).strip().lower()
                if rmse_display and metric_name == "rmse":
                    print(
                        f"[groupml]   final for {group_key}: model={model} | selector={selector} "
                        f"| cv_rmse={group_cv} | test_rmse={test_score} | test_rows={test_rows}"
                    )
                else:
                    print(
                        f"[groupml]   final for {group_key}: model={model} | selector={selector} "
                        f"| cv_score={group_cv} | test_score={test_score} | test_rows={test_rows}"
                    )
            else:
                print(
                    f"[groupml]   final for {group_key}: model={model} | selector={selector} "
                    f"| cv_score={group_cv} | test_score=n/a (no test rows)"
                )
            return
        if event_name == "group_split_shared_best":
            method = str(event.get("method_type", "per_group_models"))
            variant = str(event.get("variant", ""))
            model = str(event.get("model", ""))
            selector = str(event.get("selector", ""))
            cv_mean = _format_score(event.get("cv_mean"), positive_rmse=rmse_display)
            test_score = _format_score(event.get("test_score"), positive_rmse=rmse_display)
            if rmse_display:
                print(
                    f"[groupml] {method} ({variant}) shared-config best | model={model} | selector={selector} "
                    f"| cv_rmse={cv_mean} | test_rmse={test_score}"
                )
            else:
                print(
                    f"[groupml] {method} ({variant}) shared-config best | model={model} | selector={selector} "
                    f"| cv_score={cv_mean} | test_score={test_score}"
                )
            return
        if event_name == "group_split_variant_finished":
            method = str(event.get("method_type", "per_group_models"))
            variant = str(event.get("variant", ""))
            unique_configs = int(event.get("unique_selected_config_count", 0) or 0)
            fallback = str(event.get("fallback_config", ""))
            cv_mean = _format_score(event.get("cv_mean"), positive_rmse=rmse_display)
            test_score = _format_score(event.get("test_score"), positive_rmse=rmse_display)
            shared_model = str(event.get("shared_best_model", ""))
            shared_selector = str(event.get("shared_best_selector", ""))
            shared_cv = _format_score(event.get("shared_best_cv_mean"), positive_rmse=rmse_display)
            if rmse_display:
                print(
                    f"[groupml] {method} ({variant}) done | unique_group_configs={unique_configs} "
                    f"| fallback={fallback} | optimized_cv_rmse={cv_mean} | optimized_test_rmse={test_score} "
                    f"| shared_best={shared_model}/{shared_selector} ({shared_cv})"
                )
            else:
                print(
                    f"[groupml] {method} ({variant}) done | unique_group_configs={unique_configs} "
                    f"| fallback={fallback} | optimized_cv_score={cv_mean} | optimized_test_score={test_score} "
                    f"| shared_best={shared_model}/{shared_selector} ({shared_cv})"
                )
            return
        if event_name == "experiment_completed":
            done = int(event.get("completed_experiments", 0))
            total = int(event.get("total_experiments", 0))
            mode = event.get("mode", "unknown")
            if str(mode) in {"group_split", "group_permutations"} and str(event.get("model")) != "per_group_best":
                return
            mode_label = str(event.get("method_type", MODE_LABELS.get(str(mode), str(mode))))
            model = event.get("model", "unknown")
            selector = event.get("selector", "unknown")
            cv_mean = _format_score(event.get("cv_mean"), positive_rmse=rmse_display)
            test_score = _format_score(event.get("test_score"), positive_rmse=rmse_display)
            if rmse_display:
                print(
                    f"[groupml] [{done}/{total}] {mode_label} | model={model} | selector={selector} "
                    f"| cv_rmse={cv_mean} | test_rmse={test_score}"
                )
            else:
                print(
                    f"[groupml] [{done}/{total}] {mode_label} | model={model} | selector={selector} "
                    f"| cv_score={cv_mean} | test_score={test_score}"
                )

    def _combined_progress_callback(event: dict[str, object]) -> None:
        _capture_progress_callback(event)
        _cli_progress_callback(event)

    result: GroupMLResult | None = None
    interrupted = False
    callbacks = [_combined_progress_callback]
    try:
        result = fit_evaluate_file(args.path, config, callbacks=callbacks, **read_kwargs)
    except KeyboardInterrupt:
        interrupted = True
        partial_warnings.append(
            "Run interrupted by user. Report contains only experiments completed before interruption."
        )
        leaderboard = pd.DataFrame(partial_rows)
        if not leaderboard.empty:
            leaderboard = leaderboard.sort_values(by=["cv_mean"], ascending=rmse_display).reset_index(drop=True)
            best = leaderboard.iloc[0].to_dict()
            baseline_rows = leaderboard[leaderboard["mode"] == "full"]
            baseline = (
                baseline_rows.sort_values(by=["cv_mean"], ascending=rmse_display).iloc[0].to_dict()
                if not baseline_rows.empty
                else best
            )
        else:
            best = {}
            baseline = {}
        progress_note = (
            f"Completed {partial_completed_experiments}/{partial_total_experiments} experiments before interruption."
        )
        partial_warnings.append(progress_note)
        run_datetime_value = str(partial_split_info.get("run_datetime", ""))
        warning_details = pd.DataFrame(
            {
                "warning_datetime": [run_datetime_value] * len(partial_warnings),
                "run_datetime": [run_datetime_value] * len(partial_warnings),
                "run_experiment": [""] * len(partial_warnings),
                "warning": partial_warnings,
            }
        )
        result = GroupMLResult(
            leaderboard=leaderboard,
            recommendation=f"Run interrupted. {progress_note}",
            warnings=partial_warnings,
            best_experiment=best,
            baseline_experiment=baseline,
            split_info=partial_split_info,
            raw_report=partial_best_raw_report,
            all_runs=leaderboard.copy(),
            warning_details=warning_details,
        )
        print()
        print(
            f"[groupml] Interrupted. {progress_note} Attempting to export partial outputs."
        )

    assert result is not None
    output_lines: list[str] = []
    output_lines.append("[groupml] Finished.")
    best = result.best_experiment or (
        result.leaderboard.iloc[0].to_dict() if not result.leaderboard.empty else {}
    )
    if best:
        best_method = str(best.get("method_type", MODE_LABELS.get(str(best.get("mode", "")), best.get("mode", ""))))
        best_model = str(best.get("model", "unknown"))
        best_selector = str(best.get("selector", "unknown"))
        best_cv = _format_score(best.get("cv_mean"), positive_rmse=rmse_display)
        best_test = _format_score(best.get("test_score"), positive_rmse=rmse_display)
        if rmse_display:
            output_lines.append(
                f"[groupml] Best: {best_method} | model={best_model} | selector={best_selector} | cv_rmse={best_cv} | test_rmse={best_test}"
            )
        else:
            output_lines.append(
                f"[groupml] Best: {best_method} | model={best_model} | selector={best_selector} | cv_score={best_cv} | test_score={best_test}"
            )
    output_lines.append(f"[groupml] Recommendation: {result.recommendation}")

    saved_outputs: list[tuple[str, Path]] = []

    out_path = Path(args.out) if args.out else Path.cwd() / default_summary_filename(
        ext=preferred_tabular_extension(args.report_format)
    )
    output_suffix = out_path.suffix.lower()
    text_like_suffixes = {".txt", ".md", ".json"}
    saved_path = out_path
    used_bundle_export = output_suffix not in text_like_suffixes
    if not used_bundle_export:
        saved_path = export_summary(result, out_path, top_n=args.top)
        saved_outputs.append(("summary", saved_path))
    else:
        bundle_outputs = export_reporting_bundle(
            result=result,
            path=out_path,
            top_n=args.top,
            report_format=args.report_format,
            include_raw=not args.no_raw_report,
        )
        if "workbook" in bundle_outputs:
            saved_outputs.append(("workbook", bundle_outputs["workbook"]))
        else:
            if "summary" in bundle_outputs:
                saved_outputs.append(("summary", bundle_outputs["summary"]))
            if "recommendations" in bundle_outputs:
                saved_outputs.append(("recommendations", bundle_outputs["recommendations"]))
            if "warnings" in bundle_outputs:
                saved_outputs.append(("warnings", bundle_outputs["warnings"]))
            if "all_runs" in bundle_outputs:
                saved_outputs.append(("all_runs", bundle_outputs["all_runs"]))
            if "raw_results" in bundle_outputs:
                saved_outputs.append(("raw_results", bundle_outputs["raw_results"]))
        saved_path = bundle_outputs.get("summary", out_path)

    if args.leaderboard_out:
        leaderboard_path = export_report(result, Path(args.leaderboard_out))
        saved_outputs.append(("leaderboard", leaderboard_path))
    should_export_separate_raw = bool(args.raw_report_out) or not used_bundle_export
    if (
        should_export_separate_raw
        and not args.no_raw_report
        and isinstance(result.raw_report, pd.DataFrame)
        and (interrupted or not result.raw_report.empty)
    ):
        if args.raw_report_out:
            raw_report_path = Path(args.raw_report_out)
        else:
            base = saved_path if saved_path.suffix else Path.cwd() / default_report_filename(ext=".csv")
            raw_report_path = base.with_name(f"{base.stem}_raw.csv")
        written_raw_path = export_raw_report(result, raw_report_path)
        saved_outputs.append(("raw_report", written_raw_path))

    for line in output_lines:
        print(line)
    if saved_outputs:
        print("[groupml] Reports:")
        for label, path in saved_outputs:
            print(f"[groupml]   {label}: {path}")

    return 130 if interrupted else 0
