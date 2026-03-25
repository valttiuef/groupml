"""Command line interface for running groupml from terminal."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import GroupMLConfig
from .file_utils import (
    default_summary_filename,
    export_report,
    export_summary,
    fit_evaluate_file,
)
from .result import GroupMLResult


def _parse_cv_value(raw: str) -> int | str:
    value = str(raw).strip()
    if value.isdigit():
        return int(value)
    return value


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
        "--cv",
        type=str,
        default="5",
        help=(
            "CV setting: fold count (for example 5) or splitter name "
            "(for example groupcv, timecv, stratifycv, stratifygroupcv, stratifytimecv)."
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
    parser.add_argument("--min-group-size", type=int, default=15)
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
        help="Optional summary output path (.csv/.xls/.xlsx/.txt/.md/.json). Defaults to a timestamped .csv file.",
    )
    parser.add_argument(
        "--leaderboard-out",
        default=None,
        help="Optional raw leaderboard output path (.csv/.xls/.xlsx).",
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
        cv=_parse_cv_value(args.cv),
        split_group_column=args.split_group_column,
        split_date_column=args.split_date_column,
        split_stratify_column=args.split_stratify_column,
        scorer=args.scorer,
        task=args.task,
        test_split_strategy=args.test_split,
        test_size=test_size,
        test_size_rows=test_size_rows,
        random_state=args.random_state,
        min_group_size=args.min_group_size,
        min_improvement=args.min_improvement,
        scale_numeric=args.scale_numeric,
        dropna_base_rows=not args.keep_nans,
        drop_static_base_features=not args.keep_static_features,
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

    partial_rows: list[dict[str, object]] = []
    partial_warnings: list[str] = []
    partial_split_info: dict[str, object] = {}
    partial_total_experiments = 0
    partial_completed_experiments = 0

    def _capture_progress_callback(event: dict[str, object]) -> None:
        nonlocal partial_total_experiments, partial_completed_experiments, partial_split_info
        event_name = event.get("event")
        if event_name == "run_started":
            try:
                partial_total_experiments = int(event.get("total_experiments", 0))
            except (TypeError, ValueError):
                partial_total_experiments = 0
            partial_split_info = {
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
                }
            }
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
                    "variant": variant,
                    "experiment_name": experiment_name,
                    "model": event.get("model", "unknown"),
                    "selector": event.get("selector", "unknown"),
                    "cv_mean": event.get("cv_mean", float("nan")),
                    "cv_std": float("nan"),
                    "cv_folds_ok": float("nan"),
                    "test_score": event.get("test_score", float("nan")),
                }
            )

    def _cli_progress_callback(event: dict[str, object]) -> None:
        event_name = event.get("event")
        if event_name == "run_started":
            total = int(event.get("total_experiments", 0))
            splitter = event.get("cv_splitter", "unknown")
            strategy = event.get("cv_strategy_used", "unknown")
            folds = int(event.get("cv_n_splits", 0))
            inferred = " [inferred]" if bool(event.get("cv_inferred_from_columns")) else ""
            print(
                f"[groupml] Running {total} experiment(s) | cv={splitter} "
                f"(strategy={strategy}, folds={folds}){inferred}"
            )
            if bool(event.get("cv_fallback_applied")):
                reason = event.get("cv_fallback_reason", "constraint mismatch")
                print(f"[groupml] CV hybrid fallback applied ({reason}); check warnings in final summary.")
            return
        if event_name == "mode_started":
            mode = event.get("mode", "unknown")
            planned = int(event.get("planned_experiments", 0))
            print(f"[groupml] Mode {mode}: {planned} experiment(s)")
            return
        if event_name == "experiment_completed":
            done = int(event.get("completed_experiments", 0))
            total = int(event.get("total_experiments", 0))
            mode = event.get("mode", "unknown")
            model = event.get("model", "unknown")
            selector = event.get("selector", "unknown")
            cv_mean = _format_score(event.get("cv_mean"), positive_rmse=rmse_display)
            test_score = _format_score(event.get("test_score"), positive_rmse=rmse_display)
            if rmse_display:
                print(
                    f"[groupml] [{done}/{total}] {mode} | model={model} | selector={selector} "
                    f"| cv_rmse={cv_mean} | test_rmse={test_score}"
                )
            else:
                print(
                    f"[groupml] [{done}/{total}] {mode} | model={model} | selector={selector} "
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
            leaderboard = leaderboard.sort_values(by=["cv_mean", "test_score"], ascending=False).reset_index(drop=True)
            best = leaderboard.iloc[0].to_dict()
            baseline_rows = leaderboard[leaderboard["mode"] == "full"]
            baseline = (
                baseline_rows.sort_values(by=["cv_mean", "test_score"], ascending=False).iloc[0].to_dict()
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
        result = GroupMLResult(
            leaderboard=leaderboard,
            recommendation=f"Run interrupted. {progress_note}",
            warnings=partial_warnings,
            best_experiment=best,
            baseline_experiment=baseline,
            split_info=partial_split_info,
        )
        print()
        print(
            f"[groupml] Interrupted. {progress_note} Attempting to export partial outputs."
        )

    assert result is not None
    print(result.summary_text())
    print()
    print("Leaderboard:")
    print(result.leaderboard.head(args.top).to_string(index=False))

    out_path = Path(args.out) if args.out else Path.cwd() / default_summary_filename(ext=".csv")
    saved_path = export_summary(result, out_path, top_n=args.top)
    print()
    print(f"Saved summary: {saved_path}")

    if args.leaderboard_out:
        leaderboard_path = export_report(result, Path(args.leaderboard_out))
        print(f"Saved leaderboard: {leaderboard_path}")

    return 130 if interrupted else 0
