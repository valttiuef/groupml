"""Command line interface for running groupml from terminal."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from .config import GroupMLConfig
from .file_utils import (
    default_summary_filename,
    export_report,
    export_summary,
    fit_evaluate_file,
)


def _parse_cv_value(raw: str) -> int | str:
    value = str(raw).strip()
    if value.isdigit():
        return int(value)
    return value


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
        "--cv",
        type=str,
        default="5",
        help=(
            "CV setting: fold count (for example 5) or splitter name "
            "(for example groupcv, timecv, stratifycv, stratifygroupcv, stratifytimecv)."
        ),
    )
    parser.add_argument("--cv-group-column", default=None, help="Single group column used for CV grouping.")
    parser.add_argument("--cv-date-column", default=None, help="Datetime column used for time-based CV.")
    parser.add_argument(
        "--cv-stratify-column",
        default=None,
        help="Column used for stratified CV behavior.",
    )
    parser.add_argument(
        "--scorer",
        default="neg_root_mean_squared_error",
        help="sklearn scorer name (or 'rmse' alias).",
    )
    parser.add_argument("--task", choices=["auto", "regression", "classification"], default="auto")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test size fraction.")
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

    config = GroupMLConfig(
        target=args.target,
        feature_columns=args.features,
        group_columns=args.groups or [],
        rule_splits=args.rules or [],
        experiment_modes=args.modes or ["full", "group_as_features", "group_split", "group_permutations", "rule_split"],
        cv=_parse_cv_value(args.cv),
        cv_group_column=args.cv_group_column,
        cv_date_column=args.cv_date_column,
        cv_stratify_column=args.cv_stratify_column,
        scorer=args.scorer,
        task=args.task,
        test_size=args.test_size,
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

    result = fit_evaluate_file(args.path, config, callbacks=[_cli_progress_callback], **read_kwargs)

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

    return 0
