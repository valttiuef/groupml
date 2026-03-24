"""Utilities for loading tabular files and running groupml experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from .config import GroupMLConfig
from .result import GroupMLResult
from .runner import GroupMLRunner, compare_group_strategies
from .summaries import build_summary_payload, build_summary_tables, summary_text


def load_tabular_data(path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Load a CSV or Excel file into a pandas DataFrame."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, **read_kwargs)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(file_path, **read_kwargs)
    raise ValueError(
        f"Unsupported file extension '{file_path.suffix}'. "
        "Supported: .csv, .xls, .xlsx"
    )


def fit_evaluate_file(
    path: str | Path,
    config: GroupMLConfig,
    callbacks: Iterable[Callable[[dict[str, Any]], None]] | None = None,
    **read_kwargs: Any,
) -> GroupMLResult:
    """Load a file and run `GroupMLRunner.fit_evaluate`."""
    df = load_tabular_data(path, **read_kwargs)
    return GroupMLRunner(config).fit_evaluate(df, callbacks=callbacks)


def compare_group_strategies_file(
    path: str | Path,
    target: str,
    feature_columns: list[str] | None = None,
    group_columns: list[str] | None = None,
    rule_splits: list[str] | None = None,
    read_kwargs: dict[str, Any] | None = None,
    callbacks: Iterable[Callable[[dict[str, Any]], None]] | None = None,
    **kwargs: Any,
) -> GroupMLResult:
    """Functional API wrapper for running experiments from a tabular file."""
    df = load_tabular_data(path, **(read_kwargs or {}))
    return compare_group_strategies(
        df=df,
        target=target,
        feature_columns=feature_columns,
        group_columns=group_columns,
        rule_splits=rule_splits,
        callbacks=callbacks,
        **kwargs,
    )


def default_report_filename(prefix: str = "groupml_report", ext: str = ".csv") -> str:
    """Build a friendly timestamped report filename."""
    clean_ext = ext if ext.startswith(".") else f".{ext}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{clean_ext}"


def default_summary_filename(prefix: str = "groupml_summary", ext: str = ".csv") -> str:
    """Build a friendly timestamped summary filename."""
    return default_report_filename(prefix=prefix, ext=ext)


def export_report(
    result: GroupMLResult,
    path: str | Path,
    sheet_name: str = "leaderboard",
) -> Path:
    """Export leaderboard report to CSV or Excel based on file extension."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        result.leaderboard.to_csv(output_path, index=False)
        return output_path

    if suffix in {".xls", ".xlsx"}:
        try:
            result.leaderboard.to_excel(output_path, index=False, sheet_name=sheet_name)
        except ImportError as exc:
            raise ImportError(
                "Excel export requires an Excel writer engine (for example openpyxl). "
                "Install it or export as .csv."
            ) from exc
        return output_path

    raise ValueError(
        f"Unsupported report extension '{output_path.suffix}'. Supported: .csv, .xls, .xlsx"
    )


def export_summary(
    result: GroupMLResult,
    path: str | Path,
    top_n: int = 10,
    sheet_name: str = "summary",
) -> Path:
    """Export human-friendly summary to text/JSON/CSV/Excel."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        output_path.write_text(summary_text(result, top_n=top_n), encoding="utf-8")
        return output_path

    if suffix == ".json":
        payload = build_summary_payload(result, top_n=top_n)
        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return output_path

    if suffix == ".csv":
        tables = build_summary_tables(result, top_n=top_n)
        stacked_tables: list[pd.DataFrame] = []
        for section, table in tables.items():
            if table.empty:
                continue
            with_section = table.copy()
            with_section.insert(0, "section", section)
            stacked_tables.append(with_section)
        if stacked_tables:
            summary_csv = pd.concat(stacked_tables, ignore_index=True, sort=False)
        else:
            summary_csv = pd.DataFrame({"section": []})
        summary_csv.to_csv(output_path, index=False)
        return output_path

    if suffix in {".xls", ".xlsx"}:
        tables = build_summary_tables(result, top_n=top_n)
        try:
            with pd.ExcelWriter(output_path) as writer:
                for idx, (name, table) in enumerate(tables.items()):
                    if idx == 0:
                        safe_name = sheet_name[:31]
                    else:
                        safe_name = name[:31]
                    table.to_excel(writer, index=False, sheet_name=safe_name)
        except ImportError as exc:
            raise ImportError(
                "Excel export requires an Excel writer engine (for example openpyxl). "
                "Install it or export summary as .txt/.json/.csv."
            ) from exc
        return output_path

    raise ValueError(
        f"Unsupported summary extension '{output_path.suffix}'. Supported: .txt, .md, .json, .csv, .xls, .xlsx"
    )
