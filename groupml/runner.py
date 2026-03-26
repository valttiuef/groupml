"""Main execution engine for groupml experiments."""

from __future__ import annotations

import warnings as py_warnings
from contextlib import contextmanager
from datetime import datetime
from itertools import product
import re
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import get_scorer, mean_squared_error
from sklearn.pipeline import Pipeline

from .config import GroupMLConfig
from .estimators import (
    GroupSplitClassifier,
    GroupSplitRegressor,
    RuleSplitClassifier,
    RuleSplitRegressor,
)
from .result import GroupMLResult
from .splitting import plan_splits
from .utils import (
    build_preprocessor,
    build_selector,
    default_experiment_names,
    ensure_columns_exist,
    group_column_permutations,
    infer_task,
    normalize_models,
    normalize_selectors,
    parse_rule,
)


class GroupMLRunner:
    """Run comparable group-aware ML strategy experiments."""

    def __init__(self, config: GroupMLConfig) -> None:
        self.config = config

    @contextmanager
    def _lib_warning_filter(self) -> Iterable[None]:
        level = str(self.config.warning_verbosity).strip().lower()
        if level == "all":
            yield
            return
        with py_warnings.catch_warnings():
            if level == "quiet":
                py_warnings.filterwarnings(
                    "ignore",
                    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
                    category=UserWarning,
                )
                py_warnings.filterwarnings("ignore", category=ConvergenceWarning)
            else:
                py_warnings.filterwarnings(
                    "once",
                    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
                    category=UserWarning,
                )
            yield

    def fit_evaluate(
        self,
        df: pd.DataFrame,
        callbacks: Iterable[Callable[[dict[str, Any]], None]] | None = None,
    ) -> GroupMLResult:
        """Run configured experiments and return structured result."""
        ensure_columns_exist(df, [self.config.target], "target")
        warnings: list[str] = []
        run_datetime = datetime.now().astimezone().isoformat(timespec="seconds")
        callback_list = list(callbacks or [])
        data = df.copy()
        group_cols = list(self.config.group_columns)
        ensure_columns_exist(data, group_cols, "group")

        if self.config.feature_columns is None:
            feature_cols = [c for c in data.columns if c != self.config.target]
        else:
            feature_cols = list(self.config.feature_columns)
        ensure_columns_exist(data, feature_cols, "feature")

        parsed_rules = [parse_rule(r) for r in self.config.rule_splits]
        ensure_columns_exist(data, [r.column for r in parsed_rules], "rule")
        rule_columns = [r.column for r in parsed_rules]

        split_group_columns: list[str] = []
        if self.config.split_group_column:
            split_group_columns.append(self.config.split_group_column)
        split_group_columns.extend(list(self.config.split_group_columns or []))
        split_group_columns = list(dict.fromkeys(split_group_columns))
        split_date_column = self.config.split_date_column
        split_stratify_column = self.config.split_stratify_column
        ensure_columns_exist(data, split_group_columns, "split_group")
        ensure_columns_exist(data, [split_date_column] if split_date_column else [], "split_date")
        ensure_columns_exist(data, [split_stratify_column] if split_stratify_column else [], "split_stratify")

        task = infer_task(data[self.config.target], self.config.task)
        data, feature_cols, task, preprocessing_stats = self._preprocess_base_dataset(
            data=data,
            feature_cols=feature_cols,
            group_cols=group_cols,
            rule_columns=rule_columns,
            split_group_columns=split_group_columns,
            split_date_column=split_date_column,
            split_stratify_column=split_stratify_column,
            task=task,
            warnings=warnings,
        )
        data, feature_cols, comparability_dropped_rows = self._apply_comparability_row_filter(
            data=data,
            feature_cols=feature_cols,
            group_cols=group_cols,
            warnings=warnings,
        )
        preprocessing_stats["rows_dropped_group_comparability"] = int(comparability_dropped_rows)
        preprocessing_stats["rows_after_comparability"] = int(len(data))

        unique_groups_per_column: dict[str, int] = {}
        for col in group_cols:
            if col in data.columns:
                unique_groups_per_column[col] = int(data[col].nunique(dropna=True))
        if group_cols:
            if len(group_cols) == 1:
                unique_group_combinations = unique_groups_per_column.get(group_cols[0], 0)
            else:
                unique_group_combinations = int(data[group_cols].drop_duplicates().shape[0])
        else:
            unique_group_combinations = 0

        split_stratify_column, split_group_columns, planned_test_split_strategy = self._resolve_split_defaults(
            data=data,
            group_cols=group_cols,
            split_group_columns=split_group_columns,
            split_date_column=split_date_column,
            split_stratify_column=split_stratify_column,
            warnings=warnings,
        )

        y = data[self.config.target]

        X = data[feature_cols]
        split_plan = plan_splits(
            X=X,
            y=y,
            task=task,
            cv=self.config.cv,
            random_state=self.config.random_state,
            test_size=self.config.test_size,
            test_size_rows=self.config.test_size_rows,
            cv_params=self.config.cv_params,
            cv_fold_size_rows=self.config.cv_fold_size_rows,
            split_group_columns=split_group_columns,
            split_date_column=split_date_column,
            split_stratify_column=split_stratify_column,
            fallback_cv_group_columns=group_cols,
            test_splitter=self.config.test_splitter,
            test_split_strategy=planned_test_split_strategy,
            include_indices=self.config.include_split_indices,
            cv_source_df=data,
        )
        warnings.extend(split_plan.warnings)
        X_train, X_test = X.iloc[split_plan.train_indices], X.iloc[split_plan.test_indices]
        y_train, y_test = y.iloc[split_plan.train_indices], y.iloc[split_plan.test_indices]

        models = normalize_models(self.config.models, task, self.config.random_state)
        selectors = normalize_selectors(self.config.feature_selectors, task)
        model_selector_runs = len(models) * len(selectors)
        has_distinct_group_permutations = len(group_cols) > 1
        group_permutation_splits = (
            group_column_permutations(group_cols) if has_distinct_group_permutations else []
        )
        planned_variants = {
            "full": 1 if "full" in default_experiment_names(self.config) else 0,
            "group_as_features": 1 if "group_as_features" in default_experiment_names(self.config) and group_cols else 0,
            "group_split": 1 if "group_split" in default_experiment_names(self.config) and group_cols else 0,
            "group_permutations": (
                len(group_permutation_splits)
                if "group_permutations" in default_experiment_names(self.config) and has_distinct_group_permutations
                else 0
            ),
            "rule_split": 1 if "rule_split" in default_experiment_names(self.config) and parsed_rules else 0,
        }
        total_experiments = model_selector_runs * sum(planned_variants.values())

        cv_splits = split_plan.cv_splits
        scorer_callable = self._make_score_callable(self.config.scorer)

        rows: list[dict[str, Any]] = []
        modes = default_experiment_names(self.config)
        completed_experiments = 0
        best_so_far_row: dict[str, Any] | None = None
        best_so_far_raw_report = pd.DataFrame()

        self._emit_callbacks(
            callback_list,
            warnings,
            event="run_started",
            payload={
                "event": "run_started",
                "modes": modes,
                "total_experiments": total_experiments,
                "task": task,
                "run_datetime": run_datetime,
                "cv_splitter": split_plan.split_info.get("cv", {}).get("splitter"),
                "cv_n_splits": split_plan.split_info.get("cv", {}).get("n_splits"),
                "cv_strategy_requested": split_plan.split_info.get("cv", {}).get("strategy_requested"),
                "cv_strategy_used": split_plan.split_info.get("cv", {}).get("strategy_used"),
                "cv_fallback_applied": split_plan.split_info.get("cv", {}).get("fallback_applied"),
                "cv_fallback_reason": split_plan.split_info.get("cv", {}).get("fallback_reason"),
                "split_group_columns": split_plan.split_info.get("cv", {}).get("group_columns"),
                "split_date_column": split_plan.split_info.get("cv", {}).get("date_column"),
                "split_stratify_column": split_plan.split_info.get("cv", {}).get("stratify_column"),
                # Backward-compatible callback aliases.
                "cv_group_columns": split_plan.split_info.get("cv", {}).get("group_columns"),
                "cv_date_column": split_plan.split_info.get("cv", {}).get("date_column"),
                "cv_stratify_column": split_plan.split_info.get("cv", {}).get("stratify_column"),
                "cv_inferred_from_columns": split_plan.split_info.get("cv", {}).get("inferred_from_columns"),
                "cv_fold_size_rows": split_plan.split_info.get("cv", {}).get("fold_size_rows"),
                "cv_n_splits_derived_from_fold_size": split_plan.split_info.get("cv", {}).get(
                    "n_splits_derived_from_fold_size"
                ),
                "test_splitter": split_plan.split_info.get("test", {}).get("splitter"),
                "test_strategy": split_plan.split_info.get("test", {}).get("strategy"),
                "test_train_size": split_plan.split_info.get("test", {}).get("train_size"),
                "test_test_size": split_plan.split_info.get("test", {}).get("test_size"),
                "scorer": str(self.config.scorer),
                "preprocessing": dict(preprocessing_stats),
                "group_profile": {
                    "group_columns": list(group_cols),
                    "unique_groups_per_column": unique_groups_per_column,
                    "unique_group_combinations": unique_group_combinations,
                    "min_group_size": int(self.config.min_group_size),
                },
            },
        )

        def _emit_mode_started(mode: str, variant_count: int) -> None:
            self._emit_callbacks(
                callback_list,
                warnings,
                event="mode_started",
                payload={
                    "event": "mode_started",
                    "mode": mode,
                    "planned_experiments": variant_count * model_selector_runs,
                    "total_experiments": total_experiments,
                },
            )

        def _mark_experiment_completed(row: dict[str, Any]) -> None:
            nonlocal completed_experiments, best_so_far_row, best_so_far_raw_report
            completed_experiments += 1
            best_updated = False
            cv_mean_value = float(row.get("cv_mean", np.nan))
            cv_folds_value = int(row.get("cv_folds_ok", 0) or 0)
            candidate_is_valid = np.isfinite(cv_mean_value) and cv_folds_value > 0
            if candidate_is_valid and self._is_better_experiment_row(row, best_so_far_row):
                best_so_far_row = dict(row)
                best_updated = True
                if self.config.raw_report_enabled:
                    try:
                        best_so_far_raw_report = self._build_raw_report(
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
                            y_full=y,
                            best_row=best_so_far_row,
                            best_rows_by_method=[best_so_far_row],
                            warnings=warnings,
                        )
                    except Exception as exc:
                        warnings.append(f"Best-so-far raw report generation failed: {exc}")
            self._emit_callbacks(
                callback_list,
                warnings,
                event="experiment_completed",
                payload={
                    "event": "experiment_completed",
                    "mode": row.get("mode"),
                    "method_type": row.get("method_type"),
                    "variant": row.get("variant"),
                    "model": row.get("model"),
                    "selector": row.get("selector"),
                    "cv_mean": row.get("cv_mean"),
                    "cv_std": row.get("cv_std"),
                    "cv_folds_ok": row.get("cv_folds_ok"),
                    "test_score": row.get("test_score"),
                    "run_datetime": row.get("run_datetime", run_datetime),
                    "completed_experiments": completed_experiments,
                    "total_experiments": total_experiments,
                    "best_so_far_updated": best_updated,
                    "best_experiment_name": (best_so_far_row or {}).get("experiment_name"),
                    "best_raw_report": best_so_far_raw_report if best_updated else None,
                },
            )

        if "full" in modes:
            _emit_mode_started("full", 1)
            rows.extend(
                self._run_flat_mode(
                    mode="full",
                    variant="all_features",
                    task=task,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    feature_cols=[c for c in feature_cols if c not in group_cols],
                    group_cols=group_cols,
                    models=models,
                    selectors=selectors,
                    cv_splits=cv_splits,
                    scorer=scorer_callable,
                    warnings=warnings,
                    run_datetime=run_datetime,
                    on_experiment_completed=_mark_experiment_completed,
                )
            )

        if "group_as_features" in modes:
            if group_cols:
                _emit_mode_started("group_as_features", 1)
                rows.extend(
                    self._run_flat_mode(
                        mode="group_as_features",
                        variant="groups_onehot",
                        task=task,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        feature_cols=feature_cols,
                        group_cols=group_cols,
                        models=models,
                        selectors=selectors,
                        cv_splits=cv_splits,
                        scorer=scorer_callable,
                        warnings=warnings,
                        run_datetime=run_datetime,
                        on_experiment_completed=_mark_experiment_completed,
                    )
                )
            else:
                warnings.append("Skipping group_as_features: no group_columns provided.")

        if "group_split" in modes:
            if group_cols:
                _emit_mode_started("group_split", 1)
                rows.extend(
                    self._run_group_split_mode(
                        mode="group_split",
                        split_columns=tuple(group_cols),
                        task=task,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        feature_cols=[c for c in feature_cols if c not in group_cols],
                        models=models,
                        selectors=selectors,
                        cv_splits=cv_splits,
                        scorer=scorer_callable,
                        warnings=warnings,
                        run_datetime=run_datetime,
                        on_experiment_completed=_mark_experiment_completed,
                    )
                )
            else:
                warnings.append("Skipping group_split: no group_columns provided.")

        if "group_permutations" in modes:
            if has_distinct_group_permutations:
                _emit_mode_started("group_permutations", len(group_permutation_splits))
                for cols in group_permutation_splits:
                    rows.extend(
                        self._run_group_split_mode(
                            mode="group_permutations",
                            split_columns=tuple(cols),
                            task=task,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            feature_cols=[c for c in feature_cols if c not in group_cols],
                            models=models,
                            selectors=selectors,
                            cv_splits=cv_splits,
                            scorer=scorer_callable,
                            warnings=warnings,
                            run_datetime=run_datetime,
                            on_experiment_completed=_mark_experiment_completed,
                        )
                    )
            elif group_cols:
                warnings.append("Skipping group_permutations: requires at least two group_columns.")
            else:
                warnings.append("Skipping group_permutations: no group_columns provided.")

        if "rule_split" in modes:
            if parsed_rules:
                _emit_mode_started("rule_split", 1)
                rows.extend(
                    self._run_rule_split_mode(
                        mode="rule_split",
                        parsed_rules=parsed_rules,
                        task=task,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        feature_cols=[c for c in feature_cols if c not in group_cols],
                        models=models,
                        selectors=selectors,
                        cv_splits=cv_splits,
                        scorer=scorer_callable,
                        warnings=warnings,
                        run_datetime=run_datetime,
                        on_experiment_completed=_mark_experiment_completed,
                    )
                )
            else:
                warnings.append("Skipping rule_split: no rule_splits provided.")

        if not rows:
            raise ValueError("No experiments were executed. Check experiment_modes and inputs.")

        all_rows = pd.DataFrame(rows)
        valid_mask = np.isfinite(pd.to_numeric(all_rows["cv_mean"], errors="coerce")) & (
            pd.to_numeric(all_rows["cv_folds_ok"], errors="coerce") > 0
        )
        all_rows["run_status"] = np.where(valid_mask, "ok", "failed")
        dropped_rows = int((~valid_mask).sum())
        if dropped_rows:
            warnings.append(
                "Ignored "
                f"{dropped_rows} unsuccessful experiment(s) with invalid/unstable CV results from ranking and averages."
            )
        leaderboard = all_rows.loc[valid_mask].sort_values(
            by=["cv_mean"], ascending=self._cv_prefers_lower()
        )
        leaderboard.reset_index(drop=True, inplace=True)
        if leaderboard.empty:
            raise ValueError(
                "All experiments were unsuccessful or produced invalid/unstable CV results. "
                "Try adjusting models/selectors, scaling, or regularization."
            )

        baseline_row = self._pick_baseline(leaderboard)
        best_row = leaderboard.iloc[0].to_dict()
        recommendation = self._recommend(best_row, baseline_row, warnings)
        best_rows_by_method = self._pick_best_rows_by_method(leaderboard)
        raw_report = pd.DataFrame()
        if self.config.raw_report_enabled:
            try:
                if not best_so_far_raw_report.empty and best_so_far_row is not None:
                    if best_so_far_row.get("experiment_name") == best_row.get("experiment_name"):
                        raw_report = self._build_raw_report(
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
                            y_full=y,
                            best_row=best_row,
                            best_rows_by_method=best_rows_by_method,
                            warnings=warnings,
                        )
                    else:
                        raw_report = self._build_raw_report(
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
                            y_full=y,
                            best_row=best_row,
                            best_rows_by_method=best_rows_by_method,
                            warnings=warnings,
                        )
                else:
                    raw_report = self._build_raw_report(
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
                        y_full=y,
                        best_row=best_row,
                        best_rows_by_method=best_rows_by_method,
                        warnings=warnings,
                    )
            except Exception as exc:
                warnings.append(f"Raw report generation failed: {exc}")

        self._emit_callbacks(
            callback_list,
            warnings,
            event="run_finished",
            payload={
                "event": "run_finished",
                "total_experiments": total_experiments,
                "completed_experiments": completed_experiments,
                "best_experiment_name": best_row.get("experiment_name"),
                "recommendation": recommendation,
                "run_datetime": run_datetime,
            },
        )

        split_info_payload = dict(split_plan.split_info)
        split_info_payload["scorer"] = str(self.config.scorer)
        split_info_payload["configured_group_columns"] = list(group_cols)
        split_info_payload["preprocessing"] = dict(preprocessing_stats)
        split_info_payload["group_profile"] = {
            "group_columns": list(group_cols),
            "unique_groups_per_column": unique_groups_per_column,
            "unique_group_combinations": unique_group_combinations,
            "min_group_size": int(self.config.min_group_size),
        }
        split_info_payload["run_datetime"] = run_datetime
        warning_details = self._build_warning_details(warnings=warnings, run_datetime=run_datetime)
        return GroupMLResult(
            leaderboard=leaderboard,
            recommendation=recommendation,
            warnings=warnings,
            best_experiment=best_row,
            baseline_experiment=baseline_row,
            split_info=split_info_payload,
            raw_report=raw_report,
            all_runs=all_rows.reset_index(drop=True),
            warning_details=warning_details,
        )

    def _is_better_experiment_row(
        self,
        candidate: dict[str, Any],
        current_best: dict[str, Any] | None,
    ) -> bool:
        if current_best is None:
            return True
        candidate_cv = float(candidate.get("cv_mean", np.nan))
        current_cv = float(current_best.get("cv_mean", np.nan))
        if np.isnan(current_cv) and not np.isnan(candidate_cv):
            return True
        if np.isnan(candidate_cv) and not np.isnan(current_cv):
            return False
        if self._cv_prefers_lower():
            if candidate_cv < current_cv:
                return True
            if candidate_cv > current_cv:
                return False
        else:
            if candidate_cv > current_cv:
                return True
            if candidate_cv < current_cv:
                return False
        candidate_name = str(candidate.get("experiment_name", ""))
        current_name = str(current_best.get("experiment_name", ""))
        return candidate_name < current_name

    def _build_warning_details(self, warnings: list[str], run_datetime: str) -> pd.DataFrame:
        rows: list[dict[str, str]] = []
        for warning in warnings:
            run_experiment = ""
            match = re.search(r"(?:CV|Test) failure in ([^/]+)/(.+?) \(", str(warning))
            if match:
                run_experiment = f"{match.group(1)}:{match.group(2)}"
            else:
                model_match = re.search(r"Model warning in ([^/]+)/(.+?) \(", str(warning))
                if model_match:
                    run_experiment = f"{model_match.group(1)}:{model_match.group(2)}"
                else:
                    raw_match = re.search(r"Raw report .* failure .* for (.+?) \(", str(warning))
                    if raw_match:
                        run_experiment = str(raw_match.group(1))
            rows.append(
                {
                    "warning_datetime": run_datetime,
                    "run_datetime": run_datetime,
                    "run_experiment": run_experiment,
                    "warning": str(warning),
                }
            )
        return pd.DataFrame(
            rows,
            columns=["warning_datetime", "run_datetime", "run_experiment", "warning"],
        )

    def _is_rmse_like_scorer(self) -> bool:
        if not isinstance(self.config.scorer, str):
            return False
        scorer_name = self.config.scorer.strip().lower()
        return scorer_name in {"rmse", "neg_root_mean_squared_error", "root_mean_squared_error"}

    def _cv_prefers_lower(self) -> bool:
        return self._is_rmse_like_scorer()

    def _normalize_score_for_reporting(self, score: float) -> float:
        value = float(score)
        if self._is_rmse_like_scorer():
            return abs(value)
        return value

    def _is_unstable_score(
        self,
        score: float,
        y_ref: pd.Series,
        task: str,
    ) -> bool:
        if not np.isfinite(score):
            return True
        if task != "regression" or not self._is_rmse_like_scorer():
            return False
        y_num = pd.to_numeric(y_ref, errors="coerce").to_numpy(dtype=float)
        finite = y_num[np.isfinite(y_num)]
        if finite.size == 0:
            target_scale = 1.0
        else:
            p95 = float(np.nanpercentile(np.abs(finite), 95))
            std = float(np.nanstd(finite))
            target_scale = max(1.0, p95, std)
        # RMSE-like score magnitude exploding far beyond target scale usually indicates divergence.
        rmse_abs_cap = max(1e6, target_scale * 1e6)
        return abs(score) > rmse_abs_cap

    def _build_raw_report(
        self,
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
    ) -> pd.DataFrame:
        n_rows = len(data)
        rows_to_render = best_rows_by_method or [best_row]
        reports_by_method: list[tuple[str, dict[str, Any], pd.DataFrame]] = []
        for row in rows_to_render:
            token = self._method_token(row)
            method_report = self._build_single_experiment_prediction_report(
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
        primary_token = self._method_token(best_row)
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
        for token, row, method_report in reports_by_method:
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
            base_label = self._comparison_label(row)
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
        aliased_error_columns = [c for c in report.columns if c.startswith("error_")]
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
                list(self.config.split_group_columns or [])
                + ([self.config.split_group_column] if self.config.split_group_column else [])
                + ([self.config.split_stratify_column] if self.config.split_stratify_column else [])
                + ([self.config.split_date_column] if self.config.split_date_column else [])
                + group_cols
            )
        )
        identity_columns = [c for c in identity_columns if c in data.columns]
        sample_columns = [c for c in data.columns if c != self.config.target][: self.config.raw_report_max_columns]
        sample_columns = [c for c in sample_columns if c not in identity_columns]

        ordered_columns = (
            ["row_index"]
            + identity_columns
            + sample_columns
            + ["split_assignment", "actual"]
        )
        extra_columns = [c for c in report.columns if c not in ordered_columns]
        return report.join(data[identity_columns + sample_columns]).loc[:, ordered_columns + extra_columns]

    def _build_single_experiment_prediction_report(
        self,
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
    ) -> pd.DataFrame:
        estimator = self._build_best_estimator(
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
                with self._lib_warning_filter():
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
            with self._lib_warning_filter():
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

    def _pick_best_rows_by_method(self, leaderboard: pd.DataFrame) -> list[dict[str, Any]]:
        if leaderboard.empty:
            return []
        ordered = leaderboard.sort_values(by=["cv_mean"], ascending=self._cv_prefers_lower()).copy()
        best = ordered.drop_duplicates(subset=["mode"], keep="first")
        mode_order = ["full", "group_as_features", "group_split", "group_permutations", "rule_split"]
        order_map = {mode: idx for idx, mode in enumerate(mode_order)}
        best["__mode_order__"] = best["mode"].map(lambda mode: order_map.get(str(mode), 999))
        best = best.sort_values(
            by=["__mode_order__", "cv_mean"],
            ascending=[True, self._cv_prefers_lower()],
        )
        best = best.drop(columns=["__mode_order__"])
        return best.to_dict(orient="records")

    def _method_token(self, row: dict[str, Any]) -> str:
        mode = str(row.get("mode", "unknown")).strip().lower()
        variant = str(row.get("variant", "default")).strip().lower()
        token = f"{mode}__{variant}"
        token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
        if not token:
            return "method"
        return token[:80]

    def _method_type_for_mode(self, mode: str) -> str:
        lowered = str(mode).strip().lower()
        if lowered == "group_as_features":
            return "one_hot_group_features"
        if lowered == "group_split":
            return "per_group_models"
        if lowered == "group_permutations":
            return "per_group_models_group_combos"
        if lowered == "rule_split":
            return "rule_based_split"
        if lowered == "full":
            return "no_group_awareness"
        return lowered or "method"

    def _comparison_label(self, row: dict[str, Any]) -> str:
        mode = str(row.get("mode", "")).strip().lower()
        return self._method_type_for_mode(mode)

    def _build_best_estimator(
        self,
        best_row: dict[str, Any],
        task: str,
        models: dict[str, Any],
        selectors: dict[str, Any],
        feature_cols: list[str],
        group_cols: list[str],
        parsed_rules: list[Any],
        X_train: pd.DataFrame,
    ) -> Any:
        mode = str(best_row.get("mode"))
        variant = str(best_row.get("variant"))
        model_name = str(best_row.get("model"))
        selector_name = str(best_row.get("selector"))

        if model_name not in models:
            raise ValueError(f"Best experiment model '{model_name}' not found in configured models.")
        if selector_name not in selectors:
            raise ValueError(f"Best experiment selector '{selector_name}' not found in configured selectors.")

        if mode == "group_as_features":
            selected_feature_cols = list(feature_cols)
        else:
            selected_feature_cols = [c for c in feature_cols if c not in group_cols]

        selector = build_selector(selectors[selector_name], task, self.config.random_state)
        if mode == "group_as_features":
            base = self._build_group_as_features_pipeline(
                X_ref=X_train,
                feature_cols=selected_feature_cols,
                group_cols=group_cols,
                selector=selector,
                model=models[model_name],
            )
        else:
            preprocessor = build_preprocessor(X_train, selected_feature_cols, self.config.scale_numeric)
            steps = [("preprocess", preprocessor)]
            if selector != "passthrough":
                steps.append(("select", selector))
            steps.append(("model", models[model_name]))
            base = Pipeline(steps=steps)

        if mode in {"full", "group_as_features"}:
            return base
        if mode in {"group_split", "group_permutations"}:
            split_columns = tuple(c for c in variant.split("+") if c)
            if not split_columns:
                raise ValueError(f"Could not resolve split columns from best variant '{variant}'.")
            if task == "classification":
                return GroupSplitClassifier(
                    base_estimator=base,
                    split_columns=split_columns,
                    min_group_size=self.config.min_group_size,
                    task=task,
                )
            return GroupSplitRegressor(
                base_estimator=base,
                split_columns=split_columns,
                min_group_size=self.config.min_group_size,
                task=task,
            )
        if mode == "rule_split":
            if task == "classification":
                return RuleSplitClassifier(
                    base_estimator=base,
                    rules=parsed_rules,
                    min_group_size=self.config.min_group_size,
                    task=task,
                )
            return RuleSplitRegressor(
                base_estimator=base,
                rules=parsed_rules,
                min_group_size=self.config.min_group_size,
                task=task,
            )
        raise ValueError(f"Unsupported best experiment mode '{mode}'.")

    def _preprocess_base_dataset(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        group_cols: list[str],
        rule_columns: list[str],
        split_group_columns: list[str],
        split_date_column: str | None,
        split_stratify_column: str | None,
        task: str,
        warnings: list[str],
    ) -> tuple[pd.DataFrame, list[str], str, dict[str, Any]]:
        stats: dict[str, Any] = {
            "rows_initial": int(len(data)),
            "rows_after_target_filters": int(len(data)),
            "rows_after_dropna": int(len(data)),
            "rows_final": int(len(data)),
            "rows_dropped_min_target": 0,
            "rows_dropped_max_target": 0,
            "rows_dropped_required_na": 0,
            "columns_initial_features": int(len(feature_cols)),
            "columns_removed_static": 0,
            "columns_final_features": int(len(feature_cols)),
            "dropna_required_columns_count": 0,
            "dropna_required_columns_with_na_count": 0,
            "dropna_required_columns_with_na_preview": [],
        }
        if task == "regression":
            if (self.config.min_target is not None or self.config.max_target is not None) and not pd.api.types.is_numeric_dtype(
                data[self.config.target]
            ):
                raise ValueError("min_target/max_target require a numeric regression target.")
            if self.config.min_target is not None:
                before = len(data)
                data = data[data[self.config.target] >= self.config.min_target]
                dropped = before - len(data)
                stats["rows_dropped_min_target"] = int(dropped)
                if dropped:
                    warnings.append(
                        f"Base preprocessing dropped {dropped} rows below min_target={self.config.min_target}."
                    )
            if self.config.max_target is not None:
                before = len(data)
                data = data[data[self.config.target] <= self.config.max_target]
                dropped = before - len(data)
                stats["rows_dropped_max_target"] = int(dropped)
                if dropped:
                    warnings.append(
                        f"Base preprocessing dropped {dropped} rows above max_target={self.config.max_target}."
                    )
        stats["rows_after_target_filters"] = int(len(data))

        protected = set(group_cols) | set(rule_columns) | set(split_group_columns)
        if split_date_column:
            protected.add(split_date_column)
        if split_stratify_column:
            protected.add(split_stratify_column)
        if self.config.drop_static_base_features:
            static_features = [c for c in feature_cols if c not in protected and data[c].nunique(dropna=False) <= 1]
            if static_features:
                data = data.drop(columns=static_features)
                feature_cols = [c for c in feature_cols if c not in static_features]
                stats["columns_removed_static"] = int(len(static_features))
                preview = static_features[:10]
                preview_text = str(preview)
                if len(static_features) > 10:
                    preview_text = f"{preview} ... (+{len(static_features) - 10} more)"
                warnings.append(
                    "Base preprocessing removed "
                    f"{len(static_features)} static feature columns: {preview_text}"
                )
        stats["columns_final_features"] = int(len(feature_cols))

        if self.config.dropna_base_rows:
            required_columns = sorted(
                set(
                    [self.config.target]
                    + feature_cols
                    + group_cols
                    + rule_columns
                    + split_group_columns
                    + ([split_date_column] if split_date_column else [])
                    + ([split_stratify_column] if split_stratify_column else [])
                )
            )
            stats["dropna_required_columns_count"] = int(len(required_columns))
            if required_columns:
                na_counts = data[required_columns].isna().sum()
                na_columns = [str(col) for col in na_counts[na_counts > 0].index.tolist()]
                stats["dropna_required_columns_with_na_count"] = int(len(na_columns))
                stats["dropna_required_columns_with_na_preview"] = na_columns[:10]
            before = len(data)
            data = data.dropna(subset=required_columns)
            dropped = before - len(data)
            stats["rows_dropped_required_na"] = int(dropped)
            if dropped:
                warnings.append(
                    "Base preprocessing dropped "
                    f"{dropped} rows containing NaNs in required columns "
                    f"(required_cols={len(required_columns)}, cols_with_na={stats['dropna_required_columns_with_na_count']})."
                )
        stats["rows_after_dropna"] = int(len(data))
        stats["rows_final"] = int(len(data))

        if not feature_cols:
            raise ValueError("No feature columns left after base preprocessing.")
        if data.empty:
            raise ValueError(
                "No rows left after base preprocessing. "
                f"rows_initial={stats['rows_initial']}, "
                f"dropped_min_target={stats['rows_dropped_min_target']}, "
                f"dropped_max_target={stats['rows_dropped_max_target']}, "
                f"dropped_required_na={stats['rows_dropped_required_na']}, "
                f"removed_static_columns={stats['columns_removed_static']}. "
                "Consider narrowing --features, using --keep-nans, or relaxing target range filters."
            )

        for col in group_cols:
            if col not in feature_cols:
                feature_cols.append(col)

        stats["columns_final_features"] = int(len(feature_cols))
        task = infer_task(data[self.config.target], self.config.task)
        return data, feature_cols, task, stats

    def _apply_comparability_row_filter(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        group_cols: list[str],
        warnings: list[str],
    ) -> tuple[pd.DataFrame, list[str], int]:
        modes = set(default_experiment_names(self.config))
        uses_group_split = bool(group_cols) and (
            ("group_split" in modes) or ("group_permutations" in modes and len(group_cols) > 1)
        )
        if not uses_group_split:
            return data, feature_cols, 0

        by: Any = group_cols[0] if len(group_cols) == 1 else group_cols
        group_sizes = data.groupby(by, dropna=False).size()
        if group_sizes.empty:
            return data, feature_cols, 0
        valid_groups = set(group_sizes[group_sizes >= int(self.config.min_group_size)].index.tolist())
        if len(valid_groups) == len(group_sizes):
            return data, feature_cols, 0

        if len(group_cols) == 1:
            valid_mask = data[group_cols[0]].isin(valid_groups).to_numpy()
        else:
            row_keys = data[group_cols].apply(lambda row: tuple(row.values.tolist()), axis=1)
            valid_mask = row_keys.isin(valid_groups).to_numpy()
        dropped = int((~valid_mask).sum())
        if dropped <= 0:
            return data, feature_cols, 0
        filtered = data.loc[valid_mask].copy()
        warnings.append(
            "Dropped "
            f"{dropped} row(s) with group combination size < min_group_size={self.config.min_group_size} "
            "to keep strategy comparisons on the same usable rows."
        )
        if filtered.empty:
            raise ValueError("No rows left after group comparability filtering.")
        return filtered, feature_cols, dropped

    def _resolve_split_defaults(
        self,
        data: pd.DataFrame,
        group_cols: list[str],
        split_group_columns: list[str],
        split_date_column: str | None,
        split_stratify_column: str | None,
        warnings: list[str],
    ) -> tuple[str | None, list[str], str]:
        resolved_split_stratify = split_stratify_column
        resolved_test_split_strategy = self.config.test_split_strategy
        can_auto_stratify = (
            resolved_split_stratify is None
            and isinstance(self.config.cv, int)
            and split_date_column is None
            and self.config.test_splitter is None
        )
        stratify_source_columns = list(split_group_columns or group_cols)
        if can_auto_stratify and stratify_source_columns:
            auto_col = "__groupml_auto_stratify__"
            if len(stratify_source_columns) == 1:
                data[auto_col] = data[stratify_source_columns[0]].astype(str)
            else:
                data[auto_col] = data[stratify_source_columns].astype(str).agg("||".join, axis=1)
            resolved_split_stratify = auto_col
            if resolved_test_split_strategy == "last_rows":
                resolved_test_split_strategy = "random"
            warnings.append(
                "Auto-enabled stratification using group columns "
                f"{stratify_source_columns} for holdout and CV because no split_stratify_column/cv strategy was set."
            )
        return resolved_split_stratify, split_group_columns, resolved_test_split_strategy

    def _emit_callbacks(
        self,
        callbacks: list[Callable[[dict[str, Any]], None]],
        warnings: list[str],
        event: str,
        payload: dict[str, Any],
    ) -> None:
        for idx, callback in enumerate(callbacks):
            try:
                callback(payload)
            except Exception as exc:
                warnings.append(f"Callback failure in {event} callback[{idx}]: {exc}")

    def _make_score_callable(self, scorer: str | Callable[..., float]) -> Callable[[Any, pd.DataFrame, pd.Series], float]:
        if isinstance(scorer, str):
            if scorer.lower() == "rmse":
                def _rmse_score(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
                    y_pred = estimator.predict(X)
                    raw = float(np.sqrt(mean_squared_error(y, y_pred)))
                    return self._normalize_score_for_reporting(raw)

                return _rmse_score
            scorer_obj = get_scorer(scorer)

            def _score(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
                raw = float(scorer_obj(estimator, X, y))
                return self._normalize_score_for_reporting(raw)

            return _score

        def _score(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
            try:
                raw = float(scorer(estimator, X, y))
            except TypeError:
                y_pred = estimator.predict(X)
                raw = float(scorer(y, y_pred))
            return self._normalize_score_for_reporting(raw)

        return _score

    def _run_flat_mode(
        self,
        mode: str,
        variant: str,
        task: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_cols: list[str],
        group_cols: list[str],
        models: dict[str, Any],
        selectors: dict[str, Any],
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        scorer: Callable[[Any, pd.DataFrame, pd.Series], float],
        warnings: list[str],
        run_datetime: str,
        on_experiment_completed: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        required_group_columns = list(group_cols) if mode == "group_as_features" else []
        for (model_name, model), (selector_name, selector_spec) in product(models.items(), selectors.items()):
            selector = build_selector(selector_spec, infer_task(y_train, self.config.task), self.config.random_state)
            if mode == "group_as_features":
                estimator = self._build_group_as_features_pipeline(
                    X_ref=X_train,
                    feature_cols=feature_cols,
                    group_cols=required_group_columns,
                    selector=selector,
                    model=model,
                )
            else:
                preprocessor = build_preprocessor(X_train, feature_cols, self.config.scale_numeric)
                steps = [("preprocess", preprocessor)]
                if selector != "passthrough":
                    steps.append(("select", selector))
                steps.append(("model", model))
                estimator = Pipeline(steps=steps)
            row = self._evaluate_estimator(
                estimator=estimator,
                mode=mode,
                variant=variant,
                model_name=model_name,
                selector_name=selector_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv_splits=cv_splits,
                scorer=scorer,
                task=task,
                warnings=warnings,
                run_datetime=run_datetime,
                required_group_columns=required_group_columns,
            )
            rows.append(row)
            if on_experiment_completed is not None:
                on_experiment_completed(row)
        return rows

    def _run_group_split_mode(
        self,
        mode: str,
        split_columns: tuple[str, ...],
        task: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_cols: list[str],
        models: dict[str, Any],
        selectors: dict[str, Any],
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        scorer: Callable[[Any, pd.DataFrame, pd.Series], float],
        warnings: list[str],
        run_datetime: str,
        on_experiment_completed: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        preprocessor = build_preprocessor(X_train, feature_cols, self.config.scale_numeric)
        for (model_name, model), (selector_name, selector_spec) in product(models.items(), selectors.items()):
            selector = build_selector(selector_spec, task, self.config.random_state)
            steps = [("preprocess", preprocessor)]
            if selector != "passthrough":
                steps.append(("select", selector))
            steps.append(("model", model))
            base = Pipeline(steps=steps)
            if task == "classification":
                estimator = GroupSplitClassifier(
                    base_estimator=base,
                    split_columns=split_columns,
                    min_group_size=self.config.min_group_size,
                    task=task,
                )
            else:
                estimator = GroupSplitRegressor(
                    base_estimator=base,
                    split_columns=split_columns,
                    min_group_size=self.config.min_group_size,
                    task=task,
                )
            row = self._evaluate_estimator(
                estimator=estimator,
                mode=mode,
                variant="+".join(split_columns),
                model_name=model_name,
                selector_name=selector_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv_splits=cv_splits,
                scorer=scorer,
                task=task,
                warnings=warnings,
                run_datetime=run_datetime,
                required_group_columns=[],
            )
            rows.append(row)
            if on_experiment_completed is not None:
                on_experiment_completed(row)
        return rows

    def _run_rule_split_mode(
        self,
        mode: str,
        parsed_rules: list[Any],
        task: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_cols: list[str],
        models: dict[str, Any],
        selectors: dict[str, Any],
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        scorer: Callable[[Any, pd.DataFrame, pd.Series], float],
        warnings: list[str],
        run_datetime: str,
        on_experiment_completed: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        preprocessor = build_preprocessor(X_train, feature_cols, self.config.scale_numeric)
        variant = " | ".join([r.label() for r in parsed_rules])
        for (model_name, model), (selector_name, selector_spec) in product(models.items(), selectors.items()):
            selector = build_selector(selector_spec, task, self.config.random_state)
            steps = [("preprocess", preprocessor)]
            if selector != "passthrough":
                steps.append(("select", selector))
            steps.append(("model", model))
            base = Pipeline(steps=steps)
            if task == "classification":
                estimator = RuleSplitClassifier(
                    base_estimator=base,
                    rules=parsed_rules,
                    min_group_size=self.config.min_group_size,
                    task=task,
                )
            else:
                estimator = RuleSplitRegressor(
                    base_estimator=base,
                    rules=parsed_rules,
                    min_group_size=self.config.min_group_size,
                    task=task,
                )
            row = self._evaluate_estimator(
                estimator=estimator,
                mode=mode,
                variant=variant,
                model_name=model_name,
                selector_name=selector_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv_splits=cv_splits,
                scorer=scorer,
                task=task,
                warnings=warnings,
                run_datetime=run_datetime,
                required_group_columns=[],
            )
            rows.append(row)
            if on_experiment_completed is not None:
                on_experiment_completed(row)
        return rows

    def _evaluate_estimator(
        self,
        estimator: Any,
        mode: str,
        variant: str,
        model_name: str,
        selector_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
        scorer: Callable[[Any, pd.DataFrame, pd.Series], float],
        task: str,
        warnings: list[str],
        run_datetime: str,
        required_group_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            fold_est = clone(estimator)
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            try:
                with self._lib_warning_filter():
                    fold_est.fit(X_tr, y_tr)
                    score = float(scorer(fold_est, X_val, y_val))
                if self._is_unstable_score(score, y_val, task):
                    warnings.append(
                        f"CV failure in {mode}/{variant} ({model_name}, {selector_name}) fold={fold_idx}: "
                        f"invalid or unstable score={score:.6g}; treated as unsuccessful."
                    )
                    continue
                scores.append(score)
            except Exception as exc:
                warnings.append(
                    f"CV failure in {mode}/{variant} ({model_name}, {selector_name}) fold={fold_idx}: {exc}"
                )
        fit_est = clone(estimator)
        test_score = np.nan
        try:
            with self._lib_warning_filter():
                fit_est.fit(X_train, y_train)
                test_candidate = float(scorer(fit_est, X_test, y_test))
            if self._is_unstable_score(test_candidate, y_test, task):
                warnings.append(
                    f"Test failure in {mode}/{variant} ({model_name}, {selector_name}): "
                    f"invalid or unstable score={test_candidate:.6g}; treated as unsuccessful."
                )
            else:
                test_score = test_candidate
            if hasattr(fit_est, "warnings_"):
                for item in list(getattr(fit_est, "warnings_")):
                    warnings.append(
                        f"Model warning in {mode}/{variant} ({model_name}, {selector_name}): {item}"
                    )
        except Exception as exc:
            warnings.append(f"Test failure in {mode}/{variant} ({model_name}, {selector_name}): {exc}")
        cv_mean = float(np.mean(scores)) if scores else np.nan
        cv_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan
        group_feature_usage = self._extract_group_feature_usage(
            estimator=fit_est,
            required_group_columns=required_group_columns or [],
        )
        return {
            "mode": mode,
            "method_type": self._method_type_for_mode(mode),
            "variant": variant,
            "experiment_name": f"{mode}:{variant}",
            "model": model_name,
            "selector": selector_name,
            "run_datetime": run_datetime,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_folds_ok": len(scores),
            "test_score": float(test_score) if not np.isnan(test_score) else np.nan,
            **group_feature_usage,
        }

    def _extract_group_feature_usage(
        self,
        estimator: Any,
        required_group_columns: list[str],
    ) -> dict[str, Any]:
        base = {
            "group_features_required_count": 0,
            "group_features_selected_count": 0,
            "group_features_forced_count": 0,
            "group_features_all_selected": np.nan,
            "group_features_selected": "",
            "group_features_forced": "",
        }
        if not required_group_columns:
            return base
        if not isinstance(estimator, Pipeline):
            return base
        features_step = estimator.named_steps.get("features")
        if not isinstance(features_step, ColumnTransformer):
            return base
        forced_group_transformer = features_step.named_transformers_.get("forced_group")
        forced_group_columns: list[str] = []
        for name, _, cols in getattr(features_step, "transformers_", []):
            if name == "forced_group":
                forced_group_columns = [str(c) for c in cols]
                break
        required_names: list[str] = []
        if forced_group_transformer is not None:
            if hasattr(forced_group_transformer, "get_feature_names_out"):
                try:
                    names = forced_group_transformer.get_feature_names_out(forced_group_columns)
                    required_names = [f"forced_group__{str(n)}" for n in names]
                except Exception:
                    required_names = [f"forced_group__{col}" for col in forced_group_columns]
            else:
                required_names = [f"forced_group__{col}" for col in forced_group_columns]
        selected_base = features_step.named_transformers_.get("selected_base")
        has_selector = isinstance(selected_base, Pipeline) and ("select" in selected_base.named_steps)
        if has_selector:
            selected_names: list[str] = []
            forced_names = required_names
        else:
            selected_names = required_names
            forced_names = []
        return {
            "group_features_required_count": len(required_names),
            "group_features_selected_count": len(selected_names),
            "group_features_forced_count": len(forced_names),
            "group_features_all_selected": len(forced_names) == 0,
            "group_features_selected": "|".join(selected_names),
            "group_features_forced": "|".join(forced_names),
        }

    def _build_group_as_features_pipeline(
        self,
        X_ref: pd.DataFrame,
        feature_cols: list[str],
        group_cols: list[str],
        selector: Any,
        model: Any,
    ) -> Pipeline:
        group_feature_cols = [c for c in feature_cols if c in group_cols]
        base_feature_cols = [c for c in feature_cols if c not in group_feature_cols]
        transformers: list[tuple[str, Any, list[str]]] = []
        if base_feature_cols:
            base_preprocessor = build_preprocessor(X_ref, base_feature_cols, self.config.scale_numeric)
            if selector != "passthrough":
                base_transformer: Any = Pipeline(
                    steps=[("preprocess", base_preprocessor), ("select", selector)]
                )
            else:
                base_transformer = base_preprocessor
            transformers.append(("selected_base", base_transformer, base_feature_cols))
        if group_feature_cols:
            group_preprocessor = build_preprocessor(X_ref, group_feature_cols, self.config.scale_numeric)
            transformers.append(("forced_group", group_preprocessor, group_feature_cols))
        if not transformers:
            raise ValueError("No feature columns available for group_as_features mode.")
        feature_union = ColumnTransformer(transformers=transformers, remainder="drop")
        return Pipeline(steps=[("features", feature_union), ("model", model)])

    def _pick_baseline(self, leaderboard: pd.DataFrame) -> dict[str, Any]:
        baseline = leaderboard[leaderboard["mode"] == "full"]
        if baseline.empty:
            baseline = leaderboard.head(1)
        return baseline.sort_values(by=["cv_mean"], ascending=self._cv_prefers_lower()).iloc[0].to_dict()

    def _recommend(
        self,
        best_row: dict[str, Any],
        baseline_row: dict[str, Any],
        warnings: list[str],
    ) -> str:
        best_name = best_row["experiment_name"]
        baseline_name = baseline_row["experiment_name"]
        if self._cv_prefers_lower():
            improvement = float(baseline_row["cv_mean"] - best_row["cv_mean"])
        else:
            improvement = float(best_row["cv_mean"] - baseline_row["cv_mean"])
        stable = (
            np.isfinite(best_row.get("cv_std", np.nan))
            and np.isfinite(baseline_row.get("cv_std", np.nan))
            and best_row["cv_std"] <= baseline_row["cv_std"] * 1.25
        )
        if best_name == baseline_name:
            return f"Use baseline ({baseline_name}); no better strategy found."
        if improvement < self.config.min_improvement:
            return (
                f"Keep baseline ({baseline_name}); best alternative ({best_name}) "
                f"improves CV by only {improvement:.5f}, below threshold {self.config.min_improvement:.5f}."
            )
        if not stable:
            warnings.append(
                f"Best strategy {best_name} has higher CV variance; validate with more data/folds."
            )
        if self._cv_prefers_lower():
            return (
                f"Use {best_name}. It reduces CV RMSE by {improvement:.5f} versus baseline "
                f"{baseline_name} with test RMSE {best_row['test_score']:.5f}."
            )
        return (
            f"Use {best_name}. It improves CV score by {improvement:.5f} versus baseline "
            f"{baseline_name} with test score {best_row['test_score']:.5f}."
        )


def compare_group_strategies(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str] | None = None,
    group_columns: list[str] | None = None,
    rule_splits: list[str] | None = None,
    callbacks: Iterable[Callable[[dict[str, Any]], None]] | None = None,
    **kwargs: Any,
) -> GroupMLResult:
    """Functional API wrapper around `GroupMLRunner`."""
    config = GroupMLConfig(
        target=target,
        feature_columns=feature_columns,
        group_columns=group_columns or [],
        rule_splits=rule_splits or [],
        **kwargs,
    )
    return GroupMLRunner(config).fit_evaluate(df, callbacks=callbacks)
