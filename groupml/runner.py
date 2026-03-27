"""Main execution engine for groupml experiments."""

from __future__ import annotations

import warnings as py_warnings
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
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
from .functional import compare_group_strategies as compare_group_strategies_fn
from .group_split_utils import (
    build_group_split_candidate_estimators,
    build_group_split_progress_callback,
    build_group_split_tuned_estimator,
    parse_group_selected_configs,
)
from .mode_execution_utils import run_flat_mode, run_group_split_mode, run_rule_split_mode
from .mode_utils import comparison_label, method_token, method_type_for_mode, pick_best_rows_by_method
from .pipeline_utils import (
    build_group_as_features_pipeline,
    extract_group_config_usage,
    extract_group_feature_usage,
)
from .report_utils import build_raw_report as build_raw_report_utils
from .report_utils import build_single_experiment_prediction_report as build_single_experiment_prediction_report_utils
from .recommendation_utils import build_warning_details, pick_baseline, recommend
from .result import GroupMLResult
from .splitting import plan_splits
from .training_utils import evaluate_estimator as evaluate_estimator_utils
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

SKLEARN_PARALLEL_DELAYED_WARNING_RE = (
    r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*"
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
            py_warnings.filterwarnings(
                "ignore",
                message=SKLEARN_PARALLEL_DELAYED_WARNING_RE,
                category=UserWarning,
            )
            if level == "quiet":
                py_warnings.filterwarnings("ignore", category=ConvergenceWarning)
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
        requested_group_cols = list(self.config.group_columns)
        missing_group_cols = [c for c in requested_group_cols if c not in data.columns]
        if missing_group_cols:
            warnings.append(
                "Configured group column(s) not found and will be ignored: "
                f"{missing_group_cols}. Running non-group-aware strategies only."
            )
        group_cols = [c for c in requested_group_cols if c in data.columns]
        if requested_group_cols and not group_cols:
            warnings.append(
                "No valid group columns available after filtering missing columns. "
                "Running non-group-aware strategies only."
            )
        elif not requested_group_cols:
            warnings.append("No group columns configured. Running non-group-aware strategies only.")

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
        runs_per_mode = {
            "full": model_selector_runs,
            "group_as_features": model_selector_runs,
            "group_split": 1 + (model_selector_runs if self.config.group_split_compare_shared_candidates else 0),
            "group_permutations": 1 + (model_selector_runs if self.config.group_split_compare_shared_candidates else 0),
            "rule_split": model_selector_runs,
        }
        total_experiments = sum(
            int(planned_variants.get(mode, 0)) * int(runs_per_mode.get(mode, 0))
            for mode in planned_variants
        )

        cv_splits = split_plan.cv_splits
        scorer_callable = self._make_score_callable(self.config.scorer)

        rows: list[dict[str, Any]] = []
        per_group_rows: list[dict[str, Any]] = []
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
            per_variant_runs = int(runs_per_mode.get(mode, model_selector_runs))
            self._emit_callbacks(
                callback_list,
                warnings,
                event="mode_started",
                payload={
                    "event": "mode_started",
                    "mode": mode,
                    "planned_experiments": variant_count * per_variant_runs,
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
                # Avoid recomputing raw report on every best update; this is very expensive on larger runs.
                best_so_far_raw_report = pd.DataFrame()
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
                        callbacks=callback_list,
                        per_group_rows=per_group_rows,
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
                            callbacks=callback_list,
                            per_group_rows=per_group_rows,
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
        if "run_scope" not in all_rows.columns:
            all_rows["run_scope"] = "overall"
        else:
            all_rows["run_scope"] = all_rows["run_scope"].fillna("overall")
        if "group_key" not in all_rows.columns:
            all_rows["group_key"] = "full"
        else:
            all_rows["group_key"] = all_rows["group_key"].fillna("full")
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

        baseline_row = pick_baseline(leaderboard=leaderboard, prefers_lower=self._cv_prefers_lower())
        best_row = leaderboard.iloc[0].to_dict()
        recommendation = recommend(
            best_row=best_row,
            baseline_row=baseline_row,
            prefers_lower=self._cv_prefers_lower(),
            min_improvement=self.config.min_improvement,
            warnings=warnings,
        )
        best_rows_by_method = pick_best_rows_by_method(leaderboard, prefers_lower=self._cv_prefers_lower())
        raw_report = pd.DataFrame()
        if self.config.raw_report_enabled:
            _build_single_report = lambda **kwargs: build_single_experiment_prediction_report_utils(
                **kwargs,
                build_best_estimator=self._build_best_estimator,
                warning_filter=self._lib_warning_filter,
            )
            _build_raw = lambda: build_raw_report_utils(
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
                split_group_columns=list(self.config.split_group_columns or []),
                split_group_column=self.config.split_group_column,
                split_stratify_column=self.config.split_stratify_column,
                split_date_column=self.config.split_date_column,
                target=self.config.target,
                raw_report_max_columns=self.config.raw_report_max_columns,
                method_token=method_token,
                comparison_label=comparison_label,
                build_single_report=_build_single_report,
            )
            try:
                if not best_so_far_raw_report.empty and best_so_far_row is not None:
                    if best_so_far_row.get("experiment_name") == best_row.get("experiment_name"):
                        raw_report = _build_raw()
                    else:
                        raw_report = _build_raw()
                else:
                    raw_report = _build_raw()
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
        warning_details = build_warning_details(warnings=warnings, run_datetime=run_datetime)
        all_runs = all_rows.reset_index(drop=True)
        if per_group_rows:
            group_runs_df = pd.DataFrame(per_group_rows)
            all_runs = pd.concat([all_runs, group_runs_df], ignore_index=True, sort=False)
        return GroupMLResult(
            leaderboard=leaderboard,
            recommendation=recommendation,
            warnings=warnings,
            best_experiment=best_row,
            baseline_experiment=baseline_row,
            split_info=split_info_payload,
            raw_report=raw_report,
            all_runs=all_runs,
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
        rmse_abs_cap = max(1e6, target_scale * 1e6)
        return abs(score) > rmse_abs_cap


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

        if mode == "group_as_features":
            selected_feature_cols = list(feature_cols)
        else:
            selected_feature_cols = [c for c in feature_cols if c not in group_cols]

        if mode in {"group_split", "group_permutations"} and (
            model_name == "per_group_best" or selector_name == "per_group_best"
        ):
            split_columns = tuple(c for c in variant.split("+") if c)
            if not split_columns:
                raise ValueError(f"Could not resolve split columns from best variant '{variant}'.")
            return self._build_group_split_tuned_estimator(
                task=task,
                split_columns=split_columns,
                X_train=X_train,
                feature_cols=selected_feature_cols,
                models=models,
                selectors=selectors,
                scorer=self._make_score_callable(self.config.scorer),
            )

        if model_name not in models:
            raise ValueError(f"Best experiment model '{model_name}' not found in configured models.")
        if selector_name not in selectors:
            raise ValueError(f"Best experiment selector '{selector_name}' not found in configured selectors.")

        selector = build_selector(
            selectors[selector_name],
            task,
            self.config.random_state,
            kbest_features=self.config.kbest_features,
        )
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
        return run_flat_mode(
            mode=mode,
            variant=variant,
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
            scorer=scorer,
            warnings=warnings,
            run_datetime=run_datetime,
            config_task=self.config.task,
            random_state=self.config.random_state,
            kbest_features=self.config.kbest_features,
            scale_numeric=self.config.scale_numeric,
            evaluate_estimator=self._evaluate_estimator,
            build_group_as_features_pipeline=self._build_group_as_features_pipeline,
            infer_task=infer_task,
            on_experiment_completed=on_experiment_completed,
        )

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
        callbacks: list[Callable[[dict[str, Any]], None]] | None = None,
        per_group_rows: list[dict[str, Any]] | None = None,
        on_experiment_completed: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        return run_group_split_mode(
            mode=mode,
            method_type=method_type_for_mode(mode),
            split_columns=split_columns,
            task=task,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            models=models,
            selectors=selectors,
            cv_splits=cv_splits,
            scorer=scorer,
            warnings=warnings,
            run_datetime=run_datetime,
            callbacks=callbacks,
            per_group_rows=per_group_rows,
            compare_shared_candidates=self.config.group_split_compare_shared_candidates,
            evaluate_estimator=self._evaluate_estimator,
            emit_callbacks=self._emit_callbacks,
            build_group_split_candidate_estimators=self._build_group_split_candidate_estimators,
            build_group_split_tuned_estimator=self._build_group_split_tuned_estimator,
            build_group_split_progress_callback=self._build_group_split_progress_callback,
            parse_group_selected_configs=self._parse_group_selected_configs,
            compute_group_test_metrics=self._compute_group_test_metrics,
            is_better_experiment_row=self._is_better_experiment_row,
            on_experiment_completed=on_experiment_completed,
        )

    def _build_group_split_candidate_estimators(
        self,
        task: str,
        X_train: pd.DataFrame,
        feature_cols: list[str],
        models: dict[str, Any],
        selectors: dict[str, Any],
    ) -> dict[str, Any]:
        return build_group_split_candidate_estimators(
            task=task,
            X_train=X_train,
            feature_cols=feature_cols,
            models=models,
            selectors=selectors,
            scale_numeric=self.config.scale_numeric,
            random_state=self.config.random_state,
            kbest_features=self.config.kbest_features,
        )

    def _build_group_split_tuned_estimator(
        self,
        task: str,
        split_columns: tuple[str, ...],
        X_train: pd.DataFrame,
        feature_cols: list[str],
        models: dict[str, Any],
        selectors: dict[str, Any],
        scorer: Callable[[Any, pd.DataFrame, pd.Series], float],
        prebuilt_candidates: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_context: dict[str, Any] | None = None,
    ) -> Any:
        candidates = prebuilt_candidates or build_group_split_candidate_estimators(
            task=task,
            X_train=X_train,
            feature_cols=feature_cols,
            models=models,
            selectors=selectors,
            scale_numeric=self.config.scale_numeric,
            random_state=self.config.random_state,
            kbest_features=self.config.kbest_features,
        )
        estimator = build_group_split_tuned_estimator(
            task=task,
            split_columns=split_columns,
            scorer=scorer,
            random_state=self.config.random_state,
            min_group_size=self.config.min_group_size,
            prefers_lower=self._cv_prefers_lower(),
            cv=self.config.cv,
            prebuilt_candidates=candidates,
            progress_callback=progress_callback,
            progress_context=progress_context,
        )
        if hasattr(estimator, "set_params"):
            try:
                estimator.set_params(tune_candidates_with_cv=self.config.group_split_tune_candidates_with_cv)
            except Exception:
                pass
        return estimator

    def _build_group_split_progress_callback(
        self,
        callbacks: list[Callable[[dict[str, Any]], None]],
        warnings: list[str],
    ) -> Callable[[dict[str, Any]], None]:
        return build_group_split_progress_callback(
            emit_callback=lambda event_name, payload: self._emit_callbacks(
                callbacks=callbacks,
                warnings=warnings,
                event=event_name,
                payload=payload,
            )
        )

    def _parse_group_selected_configs(self, serialized: str) -> list[tuple[str, str]]:
        return parse_group_selected_configs(serialized)

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
        return run_rule_split_mode(
            mode=mode,
            parsed_rules=parsed_rules,
            task=task,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            models=models,
            selectors=selectors,
            cv_splits=cv_splits,
            scorer=scorer,
            warnings=warnings,
            run_datetime=run_datetime,
            random_state=self.config.random_state,
            min_group_size=self.config.min_group_size,
            scale_numeric=self.config.scale_numeric,
            kbest_features=self.config.kbest_features,
            evaluate_estimator=self._evaluate_estimator,
            on_experiment_completed=on_experiment_completed,
        )

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
        on_fitted_estimator: Callable[[Any], None] | None = None,
    ) -> dict[str, Any]:
        return evaluate_estimator_utils(
            estimator=estimator,
            mode=mode,
            method_type=method_type_for_mode(mode),
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
            warning_filter=self._lib_warning_filter,
            is_unstable_score=self._is_unstable_score,
            extract_group_feature_usage=lambda fit_est, required_cols: self._extract_group_feature_usage(
                estimator=fit_est,
                required_group_columns=required_cols,
            ),
            extract_group_config_usage=self._extract_group_config_usage,
            on_fitted_estimator=on_fitted_estimator,
        )

    def _compute_group_test_metrics(
        self,
        estimator: Any,
        split_columns: tuple[str, ...],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task: str,
    ) -> dict[str, dict[str, Any]]:
        if X_test.empty or not split_columns:
            return {}
        try:
            y_pred = estimator.predict(X_test)
        except Exception:
            return {}

        y_true = pd.Series(y_test, index=X_test.index)
        pred_series = pd.Series(y_pred, index=X_test.index)
        out: dict[str, dict[str, Any]] = {}
        by: Any = split_columns[0] if len(split_columns) == 1 else list(split_columns)
        grouped = X_test.groupby(by, dropna=False).groups
        for key, idx in grouped.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            group_key = "|".join(str(v) for v in key_tuple)
            idx_labels = list(idx)
            true_values = y_true.loc[idx_labels]
            pred_values = pred_series.loc[idx_labels]
            n_rows = int(len(true_values))
            if n_rows == 0:
                continue
            if task == "classification":
                mask = true_values.notna() & pred_values.notna()
                score = float((pred_values[mask] == true_values[mask]).mean()) if mask.any() else np.nan
                metric_name = "accuracy"
            else:
                true_num = pd.to_numeric(true_values, errors="coerce")
                pred_num = pd.to_numeric(pred_values, errors="coerce")
                mask = true_num.notna() & pred_num.notna()
                if mask.any():
                    err = pred_num[mask] - true_num[mask]
                    score = float(np.sqrt(np.mean(np.square(err))))
                else:
                    score = np.nan
                metric_name = "rmse"
            out[group_key] = {
                "group_test_rows": n_rows,
                "group_test_score": score if np.isfinite(score) else np.nan,
                "group_test_metric": metric_name,
            }
        return out

    def _extract_group_feature_usage(
        self,
        estimator: Any,
        required_group_columns: list[str],
    ) -> dict[str, Any]:
        return extract_group_feature_usage(estimator, required_group_columns)

    def _extract_group_config_usage(self, estimator: Any) -> dict[str, Any]:
        return extract_group_config_usage(estimator)

    def _build_group_as_features_pipeline(
        self,
        X_ref: pd.DataFrame,
        feature_cols: list[str],
        group_cols: list[str],
        selector: Any,
        model: Any,
    ) -> Pipeline:
        return build_group_as_features_pipeline(
            X_ref=X_ref,
            feature_cols=feature_cols,
            group_cols=group_cols,
            selector=selector,
            model=model,
            scale_numeric=self.config.scale_numeric,
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
    return compare_group_strategies_fn(
        df=df,
        target=target,
        feature_columns=feature_columns,
        group_columns=group_columns,
        rule_splits=rule_splits,
        callbacks=callbacks,
        **kwargs,
    )

