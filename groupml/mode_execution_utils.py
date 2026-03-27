"""Mode execution helpers used by GroupMLRunner orchestration."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .estimators import RuleSplitClassifier, RuleSplitRegressor
from .utils import build_preprocessor, build_selector


def run_flat_mode(
    *,
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
    config_task: str,
    random_state: int,
    kbest_features: int | str,
    scale_numeric: bool,
    evaluate_estimator: Callable[..., dict[str, Any]],
    build_group_as_features_pipeline: Callable[..., Pipeline],
    infer_task: Callable[[pd.Series, str], str],
    on_experiment_completed: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required_group_columns = list(group_cols) if mode == "group_as_features" else []
    for (model_name, model), (selector_name, selector_spec) in product(models.items(), selectors.items()):
        selector = build_selector(
            selector_spec,
            infer_task(y_train, config_task),
            random_state,
            kbest_features=kbest_features,
        )
        if mode == "group_as_features":
            estimator = build_group_as_features_pipeline(
                X_ref=X_train,
                feature_cols=feature_cols,
                group_cols=required_group_columns,
                selector=selector,
                model=model,
            )
        else:
            preprocessor = build_preprocessor(X_train, feature_cols, scale_numeric)
            steps = [("preprocess", preprocessor)]
            if selector != "passthrough":
                steps.append(("select", selector))
            steps.append(("model", model))
            estimator = Pipeline(steps=steps)
        row = evaluate_estimator(
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


def run_group_split_mode(
    *,
    mode: str,
    method_type: str,
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
    callbacks: list[Callable[[dict[str, Any]], None]] | None,
    per_group_rows: list[dict[str, Any]] | None,
    compare_shared_candidates: bool,
    evaluate_estimator: Callable[..., dict[str, Any]],
    emit_callbacks: Callable[..., None],
    build_group_split_candidate_estimators: Callable[..., dict[str, Any]],
    build_group_split_tuned_estimator: Callable[..., Any],
    build_group_split_progress_callback: Callable[..., Callable[[dict[str, Any]], None]],
    parse_group_selected_configs: Callable[[str], list[tuple[str, str]]],
    compute_group_test_metrics: Callable[..., dict[str, dict[str, Any]]],
    is_better_experiment_row: Callable[[dict[str, Any], dict[str, Any] | None], bool],
    on_experiment_completed: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    callback_list = list(callbacks or [])
    per_group_test_metrics: dict[str, dict[str, Any]] = {}
    per_group_cv_scores: dict[str, float] = {}
    per_group_sizes: dict[str, int] = {}
    variant = "+".join(split_columns)
    group_count = int(X_train.groupby(list(split_columns), dropna=False).ngroups)
    candidate_estimators = build_group_split_candidate_estimators(
        task=task,
        X_train=X_train,
        feature_cols=feature_cols,
        models=models,
        selectors=selectors,
    )
    emit_callbacks(
        callbacks=callback_list,
        warnings=warnings,
        event="group_split_variant_started",
        payload={
            "event": "group_split_variant_started",
            "mode": mode,
            "method_type": method_type,
            "variant": variant,
            "split_columns": list(split_columns),
            "group_count": group_count,
            "candidate_count": len(candidate_estimators),
            "run_datetime": run_datetime,
        },
    )
    candidate_labels = {
        f"{model_name}__{selector_name}": (model_name, selector_name)
        for (model_name, _), (selector_name, _) in product(models.items(), selectors.items())
    }
    emit_callbacks(
        callbacks=callback_list,
        warnings=warnings,
        event="group_split_optimized_search_started",
        payload={
            "event": "group_split_optimized_search_started",
            "mode": mode,
            "method_type": method_type,
            "variant": variant,
            "group_count": group_count,
            "candidate_count": len(candidate_estimators),
            "run_datetime": run_datetime,
        },
    )

    def _capture_group_progress(event: dict[str, Any]) -> None:
        if str(event.get("event", "")) != "group_tuning_group_finished":
            return
        group_key = str(event.get("group_key", ""))
        if not group_key:
            return
        per_group_sizes[group_key] = int(event.get("group_size", 0) or 0)
        try:
            score = float(event.get("best_score", np.nan))
        except (TypeError, ValueError):
            score = np.nan
        per_group_cv_scores[group_key] = score if np.isfinite(score) else np.nan

    estimator = build_group_split_tuned_estimator(
        task=task,
        split_columns=split_columns,
        X_train=X_train,
        feature_cols=feature_cols,
        models=models,
        selectors=selectors,
        scorer=scorer,
        prebuilt_candidates=candidate_estimators,
        progress_callback=build_group_split_progress_callback(
            callbacks=callback_list + [_capture_group_progress],
            warnings=warnings,
        ),
        progress_context={
            "mode": mode,
            "method_type": method_type,
            "variant": variant,
            "split_columns": list(split_columns),
            "run_datetime": run_datetime,
        },
    )
    row = evaluate_estimator(
        estimator=estimator,
        mode=mode,
        variant=variant,
        model_name="per_group_best",
        selector_name="per_group_best",
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
        on_fitted_estimator=lambda fitted_estimator: per_group_test_metrics.update(
            compute_group_test_metrics(
                estimator=fitted_estimator,
                split_columns=split_columns,
                X_test=X_test,
                y_test=y_test,
                task=task,
            )
        ),
    )
    selected_pairs = parse_group_selected_configs(str(row.get("group_selected_configs", "")))
    for idx, (group_key, selected_config) in enumerate(selected_pairs, start=1):
        group_test = per_group_test_metrics.get(str(group_key), {})
        has_group_test = bool(int(group_test.get("group_test_rows", 0) or 0) > 0)
        group_cv_mean = float(per_group_cv_scores.get(str(group_key), np.nan))
        has_group_cv = bool(np.isfinite(group_cv_mean))
        model_name_selected, selector_name_selected = str(selected_config), ""
        if "__" in str(selected_config):
            model_name_selected, selector_name_selected = str(selected_config).split("__", 1)
        if per_group_rows is not None:
            per_group_rows.append(
                {
                    "mode": mode,
                    "method_type": method_type,
                    "variant": variant,
                    "experiment_name": f"{mode}:{variant}",
                    "run_scope": "group",
                    "group_key": str(group_key),
                    "group_index": idx,
                    "group_total": len(selected_pairs),
                    "model": model_name_selected,
                    "selector": selector_name_selected,
                    "group_selected_config": str(selected_config),
                    "group_fallback_config": row.get("group_fallback_config", ""),
                    "group_train_rows": int(per_group_sizes.get(str(group_key), 0) or 0),
                    "group_test_rows": int(group_test.get("group_test_rows", 0) or 0),
                    "group_test_score": (
                        float(group_test.get("group_test_score"))
                        if has_group_test and np.isfinite(float(group_test.get("group_test_score", np.nan)))
                        else np.nan
                    ),
                    "group_test_metric": str(group_test.get("group_test_metric", "")) if has_group_test else "",
                    "split_comparable_with_global": True,
                    "split_consistency": "shared_holdout_and_cv_plan",
                    "cv_mean": group_cv_mean if has_group_cv else np.nan,
                    "cv_std": np.nan,
                    "cv_folds_ok": np.nan,
                    "test_score": (
                        float(group_test.get("group_test_score"))
                        if has_group_test and np.isfinite(float(group_test.get("group_test_score", np.nan)))
                        else np.nan
                    ),
                    "train_rows": int(per_group_sizes.get(str(group_key), 0) or 0),
                    "test_rows": int(group_test.get("group_test_rows", 0) or 0),
                    "run_datetime": run_datetime,
                    "run_status": "group_info",
                }
            )
        emit_callbacks(
            callbacks=callback_list,
            warnings=warnings,
            event="group_model_selected",
            payload={
                "event": "group_model_selected",
                "mode": mode,
                "method_type": method_type,
                "variant": variant,
                "group_key": group_key,
                "selected_config": selected_config,
                "group_index": idx,
                "group_total": len(selected_pairs),
                "fallback_config": row.get("group_fallback_config", ""),
                "group_train_rows": int(per_group_sizes.get(str(group_key), 0) or 0),
                "group_test_rows": int(group_test.get("group_test_rows", 0) or 0),
                "group_cv_mean": group_cv_mean if has_group_cv else np.nan,
                "group_test_score": (
                    float(group_test.get("group_test_score"))
                    if has_group_test and np.isfinite(float(group_test.get("group_test_score", np.nan)))
                    else np.nan
                ),
                "group_test_metric": str(group_test.get("group_test_metric", "")) if has_group_test else "",
                "split_comparable_with_global": True,
                "split_consistency": "shared_holdout_and_cv_plan",
                "run_datetime": run_datetime,
            },
        )
    rows.append(row)
    if on_experiment_completed is not None:
        on_experiment_completed(row)

    shared_rows: list[dict[str, Any]] = []
    if compare_shared_candidates:
        shared_total = len(candidate_estimators)
        emit_callbacks(
            callbacks=callback_list,
            warnings=warnings,
            event="group_split_shared_search_started",
            payload={
                "event": "group_split_shared_search_started",
                "mode": mode,
                "method_type": method_type,
                "variant": variant,
                "shared_total": shared_total,
                "run_datetime": run_datetime,
            },
        )
        for shared_index, candidate_key in enumerate(candidate_estimators.keys(), start=1):
            model_name, selector_name = candidate_labels.get(candidate_key, ("shared_model", "shared_selector"))
            shared_estimator = build_group_split_tuned_estimator(
                task=task,
                split_columns=split_columns,
                X_train=X_train,
                feature_cols=[],
                models={},
                selectors={},
                scorer=scorer,
                prebuilt_candidates={str(candidate_key): candidate_estimators[candidate_key]},
                progress_callback=None,
                progress_context=None,
            )
            shared_row = evaluate_estimator(
                estimator=shared_estimator,
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
            rows.append(shared_row)
            shared_rows.append(shared_row)
            emit_callbacks(
                callbacks=callback_list,
                warnings=warnings,
                event="group_split_shared_candidate_evaluated",
                payload={
                    "event": "group_split_shared_candidate_evaluated",
                    "mode": mode,
                    "method_type": method_type,
                    "variant": variant,
                    "shared_index": shared_index,
                    "shared_total": shared_total,
                    "model": model_name,
                    "selector": selector_name,
                    "cv_mean": shared_row.get("cv_mean"),
                    "test_score": shared_row.get("test_score"),
                    "run_datetime": run_datetime,
                },
            )
            if on_experiment_completed is not None:
                on_experiment_completed(shared_row)
    shared_best_row: dict[str, Any] | None = None
    for shared_row in shared_rows:
        cv_mean = float(shared_row.get("cv_mean", np.nan))
        cv_folds_ok = int(shared_row.get("cv_folds_ok", 0) or 0)
        if not np.isfinite(cv_mean) or cv_folds_ok <= 0:
            continue
        if is_better_experiment_row(shared_row, shared_best_row):
            shared_best_row = shared_row
    if shared_best_row is not None:
        emit_callbacks(
            callbacks=callback_list,
            warnings=warnings,
            event="group_split_shared_best",
            payload={
                "event": "group_split_shared_best",
                "mode": mode,
                "method_type": method_type,
                "variant": variant,
                "model": shared_best_row.get("model"),
                "selector": shared_best_row.get("selector"),
                "cv_mean": shared_best_row.get("cv_mean"),
                "test_score": shared_best_row.get("test_score"),
                "run_datetime": run_datetime,
            },
        )
    emit_callbacks(
        callbacks=callback_list,
        warnings=warnings,
        event="group_split_variant_finished",
        payload={
            "event": "group_split_variant_finished",
            "mode": mode,
            "method_type": method_type,
            "variant": variant,
            "split_columns": list(split_columns),
            "group_count": len(selected_pairs),
            "unique_selected_config_count": int(row.get("group_selected_config_count", 0) or 0),
            "fallback_config": row.get("group_fallback_config", ""),
            "shared_best_model": (shared_best_row or {}).get("model", ""),
            "shared_best_selector": (shared_best_row or {}).get("selector", ""),
            "shared_best_cv_mean": (shared_best_row or {}).get("cv_mean", np.nan),
            "shared_best_test_score": (shared_best_row or {}).get("test_score", np.nan),
            "cv_mean": row.get("cv_mean"),
            "test_score": row.get("test_score"),
            "run_datetime": run_datetime,
        },
    )
    return rows


def run_rule_split_mode(
    *,
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
    random_state: int,
    min_group_size: int,
    scale_numeric: bool,
    kbest_features: int | str,
    evaluate_estimator: Callable[..., dict[str, Any]],
    on_experiment_completed: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    preprocessor = build_preprocessor(X_train, feature_cols, scale_numeric)
    variant = " | ".join([r.label() for r in parsed_rules])
    for (model_name, model), (selector_name, selector_spec) in product(models.items(), selectors.items()):
        selector = build_selector(
            selector_spec,
            task,
            random_state,
            kbest_features=kbest_features,
        )
        steps = [("preprocess", preprocessor)]
        if selector != "passthrough":
            steps.append(("select", selector))
        steps.append(("model", model))
        base = Pipeline(steps=steps)
        if task == "classification":
            estimator = RuleSplitClassifier(
                base_estimator=base,
                rules=parsed_rules,
                min_group_size=min_group_size,
                task=task,
            )
        else:
            estimator = RuleSplitRegressor(
                base_estimator=base,
                rules=parsed_rules,
                min_group_size=min_group_size,
                task=task,
            )
        row = evaluate_estimator(
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
