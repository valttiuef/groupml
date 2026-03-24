"""Main execution engine for groupml experiments."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
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

    def fit_evaluate(
        self,
        df: pd.DataFrame,
        callbacks: Iterable[Callable[[dict[str, Any]], None]] | None = None,
    ) -> GroupMLResult:
        """Run configured experiments and return structured result."""
        ensure_columns_exist(df, [self.config.target], "target")
        warnings: list[str] = []
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

        cv_group_columns = list(self.config.cv_group_columns or [])
        ensure_columns_exist(data, cv_group_columns, "cv_group")

        task = infer_task(data[self.config.target], self.config.task)
        data, feature_cols, task = self._preprocess_base_dataset(
            data=data,
            feature_cols=feature_cols,
            group_cols=group_cols,
            rule_columns=rule_columns,
            cv_group_columns=cv_group_columns,
            task=task,
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
            cv_params=self.config.cv_params,
            cv_group_columns=cv_group_columns,
            fallback_cv_group_columns=group_cols,
            test_splitter=self.config.test_splitter,
            include_indices=self.config.include_split_indices,
        )
        X_train, X_test = X.iloc[split_plan.train_indices], X.iloc[split_plan.test_indices]
        y_train, y_test = y.iloc[split_plan.train_indices], y.iloc[split_plan.test_indices]

        models = normalize_models(self.config.models, task, self.config.random_state)
        selectors = normalize_selectors(self.config.feature_selectors)
        model_selector_runs = len(models) * len(selectors)
        group_permutation_splits = group_column_permutations(group_cols) if group_cols else []
        planned_variants = {
            "full": 1 if "full" in default_experiment_names(self.config) else 0,
            "group_as_features": 1 if "group_as_features" in default_experiment_names(self.config) and group_cols else 0,
            "group_split": 1 if "group_split" in default_experiment_names(self.config) and group_cols else 0,
            "group_permutations": (
                len(group_permutation_splits)
                if "group_permutations" in default_experiment_names(self.config) and group_cols
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

        self._emit_callbacks(
            callback_list,
            warnings,
            event="run_started",
            payload={
                "event": "run_started",
                "modes": modes,
                "total_experiments": total_experiments,
                "task": task,
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
            nonlocal completed_experiments
            completed_experiments += 1
            self._emit_callbacks(
                callback_list,
                warnings,
                event="experiment_completed",
                payload={
                    "event": "experiment_completed",
                    "mode": row.get("mode"),
                    "variant": row.get("variant"),
                    "model": row.get("model"),
                    "selector": row.get("selector"),
                    "completed_experiments": completed_experiments,
                    "total_experiments": total_experiments,
                },
            )

        if "full" in modes:
            _emit_mode_started("full", 1)
            rows.extend(
                self._run_flat_mode(
                    mode="full",
                    variant="all_features",
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
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        feature_cols=feature_cols,
                        models=models,
                        selectors=selectors,
                        cv_splits=cv_splits,
                        scorer=scorer_callable,
                        warnings=warnings,
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
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        feature_cols=[c for c in feature_cols if c not in group_cols],
                        models=models,
                        selectors=selectors,
                        cv_splits=cv_splits,
                        scorer=scorer_callable,
                        task=task,
                        warnings=warnings,
                        on_experiment_completed=_mark_experiment_completed,
                    )
                )
            else:
                warnings.append("Skipping group_split: no group_columns provided.")

        if "group_permutations" in modes:
            if group_cols:
                _emit_mode_started("group_permutations", len(group_permutation_splits))
                for cols in group_permutation_splits:
                    rows.extend(
                        self._run_group_split_mode(
                            mode="group_permutations",
                            split_columns=tuple(cols),
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            feature_cols=[c for c in feature_cols if c not in group_cols],
                            models=models,
                            selectors=selectors,
                            cv_splits=cv_splits,
                            scorer=scorer_callable,
                            task=task,
                            warnings=warnings,
                            on_experiment_completed=_mark_experiment_completed,
                        )
                    )
            else:
                warnings.append("Skipping group_permutations: no group_columns provided.")

        if "rule_split" in modes:
            if parsed_rules:
                _emit_mode_started("rule_split", 1)
                rows.extend(
                    self._run_rule_split_mode(
                        mode="rule_split",
                        parsed_rules=parsed_rules,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        feature_cols=[c for c in feature_cols if c not in group_cols],
                        models=models,
                        selectors=selectors,
                        cv_splits=cv_splits,
                        scorer=scorer_callable,
                        task=task,
                        warnings=warnings,
                        on_experiment_completed=_mark_experiment_completed,
                    )
                )
            else:
                warnings.append("Skipping rule_split: no rule_splits provided.")

        if not rows:
            raise ValueError("No experiments were executed. Check experiment_modes and inputs.")

        leaderboard = pd.DataFrame(rows).sort_values(
            by=["cv_mean", "test_score"], ascending=False
        )
        leaderboard.reset_index(drop=True, inplace=True)

        baseline_row = self._pick_baseline(leaderboard)
        best_row = leaderboard.iloc[0].to_dict()
        recommendation = self._recommend(best_row, baseline_row, warnings)

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
            },
        )

        return GroupMLResult(
            leaderboard=leaderboard,
            recommendation=recommendation,
            warnings=warnings,
            best_experiment=best_row,
            baseline_experiment=baseline_row,
            split_info=split_plan.split_info,
        )

    def _preprocess_base_dataset(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        group_cols: list[str],
        rule_columns: list[str],
        cv_group_columns: list[str],
        task: str,
        warnings: list[str],
    ) -> tuple[pd.DataFrame, list[str], str]:
        if task == "regression":
            if (self.config.min_target is not None or self.config.max_target is not None) and not pd.api.types.is_numeric_dtype(
                data[self.config.target]
            ):
                raise ValueError("min_target/max_target require a numeric regression target.")
            if self.config.min_target is not None:
                before = len(data)
                data = data[data[self.config.target] >= self.config.min_target]
                dropped = before - len(data)
                if dropped:
                    warnings.append(
                        f"Base preprocessing dropped {dropped} rows below min_target={self.config.min_target}."
                    )
            if self.config.max_target is not None:
                before = len(data)
                data = data[data[self.config.target] <= self.config.max_target]
                dropped = before - len(data)
                if dropped:
                    warnings.append(
                        f"Base preprocessing dropped {dropped} rows above max_target={self.config.max_target}."
                    )

        if self.config.dropna_base_rows:
            required_columns = sorted(
                set(
                    [self.config.target]
                    + feature_cols
                    + group_cols
                    + rule_columns
                    + cv_group_columns
                )
            )
            before = len(data)
            data = data.dropna(subset=required_columns)
            dropped = before - len(data)
            if dropped:
                warnings.append(
                    f"Base preprocessing dropped {dropped} rows containing NaNs in required columns."
                )

        protected = set(group_cols) | set(rule_columns) | set(cv_group_columns)
        if self.config.drop_static_base_features:
            static_features = [c for c in feature_cols if c not in protected and data[c].nunique(dropna=False) <= 1]
            if static_features:
                data = data.drop(columns=static_features)
                feature_cols = [c for c in feature_cols if c not in static_features]
                warnings.append(
                    f"Base preprocessing removed {len(static_features)} static feature columns: {static_features}"
                )

        if not feature_cols:
            raise ValueError("No feature columns left after base preprocessing.")
        if data.empty:
            raise ValueError("No rows left after base preprocessing.")

        for col in group_cols:
            if col not in feature_cols:
                feature_cols.append(col)

        task = infer_task(data[self.config.target], self.config.task)
        return data, feature_cols, task

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
                    return float(-np.sqrt(mean_squared_error(y, y_pred)))

                return _rmse_score
            scorer_obj = get_scorer(scorer)

            def _score(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
                return float(scorer_obj(estimator, X, y))

            return _score

        def _score(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
            try:
                return float(scorer(estimator, X, y))
            except TypeError:
                y_pred = estimator.predict(X)
                return float(scorer(y, y_pred))

        return _score

    def _run_flat_mode(
        self,
        mode: str,
        variant: str,
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
        on_experiment_completed: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        preprocessor = build_preprocessor(X_train, feature_cols, self.config.scale_numeric)
        for (model_name, model), (selector_name, selector_spec) in product(models.items(), selectors.items()):
            selector = build_selector(selector_spec, infer_task(y_train, self.config.task), self.config.random_state)
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
                warnings=warnings,
            )
            rows.append(row)
            if on_experiment_completed is not None:
                on_experiment_completed(row)
        return rows

    def _run_group_split_mode(
        self,
        mode: str,
        split_columns: tuple[str, ...],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_cols: list[str],
        models: dict[str, Any],
        selectors: dict[str, Any],
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        scorer: Callable[[Any, pd.DataFrame, pd.Series], float],
        task: str,
        warnings: list[str],
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
                warnings=warnings,
            )
            rows.append(row)
            if on_experiment_completed is not None:
                on_experiment_completed(row)
        return rows

    def _run_rule_split_mode(
        self,
        mode: str,
        parsed_rules: list[Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_cols: list[str],
        models: dict[str, Any],
        selectors: dict[str, Any],
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
        scorer: Callable[[Any, pd.DataFrame, pd.Series], float],
        task: str,
        warnings: list[str],
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
                warnings=warnings,
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
        warnings: list[str],
    ) -> dict[str, Any]:
        scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            fold_est = clone(estimator)
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            try:
                fold_est.fit(X_tr, y_tr)
                score = scorer(fold_est, X_val, y_val)
                scores.append(score)
            except Exception as exc:
                warnings.append(
                    f"CV failure in {mode}/{variant} ({model_name}, {selector_name}) fold={fold_idx}: {exc}"
                )
        fit_est = clone(estimator)
        test_score = np.nan
        try:
            fit_est.fit(X_train, y_train)
            test_score = scorer(fit_est, X_test, y_test)
            if hasattr(fit_est, "warnings_"):
                warnings.extend(getattr(fit_est, "warnings_"))
        except Exception as exc:
            warnings.append(f"Test failure in {mode}/{variant} ({model_name}, {selector_name}): {exc}")
        cv_mean = float(np.mean(scores)) if scores else np.nan
        cv_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan
        return {
            "mode": mode,
            "variant": variant,
            "experiment_name": f"{mode}:{variant}",
            "model": model_name,
            "selector": selector_name,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_folds_ok": len(scores),
            "test_score": float(test_score) if not np.isnan(test_score) else np.nan,
        }

    def _pick_baseline(self, leaderboard: pd.DataFrame) -> dict[str, Any]:
        baseline = leaderboard[leaderboard["mode"] == "full"]
        if baseline.empty:
            baseline = leaderboard.head(1)
        return baseline.sort_values(by=["cv_mean", "test_score"], ascending=False).iloc[0].to_dict()

    def _recommend(
        self,
        best_row: dict[str, Any],
        baseline_row: dict[str, Any],
        warnings: list[str],
    ) -> str:
        best_name = best_row["experiment_name"]
        baseline_name = baseline_row["experiment_name"]
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
