# AGENTS.md

## What This Project Is

`groupml` is a practical, sklearn-native Python package for deciding whether group-aware modeling is worth using on industrial/measurement datasets.

It compares modeling strategies such as:

- a single global model
- adding group columns as features
- separate models per group
- simple rule-based splits (for example `Temperature < 20`)
- permutations/combinations of group columns

The goal is fair, repeatable comparison with:

- one stable held-out test split
- consistent CV logic across experiments
- structured results and a clear recommendation

## Core Public API

- `GroupMLConfig`
- `GroupMLRunner`
- `GroupMLResult`
- `compare_group_strategies`

Typical flow:

1. Define `GroupMLConfig` (target, groups, rules, models/selectors/cv/scorer).
2. Run `GroupMLRunner(config).fit_evaluate(df)`.
3. Read `result.leaderboard`, `result.recommendation`, `result.summary_text()`.

## Repository Structure

- `groupml/__init__.py`
  Public exports.
- `groupml/config.py`
  Main dataclass config (`GroupMLConfig`).
- `groupml/runner.py`
  Experiment orchestration, CV/test evaluation, recommendation logic.
- `groupml/estimators.py`
  Custom sklearn-compatible split estimators:
  `GroupSplit*` and `RuleSplit*`.
- `groupml/utils.py`
  Rule parsing, preprocessing builders, selector/model presets, CV helpers.
- `groupml/result.py`
  Result dataclass (`GroupMLResult`) and summary rendering.
- `tests/`
  Smoke/API/rule parser/grouping tests.
- `examples/`
  Minimal runnable usage examples.

## Design Principles

- Keep dependencies minimal: stdlib + `numpy`, `pandas`, `scikit-learn`.
- Stay sklearn-native and easy to modify.
- Prefer readability and explicit behavior over clever abstractions.
- Keep experiments comparable and deterministic (`random_state`).
- Surface practical warnings (small groups, fold failures, instability).

## Experiment Modes (Current)

- `full`
- `group_as_features`
- `group_split`
- `group_permutations`
- `rule_split`

## Rules and Grouping Notes

- Rule parser supports: `<`, `<=`, `>`, `>=`, `==`, `!=`.
- `group_split` and `rule_split` use fallback global models for sparse subsets.
- Small subset handling is controlled by `min_group_size`.

## Dev Workflow

Use local venv:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m ensurepip --upgrade
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python -m pytest
```

## Contributor Notes for Agents

- Do not add heavy dependencies or non-sklearn ML frameworks.
- Preserve public API names and import paths.
- Add/update tests for any behavior changes.
- Keep examples runnable and aligned with API defaults.
- Favor simple, local fixes over broad refactors.

