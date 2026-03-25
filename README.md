# groupml

`groupml` is a practical, sklearn-native package for deciding whether group-aware modeling improves your results on industrial/measurement datasets.

It compares:

1. One global model (`full`)
2. Group columns as features (`group_as_features`)
3. Separate models by group (`group_split`)
4. Rule-based segmentation (`rule_split`)
5. Group-column combinations (`group_permutations`)

The framework keeps one holdout split and consistent CV across experiments for fair, repeatable comparison.
Ranking and recommendations are CV-based. Test score is always display-only.

## CLI First (Recommended)

This project is designed to be used as a command-line app first.

## Install for CLI

### Option A: `pipx` (recommended for app-style usage)

```bash
pipx install groupml
```

`openpyxl` is included by default, so Excel reporting works out of the box.

If you are developing locally:

```bash
pipx install --editable .
```

### Option B: `pip` (inside your environment)

```bash
pip install groupml
```

`openpyxl` is installed automatically with `groupml`.

For local development:

```bash
pip install -e .[dev]
```

## Basic CLI Usage

```bash
groupml --path examples/data/group_split_demo.csv --target Target --groups ActionGroup
```

Equivalent module call:

```bash
python -m groupml --path examples/data/group_split_demo.csv --target Target --groups ActionGroup
```

## What CLI Produces

- Console summary and top leaderboard rows
- Tabular report export (default):
1. Prefer one Excel workbook (`.xlsx`) with sheets:
`summary`, `recommendations`, `warnings`, `all_runs`, `raw_results`
2. Fallback to multiple CSV files if `openpyxl` is not available:
`*_summary.csv`, `*_recommendations.csv`, `*_warnings.csv`, `*_all_runs.csv`, `*_raw_results.csv`

The `summary` view is organized so full-dataset best method rows come first, then overall method comparison, then per-group comparison rows.
The `recommendations` view is intentionally compact: one recommended setup row plus top-N CV-ranked alternatives overall (default top 5; controlled by `--top`).
Excel sheets are auto-formatted for readability (column width, frozen headers, numeric formats).

## Common CLI Examples

### 1) Minimal run

```bash
groupml --path data.csv --target Target
```

### 2) Group-aware comparison

```bash
groupml ^
  --path data.csv ^
  --target Target ^
  --groups ActionGroup Material ^
  --modes full group_as_features group_split group_permutations
```

### 3) Add rule split

```bash
groupml ^
  --path data.csv ^
  --target Target ^
  --rules "Temperature < 20" "Temperature >= 20" ^
  --modes full rule_split
```

### 4) Classification task with random split and stratification column

```bash
groupml ^
  --path data.csv ^
  --target Label ^
  --task classification ^
  --test-split random ^
  --cv stratifycv ^
  --cv-stratify-column Label
```

### 5) Time-ordered split/CV

```bash
groupml ^
  --path data.csv ^
  --target Target ^
  --cv timecv ^
  --cv-date-column BatchDate ^
  --test-split last_rows
```

### 6) Force CSV report format

```bash
groupml --path data.csv --target Target --report-format csv
```

### 7) Save to explicit paths

```bash
groupml ^
  --path data.csv ^
  --target Target ^
  --out reports/run_01.xlsx ^
  --leaderboard-out reports/run_01_leaderboard.csv ^
  --raw-report-out reports/run_01_raw.csv
```

## CLI Argument Reference

### Required

- `--path`: Input dataset path (`.csv`, `.xls`, `.xlsx`)
- `--target`: Target column name

### Group/Rule setup

- `--groups`: Group columns (space-separated)
- `--rules`: Rule splits (for example `"Temperature < 20"`)

### Feature and experiment scope

- `--features`: Explicit feature column list (default: all except target)
- `--modes`: Experiment modes to run:
`full`, `group_as_features`, `group_split`, `group_permutations`, `rule_split`

### Model/selector setup

- `--models`: Model strategy or model name (for example `default_fast`, `all`, `trees`, `ridge`)
- `--feature-selectors` / `--selectors`: Selector strategy/name (for example `default_fast`, `none`, `kbest_f`)

### CV and split behavior

- `--cv`: Fold count (for example `5`) or splitter strategy (for example `groupcv`, `timecv`, `stratifycv`, `stratifygroupcv`, `stratifytimecv`)
- `--cv-fold-size-rows`: Optional fold-size control in rows
- `--cv-group-column`: Group column for split-aware CV
- `--cv-date-column`: Datetime column for time-based split/CV
- `--cv-stratify-column`: Column for stratified split/CV
- `--test-split`: `last_rows` or `random`
- `--test-size-strategy`: `auto`, `pct`, or `rows`
- `--test-size`: Value interpreted by strategy
- `--random-state`: Random seed

### Scoring/task

- `--scorer`: sklearn scorer name or `rmse` alias
- `--task`: `auto`, `regression`, `classification`

### Robustness and preprocessing

- `--warning-verbosity`: `quiet`, `default`, `all`
- `--min-group-size`: Minimum subgroup size for split estimators
- `--min-improvement`: Required CV improvement to recommend non-baseline
- `--scale-numeric`: Scale numeric features
- `--keep-nans`: Disable base-row NaN dropping
- `--keep-static-features`: Disable static feature removal
- `--min-target`: Min target filter (regression)
- `--max-target`: Max target filter (regression)

### Input/output

- `--sheet-name`: Excel sheet name when reading input `.xls/.xlsx`
- `--top`: Number of leaderboard rows printed
- `--out`: Main output path (`.xlsx/.csv/.txt/.md/.json`)
- `--report-format`: `auto` (default), `excel`, `csv`
- `--leaderboard-out`: Optional separate leaderboard file (`.csv/.xls/.xlsx`)
- `--raw-report-out`: Optional separate raw per-row report (`.csv/.xls/.xlsx`)
- `--no-raw-report`: Disable raw-report export

## Output Structure Notes

`summary` contains a standardized table with:

- Full dataset method comparison first (`full_dataset_best`)
- Combined eval-set method comparison (`overall_method_comparison`)
- Per-group method comparison next (`per_group_comparison`)

Method labels in summary/per-group comparison:

- `no_group_awareness`
- `one_hot_group_features`
- `per_group_models`
- `per_group_models_group_combos`
- `rule_based_split`

`recommendations` contains:

- One `recommended_setup` row with the selected configuration and recommendation text
- `top_n_overall` rows: top-N experiments by CV score
- Test score shown only for context, never used for ranking/recommendation

## Python API (For Embedding In Your App)

Use this after CLI when you need programmatic integration.

```python
import pandas as pd
from groupml import GroupMLConfig, GroupMLRunner

df = pd.read_csv("your_data.csv")

config = GroupMLConfig(
    target="Target",
    group_columns=["ActionGroup", "Material"],
    rule_splits=["Temperature < 20", "Temperature >= 20"],
    experiment_modes=["full", "group_as_features", "group_split", "group_permutations", "rule_split"],
    models="default_fast",
    feature_selectors="default_fast",
    cv=5,
    scorer="neg_root_mean_squared_error",
    test_size=0.2,
    random_state=42,
)

result = GroupMLRunner(config).fit_evaluate(df)
print(result.recommendation)
print(result.leaderboard.head(10))
print(result.summary_text())
```

Functional API:

```python
from groupml import compare_group_strategies

result = compare_group_strategies(
    df=df,
    target="Target",
    group_columns=["ActionGroup", "Material"],
    rule_splits=["Temperature < 20", "Temperature >= 20"],
)
```

File-based API:

```python
from groupml import GroupMLConfig, fit_evaluate_file

config = GroupMLConfig(target="Target", group_columns=["ActionGroup"], cv=3)
result = fit_evaluate_file("examples/data/group_split_demo.csv", config)
```

## Development

```bash
python -m venv .venv
.\.venv\Scripts\python -m ensurepip --upgrade
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python -m pytest
```
