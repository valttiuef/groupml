# groupml

`groupml` is a practical, sklearn-native package for testing whether group-aware modeling is worth it on your dataset.

It compares:

1. One global model (`full`)
2. Group columns added as features (`group_as_features`)
3. Separate subgroup models (`group_split`)
4. Rule-based segmentation (`rule_split`)
5. Group column permutations/combinations (`group_permutations`)

The package keeps one stable hold-out test split and reuses the same CV folds across experiments for fair comparison.

## Install

```bash
pip install -e .[dev]
```

## Quick Start

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
    # Example named sklearn splitter:
    # cv="StratifiedGroupKFold",
    # cv_params={"n_splits": 5, "shuffle": True},
    # cv_group_columns=["ActionGroup"],
    scorer="neg_mean_absolute_error",
    test_size=0.2,
    random_state=42,
)

runner = GroupMLRunner(config)
result = runner.fit_evaluate(df)

print(result.recommendation)
print(result.leaderboard.head(10))
print(result.summary_text())
print(result.split_info["cv"]["splitter"])
```

## Functional API

```python
from groupml import compare_group_strategies

result = compare_group_strategies(
    df=df,
    target="Target",
    group_columns=["ActionGroup", "Material"],
    rule_splits=["Temperature < 20", "Temperature >= 20"],
)
```

## File-Based Utilities

Run without manually calling `pd.read_csv`/`pd.read_excel`:

```python
from groupml import GroupMLConfig, fit_evaluate_file

config = GroupMLConfig(
    target="Target",
    group_columns=["ActionGroup"],
    experiment_modes=["full", "group_as_features"],
    cv=3,
)

result = fit_evaluate_file("examples/data/group_split_demo.csv", config)
```

Functional API from file path:

```python
from groupml import compare_group_strategies_file

result = compare_group_strategies_file(
    path="examples/data/rule_split_demo.csv",
    target="Target",
    rule_splits=["Temperature < 20", "Temperature >= 20"],
    experiment_modes=["full", "rule_split"],
)
```

Excel with pandas read options:

```python
from groupml import GroupMLConfig, fit_evaluate_file

config = GroupMLConfig(target="Target", group_columns=["ActionGroup"])
result = fit_evaluate_file(
    "my_data.xlsx",
    config,
    sheet_name="Sheet1",
)
```

Also available:

- `load_tabular_data(path, **read_kwargs)` for `.csv`, `.xls`, `.xlsx`
- `compare_group_strategies_file(path, target=..., ...)`

## CLI

After install (`pip install -e .[dev]`), run directly from terminal:

```bash
groupml --path examples/data/group_split_demo.csv --target Target --groups ActionGroup
```

Or without console script:

```bash
python -m groupml --path examples/data/rule_split_demo.csv --target Target --rules "Temperature < 20" "Temperature >= 20" --modes full rule_split
```

## Supported Inputs

- `cv`: int, sklearn splitter name (for example `"GroupKFold"`), sklearn splitter object, callable split function, or split iterable
- `cv_params`: parameters used when `cv` is a sklearn splitter name/dict
- `cv_group_columns`: optional columns used as group labels for group-aware CV splitters
- `test_splitter`: optional custom holdout splitter (name/object/callable/iterable)
- `include_split_indices`: include row indices for holdout + CV folds in `result.split_info`
- `scorer`: sklearn scorer string, scorer object, or callable
- `models`: `"default_fast"`, list of sklearn estimators, or dict of named estimators
- `feature_selectors`: `"default_fast"`, selector name, list, or dict

## Feature Selectors

- `kbest_f`
- `kbest_mi`
- `lasso`
- `extra_trees`

Default selector preset (`"default_fast"`):
- `kbest_f`
- `extra_trees`

## Default Model Families

Regression:
- `LinearRegression`
- `ElasticNet`
- `SGDRegressor`
- `ExtraTreesRegressor`

Classification:
- `LogisticRegression`
- `SGDClassifier` (elastic-net penalty)
- `ExtraTreesClassifier`

Default CLI behavior:
- prints a rich run summary + top leaderboard rows to console
- writes a summary file by default (`groupml_summary_*.csv`)
- optional raw leaderboard export via `--leaderboard-out path.csv`

## Recommendation Logic

Recommendation compares the best experiment against baseline (`full`) and requires:

- Meaningful CV improvement (`min_improvement`, default `0.01`)
- Stability check on CV standard deviation
- Warnings for small groups/rules and failed folds

## Examples

Run example scripts:

```bash
python examples/basic_regression.py
python examples/group_columns.py
python examples/rule_split.py
python examples/custom_cv_scorer.py
python examples/from_file.py
python examples/functional_from_file.py
```

Demo datasets are provided in `examples/data/`:

- `group_split_demo.csv`
- `rule_split_demo.csv`

## Development

```bash
pip install -e .[dev]
pytest
```

## Package API

- `GroupMLConfig`
- `GroupMLRunner`
- `GroupMLResult`
- `compare_group_strategies`
- `load_tabular_data`
- `fit_evaluate_file`
- `compare_group_strategies_file`

## Notes

- Design prioritizes readability and easy modification.
- Uses only Python stdlib + `numpy`, `pandas`, `scikit-learn`.
