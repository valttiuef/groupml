"""Built-in regression selector names and strategies."""

from __future__ import annotations


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


_SELECTOR_NAMES: set[str] = {
    "none",
    "kbest_f",
    "kbest_mi",
    "lasso",
    "extra_trees",
}

_SELECTOR_ALIASES: dict[str, str] = {
    "passthrough": "none",
    "no_selection": "none",
    "extra_trees_regressor": "extra_trees",
}

_SELECTOR_STRATEGIES: dict[str, tuple[str, ...]] = {
    "default_fast": ("kbest_f", "extra_trees"),
    "linear_default": ("none", "kbest_f", "lasso"),
    "trees": ("none", "extra_trees"),
    "mutual_info": ("none", "kbest_mi"),
    "all": ("none", "kbest_f", "kbest_mi", "lasso", "extra_trees"),
}


def available_regression_selector_names() -> list[str]:
    return sorted(_SELECTOR_NAMES)


def available_regression_selector_strategies() -> list[str]:
    return sorted(_SELECTOR_STRATEGIES.keys())


def resolve_regression_selector_name(name: str) -> str:
    key = _normalize_key(name)
    key = _SELECTOR_ALIASES.get(key, key)
    if key not in _SELECTOR_NAMES:
        names = ", ".join(available_regression_selector_names())
        raise ValueError(f"Unknown regression feature selector '{name}'. Available selector names: {names}")
    return key


def resolve_regression_selector_strategy(strategy: str) -> tuple[str, ...]:
    key = _normalize_key(strategy)
    if key not in _SELECTOR_STRATEGIES:
        strategies = ", ".join(available_regression_selector_strategies())
        raise ValueError(
            f"Unknown regression selector strategy '{strategy}'. Available strategies: {strategies}"
        )
    return _SELECTOR_STRATEGIES[key]


def get_regression_selector_names_by_strategy(strategy: str) -> tuple[str, ...]:
    return resolve_regression_selector_strategy(strategy)
