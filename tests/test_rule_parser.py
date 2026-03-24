from __future__ import annotations

import pandas as pd
import pytest

from groupml.utils import parse_rule


@pytest.mark.parametrize(
    "rule, expected",
    [
        ("Temperature < 20", ("Temperature", "<", 20)),
        ("Temperature <= 20", ("Temperature", "<=", 20)),
        ("Temperature > 20", ("Temperature", ">", 20)),
        ("Temperature >= 20", ("Temperature", ">=", 20)),
        ("Material == 'A'", ("Material", "==", "A")),
        ("Material != 'B'", ("Material", "!=", "B")),
    ],
)
def test_rule_parser_supported_ops(rule: str, expected: tuple[str, str, object]) -> None:
    parsed = parse_rule(rule)
    assert (parsed.column, parsed.op, parsed.value) == expected


def test_rule_mask_evaluates() -> None:
    df = pd.DataFrame({"Temperature": [10, 20, 30]})
    rule = parse_rule("Temperature < 20")
    mask = rule.mask(df)
    assert mask.tolist() == [True, False, False]


def test_rule_parser_invalid() -> None:
    with pytest.raises(ValueError):
        parse_rule("Temperature ~~ 20")

