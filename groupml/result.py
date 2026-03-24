"""Result objects for groupml runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .summaries import summary_text as render_summary_text


@dataclass(slots=True)
class GroupMLResult:
    """Container returned by `GroupMLRunner.fit_evaluate`."""

    leaderboard: pd.DataFrame
    recommendation: str
    warnings: list[str] = field(default_factory=list)
    best_experiment: dict[str, Any] = field(default_factory=dict)
    baseline_experiment: dict[str, Any] = field(default_factory=dict)
    split_info: dict[str, Any] = field(default_factory=dict)

    def summary_text(self) -> str:
        """Return a practical human-readable summary."""
        return render_summary_text(self)
