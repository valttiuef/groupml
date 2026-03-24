from __future__ import annotations

from pathlib import Path

import pandas as pd

from groupml.config import GroupMLConfig
from groupml.result import GroupMLResult


def test_cli_main_parses_arguments(monkeypatch) -> None:
    from groupml import cli

    captured: dict[str, object] = {}

    def _fake_fit_evaluate_file(
        path: str,
        config: GroupMLConfig,
        callbacks: object = None,
        **read_kwargs: object,
    ) -> GroupMLResult:
        captured["path"] = path
        captured["config"] = config
        captured["callbacks"] = callbacks
        captured["read_kwargs"] = read_kwargs
        leaderboard = pd.DataFrame(
            [
                {
                    "experiment_name": "full::linear::none",
                    "mode": "full",
                    "cv_mean": 0.9,
                    "test_score": 0.8,
                }
            ]
        )
        return GroupMLResult(
            leaderboard=leaderboard,
            recommendation="Use full",
            best_experiment=leaderboard.iloc[0].to_dict(),
            baseline_experiment=leaderboard.iloc[0].to_dict(),
        )

    def _fake_export_summary(
        result: GroupMLResult,
        path: str | Path,
        top_n: int = 10,
        sheet_name: str = "summary",
    ) -> Path:
        del result, top_n, sheet_name
        captured["summary_export_path"] = Path(path)
        return Path(path)

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_summary", _fake_export_summary)
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(
        [
            "--path",
            "data.csv",
            "--target",
            "Target",
            "--groups",
            "ActionGroup",
            "Material",
            "--rules",
            "Temperature < 20",
            "Temperature >= 20",
            "--modes",
            "full",
            "group_split",
            "--cv",
            "3",
            "--sheet-name",
            "SheetA",
        ]
    )

    assert exit_code == 0
    assert captured["path"] == "data.csv"
    cfg = captured["config"]
    assert isinstance(cfg, GroupMLConfig)
    assert cfg.target == "Target"
    assert cfg.group_columns == ["ActionGroup", "Material"]
    assert cfg.rule_splits == ["Temperature < 20", "Temperature >= 20"]
    assert cfg.experiment_modes == ["full", "group_split"]
    assert cfg.cv == 3
    assert captured["read_kwargs"] == {"sheet_name": "SheetA"}
    assert isinstance(captured["callbacks"], list)
    assert len(captured["callbacks"]) == 1
    assert captured["summary_export_path"] == Path.cwd() / "default_name.csv"


def test_cli_main_writes_output_file(monkeypatch, tmp_path: Path) -> None:
    from groupml import cli

    out_csv = tmp_path / "leaderboard.csv"

    def _fake_fit_evaluate_file(
        path: str,
        config: GroupMLConfig,
        callbacks: object = None,
        **read_kwargs: object,
    ) -> GroupMLResult:
        del path, config, read_kwargs
        assert callbacks is not None
        leaderboard = pd.DataFrame(
            [
                {
                    "experiment_name": "full::linear::none",
                    "mode": "full",
                    "cv_mean": 0.9,
                    "test_score": 0.8,
                }
            ]
        )
        return GroupMLResult(
            leaderboard=leaderboard,
            recommendation="Use full",
            best_experiment=leaderboard.iloc[0].to_dict(),
            baseline_experiment=leaderboard.iloc[0].to_dict(),
        )

    def _fake_export_summary(
        result: GroupMLResult,
        path: str | Path,
        top_n: int = 10,
        sheet_name: str = "summary",
    ) -> Path:
        del result, top_n, sheet_name
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"section": "overview", "metric": "best_experiment", "value": "full"}]).to_csv(
            out_path, index=False
        )
        return out_path

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_summary", _fake_export_summary)

    exit_code = cli.main(
        [
            "--path",
            "data.csv",
            "--target",
            "Target",
            "--out",
            str(out_csv),
        ]
    )

    assert exit_code == 0
    assert out_csv.exists()
    written = pd.read_csv(out_csv)
    assert "section" in written.columns
