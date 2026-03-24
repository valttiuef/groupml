from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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
            "--models",
            "trees",
            "--feature-selectors",
            "mutual_info",
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
    assert cfg.models == "trees"
    assert cfg.feature_selectors == "mutual_info"
    assert cfg.cv == 3
    assert cfg.test_split_strategy == "last_rows"
    assert cfg.test_size == 0.15
    assert cfg.test_size_rows is None
    assert captured["read_kwargs"] == {"sheet_name": "SheetA"}
    assert isinstance(captured["callbacks"], list)
    assert len(captured["callbacks"]) == 1
    assert captured["summary_export_path"] == Path.cwd() / "default_name.csv"


def test_cli_main_parses_cv_columns(monkeypatch) -> None:
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

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_summary", lambda result, path, top_n=10, sheet_name="summary": Path(path))
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(
        [
            "--path",
            "data.csv",
            "--target",
            "Target",
            "--cv",
            "stratifytimecv",
            "--cv-group-column",
            "ActionGroup",
            "--cv-date-column",
            "BatchDate",
            "--cv-stratify-column",
            "Shift",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert isinstance(cfg, GroupMLConfig)
    assert cfg.cv == "stratifytimecv"
    assert cfg.split_group_column == "ActionGroup"
    assert cfg.split_date_column == "BatchDate"
    assert cfg.split_stratify_column == "Shift"


def test_cli_main_parses_test_size_rows(monkeypatch) -> None:
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

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_summary", lambda result, path, top_n=10, sheet_name="summary": Path(path))
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(
        [
            "--path",
            "data.csv",
            "--target",
            "Target",
            "--test-size-strategy",
            "rows",
            "--test-size",
            "12",
            "--test-split",
            "random",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert isinstance(cfg, GroupMLConfig)
    assert cfg.test_size_rows == 12
    assert cfg.test_split_strategy == "random"


def test_cli_main_parses_test_size_auto_rows(monkeypatch) -> None:
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

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_summary", lambda result, path, top_n=10, sheet_name="summary": Path(path))
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(
        [
            "--path",
            "data.csv",
            "--target",
            "Target",
            "--test-size",
            "36",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert isinstance(cfg, GroupMLConfig)
    assert cfg.test_size_rows == 36
    assert cfg.test_size == 0.15


def test_cli_main_parses_test_size_pct_with_integer_percent(monkeypatch) -> None:
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

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_summary", lambda result, path, top_n=10, sheet_name="summary": Path(path))
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(
        [
            "--path",
            "data.csv",
            "--target",
            "Target",
            "--test-size-strategy",
            "pct",
            "--test-size",
            "37",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert isinstance(cfg, GroupMLConfig)
    assert cfg.test_size_rows is None
    assert cfg.test_size == 0.37


def test_cli_main_parses_test_size_auto_decimal_percent(monkeypatch) -> None:
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

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_summary", lambda result, path, top_n=10, sheet_name="summary": Path(path))
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(
        [
            "--path",
            "data.csv",
            "--target",
            "Target",
            "--test-size",
            "64.2",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert isinstance(cfg, GroupMLConfig)
    assert cfg.test_size_rows is None
    assert cfg.test_size == 0.642


def test_cli_main_rejects_invalid_manual_test_size(monkeypatch) -> None:
    from groupml import cli

    monkeypatch.setattr(cli, "fit_evaluate_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "export_summary", lambda result, path, top_n=10, sheet_name="summary": Path(path))
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    with pytest.raises(SystemExit):
        cli.main(
            [
                "--path",
                "data.csv",
                "--target",
                "Target",
                "--test-size-strategy",
                "pct",
                "--test-size",
                "200",
            ]
        )

    with pytest.raises(SystemExit):
        cli.main(
            [
                "--path",
                "data.csv",
                "--target",
                "Target",
                "--test-size-strategy",
                "rows",
                "--test-size",
                "123.546",
            ]
        )


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


def test_cli_progress_callback_prints_cv_and_test_scores(monkeypatch, capsys) -> None:
    from groupml import cli

    def _fake_fit_evaluate_file(
        path: str,
        config: GroupMLConfig,
        callbacks: object = None,
        **read_kwargs: object,
    ) -> GroupMLResult:
        del path, config, read_kwargs
        assert isinstance(callbacks, list)
        callback = callbacks[0]
        callback(
            {
                "event": "run_started",
                "total_experiments": 1,
                "cv_splitter": "KFold",
                "cv_strategy_used": "kfold",
                "cv_n_splits": 3,
            }
        )
        callback(
            {
                "event": "experiment_completed",
                "completed_experiments": 1,
                "total_experiments": 1,
                "mode": "full",
                "model": "linear",
                "selector": "none",
                "cv_mean": 0.9,
                "test_score": 0.8,
            }
        )
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

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_summary", lambda result, path, top_n=10, sheet_name="summary": Path(path))
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(["--path", "data.csv", "--target", "Target", "--scorer", "r2"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "cv_score=0.90000" in stdout
    assert "test_score=0.80000" in stdout


def test_cli_progress_callback_prints_positive_rmse(monkeypatch, capsys) -> None:
    from groupml import cli

    def _fake_fit_evaluate_file(
        path: str,
        config: GroupMLConfig,
        callbacks: object = None,
        **read_kwargs: object,
    ) -> GroupMLResult:
        del path, config, read_kwargs
        assert isinstance(callbacks, list)
        callback = callbacks[0]
        callback(
            {
                "event": "run_started",
                "total_experiments": 1,
                "cv_splitter": "KFold",
                "cv_strategy_used": "kfold",
                "cv_n_splits": 3,
            }
        )
        callback(
            {
                "event": "experiment_completed",
                "completed_experiments": 1,
                "total_experiments": 1,
                "mode": "full",
                "model": "linear",
                "selector": "none",
                "cv_mean": -1.23456,
                "test_score": -2.34567,
            }
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "experiment_name": "full::linear::none",
                    "mode": "full",
                    "cv_mean": -1.23456,
                    "test_score": -2.34567,
                }
            ]
        )
        return GroupMLResult(
            leaderboard=leaderboard,
            recommendation="Use full",
            best_experiment=leaderboard.iloc[0].to_dict(),
            baseline_experiment=leaderboard.iloc[0].to_dict(),
        )

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_summary", lambda result, path, top_n=10, sheet_name="summary": Path(path))
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(
        ["--path", "data.csv", "--target", "Target", "--scorer", "neg_root_mean_squared_error"]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "cv_rmse=1.23456" in stdout
    assert "test_rmse=2.34567" in stdout
