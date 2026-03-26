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

    def _fake_export_bundle(
        result: GroupMLResult,
        path: str | Path,
        top_n: int = 10,
        report_format: str = "auto",
        include_raw: bool = True,
    ) -> dict[str, Path]:
        del result, top_n, report_format, include_raw
        captured["summary_export_path"] = Path(path)
        return {"summary": Path(path)}

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_reporting_bundle", _fake_export_bundle)
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
            "--warning-verbosity",
            "all",
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
    assert cfg.kbest_features == "auto"
    assert cfg.cv == 3
    assert cfg.warning_verbosity == "all"
    assert cfg.cv_fold_size_rows is None
    assert cfg.test_split_strategy == "last_rows"
    assert cfg.test_size == 0.15
    assert cfg.test_size_rows is None
    assert captured["read_kwargs"] == {"sheet_name": "SheetA"}
    assert isinstance(captured["callbacks"], list)
    assert len(captured["callbacks"]) == 1
    assert captured["summary_export_path"].name.startswith("default_name.")


def test_cli_main_parses_cv_fold_size_rows(monkeypatch) -> None:
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
    monkeypatch.setattr(
        cli,
        "export_reporting_bundle",
        lambda result, path, top_n=10, report_format="auto", include_raw=True: {"summary": Path(path)},
    )
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(
        [
            "--path",
            "data.csv",
            "--target",
            "Target",
            "--cv-fold-size-rows",
            "36",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert isinstance(cfg, GroupMLConfig)
    assert cfg.cv_fold_size_rows == 36
    assert cfg.warning_verbosity == "quiet"


def test_cli_main_parses_kbest_features(monkeypatch) -> None:
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
    monkeypatch.setattr(
        cli,
        "export_reporting_bundle",
        lambda result, path, top_n=10, report_format="auto", include_raw=True: {"summary": Path(path)},
    )
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(
        [
            "--path",
            "data.csv",
            "--target",
            "Target",
            "--kbest-features",
            "9",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert isinstance(cfg, GroupMLConfig)
    assert cfg.kbest_features == 9


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
    monkeypatch.setattr(
        cli,
        "export_reporting_bundle",
        lambda result, path, top_n=10, report_format="auto", include_raw=True: {"summary": Path(path)},
    )
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
    monkeypatch.setattr(
        cli,
        "export_reporting_bundle",
        lambda result, path, top_n=10, report_format="auto", include_raw=True: {"summary": Path(path)},
    )
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
    monkeypatch.setattr(
        cli,
        "export_reporting_bundle",
        lambda result, path, top_n=10, report_format="auto", include_raw=True: {"summary": Path(path)},
    )
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
    monkeypatch.setattr(
        cli,
        "export_reporting_bundle",
        lambda result, path, top_n=10, report_format="auto", include_raw=True: {"summary": Path(path)},
    )
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
    monkeypatch.setattr(
        cli,
        "export_reporting_bundle",
        lambda result, path, top_n=10, report_format="auto", include_raw=True: {"summary": Path(path)},
    )
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
    monkeypatch.setattr(
        cli,
        "export_reporting_bundle",
        lambda result, path, top_n=10, report_format="auto", include_raw=True: {"summary": Path(path)},
    )
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

    def _fake_export_bundle(
        result: GroupMLResult,
        path: str | Path,
        top_n: int = 10,
        report_format: str = "auto",
        include_raw: bool = True,
    ) -> dict[str, Path]:
        del result, top_n, report_format, include_raw
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"section": "full_dataset_best", "method_type": "no_group_awareness"}]).to_csv(
            out_path, index=False
        )
        return {"summary": out_path}

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_reporting_bundle", _fake_export_bundle)

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
    monkeypatch.setattr(
        cli,
        "export_reporting_bundle",
        lambda result, path, top_n=10, report_format="auto", include_raw=True: {"summary": Path(path)},
    )
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
    monkeypatch.setattr(
        cli,
        "export_reporting_bundle",
        lambda result, path, top_n=10, report_format="auto", include_raw=True: {"summary": Path(path)},
    )
    monkeypatch.setattr(cli, "default_summary_filename", lambda ext=".csv": f"default_name{ext}")

    exit_code = cli.main(
        ["--path", "data.csv", "--target", "Target", "--scorer", "neg_root_mean_squared_error"]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "cv_rmse=1.23456" in stdout
    assert "test_rmse=2.34567" in stdout
    leaderboard_section = stdout.split("Leaderboard:", maxsplit=1)[1]
    assert "1.23456" in leaderboard_section
    assert "2.34567" in leaderboard_section
    assert "-1.23456" not in leaderboard_section
    assert "-2.34567" not in leaderboard_section


def test_cli_main_exports_partial_outputs_on_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    from groupml import cli

    captured: dict[str, object] = {}
    summary_out = tmp_path / "partial_summary.csv"
    leaderboard_out = tmp_path / "partial_leaderboard.csv"
    raw_out = tmp_path / "partial_raw.csv"

    def _fake_fit_evaluate_file(
        path: str,
        config: GroupMLConfig,
        callbacks: object = None,
        **read_kwargs: object,
    ) -> GroupMLResult:
        del path, config, read_kwargs
        assert isinstance(callbacks, list)
        for callback in callbacks:
            callback(
                {
                    "event": "run_started",
                    "total_experiments": 5,
                    "cv_splitter": "KFold",
                    "cv_strategy_used": "kfold",
                    "cv_n_splits": 3,
                    "test_splitter": "last_rows",
                    "test_strategy": "last_rows",
                    "test_train_size": 80,
                    "test_test_size": 20,
                }
            )
            callback(
                {
                    "event": "experiment_completed",
                    "completed_experiments": 1,
                    "total_experiments": 5,
                    "mode": "full",
                    "variant": "all_features",
                    "model": "linear",
                    "selector": "none",
                    "cv_mean": 0.85,
                    "cv_std": 0.05,
                    "cv_folds_ok": 3,
                    "test_score": 0.8,
                    "best_so_far_updated": True,
                    "best_raw_report": pd.DataFrame(
                        [
                            {
                                "row_index": 0,
                                "split_assignment": "cv_1",
                                "actual": 1.0,
                                "predicted": 0.9,
                                "error": -0.1,
                            }
                        ]
                    ),
                }
            )
        raise KeyboardInterrupt()

    def _fake_export_bundle(
        result: GroupMLResult,
        path: str | Path,
        top_n: int = 10,
        report_format: str = "auto",
        include_raw: bool = True,
    ) -> dict[str, Path]:
        del top_n, report_format, include_raw
        captured["summary_result"] = result
        captured["summary_export_path"] = Path(path)
        return {"summary": Path(path)}

    def _fake_export_report(
        result: GroupMLResult,
        path: str | Path,
        sheet_name: str = "leaderboard",
    ) -> Path:
        del sheet_name
        captured["report_result"] = result
        captured["report_export_path"] = Path(path)
        return Path(path)

    def _fake_export_raw_report(
        result: GroupMLResult,
        path: str | Path,
        sheet_name: str = "raw_report",
    ) -> Path:
        del sheet_name
        captured["raw_result"] = result
        captured["raw_export_path"] = Path(path)
        return Path(path)

    monkeypatch.setattr(cli, "fit_evaluate_file", _fake_fit_evaluate_file)
    monkeypatch.setattr(cli, "export_reporting_bundle", _fake_export_bundle)
    monkeypatch.setattr(cli, "export_report", _fake_export_report)
    monkeypatch.setattr(cli, "export_raw_report", _fake_export_raw_report)

    exit_code = cli.main(
        [
            "--path",
            "data.csv",
            "--target",
            "Target",
            "--out",
            str(summary_out),
            "--leaderboard-out",
            str(leaderboard_out),
            "--raw-report-out",
            str(raw_out),
        ]
    )

    assert exit_code == 130
    assert captured["summary_export_path"] == summary_out
    assert captured["report_export_path"] == leaderboard_out
    assert captured["raw_export_path"] == raw_out

    partial_result = captured["summary_result"]
    assert isinstance(partial_result, GroupMLResult)
    assert len(partial_result.leaderboard) == 1
    assert partial_result.leaderboard.iloc[0]["experiment_name"] == "full:all_features"
    assert "Run interrupted" in partial_result.recommendation
    assert isinstance(partial_result.raw_report, pd.DataFrame)
    assert not partial_result.raw_report.empty
    assert partial_result.leaderboard.iloc[0]["cv_std"] == 0.05
    assert partial_result.leaderboard.iloc[0]["cv_folds_ok"] == 3
    assert partial_result.split_info["test"]["splitter"] == "last_rows"
    assert partial_result.split_info["test"]["train_size"] == 80
