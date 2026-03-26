from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from groupml import (
    GroupMLConfig,
    compare_group_strategies_file,
    default_report_filename,
    default_summary_filename,
    export_reporting_bundle,
    export_raw_report,
    export_report,
    export_summary,
    fit_evaluate_file,
    load_tabular_data,
)


def test_load_tabular_data_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "tiny.csv"
    pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(input_path, index=False)

    loaded = load_tabular_data(input_path)
    assert list(loaded.columns) == ["x", "y"]
    assert loaded.shape == (2, 2)


def test_load_tabular_data_csv_falls_back_from_utf8_to_cp1252(tmp_path: Path) -> None:
    input_path = tmp_path / "ansi.csv"
    input_path.write_bytes("name,value\nM\xe4nty,1\n".encode("latin-1"))

    loaded = load_tabular_data(input_path)
    assert list(loaded.columns) == ["name", "value"]
    assert loaded.loc[0, "name"] == "Mänty"
    assert loaded.loc[0, "value"] == 1


def test_load_tabular_data_csv_with_explicit_encoding_does_not_fallback(tmp_path: Path) -> None:
    input_path = tmp_path / "ansi.csv"
    input_path.write_bytes("name,value\nM\xe4nty,1\n".encode("latin-1"))

    with pytest.raises(UnicodeDecodeError):
        load_tabular_data(input_path, encoding="utf-8")


def test_load_tabular_data_excel_uses_pandas(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = pd.DataFrame({"a": [1], "b": [2]})
    captured: dict[str, object] = {}

    def _fake_read_excel(path: Path, **kwargs: object) -> pd.DataFrame:
        captured["path"] = path
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr("groupml.file_utils.pd.read_excel", _fake_read_excel)
    path = tmp_path / "demo.xlsx"
    loaded = load_tabular_data(path, sheet_name="SheetA")

    assert loaded.equals(expected)
    assert captured["path"] == path
    assert captured["kwargs"] == {"sheet_name": "SheetA"}


def test_load_tabular_data_unsupported_extension(tmp_path: Path) -> None:
    bad = tmp_path / "data.json"
    bad.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_tabular_data(bad)


def test_fit_evaluate_file_runs_on_csv() -> None:
    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full", "group_as_features"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
    )
    result = fit_evaluate_file("examples/data/group_split_demo.csv", config)
    assert not result.leaderboard.empty
    assert (result.leaderboard["mode"] == "group_as_features").any()


def test_fit_evaluate_file_passes_callbacks() -> None:
    config = GroupMLConfig(
        target="Target",
        group_columns=["ActionGroup"],
        experiment_modes=["full"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
    )
    events: list[str] = []

    def _callback(event: dict[str, object]) -> None:
        events.append(str(event.get("event")))

    result = fit_evaluate_file("examples/data/group_split_demo.csv", config, callbacks=[_callback])
    assert not result.leaderboard.empty
    assert "run_started" in events
    assert "run_finished" in events


def test_compare_group_strategies_file_runs_on_csv() -> None:
    result = compare_group_strategies_file(
        path="examples/data/rule_split_demo.csv",
        target="Target",
        rule_splits=["Temperature < 20", "Temperature >= 20"],
        experiment_modes=["full", "rule_split"],
        models=[LinearRegression()],
        feature_selectors=["none"],
        cv=3,
        random_state=42,
    )
    assert not result.leaderboard.empty
    assert (result.leaderboard["mode"] == "rule_split").any()


def test_export_report_csv(tmp_path: Path) -> None:
    leaderboard = pd.DataFrame([{"mode": "full", "cv_mean": 0.1, "test_score": 0.2}])
    from groupml.result import GroupMLResult

    result = GroupMLResult(leaderboard=leaderboard, recommendation="ok")
    path = tmp_path / "report.csv"
    out = export_report(result, path)
    assert out == path
    assert path.exists()
    written = pd.read_csv(path)
    assert list(written["mode"]) == ["full"]


def test_export_report_unsupported_extension(tmp_path: Path) -> None:
    leaderboard = pd.DataFrame([{"mode": "full", "cv_mean": 0.1, "test_score": 0.2}])
    from groupml.result import GroupMLResult

    result = GroupMLResult(leaderboard=leaderboard, recommendation="ok")
    with pytest.raises(ValueError, match="Unsupported report extension"):
        export_report(result, tmp_path / "report.json")


def test_export_raw_report_csv(tmp_path: Path) -> None:
    raw = pd.DataFrame([{"row_index": 0, "split_assignment": "cv_1", "actual": 1.0, "predicted": 0.9, "error": -0.1}])
    leaderboard = pd.DataFrame([{"mode": "full", "cv_mean": 0.1, "test_score": 0.2}])
    from groupml.result import GroupMLResult

    result = GroupMLResult(leaderboard=leaderboard, recommendation="ok", raw_report=raw)
    path = tmp_path / "raw.csv"
    out = export_raw_report(result, path)
    assert out == path
    assert path.exists()
    written = pd.read_csv(path)
    assert list(written["split_assignment"]) == ["cv_1"]


def test_export_summary_csv(tmp_path: Path) -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "experiment_name": "full:all_features",
                "mode": "full",
                "model": "linear_regression",
                "selector": "kbest_f",
                "cv_mean": 0.1,
                "test_score": 0.2,
            }
        ]
    )
    from groupml.result import GroupMLResult

    result = GroupMLResult(
        leaderboard=leaderboard,
        recommendation="ok",
        best_experiment=leaderboard.iloc[0].to_dict(),
        baseline_experiment=leaderboard.iloc[0].to_dict(),
    )
    path = tmp_path / "summary.csv"
    out = export_summary(result, path)
    assert out == path
    assert path.exists()
    written = pd.read_csv(path)
    assert "section" in written.columns
    assert "overview" in set(written["section"])


def test_export_summary_unsupported_extension(tmp_path: Path) -> None:
    leaderboard = pd.DataFrame([{"mode": "full", "cv_mean": 0.1, "test_score": 0.2}])
    from groupml.result import GroupMLResult

    result = GroupMLResult(leaderboard=leaderboard, recommendation="ok")
    with pytest.raises(ValueError, match="Unsupported summary extension"):
        export_summary(result, tmp_path / "summary.bin")


def test_default_report_filename_has_expected_shape() -> None:
    value = default_report_filename(prefix="demo", ext=".xlsx")
    assert value.startswith("demo_")
    assert value.endswith(".xlsx")


def test_default_summary_filename_has_expected_shape() -> None:
    value = default_summary_filename(prefix="demo_summary", ext=".csv")
    assert value.startswith("demo_summary_")
    assert value.endswith(".csv")


def test_export_reporting_bundle_uses_all_runs_with_failed_rows(tmp_path: Path) -> None:
    leaderboard = pd.DataFrame(
        [{"experiment_name": "full:all_features", "mode": "full", "cv_mean": 0.1, "test_score": 0.2}]
    )
    all_runs = pd.DataFrame(
        [
            {"experiment_name": "full:all_features", "mode": "full", "cv_mean": 0.1, "run_status": "ok"},
            {"experiment_name": "group_split:A", "mode": "group_split", "cv_mean": float("nan"), "run_status": "failed"},
        ]
    )
    from groupml.result import GroupMLResult

    result = GroupMLResult(leaderboard=leaderboard, recommendation="ok", all_runs=all_runs)
    outputs = export_reporting_bundle(result, tmp_path / "bundle.csv", report_format="csv")
    runs_path = outputs["all_runs"]
    written_runs = pd.read_csv(runs_path)
    assert "run_status" in written_runs.columns
    assert "failed" in set(written_runs["run_status"].astype(str))
