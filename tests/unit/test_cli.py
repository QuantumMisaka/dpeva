import json
import sys
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import dpeva.cli as cli
from dpeva.run.doctor import DoctorCheck, DoctorReport
from dpeva.config_migration import MigrationResult


def _write_config(tmp_path, content='{}'):
    config_path = Path(tmp_path) / "config.json"
    config_path.write_text(content, encoding="utf-8")
    return str(config_path)


def test_cli_dispatch_train_without_banner(monkeypatch, tmp_path):
    called = {"train": False}
    config_path = _write_config(tmp_path)

    def fake_train(args):
        called["train"] = True
        assert args.config == config_path

    monkeypatch.setattr(cli, "handle_train", fake_train)
    monkeypatch.setattr(cli, "show_banner", lambda: (_ for _ in ()).throw(AssertionError("banner should not be called")))
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "train", config_path])

    cli.main()
    assert called["train"] is True


def test_cli_dispatch_eval_card_without_banner(monkeypatch, tmp_path):
    called = {"eval_card": False}
    config_path = _write_config(tmp_path)

    def fake_eval_card(args):
        called["eval_card"] = True
        assert args.config == config_path

    monkeypatch.setattr(cli, "handle_eval_card", fake_eval_card)
    monkeypatch.setattr(cli, "show_banner", lambda: (_ for _ in ()).throw(AssertionError("banner should not be called")))
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "eval-card", config_path])

    cli.main()
    assert called["eval_card"] is True


def test_cli_exit_on_handler_error(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)

    def fake_train(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "handle_train", fake_train)
    monkeypatch.setattr(cli, "show_banner", lambda: None)
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "train", config_path])

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_cli_rejects_missing_config_early(monkeypatch, capsys):
    missing_path = "not_exists_config.json"
    monkeypatch.setattr(cli, "show_banner", lambda: None)
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "train", missing_path])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert f"Config file not found: {missing_path}" in captured.err


def test_cli_rejects_label_stage_token_as_config(monkeypatch, capsys):
    monkeypatch.setattr(cli, "show_banner", lambda: None)
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "label", "prepare"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "--stage prepare" in captured.err


def test_doctor_json_exit_zero(monkeypatch, capsys):
    report = DoctorReport(status="ok", checks=[])
    monkeypatch.setattr("dpeva.run.doctor.build_doctor_report", lambda: report)
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "doctor", "--json"])

    cli.main()

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "schema_version": "1.0",
        "status": "ok",
        "checks": [],
    }


def test_doctor_json_exit_one(monkeypatch, capsys):
    report = DoctorReport(
        status="failed",
        checks=[
            DoctorCheck(
                name="deepmd",
                status="missing",
                detail="dp executable not found",
            )
        ],
    )
    monkeypatch.setattr("dpeva.run.doctor.build_doctor_report", lambda: report)
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "doctor", "--json"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["status"] == "failed"


def test_doctor_json_suppresses_banner(monkeypatch, capsys):
    monkeypatch.setattr(
        "dpeva.run.doctor.build_doctor_report",
        lambda: DoctorReport(status="ok", checks=[]),
    )
    monkeypatch.setattr(
        cli,
        "show_banner",
        lambda: (_ for _ in ()).throw(AssertionError("JSON doctor must not show banner")),
    )
    monkeypatch.setattr(sys, "argv", ["dpeva", "doctor", "--json"])

    cli.main()

    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_doctor_human_output_is_default(monkeypatch, capsys):
    monkeypatch.setattr(
        "dpeva.run.doctor.build_doctor_report",
        lambda: DoctorReport(
            status="ok",
            checks=[
                DoctorCheck(
                    name="deepmd",
                    status="ok",
                    version="3.2.0",
                    detail="required >= 3.2.0, < 3.3",
                )
            ],
        ),
    )
    monkeypatch.setattr(cli, "show_banner", lambda: None)
    monkeypatch.setattr(sys, "argv", ["dpeva", "doctor"])

    cli.main()

    assert capsys.readouterr().out == (
        "deepmd: ok - required >= 3.2.0, < 3.3\n"
    )


def test_doctor_human_output_failed_exits_one(monkeypatch, capsys):
    monkeypatch.setattr(
        "dpeva.run.doctor.build_doctor_report",
        lambda: DoctorReport(
            status="failed",
            checks=[
                DoctorCheck(
                    name="deepmd",
                    status="missing",
                    detail="dp executable not found",
                )
            ],
        ),
    )
    monkeypatch.setattr(cli, "show_banner", lambda: None)
    monkeypatch.setattr(sys, "argv", ["dpeva", "doctor"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == "deepmd: missing - dp executable not found\n"


def test_load_and_resolve_config_reports_invalid_json(tmp_path):
    config_path = _write_config(tmp_path, "{invalid_json")
    with pytest.raises(cli.CLIUserInputError, match="Invalid JSON in config file"):
        cli.load_and_resolve_config(config_path)


def test_load_and_resolve_config_migrates_and_warns_without_overwriting_source(
    tmp_path, caplog
):
    config_path = _write_config(
        tmp_path,
        json.dumps({"backend": "slurm", "data_path": "data"}),
    )

    with caplog.at_level("WARNING"):
        result = cli.load_config_with_metadata(config_path)

    assert isinstance(result, MigrationResult)
    assert result.normalized["submission"]["backend"] == "slurm"
    assert "backend" not in result.normalized
    assert json.loads(Path(config_path).read_text(encoding="utf-8")) == {
        "backend": "slurm",
        "data_path": "data",
    }
    assert "legacy config field backend; use submission.backend; removal target 1.0" in caplog.text


def test_load_and_resolve_config_preserves_legacy_dict_contract(tmp_path):
    config_path = _write_config(
        tmp_path,
        json.dumps({"backend": "slurm", "data_path": "data"}),
    )

    result = cli.load_and_resolve_config(config_path)

    assert isinstance(result, dict)
    assert result["submission"]["backend"] == "slurm"
    assert "backend" not in result


def _label_config_dict(tmp_path):
    return {
        "work_dir": str(tmp_path / "work"),
        "input_data_path": str(tmp_path / "data"),
        "submission": {"backend": "local"},
        "dft_params": {},
        "pp_dir": "/tmp/pp",
        "orb_dir": "/tmp/orb",
    }


def test_handle_label_stage_prepare(monkeypatch, tmp_path):
    args = SimpleNamespace(config="config.json", stage="prepare")
    monkeypatch.setattr(cli, "load_and_resolve_config", lambda _p: _label_config_dict(tmp_path))

    with patch("dpeva.workflows.labeling.LabelingWorkflow") as MockWorkflow:
        cli.handle_label(args)
        MockWorkflow.return_value.run_prepare.assert_called_once()
        MockWorkflow.return_value.run_execute.assert_not_called()
        MockWorkflow.return_value.run_postprocess.assert_not_called()
        MockWorkflow.return_value.run.assert_not_called()


def test_handle_label_stage_execute(monkeypatch, tmp_path):
    args = SimpleNamespace(config="config.json", stage="execute")
    monkeypatch.setattr(cli, "load_and_resolve_config", lambda _p: _label_config_dict(tmp_path))

    with patch("dpeva.workflows.labeling.LabelingWorkflow") as MockWorkflow:
        cli.handle_label(args)
        MockWorkflow.return_value.run_execute.assert_called_once()
        MockWorkflow.return_value.run_prepare.assert_not_called()
        MockWorkflow.return_value.run_postprocess.assert_not_called()
        MockWorkflow.return_value.run.assert_not_called()


def test_handle_label_stage_postprocess(monkeypatch, tmp_path):
    args = SimpleNamespace(config="config.json", stage="postprocess")
    monkeypatch.setattr(cli, "load_and_resolve_config", lambda _p: _label_config_dict(tmp_path))

    with patch("dpeva.workflows.labeling.LabelingWorkflow") as MockWorkflow:
        cli.handle_label(args)
        MockWorkflow.return_value.run_postprocess.assert_called_once()
        MockWorkflow.return_value.run_prepare.assert_not_called()
        MockWorkflow.return_value.run_execute.assert_not_called()
        MockWorkflow.return_value.run.assert_not_called()


def test_handle_label_stage_extract(monkeypatch, tmp_path):
    args = SimpleNamespace(config="config.json", stage="extract")
    monkeypatch.setattr(cli, "load_and_resolve_config", lambda _p: _label_config_dict(tmp_path))

    with patch("dpeva.workflows.labeling.LabelingWorkflow") as MockWorkflow:
        cli.handle_label(args)
        MockWorkflow.return_value.run_extract.assert_called_once()
        MockWorkflow.return_value.run_prepare.assert_not_called()
        MockWorkflow.return_value.run_execute.assert_not_called()
        MockWorkflow.return_value.run_postprocess.assert_not_called()
        MockWorkflow.return_value.run.assert_not_called()


def test_handle_label_stage_all(monkeypatch, tmp_path):
    args = SimpleNamespace(config="config.json", stage="all")
    monkeypatch.setattr(cli, "load_and_resolve_config", lambda _p: _label_config_dict(tmp_path))

    with patch("dpeva.workflows.labeling.LabelingWorkflow") as MockWorkflow:
        cli.handle_label(args)
        MockWorkflow.return_value.run.assert_called_once()


def test_handle_analysis_passes_absolute_config_path(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    args = SimpleNamespace(config=config_path)
    monkeypatch.setattr(cli, "load_and_resolve_config", lambda _p: {"mode": "dataset", "dataset_dir": str(tmp_path)})
    with patch("dpeva.workflows.analysis.AnalysisWorkflow") as MockWorkflow:
        cli.handle_analysis(args)
        MockWorkflow.assert_called_once()
        call_args = MockWorkflow.call_args
        assert call_args.kwargs["config_path"] == str(Path(config_path).resolve())
        MockWorkflow.return_value.run.assert_called_once()


def test_handle_clean_runs_workflow(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path)
    args = SimpleNamespace(config=config_path)
    monkeypatch.setattr(
        cli,
        "load_and_resolve_config",
        lambda _p: {"dataset_dir": str(tmp_path), "result_dir": str(tmp_path)},
    )
    with patch("dpeva.workflows.data_cleaning.DataCleaningWorkflow") as MockWorkflow:
        cli.handle_clean(args)
        MockWorkflow.assert_called_once()
        MockWorkflow.return_value.run.assert_called_once()
