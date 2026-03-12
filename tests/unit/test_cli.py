import sys
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import dpeva.cli as cli


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


def test_load_and_resolve_config_reports_invalid_json(tmp_path):
    config_path = _write_config(tmp_path, "{invalid_json")
    with pytest.raises(cli.CLIUserInputError, match="Invalid JSON in config file"):
        cli.load_and_resolve_config(config_path)


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


def test_handle_label_stage_all(monkeypatch, tmp_path):
    args = SimpleNamespace(config="config.json", stage="all")
    monkeypatch.setattr(cli, "load_and_resolve_config", lambda _p: _label_config_dict(tmp_path))

    with patch("dpeva.workflows.labeling.LabelingWorkflow") as MockWorkflow:
        cli.handle_label(args)
        MockWorkflow.return_value.run.assert_called_once()
