from pathlib import Path
import json
from types import SimpleNamespace

from ase import Atoms
from ase.io import write
import pytest

from dpeva.exploration.atst_backend import ATSTToolsBackend
from dpeva.exploration.base import ExplorationRequest


def test_atst_backend_reports_missing_cli(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("dpeva.exploration.atst_backend.shutil.which", lambda _name: None)
    backend = ATSTToolsBackend()

    request = ExplorationRequest(
        backend="atst-tools",
        workflow_type="md",
        work_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
    )

    with pytest.raises(RuntimeError, match="Install dpeva\\[explore\\] or pip install atst-tools"):
        backend.run(request)


def test_atst_backend_runs_cli_and_collects_result_structures(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("workflow: md\n", encoding="utf-8")
    result_path = tmp_path / "result.extxyz"
    write(result_path, Atoms("H", positions=[[0.0, 0.0, 0.0]]))
    calls = []

    def fake_run(command, cwd, check, capture_output, text):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "check": check,
                "capture_output": capture_output,
                "text": text,
            }
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("dpeva.exploration.atst_backend.shutil.which", lambda _name: "/usr/bin/atst")
    monkeypatch.setattr("dpeva.exploration.atst_backend.subprocess.run", fake_run)
    backend = ATSTToolsBackend()

    result = backend.run(
        ExplorationRequest(
            backend="atst-tools",
            workflow_type="md",
            work_dir=tmp_path,
            config_path=config_path,
            metadata={"result_structure_paths": [result_path]},
        )
    )

    assert calls == [
        {
            "command": ["atst", "run", str(config_path)],
            "cwd": tmp_path,
            "check": False,
            "capture_output": True,
            "text": True,
        }
    ]
    assert result.status == "success"
    assert len(result.structures) == 1
    assert result.metrics["returncode"] == 0


def test_atst_backend_writes_input_artifacts_and_manifest(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("workflow: relax\n", encoding="utf-8")
    result_path = tmp_path / "result.extxyz"
    write(result_path, Atoms("H", positions=[[0.0, 0.0, 0.0]]))

    def fake_run(command, cwd, check, capture_output, text):
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("dpeva.exploration.atst_backend.shutil.which", lambda _name: "/usr/bin/atst")
    monkeypatch.setattr("dpeva.exploration.atst_backend.subprocess.run", fake_run)

    result = ATSTToolsBackend().run(
        ExplorationRequest(
            backend="atst-tools",
            workflow_type="relax",
            work_dir=tmp_path,
            config_path=config_path,
            input_structures=[Atoms("He", positions=[[0.0, 0.0, 0.0]])],
            metadata={"result_structure_paths": [result_path]},
        )
    )

    manifest_path = tmp_path / "dpeva_exploration_result.json"
    input_path = tmp_path / "dpeva_inputs" / "input_000000.extxyz"
    assert input_path.exists()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["returncode"] == 0
    assert manifest["input_artifacts"] == [str(input_path)]
    assert manifest["result_structure_artifacts"] == [str(result_path)]
    assert any(artifact.kind == "manifest" and artifact.path == manifest_path for artifact in result.artifacts)
    assert any(artifact.kind == "input_structure" and artifact.path == input_path for artifact in result.artifacts)


def test_atst_backend_fails_when_configured_result_structure_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("workflow: md\n", encoding="utf-8")
    missing_result = tmp_path / "missing.extxyz"

    def fake_run(command, cwd, check, capture_output, text):
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("dpeva.exploration.atst_backend.shutil.which", lambda _name: "/usr/bin/atst")
    monkeypatch.setattr("dpeva.exploration.atst_backend.subprocess.run", fake_run)

    result = ATSTToolsBackend().run(
        ExplorationRequest(
            backend="atst-tools",
            workflow_type="md",
            work_dir=tmp_path,
            config_path=config_path,
            metadata={"result_structure_paths": [missing_result]},
        )
    )

    assert result.status == "failed"
    assert str(missing_result) in result.error_message
    assert result.structures == []


def test_atst_backend_rejects_unsupported_workflow(tmp_path: Path) -> None:
    backend = ATSTToolsBackend()
    request = ExplorationRequest(
        backend="atst-tools",
        workflow_type="neb",
        work_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
    )

    with pytest.raises(ValueError, match="Unsupported atst-tools workflow_type"):
        backend.run(request)
