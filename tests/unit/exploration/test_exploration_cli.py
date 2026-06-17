from pathlib import Path
from types import SimpleNamespace

import dpeva.cli as cli


def test_handle_explore_runs_exploration(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "dpeva-explore.json"
    backend_config = tmp_path / "atst.yaml"
    backend_config.write_text("workflow: md\n", encoding="utf-8")
    config_path.write_text(
        (
            "{"
            f'"work_dir": "{tmp_path}", '
            '"backend": "atst-tools", '
            '"workflow_type": "md", '
            f'"backend_config_path": "{backend_config}"'
            "}"
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_exploration(request):
        captured["request"] = request
        return SimpleNamespace(status="success")

    monkeypatch.setattr("dpeva.exploration.manager.run_exploration", fake_run_exploration)

    cli.handle_explore(SimpleNamespace(config=str(config_path)))

    request = captured["request"]
    assert request.backend == "atst-tools"
    assert request.workflow_type == "md"
    assert request.config_path == backend_config
