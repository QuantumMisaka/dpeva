import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from ase import Atoms
from ase.io import write

import dpeva.cli as cli


class _FakeMultiSystems(list):
    def append(self, system):  # noqa: D102 - mirrors the dpdata API used by LabelingWorkflow.
        super().append(system)


def test_v080_label_prepare_cli_acceptance(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "sampled_dpdata"
    data_dir.mkdir()
    (data_dir / "type.raw").write_text("0\n", encoding="utf-8")

    config_path = tmp_path / "label.json"
    config_path.write_text(
        json.dumps(
            {
                "work_dir": str(tmp_path / "label_work"),
                "input_data_path": str(data_dir),
                "pp_dir": str(tmp_path / "pp"),
                "orb_dir": str(tmp_path / "orb"),
                "submission": {"backend": "local"},
                "dft_params": {"ecutwfc": 100},
                "pp_map": {"H": "H.upf"},
                "orb_map": {"H": "H.orb"},
                "attempt_params": [],
            }
        ),
        encoding="utf-8",
    )

    prepared_bundle = tmp_path / "label_work" / "inputs" / "N_000000"
    prepared_bundle.mkdir(parents=True)
    mock_manager = MagicMock()
    mock_manager.prepare_tasks.return_value = [prepared_bundle]

    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "label", str(config_path), "--stage", "prepare"])
    monkeypatch.setattr("dpeva.workflows.labeling.LabelingManager", lambda _config: mock_manager)
    monkeypatch.setattr("dpeva.workflows.labeling.dpdata.MultiSystems", _FakeMultiSystems)
    monkeypatch.setattr(
        "dpeva.workflows.labeling.load_systems",
        lambda *_args, **_kwargs: [SimpleNamespace()],
    )

    cli.main()

    mock_manager.prepare_tasks.assert_called_once()
    dataset_map = mock_manager.prepare_tasks.call_args.args[0]
    assert list(dataset_map) == ["sampled_dpdata"]


def test_v080_explore_cli_acceptance_writes_manifest(monkeypatch, tmp_path: Path) -> None:
    work_dir = tmp_path / "explore_work"
    result_path = work_dir / "result.extxyz"
    input_path = tmp_path / "input.extxyz"
    backend_config_path = tmp_path / "atst.yaml"
    config_path = tmp_path / "explore.json"
    bin_dir = tmp_path / "bin"
    fake_atst = bin_dir / "atst"

    work_dir.mkdir()
    bin_dir.mkdir()
    write(input_path, Atoms("He", positions=[[0.0, 0.0, 0.0]]))
    backend_config_path.write_text("calculation:\n  type: relax\n", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "work_dir": str(work_dir),
                "backend": "atst-tools",
                "workflow_type": "relax",
                "backend_config_path": str(backend_config_path),
                "input_structure_paths": [str(input_path)],
                "result_structure_paths": [str(result_path)],
            }
        ),
        encoding="utf-8",
    )
    fake_atst.write_text(
        "#!/usr/bin/env python\n"
        "from pathlib import Path\n"
        "Path('result.extxyz').write_text('1\\nProperties=species:S:1:pos:R:3\\nH 0 0 0\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_atst.chmod(0o755)

    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "explore", str(config_path)])
    monkeypatch.setenv("PATH", f"{bin_dir}:{str(Path('/usr/bin'))}")

    cli.main()

    manifest_path = work_dir / "dpeva_exploration_result.json"
    input_snapshot = work_dir / "dpeva_inputs" / "input_000000.extxyz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["returncode"] == 0
    assert manifest["result_structure_artifacts"] == [str(result_path)]
    assert input_snapshot.exists()


def test_v080_progress_audit_records_tg7_acceptance() -> None:
    audit = Path(
        "docs/archive/v0.8.0/reports/2026-06-11-v0.8.0-atst-integration-progress-audit.md"
    ).read_text(encoding="utf-8")

    assert "| TG7 最小集成验证 | 已完成 |" in audit
    assert "tests/integration/test_v080_atst_acceptance.py" in audit
