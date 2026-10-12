import json
import sys
from pathlib import Path
from ase import Atoms
from ase.io import write

import dpeva.cli as cli


def test_explore_cli_writes_manifest_from_executed_backend(monkeypatch, tmp_path: Path) -> None:
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
        f"#!{sys.executable}\n"
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
