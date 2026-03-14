from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from dpeva.labeling.postprocess import AbacusPostProcessor


def test_check_convergence_true(tmp_path: Path):
    task_dir = tmp_path / "task"
    out_dir = task_dir / "OUT.ABACUS"
    out_dir.mkdir(parents=True)
    (out_dir / "running_scf.log").write_text(
        "charge density convergence is achieved\nTOTAL-FORCE (eV/Angstrom)\n"
    )

    pp = AbacusPostProcessor({})
    assert pp.check_convergence(task_dir) is True


def test_check_convergence_false(tmp_path: Path):
    task_dir = tmp_path / "task"
    out_dir = task_dir / "OUT.ABACUS"
    out_dir.mkdir(parents=True)
    (out_dir / "running_scf.log").write_text("convergence has not been achieved\n")

    pp = AbacusPostProcessor({})
    assert pp.check_convergence(task_dir) is False


def test_check_convergence_missing_force_block(tmp_path: Path):
    task_dir = tmp_path / "task"
    out_dir = task_dir / "OUT.ABACUS"
    out_dir.mkdir(parents=True)
    (out_dir / "running_scf.log").write_text("charge density convergence is achieved\n")

    pp = AbacusPostProcessor({})
    assert pp.check_convergence(task_dir) is False


@patch("dpeva.labeling.postprocess.dpdata.LabeledSystem")
def test_load_data_skips_incomplete_system(mock_labeled_system):
    mock_system = MagicMock()
    mock_labeled_system.return_value = mock_system
    mock_system.__len__.return_value = 1

    pp = AbacusPostProcessor({})
    with patch.object(pp, "_is_labeled_system_complete", return_value=(False, "short_array:forces")):
        assert pp.load_data(Path("/tmp/fake-task")) is None


def test_compute_metrics_returns_empty_dataframe_for_empty_input():
    pp = AbacusPostProcessor({})
    df = pp.compute_metrics([])
    assert df.empty
    assert "cohesive_energy_per_atom" in df.columns


def test_classify_task_status_bad_converged(tmp_path: Path):
    task_dir = tmp_path / "task"
    out_dir = task_dir / "OUT.ABACUS"
    out_dir.mkdir(parents=True)
    (out_dir / "running_scf.log").write_text("charge density convergence is achieved\n")

    pp = AbacusPostProcessor({})
    status, reason = pp.classify_task_status(task_dir)
    assert status == "bad_converged"
    assert reason == "missing_total_force_block"


@patch("dpeva.labeling.postprocess.dpdata.MultiSystems")
def test_export_data_uses_indexable_system_list(mock_multi_systems, tmp_path: Path):
    pp = AbacusPostProcessor({})
    sys0 = MagicMock()
    sys1 = MagicMock()
    sys0.sub_system.return_value = MagicMock()
    sys1.sub_system.return_value = MagicMock()
    systems = [sys0, sys1]
    df_clean = pd.DataFrame(
        [
            {"sys_idx": 0, "frame_idx": 0},
            {"sys_idx": 1, "frame_idx": 0},
        ]
    )

    pp.export_data(systems, df_clean, tmp_path / "cleaned")

    sys0.sub_system.assert_called_once()
    sys1.sub_system.assert_called_once()
