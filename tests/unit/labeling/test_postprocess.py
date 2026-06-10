from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from dpeva.labeling.postprocess import AbacusPostProcessor


class FakeLabeledSystem:
    def __init__(self, data, nframes=1):
        self.data = data
        self._nframes = nframes

    def __len__(self):
        return self._nframes

    def get_nframes(self):
        return self._nframes

    def __getitem__(self, key):
        return self.data[key]


def test_is_labeled_system_complete_detects_missing_key():
    system = FakeLabeledSystem({"energies": [1.0], "forces": [1.0], "cells": [1.0]})
    valid, reason = AbacusPostProcessor._is_labeled_system_complete(system)
    assert valid is False
    assert reason == "missing_key:virials"


def test_is_labeled_system_complete_detects_short_array():
    system = FakeLabeledSystem(
        {
            "energies": [1.0],
            "forces": [1.0],
            "virials": [],
            "cells": [1.0],
        }
    )
    valid, reason = AbacusPostProcessor._is_labeled_system_complete(system)
    assert valid is False
    assert reason == "short_array:virials"


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
def test_load_data_skips_incomplete_system(mock_labeled_system, tmp_path: Path):
    mock_system = MagicMock()
    mock_labeled_system.return_value = mock_system
    mock_system.__len__.return_value = 1

    pp = AbacusPostProcessor({})
    with patch.object(pp, "_is_labeled_system_complete", return_value=(False, "short_array:forces")):
        assert pp.load_data(tmp_path / "fake-task") is None


@patch("dpeva.labeling.postprocess.dpdata.LabeledSystem")
def test_load_data_returns_complete_system(mock_labeled_system, tmp_path: Path):
    mock_system = MagicMock()
    mock_system.__len__.return_value = 1
    mock_labeled_system.return_value = mock_system
    pp = AbacusPostProcessor({})

    with patch.object(pp, "_is_labeled_system_complete", return_value=(True, "")):
        result = pp.load_data(tmp_path / "task")

    assert result is mock_system


@patch("dpeva.labeling.postprocess.dpdata.LabeledSystem", side_effect=RuntimeError("boom"))
def test_load_data_logs_error_when_dpdata_fails(mock_labeled_system, tmp_path: Path):
    pp = AbacusPostProcessor({})

    with patch("dpeva.labeling.postprocess.logger.error") as mock_error:
        result = pp.load_data(tmp_path / "task")

    assert result is None
    mock_error.assert_called_once()


def test_compute_metrics_returns_empty_dataframe_for_empty_input():
    pp = AbacusPostProcessor({})
    df = pp.compute_metrics([])
    assert df.empty
    assert "cohesive_energy_per_atom" in df.columns


def test_compute_metrics_fits_missing_reference_energy_for_single_element():
    system = FakeLabeledSystem(
        {
            "atom_names": ["H"],
            "atom_types": np.array([0, 0]),
            "energies": np.array([-2.0]),
            "forces": np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]),
            "virials": np.array([np.eye(3)]),
            "cells": np.array([np.eye(3)]),
        }
    )
    pp = AbacusPostProcessor({})

    df = pp.compute_metrics([system])

    assert len(df) == 1
    assert df.iloc[0]["num_atoms"] == 2
    assert df.iloc[0]["max_force"] == 2.0
    assert abs(df.iloc[0]["cohesive_energy_per_atom"]) < 1e-8


def test_compute_metrics_skips_system_missing_required_key():
    system = FakeLabeledSystem(
        {
            "atom_names": ["H"],
            "atom_types": np.array([0]),
            "energies": np.array([-1.0]),
            "forces": np.array([[[0.0, 0.0, 0.0]]]),
            "cells": np.array([np.eye(3)]),
        }
    )
    pp = AbacusPostProcessor({})

    with patch("dpeva.labeling.postprocess.logger.warning") as mock_warning:
        df = pp.compute_metrics([system])

    assert df.empty
    assert any("Skip system 0 due to missing key" in call.args[0] for call in mock_warning.call_args_list)


def test_compute_metrics_skips_frame_with_empty_force_array():
    system = FakeLabeledSystem(
        {
            "atom_names": ["H"],
            "atom_types": np.array([0]),
            "energies": np.array([-1.0]),
            "forces": np.array([[]]),
            "virials": np.array([np.eye(3)]),
            "cells": np.array([np.eye(3)]),
        }
    )
    pp = AbacusPostProcessor({"ref_energies": {"H": -1.0}})

    df = pp.compute_metrics([system])

    assert df.empty


def test_compute_metrics_sets_zero_pressure_for_singular_cell():
    system = FakeLabeledSystem(
        {
            "atom_names": ["H"],
            "atom_types": np.array([0]),
            "energies": np.array([-1.0]),
            "forces": np.array([[[0.0, 0.0, 1.5]]]),
            "virials": np.array([np.eye(3)]),
            "cells": np.array([np.zeros((3, 3))]),
        }
    )
    pp = AbacusPostProcessor({"ref_energies": {"H": -1.0}})

    df = pp.compute_metrics([system])

    assert len(df) == 1
    assert df.iloc[0]["pressure_gpa"] == 0.0
    assert df.iloc[0]["volume"] == 0.0


def test_compute_metrics_handles_multiple_frames_and_systems():
    system_a = FakeLabeledSystem(
        {
            "atom_names": ["H"],
            "atom_types": np.array([0]),
            "energies": np.array([-1.0, -1.2]),
            "forces": np.array(
                [
                    [[0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 2.0]],
                ]
            ),
            "virials": np.array([np.eye(3), np.eye(3) * 2]),
            "cells": np.array([np.eye(3), np.eye(3)]),
        },
        nframes=2,
    )
    system_b = FakeLabeledSystem(
        {
            "atom_names": ["H"],
            "atom_types": np.array([0]),
            "energies": np.array([-0.5]),
            "forces": np.array([[[0.0, 0.0, 0.5]]]),
            "virials": np.array([np.eye(3)]),
            "cells": np.array([np.eye(3)]),
        }
    )
    pp = AbacusPostProcessor({"ref_energies": {"H": -1.0}})

    df = pp.compute_metrics([system_a, system_b])

    assert len(df) == 3
    assert list(df["sys_idx"]) == [0, 0, 1]
    assert list(df["frame_idx"]) == [0, 1, 0]


def test_classify_task_status_bad_converged(tmp_path: Path):
    task_dir = tmp_path / "task"
    out_dir = task_dir / "OUT.ABACUS"
    out_dir.mkdir(parents=True)
    (out_dir / "running_scf.log").write_text("charge density convergence is achieved\n")

    pp = AbacusPostProcessor({})
    status, reason = pp.classify_task_status(task_dir)
    assert status == "bad_converged"
    assert reason == "missing_total_force_block"


def test_classify_task_status_failed_when_log_missing(tmp_path: Path):
    pp = AbacusPostProcessor({})
    status, reason = pp.classify_task_status(tmp_path / "missing")
    assert status == "failed"
    assert reason == "missing_log"


def test_classify_task_status_failed_without_converged_marker(tmp_path: Path):
    task_dir = tmp_path / "task"
    out_dir = task_dir / "OUT.ABACUS"
    out_dir.mkdir(parents=True)
    (out_dir / "running_scf.log").write_text("random line\n")

    pp = AbacusPostProcessor({})
    status, reason = pp.classify_task_status(task_dir)
    assert status == "failed"
    assert reason == "no_converged_marker"


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


def test_filter_data_skips_missing_cohesive_energy_with_warning():
    pp = AbacusPostProcessor({"cleaning_thresholds": {"cohesive_energy": 0.2, "force": 1.0}})
    df = pd.DataFrame(
        {
            "max_force": [0.2, 2.0],
            "pressure_gpa": [0.0, 0.0],
            "num_atoms": [2, 2],
            "cohesive_energy_per_atom": [np.nan, np.nan],
        }
    )

    with patch("dpeva.labeling.postprocess.logger.warning") as mock_warning:
        cleaned = pp.filter_data(df)

    mock_warning.assert_called_once()
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["max_force"] == 0.2


def test_filter_data_supports_legacy_energy_threshold_key():
    pp = AbacusPostProcessor({"cleaning_thresholds": {"energy": -0.1, "force": 10.0, "stress": 10.0}})
    df = pd.DataFrame(
        {
            "cohesive_energy_per_atom": [-0.2, 0.1],
            "max_force": [0.1, 0.1],
            "pressure_gpa": [0.0, 0.0],
            "num_atoms": [2, 2],
        }
    )

    cleaned = pp.filter_data(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["cohesive_energy_per_atom"] == -0.2


def test_build_no_contribution_hint_covers_zero_cases():
    assert AbacusPostProcessor.build_no_contribution_hint(conv=0, clean=3) == (
        "Hint: no converged tasks, this branch contributes no training data."
    )
    assert AbacusPostProcessor.build_no_contribution_hint(conv=2, clean=0) == (
        "Hint: converged tasks were fully filtered, this branch contributes no training data."
    )
    assert AbacusPostProcessor.build_no_contribution_hint(conv=2, clean=1) is None


@patch("dpeva.labeling.postprocess.dpdata.MultiSystems")
def test_export_data_logs_slice_failures_and_continues(mock_multi_systems, tmp_path: Path):
    pp = AbacusPostProcessor({})
    sys0 = MagicMock()
    sys1 = MagicMock()
    sys0.sub_system.side_effect = RuntimeError("slice failed")
    sys1.sub_system.return_value = MagicMock()
    systems = [sys0, sys1]
    df_clean = pd.DataFrame(
        [
            {"sys_idx": 0, "frame_idx": 0},
            {"sys_idx": 1, "frame_idx": 1},
        ]
    )

    with patch("dpeva.labeling.postprocess.logger.error") as mock_error:
        pp.export_data(systems, df_clean, tmp_path / "cleaned")

    mock_error.assert_any_call("Failed to slice system 0: slice failed")
    sys1.sub_system.assert_called_once()


@patch("dpeva.labeling.postprocess.dpdata.MultiSystems")
def test_export_data_logs_export_failure(mock_multi_systems, tmp_path: Path):
    pp = AbacusPostProcessor({})
    system = MagicMock()
    system.sub_system.return_value = MagicMock()
    df_clean = pd.DataFrame([{"sys_idx": 0, "frame_idx": 0}])
    new_ms = mock_multi_systems.return_value
    new_ms.to.side_effect = RuntimeError("write failed")

    with patch("dpeva.labeling.postprocess.logger.error") as mock_error:
        pp.export_data([system], df_clean, tmp_path / "cleaned")

    mock_error.assert_any_call("Export failed: write failed")
