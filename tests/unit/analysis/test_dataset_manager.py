import json
from unittest.mock import patch

from dpeva.analysis.dataset import DatasetAnalysisManager


class _FakeSystem:
    def __init__(self):
        self.target_name = "sys_a"
        self.data = {
            "atom_names": ["H", "O"],
            "energies": [10.0, 12.0],
            "forces": [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]],
            ],
            "virials": [
                [[-3.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, -3.0]],
                [[-6.0, 0.0, 0.0], [0.0, -6.0, 0.0], [0.0, 0.0, -6.0]],
            ],
            "cells": [
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            ],
        }

    def __getitem__(self, key):
        if key == "atom_types":
            return [0, 1]
        raise KeyError(key)

    def get_nframes(self):
        return 2


@patch("dpeva.analysis.dataset.InferenceVisualizer")
@patch("dpeva.analysis.dataset.load_systems")
def test_dataset_manager_analyze_success(mock_load_systems, MockVisualizer, tmp_path):
    mock_load_systems.return_value = [_FakeSystem()]
    manager = DatasetAnalysisManager(ref_energies={"H": -1.0, "O": -2.0})
    output_dir = tmp_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = manager.analyze(tmp_path / "dataset", output_dir)

    assert summary["n_systems"] == 1
    assert summary["n_frames"] == 2
    assert summary["energy_per_atom"]["count"] == 2
    assert summary["force_magnitude"]["count"] == 4
    assert summary["pressure_gpa"]["count"] == 2
    assert summary["element_categories"] == ["H", "O"]
    assert summary["element_count_by_atom"]["H"] == 2
    assert summary["element_count_by_atom"]["O"] == 2
    assert summary["element_ratio_by_atom"]["H"] == 0.5
    assert summary["system_element_presence"]["H"]["system_count"] == 1
    assert summary["frame_element_presence"]["H"]["frame_count"] == 2
    assert summary["frame_element_presence"]["H"]["frame_ratio"] == 1.0
    assert "cohesive_energy_per_atom" in summary
    assert (output_dir / "dataset_stats.json").exists()
    assert (output_dir / "dataset_frame_summary.csv").exists()
    assert (output_dir / "dataset_element_ratio.png").exists()
    assert (output_dir / "dataset_element_presence.png").exists()
    MockVisualizer.return_value.plot_distribution.assert_called()

    with open(output_dir / "dataset_stats.json") as f:
        written = json.load(f)
    assert "pressure_gpa" in written


@patch("dpeva.analysis.dataset.load_systems")
def test_dataset_manager_analyze_empty(mock_load_systems, tmp_path):
    mock_load_systems.return_value = []
    manager = DatasetAnalysisManager()
    import pytest
    with pytest.raises(ValueError, match="No valid systems found"):
        manager.analyze(tmp_path / "dataset", tmp_path / "analysis")
