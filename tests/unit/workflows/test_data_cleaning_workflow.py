import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dpeva.workflows.data_cleaning import DataCleaningWorkflow


class FakeSubSystem:
    def __init__(self, output):
        self.output = output

    def to_deepmd_npy(self, out_path):
        self.output.append(str(out_path))
        path = Path(out_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "set.000").mkdir(exist_ok=True)
        (path / "type.raw").write_text("0\n", encoding="utf-8")


class FakeSystem:
    def __init__(self, target_name, n_frames):
        self.target_name = target_name
        self.short_name = target_name
        self.n_frames = n_frames
        self.calls = []
        self.export_paths = []

    def __len__(self):
        return self.n_frames

    def sub_system(self, indices):
        self.calls.append(list(indices))
        return FakeSubSystem(self.export_paths)


def _build_parsed_result():
    energy = np.array([(0.00, 0.01), (0.00, 0.20)], dtype=[("data_e", float), ("pred_e", float)])
    force = np.array(
        [
            (0.0, 0.0, 0.0, 0.1, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.4, 0.0, 0.0),
        ],
        dtype=[
            ("data_fx", float),
            ("data_fy", float),
            ("data_fz", float),
            ("pred_fx", float),
            ("pred_fy", float),
            ("pred_fz", float),
        ],
    )
    return {
        "energy": energy,
        "force": force,
        "virial": None,
        "has_ground_truth": True,
        "dataname_list": [["sysA", 0, 1], ["sysA", 1, 1]],
        "datanames_nframe": {"sysA": 2},
    }


def test_data_cleaning_filters_and_exports(tmp_path):
    dataset_dir = tmp_path / "dataset"
    result_dir = tmp_path / "result"
    output_dir = tmp_path / "out"
    dataset_dir.mkdir()
    result_dir.mkdir()
    fake_system = FakeSystem("sysA", 2)
    config = {
        "dataset_dir": str(dataset_dir),
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "force_max_diff_threshold": 0.3,
    }
    with patch("dpeva.workflows.data_cleaning.setup_workflow_logger"), patch(
        "dpeva.workflows.data_cleaning.DPTestResultParser.parse", return_value=_build_parsed_result()
    ), patch("dpeva.workflows.data_cleaning.load_systems", return_value=[fake_system]):
        workflow = DataCleaningWorkflow(config)
        workflow.run()

    summary = json.loads((output_dir / "cleaning_summary.json").read_text(encoding="utf-8"))
    assert summary["frames"]["total"] == 2
    assert summary["frames"]["kept"] == 1
    assert summary["frames"]["dropped"] == 1
    assert summary["drop_trigger_counts"]["force_max_diff"] == 1
    assert fake_system.calls[0] == [0]
    assert fake_system.calls[1] == [1]
    frame_report = (output_dir / "frame_metrics.csv").read_text(encoding="utf-8")
    assert "sysA-0" in frame_report
    assert "sysA-1" in frame_report
    assert "force_max_diff" in frame_report


def test_data_cleaning_strict_alignment_fails_on_missing_frame(tmp_path):
    dataset_dir = tmp_path / "dataset"
    result_dir = tmp_path / "result"
    dataset_dir.mkdir()
    result_dir.mkdir()
    fake_system = FakeSystem("sysA", 3)
    config = {
        "dataset_dir": str(dataset_dir),
        "result_dir": str(result_dir),
        "output_dir": str(tmp_path / "out"),
        "force_max_diff_threshold": 0.3,
        "strict_alignment": True,
    }
    with patch("dpeva.workflows.data_cleaning.setup_workflow_logger"), patch(
        "dpeva.workflows.data_cleaning.DPTestResultParser.parse", return_value=_build_parsed_result()
    ), patch("dpeva.workflows.data_cleaning.load_systems", return_value=[fake_system]):
        workflow = DataCleaningWorkflow(config)
        with pytest.raises(ValueError, match="Missing frame alignment"):
            workflow.run()


def test_data_cleaning_passthrough_when_no_threshold(tmp_path):
    dataset_dir = tmp_path / "dataset"
    result_dir = tmp_path / "result"
    output_dir = tmp_path / "out"
    dataset_dir.mkdir()
    result_dir.mkdir()
    fake_system = FakeSystem("sysA", 2)
    config = {
        "dataset_dir": str(dataset_dir),
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
    }
    with patch("dpeva.workflows.data_cleaning.setup_workflow_logger"), patch(
        "dpeva.workflows.data_cleaning.DPTestResultParser.parse", return_value=_build_parsed_result()
    ), patch("dpeva.workflows.data_cleaning.load_systems", return_value=[fake_system]):
        workflow = DataCleaningWorkflow(config)
        workflow.run()
    summary = json.loads((output_dir / "cleaning_summary.json").read_text(encoding="utf-8"))
    assert summary["frames"]["kept"] == 2
    assert summary["frames"]["dropped"] == 0


def test_data_cleaning_all_thresholds_enabled(tmp_path):
    dataset_dir = tmp_path / "dataset"
    result_dir = tmp_path / "result"
    output_dir = tmp_path / "out"
    dataset_dir.mkdir()
    result_dir.mkdir()
    fake_system = FakeSystem("sysA", 2)
    config = {
        "dataset_dir": str(dataset_dir),
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "energy_diff_threshold": 0.05,
        "force_max_diff_threshold": 0.30,
        "stress_max_diff_threshold": 0.10,
    }
    virial = np.array(
        [
            tuple([0.0] * 9 + [0.01] * 9),
            tuple([0.0] * 9 + [0.50] * 9),
        ],
        dtype=[(f"data_v{i}", float) for i in range(9)] + [(f"pred_v{i}", float) for i in range(9)],
    )
    parsed = _build_parsed_result()
    parsed["virial"] = virial
    with patch("dpeva.workflows.data_cleaning.setup_workflow_logger"), patch(
        "dpeva.workflows.data_cleaning.DPTestResultParser.parse", return_value=parsed
    ), patch("dpeva.workflows.data_cleaning.load_systems", return_value=[fake_system]):
        workflow = DataCleaningWorkflow(config)
        workflow.run()
    summary = json.loads((output_dir / "cleaning_summary.json").read_text(encoding="utf-8"))
    assert summary["frames"]["kept"] == 1
    assert summary["frames"]["dropped"] == 1
    assert summary["drop_trigger_counts"]["energy_diff"] == 1
    assert summary["drop_trigger_counts"]["force_max_diff"] == 1
    assert summary["drop_trigger_counts"]["stress_max_diff"] == 1


def test_data_cleaning_propagates_missing_result_file_error(tmp_path):
    dataset_dir = tmp_path / "dataset"
    result_dir = tmp_path / "result"
    dataset_dir.mkdir()
    result_dir.mkdir()
    config = {
        "dataset_dir": str(dataset_dir),
        "result_dir": str(result_dir),
        "output_dir": str(tmp_path / "out"),
        "force_max_diff_threshold": 0.30,
    }
    with patch("dpeva.workflows.data_cleaning.setup_workflow_logger"), patch(
        "dpeva.workflows.data_cleaning.DPTestResultParser.parse",
        side_effect=FileNotFoundError("Energy file not found"),
    ):
        workflow = DataCleaningWorkflow(config)
        with pytest.raises(FileNotFoundError, match="Energy file not found"):
            workflow.run()
