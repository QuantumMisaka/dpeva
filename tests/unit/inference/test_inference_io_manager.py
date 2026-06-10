import json

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from dpeva.constants import FILENAME_STATS_JSON
from dpeva.inference.managers import InferenceIOManager


def test_discover_models_stops_at_first_missing_index(tmp_path):
    work_dir = tmp_path / "work"
    (work_dir / "0").mkdir(parents=True)
    (work_dir / "0" / "model.ckpt.pt").touch()
    (work_dir / "2").mkdir(parents=True)
    (work_dir / "2" / "model.ckpt.pt").touch()

    manager = InferenceIOManager(str(work_dir))

    models = manager.discover_models()

    assert models == [str(work_dir / "0" / "model.ckpt.pt")]


def test_load_composition_info_returns_none_for_invalid_path(tmp_path):
    manager = InferenceIOManager(str(tmp_path))

    atom_counts, atom_nums = manager.load_composition_info(str(tmp_path / "missing"))

    assert atom_counts is None
    assert atom_nums is None


@patch("dpeva.inference.managers.load_systems", side_effect=RuntimeError("dpdata failed"))
def test_load_composition_info_falls_back_to_none_on_exception(mock_load_systems, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manager = InferenceIOManager(str(tmp_path))

    atom_counts, atom_nums = manager.load_composition_info(str(data_dir))

    assert atom_counts is None
    assert atom_nums is None


@patch("dpeva.inference.managers.load_systems")
def test_load_composition_info_replicates_counts_per_frame(mock_load_systems, tmp_path):
    system = MagicMock()
    system.__getitem__.side_effect = lambda key: {
        "atom_names": ["H", "O"],
        "atom_types": np.array([0, 1, 1]),
    }[key]
    system.get_nframes.return_value = 2
    mock_load_systems.return_value = [system]
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    manager = InferenceIOManager(str(tmp_path))
    atom_counts, atom_nums = manager.load_composition_info(str(data_dir))

    assert atom_counts == [{"H": 1, "O": 2}, {"H": 1, "O": 2}]
    assert atom_nums == [3, 3]


def test_save_statistics_serializes_numpy_types(tmp_path):
    manager = InferenceIOManager(str(tmp_path))
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    manager.save_statistics(
        str(analysis_dir),
        {"count": np.int64(2), "values": np.array([1.0, 2.0]), "mean": np.float64(1.5)},
    )

    saved = json.loads((analysis_dir / FILENAME_STATS_JSON).read_text())
    assert saved == {"count": 2, "values": [1.0, 2.0], "mean": 1.5}


def test_save_summary_writes_csv_only_when_metrics_present(tmp_path):
    manager = InferenceIOManager(str(tmp_path))
    manager.save_summary([])
    assert not (tmp_path / "inference_summary.csv").exists()

    manager.save_summary([{"model_idx": 0, "e_mae": 0.1}])

    saved = pd.read_csv(tmp_path / "inference_summary.csv")
    assert list(saved.columns) == ["model_idx", "e_mae"]
    assert saved.iloc[0]["model_idx"] == 0
