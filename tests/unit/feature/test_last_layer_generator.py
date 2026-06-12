import numpy as np
from unittest.mock import MagicMock, patch

from dpeva.feature.generator import DescriptorGenerator
from dpeva.feature.managers import FeatureExecutionManager


@patch("dpeva.feature.generator.DeepPot", create=True)
@patch("dpeva.feature.generator._DEEPMD_AVAILABLE", True)
def test_compute_fitting_last_layer_uses_deepmd_api(_deep_pot):
    generator = DescriptorGenerator(model_path="model.pt")
    generator.model = MagicMock()
    generator.model.get_type_map.return_value = ["H", "O"]
    generator.model.eval_fitting_last_layer.return_value = np.ones((1, 2, 3))

    system = MagicMock()
    system.data = {
        "coords": np.zeros((1, 2, 3)),
        "cells": np.eye(3).reshape(1, 3, 3),
        "atom_names": ["H", "O"],
        "atom_types": np.array([0, 1]),
        "nopbc": False,
    }
    system.__len__.return_value = 1
    system.__getitem__.return_value = system

    with patch("dpeva.feature.generator.load_systems", return_value=[system]):
        result = generator.compute_fitting_last_layer("data", output_mode="structural")

    generator.model.eval_fitting_last_layer.assert_called_once()
    assert result.shape == (1, 3)
    np.testing.assert_allclose(result, np.ones((1, 3)))


def test_execution_manager_uses_fitting_last_layer_feature_kind(tmp_path):
    manager = FeatureExecutionManager(
        backend="local",
        slurm_config={},
        env_setup="",
        dp_backend="pt",
        omp_threads=1,
    )
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    (data_dir / "type.raw").write_text("0\n")
    generator = MagicMock()
    generator.compute_fitting_last_layer.return_value = np.ones((1, 3))

    manager.run_local_python_recursion(
        generator,
        str(data_dir),
        str(out_dir),
        output_mode="structural",
        feature_kind="fitting_last_layer",
    )

    generator.compute_fitting_last_layer.assert_called_once_with(str(data_dir), "structural")
    assert (out_dir / "data.npy").exists()
