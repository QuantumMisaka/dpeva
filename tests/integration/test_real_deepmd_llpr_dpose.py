import logging
import os
from pathlib import Path

import numpy as np
import pytest

from dpeva.constants import (
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_MEAN,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM,
    COL_UQ_LLPR_ALPHA,
    COL_UQ_LLPR_CALIBRATED,
    COL_UQ_LLPR_ENERGY_PER_ATOM,
    COL_UQ_LLPR_ENERGY_TOTAL,
)
from dpeva.uncertain.dpose import resolve_last_layer_weights
from dpeva.uncertain.manager import UQManager


pytestmark = pytest.mark.skipif(
    os.environ.get("DPEVA_RUN_DEEPMD_DPOSE_REAL") != "1",
    reason="Real DeePMD LLPR/DPOSE validation is GPU-node only.",
)


def _fixture_model_path(env_name: str, default_name: str) -> Path:
    return Path(os.environ.get(env_name, f"tests/integration/data/{default_name}"))


@pytest.mark.parametrize(
    ("model_env", "default_model_name", "head"),
    [
        ("DPEVA_DPA3_MODEL_PATH", "DPA-3.1-3M.pt", "Omat24"),
        ("DPEVA_DPA4_MODEL_PATH", "DPA4-Mini-OMat24.pt", "Omat24"),
    ],
    ids=["dpa3_1_3m", "dpa4_mini_omat24"],
)
def test_real_deepmd_last_layer_energy_llpr_and_runtime_logs(
    model_env,
    default_model_name,
    head,
    caplog,
):
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "Real DeePMD LLPR/DPOSE validation must run on a GPU node."

    deepmd_infer = pytest.importorskip("deepmd.infer")
    model_path = _fixture_model_path(model_env, default_model_name)
    data_dir = Path("tests/integration/data/sampled_dpdata/122")
    if not model_path.exists():
        pytest.skip(
            f"Real DeePMD fixture model is not available: {model_path}. "
            f"Set {model_env} to an external model path."
        )
    assert data_dir.exists()

    deep_pot = deepmd_infer.DeepPot(
        str(model_path),
        head=os.environ.get("DPEVA_DEEPMD_HEAD", head),
    )
    coord = np.load(data_dir / "set.000" / "coord.npy")
    cell = np.load(data_dir / "set.000" / "box.npy")
    atom_types = np.loadtxt(data_dir / "type.raw", dtype=int)

    energy, force, _ = deep_pot.eval(coord, cell, atom_types, atomic=False)
    last_layer = np.asarray(
        deep_pot.eval_fitting_last_layer(coord, cell, atom_types),
        dtype=float,
    )

    assert np.asarray(energy).shape == (1, 1)
    assert np.asarray(force).shape == (1, atom_types.shape[0], 3)
    assert last_layer.shape[0] == 1
    assert last_layer.shape[1] == atom_types.shape[0]
    assert last_layer.ndim == 3
    assert np.isfinite(last_layer).all()

    manager = UQManager(
        project_dir="tests/integration/data",
        testing_dir="unused",
        testing_head="unused",
        uq_config={
            "llpr_targets": "energy",
            "llpr_regularizer": 1e-6,
            "llpr_feature_normalization": "mean",
        },
        num_models=1,
    )
    caplog.set_level(logging.INFO, logger="dpeva.uncertain.manager")
    result = manager.run_llpr_analysis(
        train_features=last_layer,
        candidate_features=last_layer,
        candidate_atom_counts=np.array([atom_types.shape[0]]),
    )

    assert set(result) >= {
        COL_UQ_LLPR_ENERGY_TOTAL,
        COL_UQ_LLPR_ENERGY_PER_ATOM,
        COL_UQ_LLPR_ALPHA,
        COL_UQ_LLPR_CALIBRATED,
    }
    assert np.isfinite(result[COL_UQ_LLPR_ENERGY_TOTAL]).all()
    assert np.isfinite(result[COL_UQ_LLPR_ENERGY_PER_ATOM]).all()
    assert result[COL_UQ_LLPR_CALIBRATED] is False
    assert "Running LLPR/DPOSE UQ" in caplog.text
    assert "LLPR covariance settings" in caplog.text
    assert "LLPR energy UQ completed" in caplog.text
    assert "LLPR output columns" in caplog.text


def test_real_deepmd_detached_last_layer_rejects_force_dpose():
    manager = UQManager(
        project_dir="tests/integration/data",
        testing_dir="unused",
        testing_head="unused",
        uq_config={
            "llpr_targets": "force",
            "llpr_regularizer": 1e-6,
        },
        num_models=1,
    )

    with pytest.raises(RuntimeError, match="DeepMDTorchDPOSEAdapter"):
        manager.run_llpr_analysis(
            train_features=np.zeros((1, 2, 4)),
            candidate_features=np.zeros((1, 2, 4)),
        )


@pytest.mark.parametrize(
    ("model_env", "weights_env", "default_model_name", "head"),
    [
        ("DPEVA_DPA3_MODEL_PATH", "DPEVA_DPA3_LAST_LAYER_WEIGHTS_PATH", "DPA-3.1-3M.pt", "Omat24"),
        ("DPEVA_DPA4_MODEL_PATH", "DPEVA_DPA4_LAST_LAYER_WEIGHTS_PATH", "DPA4-Mini-OMat24.pt", "Omat24"),
    ],
    ids=["dpa3_1_3m", "dpa4_mini_omat24"],
)
def test_real_deepmd_energy_dpose_ensemble_mean_recenters_to_base_energy(
    model_env,
    weights_env,
    default_model_name,
    head,
    tmp_path,
):
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "Real DeePMD LLPR/DPOSE validation must run on a GPU node."

    deepmd_infer = pytest.importorskip("deepmd.infer")
    model_path = _fixture_model_path(model_env, default_model_name)
    data_dir = Path("tests/integration/data/sampled_dpdata/122")
    if not model_path.exists():
        pytest.skip(
            f"Real DeePMD fixture model is not available: {model_path}. "
            f"Set {model_env} to an external model path."
        )

    deep_pot = deepmd_infer.DeepPot(
        str(model_path),
        head=os.environ.get("DPEVA_DEEPMD_HEAD", head),
    )
    coord = np.load(data_dir / "set.000" / "coord.npy")
    cell = np.load(data_dir / "set.000" / "box.npy")
    atom_types = np.loadtxt(data_dir / "type.raw", dtype=int)

    base_energy, _, _ = deep_pot.eval(coord, cell, atom_types, atomic=False)
    last_layer = np.asarray(
        deep_pot.eval_fitting_last_layer(coord, cell, atom_types),
        dtype=float,
    )
    feature_dimension = int(last_layer.shape[-1])
    explicit_weights = os.environ.get(weights_env)
    weights = resolve_last_layer_weights(
        feature_dimension=feature_dimension,
        last_layer_weights_path=explicit_weights,
        model_path=model_path if explicit_weights is None else None,
        model_head=os.environ.get("DPEVA_DEEPMD_HEAD", head),
    )

    ensemble_path = tmp_path / "energy_ensemble.npy"
    manager = UQManager(
        project_dir="tests/integration/data",
        testing_dir="unused",
        testing_head="unused",
        uq_config={
            "llpr_targets": "energy",
            "llpr_regularizer": 1e-6,
            "llpr_feature_normalization": "mean",
            "llpr_num_ensemble_members": 8,
            "llpr_random_seed": 17,
            "llpr_ensemble_output_path": ensemble_path,
            "llpr_weight_source": explicit_weights or str(model_path),
        },
        num_models=1,
    )
    result = manager.run_llpr_analysis(
        train_features=last_layer,
        candidate_features=last_layer,
        candidate_atom_counts=np.array([atom_types.shape[0]]),
        mean_energy=np.asarray(base_energy).reshape(-1),
        last_layer_weights=weights,
    )

    assert ensemble_path.exists()
    assert result["energy_ensemble"].shape == (1, 8)
    np.testing.assert_allclose(
        result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_MEAN],
        np.asarray(base_energy).reshape(-1),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.isfinite(result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD]).all()
    assert np.isfinite(result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM]).all()
    assert np.isfinite(result[COL_UQ_LLPR_ENERGY_PER_ATOM]).all()
