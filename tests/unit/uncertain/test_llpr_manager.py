import numpy as np
from unittest.mock import patch

from dpeva.constants import (
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_MEAN,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_N_MEMBERS,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_PATH,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM,
    COL_UQ_LLPR_ALPHA,
    COL_UQ_LLPR_CALIBRATED,
    COL_UQ_LLPR_ENERGY_PER_ATOM,
    COL_UQ_LLPR_ENERGY_TOTAL,
)
from dpeva.uncertain.manager import UQManager


def test_uq_manager_runs_llpr_analysis_for_collect_columns(tmp_path):
    manager = UQManager(
        project_dir=str(tmp_path),
        testing_dir="test",
        testing_head="head",
        uq_config={"llpr_regularizer": 0.5},
        num_models=3,
    )
    train_features = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    candidate_features = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 2.0]],
        ]
    )

    result = manager.run_llpr_analysis(
        train_features=train_features,
        candidate_features=candidate_features,
    )

    assert set(result) >= {
        COL_UQ_LLPR_ENERGY_TOTAL,
        COL_UQ_LLPR_ENERGY_PER_ATOM,
        COL_UQ_LLPR_ALPHA,
        COL_UQ_LLPR_CALIBRATED,
    }
    assert result[COL_UQ_LLPR_ENERGY_TOTAL].shape == (2,)
    np.testing.assert_allclose(
        result[COL_UQ_LLPR_ENERGY_PER_ATOM],
        result[COL_UQ_LLPR_ENERGY_TOTAL] / 2,
    )
    assert result[COL_UQ_LLPR_ALPHA] == 1.0
    assert result[COL_UQ_LLPR_CALIBRATED] is False


def test_uq_manager_scales_llpr_per_atom_with_candidate_atom_counts(tmp_path):
    manager = UQManager(
        project_dir=str(tmp_path),
        testing_dir="test",
        testing_head="head",
        uq_config={"llpr_regularizer": 0.5},
        num_models=3,
    )
    train_features = np.array([[1.0, 1.0], [2.0, 0.0]])
    candidate_features = np.array([[1.0, 1.0], [2.0, 2.0]])

    result = manager.run_llpr_analysis(
        train_features=train_features,
        candidate_features=candidate_features,
        candidate_atom_counts=np.array([2, 4]),
    )

    np.testing.assert_allclose(
        result[COL_UQ_LLPR_ENERGY_PER_ATOM],
        result[COL_UQ_LLPR_ENERGY_TOTAL] / np.array([2.0, 4.0]),
    )


def test_uq_manager_energy_only_llpr_does_not_require_dpose_adapter(tmp_path):
    manager = UQManager(
        project_dir=str(tmp_path),
        testing_dir="test",
        testing_head="head",
        uq_config={"llpr_targets": "energy", "llpr_regularizer": 0.5},
        num_models=3,
    )

    result = manager.run_llpr_analysis(
        train_features=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        candidate_features=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
    )

    assert "force_ensemble" not in result
    assert "force_uncertainty" not in result


def test_uq_manager_force_llpr_requires_dpose_adapter(tmp_path):
    manager = UQManager(
        project_dir=str(tmp_path),
        testing_dir="test",
        testing_head="head",
        uq_config={"llpr_targets": "force", "llpr_regularizer": 0.5},
        num_models=3,
    )

    try:
        manager.run_llpr_analysis(
            train_features=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
            candidate_features=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        )
    except RuntimeError as exc:
        assert "DeepMDTorchDPOSEAdapter" in str(exc)
    else:
        raise AssertionError("force LLPR without adapter should fail")


def test_uq_manager_dpose_energy_ensemble_requires_last_layer_weights(tmp_path):
    manager = UQManager(
        project_dir=str(tmp_path),
        testing_dir="test",
        testing_head="head",
        uq_config={"llpr_regularizer": 0.5, "llpr_num_ensemble_members": 4},
        num_models=3,
    )

    try:
        manager.run_llpr_analysis(
            train_features=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
            candidate_features=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
            mean_energy=np.array([1.0]),
        )
    except RuntimeError as exc:
        assert "last-layer weights" in str(exc)
    else:
        raise AssertionError("DPOSE ensemble without last-layer weights should fail")


def test_uq_manager_dpose_energy_ensemble_outputs_summary_and_file(tmp_path):
    ensemble_path = tmp_path / "energy_ensemble.npy"
    manager = UQManager(
        project_dir=str(tmp_path),
        testing_dir="test",
        testing_head="head",
        uq_config={
            "llpr_regularizer": 0.5,
            "llpr_num_ensemble_members": 4,
            "llpr_random_seed": 5,
            "llpr_ensemble_output_path": ensemble_path,
        },
        num_models=3,
    )

    result = manager.run_llpr_analysis(
        train_features=np.array([[1.0, 0.0], [0.0, 1.0]]),
        candidate_features=np.array([[1.0, 0.0], [2.0, 1.0]]),
        candidate_atom_counts=np.array([2, 4]),
        mean_energy=np.array([10.0, 20.0]),
        last_layer_weights=np.array([1.0, 1.0]),
    )

    assert ensemble_path.exists()
    energy_ensemble = np.load(ensemble_path)
    assert energy_ensemble.shape == (2, 4)
    np.testing.assert_allclose(result["energy_ensemble"], energy_ensemble)
    np.testing.assert_allclose(result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_MEAN], np.array([10.0, 20.0]))
    np.testing.assert_allclose(
        result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD],
        np.std(energy_ensemble, axis=1, ddof=1),
    )
    np.testing.assert_allclose(
        result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM],
        result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD] / np.array([2.0, 4.0]),
    )
    assert result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_N_MEMBERS] == 4
    assert result[COL_UQ_DPOSE_ENERGY_ENSEMBLE_PATH] == str(ensemble_path)


def test_uq_manager_can_save_and_reuse_llpr_state(tmp_path):
    state_path = tmp_path / "state.npz"
    train_features = np.array([[1.0, 0.0], [0.0, 1.0]])
    candidate_features = np.array([[1.0, 0.0], [2.0, 1.0]])
    manager = UQManager(
        project_dir=str(tmp_path),
        testing_dir="test",
        testing_head="head",
        uq_config={"llpr_regularizer": 0.5, "llpr_save_state_path": state_path},
        num_models=3,
    )
    first = manager.run_llpr_analysis(
        train_features=train_features,
        candidate_features=candidate_features,
    )

    reuse_manager = UQManager(
        project_dir=str(tmp_path),
        testing_dir="test",
        testing_head="head",
        uq_config={"llpr_state_path": state_path},
        num_models=3,
    )
    reused = reuse_manager.run_llpr_analysis(
        train_features=np.empty((0, 2)),
        candidate_features=candidate_features,
    )

    assert state_path.exists()
    np.testing.assert_allclose(
        reused[COL_UQ_LLPR_ENERGY_TOTAL],
        first[COL_UQ_LLPR_ENERGY_TOTAL],
    )


def test_uq_manager_logs_llpr_runtime_contract(tmp_path):
    manager = UQManager(
        project_dir=str(tmp_path),
        testing_dir="test",
        testing_head="head",
        uq_config={
            "llpr_regularizer": 0.5,
            "llpr_feature_normalization": "mean",
        },
        num_models=3,
    )

    with patch.object(manager.logger, "info") as mock_info:
        manager.run_llpr_analysis(
            train_features=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
            candidate_features=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        )

    messages = [str(call.args[0]) for call in mock_info.call_args_list]
    assert any("Running LLPR/DPOSE UQ" in msg for msg in messages)
    assert any("LLPR covariance settings" in msg for msg in messages)
    assert any("LLPR energy UQ completed" in msg for msg in messages)
    assert any("LLPR output columns" in msg for msg in messages)
