import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from dpeva.constants import (
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_PATH,
    COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM,
    COL_UQ_LLPR_ENERGY_PER_ATOM,
)
from dpeva.workflows.collect import CollectionWorkflow

def test_collect_single_pool_routing(tmp_path):
    """
    Test routing for Single Data Pool configuration.
    """
    # Create dummy dirs
    (tmp_path / "project").mkdir()
    (tmp_path / "desc_pool").mkdir()
    (tmp_path / "test_data").mkdir()
    
    config = {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc_pool"),
        "testdata_dir": str(tmp_path / "test_data"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_select_scheme": "tangent_lo",
        "uq_trust_mode": "auto",
        "uq_trust_ratio": 0.5,
        "backend": "local"
    }
    
    # Init workflow
    with patch("dpeva.workflows.collect.UQManager"):
        workflow = CollectionWorkflow(config)
        
        # Verify Single Pool characteristics
        assert workflow.config.uq_select_scheme == "tangent_lo"
        assert workflow.config.uq_trust_mode == "auto"
        assert workflow.config.uq_trust_ratio is not None

def test_collect_multi_pool_routing(tmp_path):
    """
    Test routing for Multi Data Pool configuration.
    """
    # Create dummy dirs
    (tmp_path / "project").mkdir()
    (tmp_path / "desc_pool").mkdir()
    (tmp_path / "test_data").mkdir()
    (tmp_path / "train_data").mkdir()
    (tmp_path / "desc_train").mkdir()
    
    config = {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc_pool"),
        "testdata_dir": str(tmp_path / "test_data"),
        "training_data_dir": str(tmp_path / "train_data"),
        "training_desc_dir": str(tmp_path / "desc_train"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_select_scheme": "tangent_lo",
        "backend": "local"
    }
    
    with patch("dpeva.workflows.collect.UQManager"):
        workflow = CollectionWorkflow(config)
        
        # Verify separation
        if workflow.config.training_data_dir:
            assert str(workflow.config.training_data_dir) == config["training_data_dir"]


def test_no_filter_uq_phase_with_multi_pool_names(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "desc_pool").mkdir()
    (tmp_path / "test_data").mkdir()

    config = {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc_pool"),
        "testdata_dir": str(tmp_path / "test_data"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_trust_mode": "no_filter",
        "sampler_type": "direct",
        "backend": "local",
    }

    with patch("dpeva.workflows.collect.UQManager"):
        workflow = CollectionWorkflow(config)

    datanames = ["poolA/sys1-0", "poolA/sys1-1", "poolB/sys2-0"]
    desc = np.random.rand(3, 6)
    with patch.object(workflow.io_manager, "load_descriptors", return_value=(datanames, desc)):
        _, df_candidate, unique_system_names = workflow._run_no_filter_uq_phase()

    assert len(df_candidate) == 3
    assert set(unique_system_names) == {"poolA/sys1", "poolB/sys2"}


def test_no_filter_uq_phase_with_single_pool_names(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "desc_pool").mkdir()
    (tmp_path / "test_data").mkdir()

    config = {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc_pool"),
        "testdata_dir": str(tmp_path / "test_data"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_trust_mode": "no_filter",
        "sampler_type": "direct",
        "backend": "local",
    }

    with patch("dpeva.workflows.collect.UQManager"):
        workflow = CollectionWorkflow(config)

    datanames = ["sys1-0", "sys1-1", "sys2-0"]
    desc = np.random.rand(3, 6)
    with patch.object(workflow.io_manager, "load_descriptors", return_value=(datanames, desc)):
        _, df_candidate, unique_system_names = workflow._run_no_filter_uq_phase()

    assert len(df_candidate) == 3
    assert set(unique_system_names) == {"sys1", "sys2"}


def test_no_filter_uq_phase_passes_last_layer_hdf5_dataset(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "desc_pool").mkdir()
    (tmp_path / "test_data").mkdir()

    config = {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc_pool"),
        "testdata_dir": str(tmp_path / "test_data"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_trust_mode": "no_filter",
        "sampler_type": "direct",
        "desc_feature_kind": "fitting_last_layer",
        "backend": "local",
    }

    with patch("dpeva.workflows.collect.UQManager"):
        workflow = CollectionWorkflow(config)

    datanames = ["sys1-0", "sys1-1"]
    desc = np.random.rand(2, 3)
    with patch.object(workflow.io_manager, "load_descriptors", return_value=(datanames, desc)) as load_descriptors:
        workflow._run_no_filter_uq_phase()

    assert load_descriptors.call_args.kwargs["hdf5_dataset"] == "atomic_feature"


def test_sampling_phase_passes_last_layer_hdf5_dataset_to_training_and_2direct(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "desc_pool").mkdir()
    (tmp_path / "test_data").mkdir()
    (tmp_path / "desc_train").mkdir()

    config = {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc_pool"),
        "testdata_dir": str(tmp_path / "test_data"),
        "training_desc_dir": str(tmp_path / "desc_train"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_trust_mode": "no_filter",
        "sampler_type": "2-direct",
        "desc_feature_kind": "fitting_last_layer",
        "backend": "local",
    }

    with patch("dpeva.workflows.collect.UQManager"):
        workflow = CollectionWorkflow(config)

    workflow.sampling_manager = MagicMock()
    workflow.sampling_manager.sampler_type = "2-direct"
    workflow.sampling_manager.prepare_features.return_value = (np.array([[1.0, 0.0]]), False, 1)
    workflow.sampling_manager.execute_sampling.return_value = {
        "selected_indices": [0],
        "pca_features": np.array([[1.0, 0.0]]),
        "explained_variance": np.array([1.0]),
        "random_indices": [0],
        "scores_direct": np.array([0.0]),
        "scores_random": np.array([0.0]),
    }

    df_candidate = pd.DataFrame({"dataname": ["sys1-0"], "uq_identity": ["candidate"]})
    df_desc = pd.DataFrame({"dataname": ["sys1-0"], "desc_0": [1.0], "desc_1": [0.0]})
    vis = MagicMock()

    with patch.object(workflow.io_manager, "load_descriptors", return_value=(["train-0"], np.array([[0.0, 1.0]]))) as load_descriptors:
        with patch.object(workflow.io_manager, "load_atomic_features", return_value=([np.ones((2, 3))], [2])) as load_atomic:
            workflow._run_sampling_phase(df_candidate, df_desc, vis)

    assert load_descriptors.call_args.kwargs["hdf5_dataset"] == "atomic_feature"
    assert load_atomic.call_args.kwargs["hdf5_dataset"] == "atomic_feature"


def test_llpr_uq_phase_routes_last_layer_features(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "desc_pool").mkdir()
    (tmp_path / "test_data").mkdir()
    train_feature_dir = tmp_path / "train_ll"
    candidate_feature_dir = tmp_path / "candidate_ll"
    train_feature_dir.mkdir()
    candidate_feature_dir.mkdir()

    np.save(train_feature_dir / "train.npy", np.array([[[1.0, 0.0], [0.0, 1.0]]]))
    np.save(
        candidate_feature_dir / "sys.npy",
        np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[3.0, 0.0], [0.0, 3.0]],
            ]
        ),
    )

    config = {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc_pool"),
        "testdata_dir": str(tmp_path / "test_data"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_backend": "llpr",
        "uq_trust_mode": "manual",
        "uq_llpr_energy_trust_lo": 0.0,
        "uq_llpr_energy_trust_hi": 10.0,
        "llpr_train_feature_dir": str(train_feature_dir),
        "llpr_candidate_feature_dir": str(candidate_feature_dir),
        "sampler_type": "direct",
        "backend": "local",
    }

    workflow = CollectionWorkflow(config)

    datanames = ["sys-0", "sys-1"]
    desc = np.random.rand(2, 6)
    with patch.object(workflow.io_manager, "load_descriptors", return_value=(datanames, desc)):
        df_desc, df_candidate, unique_system_names = workflow._run_uq_phase(vis=None)

    assert len(df_desc) == 2
    assert len(df_candidate) == 2
    assert set(unique_system_names) == {"sys"}
    assert COL_UQ_LLPR_ENERGY_PER_ATOM in df_candidate.columns
    assert set(df_candidate["uq_identity"]) == {"candidate"}


def test_llpr_uq_phase_writes_energy_ensemble_outputs_and_scores_by_std(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "desc_pool").mkdir()
    (tmp_path / "test_data").mkdir()
    train_feature_dir = tmp_path / "train_ll"
    candidate_feature_dir = tmp_path / "candidate_ll"
    train_feature_dir.mkdir()
    candidate_feature_dir.mkdir()
    weights_path = tmp_path / "weights.npy"
    energy_path = tmp_path / "base_energy.npy"

    np.save(train_feature_dir / "train.npy", np.array([[1.0, 0.0], [0.0, 1.0]]))
    np.save(candidate_feature_dir / "sys.npy", np.array([[1.0, 0.0], [3.0, 0.0]]))
    np.save(weights_path, np.array([1.0, 0.0]))
    np.save(energy_path, np.array([10.0, 20.0]))

    config = {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc_pool"),
        "testdata_dir": str(tmp_path / "test_data"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_backend": "llpr",
        "uq_trust_mode": "manual",
        "uq_llpr_energy_trust_lo": 0.0,
        "uq_llpr_energy_trust_hi": 0.8,
        "llpr_train_feature_dir": str(train_feature_dir),
        "llpr_candidate_feature_dir": str(candidate_feature_dir),
        "llpr_last_layer_weights_path": str(weights_path),
        "llpr_candidate_energy_path": str(energy_path),
        "llpr_num_ensemble_members": 4,
        "llpr_random_seed": 13,
        "llpr_collect_score": "energy_ensemble_std_per_atom",
        "sampler_type": "direct",
        "backend": "local",
    }

    workflow = CollectionWorkflow(config)

    datanames = ["sys-0", "sys-1"]
    desc = np.random.rand(2, 6)
    with patch.object(workflow.io_manager, "load_descriptors", return_value=(datanames, desc)):
        _, df_candidate, _ = workflow._run_uq_phase(vis=None)

    ensemble_path = tmp_path / "savedir" / "energy_ensemble.npy"
    assert ensemble_path.exists()
    assert COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM in df_candidate.columns
    assert set(df_candidate[COL_UQ_DPOSE_ENERGY_ENSEMBLE_PATH]) == {str(ensemble_path)}

    df_uq = pd.read_csv(tmp_path / "savedir" / "dataframe" / "df_uq.csv", index_col=0)
    df_uq_desc = pd.read_csv(tmp_path / "savedir" / "dataframe" / "df_uq_desc.csv", index_col=0)
    assert COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM in df_uq.columns
    assert COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM in df_uq_desc.columns
    np.testing.assert_allclose(
        df_uq[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM],
        df_uq_desc[COL_UQ_DPOSE_ENERGY_ENSEMBLE_STD_PER_ATOM],
    )
