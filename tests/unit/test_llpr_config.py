import pytest

from dpeva.config import CollectionConfig


def test_collection_config_accepts_llpr_backend(tmp_path):
    cfg = CollectionConfig(
        desc_dir=tmp_path,
        testdata_dir=tmp_path,
        uq_backend="llpr",
        uq_trust_mode="no_filter",
    )

    assert cfg.uq_backend == "llpr"


def test_collection_config_rejects_llpr_manual_without_llpr_bounds(tmp_path):
    with pytest.raises(ValueError, match="uq_llpr_energy_trust_lo"):
        CollectionConfig(
            desc_dir=tmp_path,
            testdata_dir=tmp_path,
            uq_backend="llpr",
            uq_trust_mode="manual",
        )


def test_collection_config_defaults_llpr_to_energy_target(tmp_path):
    cfg = CollectionConfig(
        desc_dir=tmp_path,
        testdata_dir=tmp_path,
        uq_backend="llpr",
        uq_trust_mode="no_filter",
    )

    assert cfg.llpr_targets == "energy"
    assert cfg.llpr_collect_score == "energy_uncertainty_per_atom"


def test_collection_config_accepts_desc_feature_kind_for_hdf5_routing(tmp_path):
    cfg = CollectionConfig(
        desc_dir=tmp_path,
        testdata_dir=tmp_path,
        desc_feature_kind="fitting_last_layer",
    )

    assert cfg.desc_feature_kind == "fitting_last_layer"


def test_collection_config_accepts_energy_ensemble_collect_score(tmp_path):
    cfg = CollectionConfig(
        desc_dir=tmp_path,
        testdata_dir=tmp_path,
        uq_backend="llpr",
        uq_trust_mode="no_filter",
        llpr_num_ensemble_members=4,
        llpr_candidate_energy_path=tmp_path / "energy.npy",
        llpr_last_layer_weights_path=tmp_path / "weights.npy",
        llpr_collect_score="energy_ensemble_std_per_atom",
    )

    assert cfg.llpr_collect_score == "energy_ensemble_std_per_atom"


def test_collection_config_rejects_energy_ensemble_score_without_ensemble(tmp_path):
    with pytest.raises(ValueError, match="energy_ensemble_std_per_atom"):
        CollectionConfig(
            desc_dir=tmp_path,
            testdata_dir=tmp_path,
            uq_backend="llpr",
            uq_trust_mode="no_filter",
            llpr_collect_score="energy_ensemble_std_per_atom",
        )


def test_collection_config_rejects_force_score_for_energy_only_llpr(tmp_path):
    with pytest.raises(ValueError, match="llpr_collect_score='force_uncertainty_max'"):
        CollectionConfig(
            desc_dir=tmp_path,
            testdata_dir=tmp_path,
            uq_backend="llpr",
            uq_trust_mode="no_filter",
            llpr_targets="energy",
            llpr_collect_score="force_uncertainty_max",
        )
