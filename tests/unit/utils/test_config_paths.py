import os

from dpeva.utils.config import resolve_config_paths


def test_resolve_config_paths_resolves_relative_paths(tmp_path):
    config_path = tmp_path / "configs" / "train.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}", encoding="utf-8")

    cfg = {
        "data_path": "data",
        "model_path": "/abs/model.pt",
        "work_dir": "./work",
        "dataset_dir": "dataset",
        "empty": "",
    }

    out = resolve_config_paths(cfg, str(config_path))

    assert out is cfg
    assert out["data_path"] == os.path.join(str(config_path.parent), "data")
    assert out["work_dir"] == os.path.join(str(config_path.parent), "work")
    assert out["dataset_dir"] == os.path.join(str(config_path.parent), "dataset")
    assert out["model_path"] == "/abs/model.pt"
    assert out["empty"] == ""


def test_resolve_config_paths_resolves_llpr_paths(tmp_path):
    config_path = tmp_path / "configs" / "collect.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}", encoding="utf-8")

    cfg = {
        "llpr_train_feature_dir": "train_features.hdf5",
        "llpr_candidate_feature_dir": "candidate_features.hdf5",
        "llpr_model_path": "model.pt",
        "llpr_last_layer_weights_path": "weights.npy",
        "llpr_candidate_energy_path": "energy.npy",
        "llpr_state_path": "state.npz",
        "llpr_save_state_path": "out/state.npz",
        "llpr_ensemble_output_path": "out/ensemble.npz",
    }

    expected = {
        key: os.path.join(str(config_path.parent), value)
        for key, value in cfg.items()
    }

    out = resolve_config_paths(cfg, str(config_path))

    for key, value in expected.items():
        assert out[key] == value


def test_resolve_config_paths_without_config_file_path_returns_input_unchanged():
    cfg = {"data_path": "data"}
    out = resolve_config_paths(cfg, "")
    assert out is cfg
    assert out["data_path"] == "data"
