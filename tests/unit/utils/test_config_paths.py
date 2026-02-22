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
        "empty": "",
    }

    out = resolve_config_paths(cfg, str(config_path))

    assert out is cfg
    assert out["data_path"] == os.path.join(str(config_path.parent), "data")
    assert out["work_dir"] == os.path.join(str(config_path.parent), "work")
    assert out["model_path"] == "/abs/model.pt"
    assert out["empty"] == ""


def test_resolve_config_paths_without_config_file_path_returns_input_unchanged():
    cfg = {"data_path": "data"}
    out = resolve_config_paths(cfg, "")
    assert out is cfg
    assert out["data_path"] == "data"

