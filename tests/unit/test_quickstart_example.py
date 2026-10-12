"""The published local recipe must survive CLI loading from another cwd."""

from pathlib import Path
from shutil import copyfile

from dpeva.cli import load_and_resolve_config
from dpeva.config import InferenceConfig
from dpeva.workflows.collect import CollectionWorkflow
from dpeva.workflows.feature import FeatureWorkflow
from dpeva.workflows.train import TrainingWorkflow


RECIPE_DIR = Path(__file__).resolve().parents[2] / "examples/recipes/quickstart"


def test_quickstart_recipe_connects_all_four_stages_from_another_cwd(tmp_path, monkeypatch):
    mirror_root = tmp_path / "repository"
    recipe_dir = mirror_root / "examples/recipes/quickstart"
    recipe_dir.mkdir(parents=True)
    for source in RECIPE_DIR.glob("*.json"):
        copyfile(source, recipe_dir / source.name)
    fixture_link = mirror_root / "tests/integration/data"
    fixture_link.parent.mkdir(parents=True)
    fixture_link.symlink_to(RECIPE_DIR.parents[2] / "tests/integration/data", target_is_directory=True)

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)
    configs = {
        stage: load_and_resolve_config(recipe_dir / f"{stage}.json")
        for stage in ("feature", "train", "infer", "collect")
    }

    feature = FeatureWorkflow(configs["feature"])
    train = TrainingWorkflow(configs["train"])
    infer = InferenceConfig(**configs["infer"])
    work = recipe_dir / "work"
    (work / "desc_pool").mkdir(parents=True)
    collect = CollectionWorkflow(configs["collect"])

    assert Path(feature.data_path).is_dir()
    assert Path(feature.model_path).is_file()
    assert Path(feature.output_dir) == collect.config.desc_dir
    assert Path(train.input_json_path).is_file()
    assert Path(train.base_model_path).is_file()
    assert Path(train.work_dir) == infer.work_dir == Path(collect.project) == work
    assert Path(infer.data_path) == collect.config.testdata_dir == Path(feature.data_path)
    assert collect.config.root_savedir == work / "dpeva_uq_result"
    assert collect.config.testing_dir == infer.task_name
    assert collect.config.results_prefix == infer.results_prefix
    assert collect.config.num_models == train.num_models
    assert collect.io_manager.dpdata_savedir == str(
        work / "dpeva_uq_result" / "dpdata"
    )

    seeds = train.config_manager.generate_seeds(train.num_models, train.config.seeds)
    training_seeds = train.config_manager.generate_seeds(
        train.num_models, train.config.training_seeds
    )
    heads = train.config_manager.get_finetune_heads(
        train.mode, train.finetune_head_name, train.num_models
    )
    task_configs = train.config_manager.prepare_task_configs(
        train.num_models, seeds, training_seeds, heads, train.training_data_path
    )
    for task_config in task_configs:
        systems = task_config["training"]["training_data"]["systems"]
        assert all(Path(system).is_dir() for system in systems)
        assert task_config["training"]["numb_steps"] <= 30
