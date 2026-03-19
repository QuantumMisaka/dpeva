import json
from pathlib import Path

from dpeva.config import AnalysisConfig


ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_analysis_model_recipe_aligns_with_collection_slurm_fields():
    collect_cfg = _load_json(ROOT / "examples/recipes/collection/config_collect_normal.json")
    analysis_cfg = _load_json(ROOT / "examples/recipes/analysis/config_analysis.json")

    collect_submission = collect_cfg["submission"]
    analysis_submission = analysis_cfg["submission"]

    assert analysis_submission["backend"] == "slurm"
    assert analysis_submission["env_setup"] == collect_submission["env_setup"]
    assert analysis_submission["slurm_config"]["partition"] == collect_submission["slurm_config"]["partition"]
    assert analysis_submission["slurm_config"]["qos"] == collect_submission["slurm_config"]["qos"]
    assert analysis_submission["slurm_config"]["ntasks"] == collect_submission["slurm_config"]["ntasks"]
    assert analysis_submission["slurm_config"]["cpus_per_task"] == collect_submission["slurm_config"]["cpus_per_task"]
    assert analysis_cfg["results_prefix"] == collect_cfg["results_prefix"]
    assert analysis_cfg["result_dir"].split("/", 1)[1] == collect_cfg["testing_dir"]
    assert analysis_cfg["data_path"] == collect_cfg["testdata_dir"]
    parsed = AnalysisConfig(**analysis_cfg)
    assert parsed.submission.backend == "slurm"


def test_analysis_dataset_recipe_has_slurm_submission():
    collect_cfg = _load_json(ROOT / "examples/recipes/collection/config_collect_normal.json")
    analysis_cfg = _load_json(ROOT / "examples/recipes/analysis/config_analysis_dataset.json")

    collect_submission = collect_cfg["submission"]
    analysis_submission = analysis_cfg["submission"]

    assert analysis_submission["backend"] == "slurm"
    assert analysis_submission["env_setup"] == collect_submission["env_setup"]
    assert analysis_submission["slurm_config"]["partition"] == collect_submission["slurm_config"]["partition"]
    assert analysis_submission["slurm_config"]["qos"] == collect_submission["slurm_config"]["qos"]
    assert analysis_submission["slurm_config"]["ntasks"] == collect_submission["slurm_config"]["ntasks"]
    assert analysis_submission["slurm_config"]["cpus_per_task"] == collect_submission["slurm_config"]["cpus_per_task"]
    parsed = AnalysisConfig(**analysis_cfg)
    assert parsed.mode == "dataset"
    assert parsed.submission.backend == "slurm"
