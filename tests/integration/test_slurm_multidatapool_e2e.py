import json
import os
import shutil
from pathlib import Path

import pytest

from tests.integration.slurm_multidatapool.data_minimizer import (
    MinimalDatasetSpec,
    prepare_minimal_dataset,
    write_minimal_training_input,
)
from tests.integration.slurm_multidatapool.orchestrator import OrchestratorEnv, SlurmWorkflowOrchestrator


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_production_root() -> Path:
    return _project_root() / "test" / "test-for-multiple-datapool"


def _templates_root() -> Path:
    return _project_root() / "tests" / "integration" / "slurm_multidatapool" / "configs"


def _require_slurm():
    if os.environ.get("DPEVA_RUN_SLURM_ITEST") != "1":
        pytest.skip("Set DPEVA_RUN_SLURM_ITEST=1 to enable Slurm integration tests")
    if shutil.which("sbatch") is None:
        pytest.skip("sbatch not found in PATH")
    if shutil.which("squeue") is None:
        pytest.skip("squeue not found in PATH")


def _env_setup_lines() -> list[str]:
    raw = os.environ.get("DPEVA_TEST_ENV_SETUP", "").strip()
    if not raw:
        return [
            "source /opt/envs/deepmd3.1.2.env",
            "export DP_INTERFACE_PREC=high",
        ]
    return [line for line in raw.splitlines() if line.strip()]


def _maybe_set_partition(slurm_cfg: dict, key: str) -> None:
    val = os.environ.get(key, "").strip()
    if val:
        slurm_cfg["partition"] = val


def _maybe_set_qos(slurm_cfg: dict, key: str) -> None:
    val = os.environ.get(key, "").strip()
    if val:
        slurm_cfg["qos"] = val


def _load_template(name: str) -> dict:
    return json.loads((_templates_root() / name).read_text())


def _write_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, indent=2))


@pytest.mark.integration
def test_slurm_multidatapool_chain_smoke(tmp_path: Path):
    _require_slurm()

    src_root = _source_production_root()
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    spec = MinimalDatasetSpec(
        candidate_pool="mptrj-FeCOH",
        candidate_system="C0Fe4H0O8",
        candidate_frames=20,
        train_system="122",
    )
    prepare_minimal_dataset(src_root=src_root, dst_root=work_dir, spec=spec)

    base_input = src_root / "0" / "input.json"
    write_minimal_training_input(
        base_input_path=base_input,
        out_path=work_dir / "input.json",
        systems_path="./sampled_dpdata",
    )

    num_models = 3
    task_name = "test_val"

    cfg_feature_pool = _load_template("feature_pool.json")
    cfg_feature_pool["model_head"] = "DPEVA_TEST"
    cfg_feature_pool["submission"]["env_setup"] = _env_setup_lines()
    _maybe_set_partition(cfg_feature_pool["submission"]["slurm_config"], "DPEVA_TEST_GPU_PARTITION")
    _maybe_set_qos(cfg_feature_pool["submission"]["slurm_config"], "DPEVA_TEST_GPU_QOS")
    _write_config(work_dir / "feature_pool.json", cfg_feature_pool)

    cfg_feature_train = _load_template("feature_train.json")
    cfg_feature_train["model_head"] = "DPEVA_TEST"
    cfg_feature_train["submission"]["env_setup"] = _env_setup_lines()
    _maybe_set_partition(cfg_feature_train["submission"]["slurm_config"], "DPEVA_TEST_GPU_PARTITION")
    _maybe_set_qos(cfg_feature_train["submission"]["slurm_config"], "DPEVA_TEST_GPU_QOS")
    _write_config(work_dir / "feature_train.json", cfg_feature_train)

    cfg_train = _load_template("train.json")
    cfg_train["model_head"] = "DPEVA_TEST"
    cfg_train["num_models"] = num_models
    cfg_train["submission"]["env_setup"] = _env_setup_lines()
    _maybe_set_partition(cfg_train["submission"]["slurm_config"], "DPEVA_TEST_GPU_PARTITION")
    _maybe_set_qos(cfg_train["submission"]["slurm_config"], "DPEVA_TEST_GPU_QOS")
    _write_config(work_dir / "train.json", cfg_train)

    cfg_infer = _load_template("infer.json")
    cfg_infer["model_head"] = "DPEVA_TEST"
    cfg_infer["task_name"] = task_name
    cfg_infer["submission"]["env_setup"] = _env_setup_lines()
    _maybe_set_partition(cfg_infer["submission"]["slurm_config"], "DPEVA_TEST_GPU_PARTITION")
    _maybe_set_qos(cfg_infer["submission"]["slurm_config"], "DPEVA_TEST_GPU_QOS")
    _write_config(work_dir / "infer.json", cfg_infer)

    cfg_collect = _load_template("collect_normal.json")
    cfg_collect["num_models"] = num_models
    cfg_collect["direct_n_clusters"] = 8
    cfg_collect["submission"]["env_setup"] = _env_setup_lines()
    _maybe_set_partition(cfg_collect["submission"]["slurm_config"], "DPEVA_TEST_CPU_PARTITION")
    _maybe_set_qos(cfg_collect["submission"]["slurm_config"], "DPEVA_TEST_CPU_QOS")
    _write_config(work_dir / "collect.json", cfg_collect)

    pythonpath = _project_root() / "src"
    orch_env = OrchestratorEnv(project_root=_project_root(), pythonpath=pythonpath)
    orch = SlurmWorkflowOrchestrator(work_dir=work_dir, env=orch_env)

    timeout_s = float(os.environ.get("DPEVA_SLURM_ITEST_TIMEOUT_S", "7200"))

    orch.run_feature(config_path=work_dir / "feature_pool.json", savedir=work_dir / "desc_pool", timeout_s=timeout_s)
    assert (work_dir / "desc_pool" / spec.candidate_pool / f"{spec.candidate_system}.npy").exists()

    orch.run_feature(config_path=work_dir / "feature_train.json", savedir=work_dir / "desc_train", timeout_s=timeout_s)
    assert (work_dir / "desc_train" / f"{spec.train_system}.npy").exists()

    orch.run_training(config_path=work_dir / "train.json", num_models=num_models, timeout_s=timeout_s)
    for i in range(num_models):
        assert (work_dir / str(i) / "model.ckpt.pt").exists()

    orch.run_inference(config_path=work_dir / "infer.json", num_models=num_models, task_name=task_name, timeout_s=timeout_s)
    for i in range(num_models):
        assert (work_dir / str(i) / task_name / "results.e.out").exists()

    orch.run_collect(config_path=work_dir / "collect.json", timeout_s=timeout_s)
    assert (work_dir / "dpeva_uq_result" / "dataframe" / "df_uq_desc_sampled-final.csv").exists()

