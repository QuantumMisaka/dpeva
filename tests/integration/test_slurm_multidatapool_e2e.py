import json
import os
import shutil
import logging
import pandas as pd
from pathlib import Path

import pytest

from tests.integration.slurm_multidatapool.data_minimizer import (
    MinimalDatasetSpec,
    prepare_minimal_dataset,
    write_minimal_training_input,
)
from tests.integration.slurm_multidatapool.orchestrator import OrchestratorEnv, WorkflowOrchestrator


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_data_root() -> Path:
    # Points to the local integration test data (golden set)
    # If not present, one should run setup_data.py
    data_dir = _project_root() / "tests" / "integration" / "data"
    if not (data_dir / "input.json").exists():
        pytest.fail(f"Integration test data not found at {data_dir}. Run tests/integration/setup_data.py first.")
    return data_dir


def _templates_root() -> Path:
    return _project_root() / "tests" / "integration" / "slurm_multidatapool" / "configs"


def _require_slurm():
    if os.environ.get("DPEVA_RUN_SLURM_ITEST") != "1":
        pytest.skip("Set DPEVA_RUN_SLURM_ITEST=1 to enable Slurm integration tests")
    if shutil.which("sbatch") is None:
        pytest.skip("sbatch not found in PATH")
    if shutil.which("squeue") is None:
        pytest.skip("squeue not found in PATH")


def _env_setup_lines(backend: str) -> list[str]:
    raw = os.environ.get("DPEVA_TEST_ENV_SETUP", "").strip()
    
    # For local backend, we might need specific setup or just rely on current env
    if backend == "local" and not raw:
        return []

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
@pytest.mark.parametrize("backend", ["local", "slurm"])
def test_multidatapool_e2e(tmp_path: Path, backend: str):
    if backend == "slurm":
        _require_slurm()
    
    # For local execution, we might want to skip if dependencies (like deepmd) are not installed in current env
    # But we assume the dev env has them.

    src_root = _source_data_root()
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    spec = MinimalDatasetSpec(
        candidate_pool="amourC",
        candidate_system="C2",
        candidate_frames=5,
        train_system="122",
    )
    prepare_minimal_dataset(src_root=src_root, dst_root=work_dir, spec=spec)

    base_input = src_root / "input.json"
    write_minimal_training_input(
        base_input_path=base_input,
        out_path=work_dir / "input.json",
        systems_path="./sampled_dpdata",
    )

    num_models = 3 # Minimum required by TrainingConfig
    task_name = "test_val"
    
    # Helper to configure backend
    def _configure_backend(cfg):
        cfg["submission"]["backend"] = backend
        if backend == "local":
            # Clear slurm config for clarity, though it should be ignored
            cfg["submission"]["slurm_config"] = {}
        else:
             cfg["submission"]["env_setup"] = _env_setup_lines(backend)
             # Use test-specific partition/qos
             _maybe_set_partition(cfg["submission"]["slurm_config"], "DPEVA_TEST_GPU_PARTITION")
             _maybe_set_qos(cfg["submission"]["slurm_config"], "DPEVA_TEST_GPU_QOS")

    # 1. Feature Config
    cfg_feature_pool = _load_template("feature_pool.json")
    cfg_feature_pool["model_head"] = "MP_traj_v024_alldata_mixu" # Use existing head from DPA-3.1-3M.pt
    _configure_backend(cfg_feature_pool)
    _write_config(work_dir / "feature_pool.json", cfg_feature_pool)

    # 2. Feature Train Config
    cfg_feature_train = _load_template("feature_train.json")
    cfg_feature_train["model_head"] = "MP_traj_v024_alldata_mixu" # Use existing head
    _configure_backend(cfg_feature_train)
    _write_config(work_dir / "feature_train.json", cfg_feature_train)

    # 3. Train Config
    cfg_train = _load_template("train.json")
    cfg_train["model_head"] = "MP_traj_v024_alldata_mixu" # Use existing head
    cfg_train["num_models"] = num_models
    cfg_train["seeds"] = cfg_train["seeds"][:num_models]
    cfg_train["training_seeds"] = cfg_train["training_seeds"][:num_models]
    _configure_backend(cfg_train)
    _write_config(work_dir / "train.json", cfg_train)

    # 4. Infer Config
    cfg_infer = _load_template("infer.json")
    cfg_infer["model_head"] = "MP_traj_v024_alldata_mixu" # Use existing head
    cfg_infer["task_name"] = task_name
    _configure_backend(cfg_infer)
    _write_config(work_dir / "infer.json", cfg_infer)

    # 5. Collect Config
    cfg_collect = _load_template("collect_normal.json")
    cfg_collect["num_models"] = num_models
    cfg_collect["direct_n_clusters"] = 2 # Reduced for small dataset
    _configure_backend(cfg_collect)
    # Collect might use CPU partition for Slurm
    if backend == "slurm":
         _maybe_set_partition(cfg_collect["submission"]["slurm_config"], "DPEVA_TEST_CPU_PARTITION")
         _maybe_set_qos(cfg_collect["submission"]["slurm_config"], "DPEVA_TEST_CPU_QOS")
    _write_config(work_dir / "collect.json", cfg_collect)

    # Execution
    # Ensure PYTHONPATH includes src
    pythonpath = _project_root() / "src"
    
    orch_env = OrchestratorEnv(project_root=_project_root(), pythonpath=pythonpath)
    orch = WorkflowOrchestrator(work_dir=work_dir, env=orch_env, backend=backend)

    timeout_s = float(os.environ.get("DPEVA_ITEST_TIMEOUT_S", "300")) # 5 mins default for local

    # --- Feature Generation ---
    orch.run_feature(config_path=work_dir / "feature_pool.json", savedir=work_dir / "desc_pool", timeout_s=timeout_s)
    
    desc_file = work_dir / "desc_pool" / spec.candidate_pool / f"{spec.candidate_system}.npy"
    
    assert desc_file.exists()
    # Check content size
    import numpy as np
    desc_data = np.load(desc_file)
    assert len(desc_data) == spec.candidate_frames, f"Expected {spec.candidate_frames} frames, got {len(desc_data)}"

    orch.run_feature(config_path=work_dir / "feature_train.json", savedir=work_dir / "desc_train", timeout_s=timeout_s)
    assert (work_dir / "desc_train" / f"{spec.train_system}.npy").exists()

    # --- Training ---
    orch.run_training(config_path=work_dir / "train.json", num_models=num_models, timeout_s=timeout_s)
    for i in range(num_models):
        ckpt = work_dir / str(i) / "model.ckpt.pt"
        if not ckpt.exists():
             print(f"\n--- DEBUG: Checkpoint missing for model {i}")
             # Check for log
             log = work_dir / str(i) / "train.log"
             if log.exists():
                 print(f"--- Log {log} tail ---:\n{log.read_text()[-1000:]}")
        assert ckpt.exists()
        # Check lcurve
        lcurve = work_dir / str(i) / "lcurve.out"
        assert lcurve.exists()
        with open(lcurve) as f:
            lines = f.readlines()
            assert len(lines) > 1 # Header + at least one step

    # --- Inference ---
    orch.run_inference(config_path=work_dir / "infer.json", num_models=num_models, task_name=task_name, timeout_s=timeout_s)
    for i in range(num_models):
        e_out = work_dir / str(i) / task_name / "results.e.out"
        assert e_out.exists()
        # Verify content not empty
        assert e_out.stat().st_size > 0

    # --- Collection ---
    orch.run_collect(config_path=work_dir / "collect.json", timeout_s=timeout_s)
    
    final_df_path = work_dir / "dpeva_uq_result" / "dataframe" / "final_df.csv"
    
    if final_df_path.exists():
        # Enhanced Assertion: Check CSV content
        df = pd.read_csv(final_df_path)
        assert not df.empty, "Final dataframe is empty"
        assert "dataname" in df.columns
    else:
        # Check if it was because no candidates were selected
        col_log = work_dir / "dpeva_uq_result" / "collection.log"
        if col_log.exists() and "No candidates selected" in col_log.read_text():
            print("--- INFO: No candidates selected, skipping final_df.csv check ---")
            
            # Check intermediate files
            df_uq_path = work_dir / "dpeva_uq_result" / "dataframe" / "df_uq.csv"
            assert df_uq_path.exists(), "df_uq.csv should exist even if no candidates selected"
            df_uq = pd.read_csv(df_uq_path)
            assert not df_uq.empty
        else:
            pytest.fail("final_df.csv missing and 'No candidates selected' not found in log")
    # Check if we selected something (might be empty if UQ filters everything, but with minimal dataset we hope for some)
    # Actually, with 5 frames and 2 models, we might or might not select.
    # But the file should exist and be valid CSV.


