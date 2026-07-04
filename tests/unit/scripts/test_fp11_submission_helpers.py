import copy
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "fp11_submission_helpers.py"


def load_module():
    assert MODULE_PATH.exists(), f"missing helper module: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("fp11_submission_helpers", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fp11_audit_requires_normal_single_gpu_without_rank_map():
    helpers = load_module()
    config = {
        "submission": {
            "env_setup": [
                "module load abacus/LTSv3.10.1-sm70-auto",
            ]
        },
        "labeling_task_classes": [
            {
                "name": "normal",
                "launcher_mode": "abacus",
                "resource_mode": "single_gpu",
                "slurm_config": {"ntasks": 1, "gpus_per_node": 1, "qos": "flood-1o2gpu"},
            },
            {
                "name": "highmem",
                "launcher_mode": "mpi_abacus",
                "resource_mode": "multi_gpu_mpi",
                "slurm_config": {"ntasks": 4, "gpus_per_node": 4, "qos": "flood-gpu"},
            },
        ],
    }

    assert helpers.audit_single_gpu_class(config, "normal") == []

    drifted = copy.deepcopy(config)
    drifted["labeling_task_classes"][0]["slurm_config"]["ntasks"] = 2
    drifted["submission"]["env_setup"].insert(
        0, "source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash"
    )
    issues = helpers.audit_single_gpu_class(drifted, "normal")
    assert any("ntasks=1" in issue for issue in issues)
    assert any("rank-map" in issue for issue in issues)


def test_fp11_patch_config_uses_summary_selection_not_g2_defaults():
    helpers = load_module()
    base = {
        "work_dir": "labeling_workdir_g2",
        "omp_threads": 8,
        "tasks_per_job": 4,
        "submission": {
            "slurm_config": {"ntasks": 2, "gpus_per_node": 2, "qos": "flood-1o2gpu"},
            "env_setup": [
                "source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash",
                "module load abacus/LTSv3.10.1-sm70-auto",
            ],
        },
        "dft_params": {"xc": "pbe"},
    }
    summary = {
        "selected": {
            "work_dir": "labeling_workdir_normal_g1",
            "omp_threads": 4,
            "tasks_per_job": 1,
            "launcher_mode": "abacus",
            "resource_mode": "single_gpu",
            "slurm_config": {
                "partition": "4V100",
                "ntasks": 1,
                "gpus_per_node": 1,
                "qos": "flood-1o2gpu",
                "walltime": "02:00:00",
            },
            "env_setup": ["module load abacus/LTSv3.10.1-sm70-auto"],
        }
    }

    patched = helpers.patch_config_from_benchmark_summary(base, summary)

    assert patched["work_dir"] == "labeling_workdir_normal_g1"
    assert patched["omp_threads"] == 4
    assert patched["tasks_per_job"] == 1
    assert patched["submission"]["slurm_config"]["ntasks"] == 1
    assert patched["submission"]["slurm_config"]["gpus_per_node"] == 1
    assert patched["submission"]["env_setup"] == ["module load abacus/LTSv3.10.1-sm70-auto"]
    assert "xc" not in patched["dft_params"]
    assert patched["dft_params"]["dft_functional"] == "pbe"


def test_fp11_benchmark_summary_selected_fields_match_patcher_contract():
    helpers = load_module()
    cases = [
        {"omp_threads": 1, "mean_scf_iter_s_excluding_cu1": 8.0},
        {"omp_threads": 4, "mean_scf_iter_s_excluding_cu1": 3.0},
        {"omp_threads": 8, "mean_scf_iter_s_excluding_cu1": 3.5},
    ]

    summary = helpers.build_benchmark_summary(cases, sample={"system": "x", "frame": 0, "atoms": 104})

    assert summary["selected"]["omp_threads"] == 4
    assert summary["selected"]["launcher_mode"] == "abacus"
    assert summary["selected"]["resource_mode"] == "single_gpu"
    assert summary["selected"]["slurm_config"]["ntasks"] == 1
    assert summary["selected"]["slurm_config"]["gpus_per_node"] == 1
    assert summary["selected"]["tasks_per_job"] == 1
