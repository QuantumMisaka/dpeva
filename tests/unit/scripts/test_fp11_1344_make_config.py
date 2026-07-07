import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "fp11_1344_make_config.py"


def load_module():
    assert MODULE_PATH.exists(), f"missing script module: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("fp11_1344_make_config", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def base_config():
    return {
        "work_dir": "labeling_workdir",
        "input_data_path": "sampled_dpdata",
        "pp_dir": "pp",
        "orb_dir": "orb",
        "tasks_per_job": 2,
        "submission": {
            "backend": "local",
            "slurm_array": False,
            "slurm_config": {"partition": "4V100", "walltime": "02:00:00"},
            "env_setup": ["module load old/module"],
        },
    }


def passing_probe():
    return {
        "flood_gpu": {
            "accepted_by_sbatch": True,
            "completed": True,
            "exit_code": "0:0",
        }
    }


def failing_probe():
    return {
        "flood_gpu": {
            "accepted_by_sbatch": False,
            "completed": False,
            "exit_code": "QOSMinGRES",
        }
    }


def task_class(config, name):
    return next(item for item in config["labeling_task_classes"] if item["name"] == name)


def assert_mpi_4gpu(task_class_config, expected_qos):
    assert task_class_config["launcher_mode"] == "mpi_abacus"
    assert task_class_config["resource_mode"] == "multi_gpu_mpi"
    assert task_class_config["slurm_config"]["partition"] == "16V100"
    assert task_class_config["slurm_config"]["ntasks"] == 4
    assert task_class_config["slurm_config"]["gpus_per_node"] == 4
    assert task_class_config["slurm_config"]["qos"] == expected_qos


def test_patcher_uses_single_gpu_normal_class_when_flood_probe_passes():
    module = load_module()

    patched = module.patch_config_for_1344(base_config(), passing_probe())

    assert patched["work_dir"] == "labeling_workdir_1344"
    assert patched["fp11_1344_resource_strategy"] == "single_gpu_normal"
    assert patched["submission"]["backend"] == "slurm"
    assert patched["submission"]["slurm_array"] is True
    assert patched["submission"]["slurm_array_task_limit"] == 128
    assert patched["submission"]["slurm_config"]["partition"] == "16V100"
    assert patched["submission"]["slurm_config"]["walltime"] == "04:00:00"
    assert patched["submission"]["env_setup"][0] == "source /etc/profile"
    assert "export ABACUS_PP_PATH=$HOME/PP_ORB/PP" in patched["submission"]["env_setup"]
    assert "export ABACUS_ORB_PATH=$HOME/PP_ORB/ORB" in patched["submission"]["env_setup"]

    normal = task_class(patched, "normal")
    assert normal["launcher_mode"] == "abacus"
    assert normal["resource_mode"] == "single_gpu"
    assert normal["slurm_config"]["partition"] == "16V100"
    assert normal["slurm_config"]["gpus_per_node"] == 1
    assert normal["slurm_config"]["ntasks"] == 1
    assert normal["slurm_config"]["qos"] == "flood-gpu"

    assert_mpi_4gpu(task_class(patched, "highmem"), "flood-gpu")


def test_patcher_falls_back_all_classes_to_4gpu_mpi_when_flood_probe_fails():
    module = load_module()

    patched = module.patch_config_for_1344(base_config(), failing_probe())

    assert patched["fp11_1344_resource_strategy"] == "multi_gpu_mpi_fallback"
    assert patched["submission"]["slurm_config"]["partition"] == "16V100"
    for class_config in patched["labeling_task_classes"]:
        assert_mpi_4gpu(class_config, "flood-gpu")


def test_patcher_preserves_existing_classes_and_applies_fallback_to_each():
    module = load_module()
    config = base_config()
    config["labeling_task_classes"] = [
        {
            "name": "normal",
            "selector": {"max_atoms": 180},
            "launcher_mode": "abacus",
            "resource_mode": "single_gpu",
            "slurm_config": {"partition": "4V100", "ntasks": 1, "gpus_per_node": 1},
            "env_setup": [
                "module load abacus/LTSv3.10.1-sm70-auto",
                "export ABACUS_PP_PATH=/old/pp",
                "export OMP_NUM_THREADS=8",
            ],
        },
        {
            "name": "multi_gpu",
            "selector": {"min_atoms": 181},
            "launcher_mode": "mpi_abacus",
            "resource_mode": "multi_gpu_mpi",
            "slurm_config": {"partition": "4V100", "ntasks": 4, "gpus_per_node": 4},
        },
    ]

    patched = module.patch_config_for_1344(config, failing_probe())

    assert [item["name"] for item in patched["labeling_task_classes"]] == ["normal", "multi_gpu"]
    assert_mpi_4gpu(task_class(patched, "normal"), "flood-gpu")
    assert_mpi_4gpu(task_class(patched, "multi_gpu"), "flood-gpu")
    assert task_class(patched, "normal")["env_setup"] == ["export OMP_NUM_THREADS=8"]
    assert task_class(patched, "multi_gpu")["env_setup"] == []


def test_patcher_defaults_none_array_limit_and_rejects_non_positive_limit():
    module = load_module()
    config = base_config()
    config["submission"]["slurm_array_task_limit"] = None

    patched = module.patch_config_for_1344(config, failing_probe())

    assert patched["submission"]["slurm_array_task_limit"] == 128

    config["submission"]["slurm_array_task_limit"] = 0
    try:
        module.patch_config_for_1344(config, failing_probe())
    except ValueError as exc:
        assert "slurm_array_task_limit" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    for invalid in (True, 1.5):
        config["submission"]["slurm_array_task_limit"] = invalid
        try:
            module.patch_config_for_1344(config, failing_probe())
        except ValueError as exc:
            assert "slurm_array_task_limit" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {invalid!r}")

    config["submission"]["slurm_array_task_limit"] = "64"
    patched = module.patch_config_for_1344(config, failing_probe())
    assert patched["submission"]["slurm_array_task_limit"] == 64


def test_patcher_keeps_explicit_multi_gpu_slurm_class_multi_when_probe_passes():
    module = load_module()
    config = base_config()
    config["labeling_task_classes"] = [
        {
            "name": "large",
            "selector": {"max_atoms": 220},
            "launcher_mode": "abacus",
            "resource_mode": "single_gpu",
            "slurm_config": {"ntasks": 4, "gpus_per_node": 4},
        }
    ]

    patched = module.patch_config_for_1344(config, passing_probe())

    assert_mpi_4gpu(task_class(patched, "large"), "flood-gpu")


def test_ensure_profile_normalizes_env_setup_and_orders_profile_before_module():
    module = load_module()

    from_string = module._ensure_profile(
        "module load abacus/LTSv3.10.1-sm70-auto\nexport OMP_NUM_THREADS=8"
    )
    assert from_string == [
        "source /etc/profile",
        "module load abacus/LTSv3.10.1-sm70-auto",
        "export ABACUS_PP_PATH=$HOME/PP_ORB/PP",
        "export ABACUS_ORB_PATH=$HOME/PP_ORB/ORB",
        "export OMP_NUM_THREADS=8",
    ]

    from_list = module._ensure_profile(
        [
            "source /etc/profile",
            "module load abacus/LTSv3.10.1-sm70-auto",
            "export ABACUS_PP_PATH=/old/pp",
            "export ABACUS_ORB_PATH=/old/orb",
            "source /etc/profile",
            "module load abacus/LTSv3.10.1-sm70-auto",
        ]
    )
    assert from_list == [
        "source /etc/profile",
        "module load abacus/LTSv3.10.1-sm70-auto",
        "export ABACUS_PP_PATH=$HOME/PP_ORB/PP",
        "export ABACUS_ORB_PATH=$HOME/PP_ORB/ORB",
    ]

    inserted = module._ensure_profile(["export OMP_NUM_THREADS=4"])
    assert inserted == [
        "source /etc/profile",
        "module load abacus/LTSv3.10.1-sm70-auto",
        "export ABACUS_PP_PATH=$HOME/PP_ORB/PP",
        "export ABACUS_ORB_PATH=$HOME/PP_ORB/ORB",
        "export OMP_NUM_THREADS=4",
    ]


def test_main_writes_config_gpu_1344_from_cwd_files(tmp_path, monkeypatch):
    module = load_module()
    probe_dir = tmp_path / "probes" / "qos-single-gpu"
    probe_dir.mkdir(parents=True)
    (tmp_path / "config_gpu.json").write_text(json.dumps(base_config()), encoding="utf-8")
    (probe_dir / "qos_single_gpu_probe_result.json").write_text(
        json.dumps(failing_probe()),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert module.main() == 0

    output_path = tmp_path / "config_gpu_1344.json"
    assert output_path.exists()
    patched = json.loads(output_path.read_text(encoding="utf-8"))
    assert patched["work_dir"] == "labeling_workdir_1344"
    assert patched["fp11_1344_resource_strategy"] == "multi_gpu_mpi_fallback"
