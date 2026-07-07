import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "fp11_1344_backend_report.py"


def load_module():
    assert MODULE_PATH.exists(), f"missing script module: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("fp11_1344_backend_report", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_job_ids_merge_execute_and_recovery_logs_without_duplicates(tmp_path):
    module = load_module()
    execute_log = tmp_path / "labeling_execute.log"
    recovery_log = tmp_path / "fp11_recover_1344.log"
    execute_log.write_text(
        "2026-07-06 00:00:00,000 INFO Submission result: Submitted batch job 577510\n"
        "2026-07-06 00:01:00,000 INFO Submission result: Submitted batch job 586728\n",
        encoding="utf-8",
    )
    recovery_log.write_text(
        "2026-07-06 03:56:35,395 INFO Submission result: Submitted batch job 590738\n"
        "2026-07-06 03:56:36,802 INFO Submission result: Submitted batch job 586728\n"
        "2026-07-06 03:56:36,802 INFO Submission result: Submitted batch job 590758\n",
        encoding="utf-8",
    )

    job_ids, sources = module._job_ids_from_logs([execute_log, recovery_log])

    assert job_ids == ["577510", "586728", "590738", "590758"]
    assert sources[str(execute_log)] == ["577510", "586728"]
    assert sources[str(recovery_log)] == ["590738", "586728", "590758"]


def test_submit_elapsed_seconds_include_recovery_log(tmp_path):
    module = load_module()
    recovery_log = tmp_path / "fp11_recover_1344.log"
    recovery_log.write_text(
        "2026-07-06 03:56:34,233 INFO fp11_1344_recover: Submitting 616 recovered job bundles for attempt 2\n"
        "2026-07-06 03:56:36,802 INFO dpeva.workflows.labeling: Monitoring 2 Slurm jobs...\n",
        encoding="utf-8",
    )

    assert module._submit_elapsed_seconds_from_logs([recovery_log]) == [2.569]
