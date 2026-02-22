import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from dpeva.constants import WORKFLOW_FINISHED_TAG

from .slurm_utils import submit_cli_and_capture_job_id
from ..utils.logs import wait_for_text_in_file


@dataclass(frozen=True)
class OrchestratorEnv:
    project_root: Path
    pythonpath: Path

    def as_subprocess_env(self) -> Mapping[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.pythonpath)
        return env


class SlurmWorkflowOrchestrator:
    def __init__(self, work_dir: Path, env: OrchestratorEnv):
        self.work_dir = work_dir
        self.env = env

    def run_feature(self, config_path: Path, savedir: Path, timeout_s: float) -> None:
        submit_cli_and_capture_job_id(
            [sys.executable, "-m", "dpeva.cli", "feature", str(config_path)],
            cwd=self.work_dir,
            env=self.env.as_subprocess_env(),
        )
        wait_for_text_in_file(savedir / "eval_desc.log", WORKFLOW_FINISHED_TAG, timeout_s=timeout_s)

    def run_training(self, config_path: Path, num_models: int, timeout_s: float) -> None:
        submit_cli_and_capture_job_id(
            [sys.executable, "-m", "dpeva.cli", "train", str(config_path)],
            cwd=self.work_dir,
            env=self.env.as_subprocess_env(),
        )
        for i in range(num_models):
            wait_for_text_in_file(self.work_dir / str(i) / "train.out", WORKFLOW_FINISHED_TAG, timeout_s=timeout_s)

    def run_inference(self, config_path: Path, num_models: int, task_name: str, timeout_s: float) -> None:
        submit_cli_and_capture_job_id(
            [sys.executable, "-m", "dpeva.cli", "infer", str(config_path)],
            cwd=self.work_dir,
            env=self.env.as_subprocess_env(),
        )
        for i in range(num_models):
            wait_for_text_in_file(
                self.work_dir / str(i) / task_name / "test_job.log",
                WORKFLOW_FINISHED_TAG,
                timeout_s=timeout_s,
            )

    def run_collect(self, config_path: Path, timeout_s: float, log_path: Optional[Path] = None) -> None:
        submit_cli_and_capture_job_id(
            [sys.executable, "-m", "dpeva.cli", "collect", str(config_path)],
            cwd=self.work_dir,
            env=self.env.as_subprocess_env(),
        )
        target_log = log_path or (self.work_dir / "collect_slurm.out")
        wait_for_text_in_file(target_log, WORKFLOW_FINISHED_TAG, timeout_s=timeout_s)
