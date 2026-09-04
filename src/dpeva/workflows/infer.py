import os
import json
import logging
from pathlib import Path
from typing import Any, Union, Dict, Optional

from dpeva.config import InferenceConfig
from dpeva.inference.managers import InferenceIOManager, InferenceExecutionManager
from dpeva.constants import WORKFLOW_FINISHED_TAG, LOG_FILE_INFER, FILENAME_METRICS_JSON
from dpeva.utils.logs import setup_workflow_logger
from dpeva.utils.exceptions import PartialWorkflowError, WorkflowError
from dpeva.run.context import RunContext, RunOptions
from dpeva.run.models import JobRecord
from dpeva.run.status import RunEventKind, RunState

class InferenceWorkflow:
    """
    Workflow for running inference using an ensemble of DPA models.
    Supports both Local and Slurm backends via JobManager.
    Refactored using DDD Managers.
    """
    
    def __init__(
        self,
        config: Union[Dict, InferenceConfig],
        config_path: Optional[str] = None,
        *,
        original_config: dict[str, Any] | None = None,
        run_options: RunOptions | None = None,
    ):
        """
        Initialize the Inference Workflow.

        Args:
            config (Union[Dict, InferenceConfig]): Configuration object or dictionary.
            config_path (Optional[str]): Path to the configuration file (required for Slurm self-submission).
        """
        if isinstance(config, dict):
            self.config = InferenceConfig(**config)
        else:
            self.config = config

        self.original_config = original_config or self.config.model_dump(mode="json")
        self.run_options = run_options or RunOptions()

        self.config_path = config_path
        self._setup_logger()
        
        # Core Configurations
        self.work_dir = str(self.config.work_dir)
        self.data_path = str(self.config.data_path)
        self.results_prefix = self.config.results_prefix
        
        # Managers
        self.io_manager = InferenceIOManager(self.work_dir)
        self.execution_manager = InferenceExecutionManager(
            backend=self.config.submission.backend,
            slurm_config=self.config.submission.slurm_config,
            env_setup=self.config.submission.env_setup,
            dp_backend=self.config.dp_backend,
            omp_threads=self.config.omp_threads
        )
        # Models
        self.task_name = self.config.task_name
        self.head = self.config.model_head
        
        # Model discovery
        self.models_paths = self.io_manager.discover_models()
        self.logger.info(f"Discovered {len(self.models_paths)} models in {self.work_dir}")

    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)

    def run(self):
        """
        Executes the inference workflow.
        
        1. Configures workflow logging.
        2. Discovers trained models.
        3. Submits parallel inference jobs (dp test) via Slurm or Local backend.
        4. Optionally triggers analysis based on explicit config.
        """
        env_backend = os.environ.get("DPEVA_INTERNAL_BACKEND")
        if env_backend:
            self.logger.info(f"Overriding backend to '{env_backend}'")
            self.execution_manager.backend = env_backend

        context = RunContext.create(
            work_dir=self.work_dir,
            workflow="infer",
            options=self.run_options,
            original_config=self.original_config,
            normalized_config=self.config.model_dump(mode="json"),
        )
        try:
            backend = self.execution_manager.backend
            state = context.recorder.manifest.status
            if state is RunState.CREATED:
                context.recorder.transition(RunState.VALIDATED)
                state = RunState.VALIDATED
            if state is RunState.PARTIAL:
                context.recorder.transition(RunState.RUNNING, RunEventKind.RESUME)
                state = RunState.RUNNING
            if state is RunState.VALIDATED:
                context.recorder.transition(
                    RunState.SUBMITTED if backend == "slurm" else RunState.RUNNING
                )
            elif state is RunState.SUBMITTED and backend == "local":
                context.recorder.transition(RunState.RUNNING)
            records = self._run_body()
            for record in records:
                context.recorder.add_job(record)

            if self.execution_manager.backend == "slurm":
                if self.config.auto_analysis:
                    self.logger.warning("auto_analysis=true is ignored when backend is not local.")
                    self.logger.info("Inference jobs submitted. Run analysis workflow separately after jobs finish.")
                return

            successful = [record for record in records if record.status is RunState.FINISHED]
            failed = [record for record in records if record.status is RunState.FAILED]
            for artifact_paths in self.execution_manager.last_artifacts.values():
                context.register_verified_artifacts(
                    "inference", [Path(path) for path in artifact_paths]
                )
            if not successful:
                message = "all inference jobs failed"
                context.recorder.fail(category="EXECUTION", message=message)
                raise WorkflowError(message)
            if failed:
                context.recorder.partial(
                    category="EXECUTION",
                    message="one or more inference jobs failed",
                )
                raise PartialWorkflowError("one or more inference jobs failed")

            context.recorder.transition(RunState.FINISHED)

            if self.config.auto_analysis:
                self.logger.info("Auto analysis enabled. Starting analysis...")
                self.analyze_results()
            else:
                self.logger.info("Auto analysis disabled. Run analysis workflow separately after jobs finish.")
        except Exception:
            # Terminal states written above must remain the original exception;
            # only unrecorded execution errors need a generic failure event.
            if context.recorder.manifest.status not in {
                RunState.FAILED,
                RunState.PARTIAL,
                RunState.FINISHED,
            }:
                context.recorder.fail(category="EXECUTION", message="inference execution failed")
            raise

    def _run_body(self) -> list[JobRecord]:
        """Validate inputs and submit all model jobs."""
        # Configure logging: log to inference.log, but DO NOT capture stdout (propagate=True)
        setup_workflow_logger(
            logger_name="dpeva",
            work_dir=self.work_dir,
            filename=LOG_FILE_INFER,
            capture_stdout=False
        )

        self.logger.info(f"Initializing Inference Workflow (Backend: {self.execution_manager.backend})")
        
        if not self.data_path or not os.path.exists(self.data_path):
            self.logger.error(f"Test data path not found: {self.data_path}")
            raise WorkflowError(f"Test data path not found: {self.data_path}")

        if not self.models_paths:
            self.logger.error("No models provided for inference.")
            raise WorkflowError("No models provided for inference.")

        # Submit Jobs
        records = self.execution_manager.submit_jobs(
            models_paths=self.models_paths,
            data_path=self.data_path,
            work_dir=self.work_dir,
            task_name=self.task_name,
            head=self.head,
            results_prefix=self.results_prefix
        )
        
        return records

    def analyze_results(self):
        """Analyze results for all models."""
        self.logger.info("Starting result analysis...")
        from dpeva.workflows.analysis import AnalysisWorkflow

        summary_metrics = []
        failed_count = 0
        
        for i, model_path in enumerate(self.models_paths):
            if self.task_name:
                job_work_dir = os.path.join(self.work_dir, str(i), self.task_name)
            else:
                job_work_dir = os.path.join(self.work_dir, str(i))
                
            if not os.path.exists(job_work_dir):
                self.logger.warning(f"Job directory not found: {job_work_dir}")
                failed_count += 1
                continue
                
            try:
                analysis_output_dir = os.path.join(job_work_dir, "analysis")
                analysis_config = {
                    "mode": "model_test",
                    "result_dir": job_work_dir,
                    "output_dir": analysis_output_dir,
                    "results_prefix": self.results_prefix,
                    "data_path": self.data_path,
                    "ref_energies": self.config.ref_energies,
                    "enable_cohesive_energy": True,
                    "allow_ref_energy_lstsq_completion": False,
                }
                workflow = AnalysisWorkflow(analysis_config)
                workflow.run()

                metrics_path = os.path.join(analysis_output_dir, FILENAME_METRICS_JSON)
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    metrics["model_idx"] = i
                    summary_metrics.append(metrics)
                else:
                    self.logger.warning(
                        f"Metrics file not found for model {i}: {metrics_path}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Analysis failed for model {i}: {e}")
                failed_count += 1
                continue
        
        # Save Global Summary
        self.io_manager.save_summary(summary_metrics)
        
        if failed_count > 0:
            msg = f"Analysis completed with {failed_count} failures out of {len(self.models_paths)} models."
            self.logger.error(msg)
            # Raise exception to signal workflow failure (non-zero exit code)
            raise WorkflowError(msg)
        else:
            self.logger.info(WORKFLOW_FINISHED_TAG)
