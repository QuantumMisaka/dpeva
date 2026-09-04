import os
import json
import logging
import shlex
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

from dpeva.constants import FILENAME_STATS_JSON
from dpeva.submission import JobManager, JobConfig
from dpeva.submission.guards import guarded_command
from dpeva.io.dataproc import DPTestResultParser
from dpeva.io.dataset import load_systems
from dpeva.utils.command import DPCommandBuilder
from dpeva.run.artifacts import ArtifactValidationError, validate_inference_outputs
from dpeva.run.models import JobRecord
from dpeva.run.status import RunState

logger = logging.getLogger(__name__)

class InferenceIOManager:
    """
    Manages IO operations for Inference Workflow:
    - Model discovery
    - Data loading (composition info)
    - Result parsing
    - Statistics saving
    """
    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.logger = logging.getLogger(__name__)

    def discover_models(self) -> List[str]:
        """Discover models in work_dir subdirectories (0/, 1/, ...)."""
        models_paths = []
        if os.path.exists(self.work_dir):
            i = 0
            while True:
                possible_model = os.path.join(self.work_dir, str(i), "model.ckpt.pt")
                if os.path.exists(possible_model):
                    models_paths.append(possible_model)
                    i += 1
                else:
                    break
        return models_paths

    def load_composition_info(self, data_path: str) -> Tuple[Optional[List[Dict]], Optional[List[int]]]:
        """Load composition info using dpdata."""
        if not data_path or not os.path.exists(data_path):
            self.logger.warning("dpdata not available or test_data_path invalid. Skipping composition loading.")
            return None, None

        try:
            self.logger.info(f"Loading system composition from {data_path} using dpdata...")
            loaded_systems = load_systems(data_path)
                
            atom_counts_list = []
            atom_num_list = []
            
            # Iterate in the same order as dp test usually does (assumed alphabetical/system order)
            # load_systems returns a list, usually sorted by how it discovered them.
            # Ideally we trust load_systems order matches dp test order if directory traversal is consistent.
            for s in loaded_systems:
                atom_names = s["atom_names"]
                atom_types = s["atom_types"]
                elements = [atom_names[t] for t in atom_types]
                counts = Counter(elements)
                n_atoms = len(elements)
                
                # Replicate for each frame in the system
                for _ in range(s.get_nframes()):
                    atom_counts_list.append(counts)
                    atom_num_list.append(n_atoms)
                    
            self.logger.info(f"Loaded composition info for {len(atom_counts_list)} frames.")
            return atom_counts_list, atom_num_list
        except Exception as e:
            self.logger.warning(f"Failed to load composition info: {e}. Relative energy will use mean subtraction.")
            return None, None

    def parse_results(self, job_work_dir: str, prefix: str) -> Dict:
        """Parse dp test results."""
        parser = DPTestResultParser(job_work_dir, head=prefix)
        return parser.parse()

    def save_statistics(self, analysis_dir: str, stats_data: Dict):
        """Save statistics to JSON."""
        def default(o):
            """JSON serializer for NumPy types."""
            if isinstance(o, (np.integer, int)):
                return int(o)
            if isinstance(o, (np.floating, float)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
            
        with open(os.path.join(analysis_dir, FILENAME_STATS_JSON), "w") as f:
            json.dump(stats_data, f, indent=4, default=default)

    def save_summary(self, summary_metrics: List[Dict]):
        """Save summary CSV."""
        if summary_metrics:
            summary_path = os.path.join(self.work_dir, "inference_summary.csv")
            pd.DataFrame(summary_metrics).to_csv(summary_path, index=False)
            self.logger.info(f"Analysis completed. Summary saved to {summary_path}")


class InferenceExecutionManager:
    """
    Manages Execution for Inference Workflow:
    - Command construction
    - Job submission
    """
    def __init__(self, backend: str, slurm_config: Dict, env_setup: str, dp_backend: str, omp_threads: int):
        self.backend = backend
        self.slurm_config = slurm_config or {}
        self.env_setup = env_setup
        self.dp_backend = dp_backend
        self.omp_threads = omp_threads
        
        DPCommandBuilder.set_backend(self.dp_backend)
        self.job_manager = JobManager(mode=backend)
        self.logger = logging.getLogger(__name__)
        self.last_artifacts: dict[str, list[str]] = {}

    def _get_default_env_setup(self):
        """Provide default environment variables if user didn't specify any."""
        return f"export OMP_NUM_THREADS={self.omp_threads}"

    def submit_jobs(self, models_paths: List[str], data_path: str, work_dir: str, task_name: str, 
                   head: str, results_prefix: str):
        """Submit every inference job and return an evidence-bearing job record."""
        final_env_setup = self.env_setup if self.env_setup.strip() else self._get_default_env_setup()
        
        self.logger.info(f"Submitting {len(models_paths)} inference jobs...")
        
        records: list[JobRecord] = []
        self.last_artifacts = {}

        for i, model_path in enumerate(models_paths):
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}, skipping.")
                records.append(
                    JobRecord(
                        name=f"model-{i}",
                        backend=self.backend,
                        status=RunState.FAILED,
                        failure=f"model file not found: {model_path}",
                        failure_category="EXECUTION",
                    )
                )
                continue
                
            # Define output directory structure: work_dir/i/task_name
            if task_name:
                job_work_dir = os.path.join(work_dir, str(i), task_name)
            else:
                job_work_dir = os.path.join(work_dir, str(i))
                
            os.makedirs(job_work_dir, exist_ok=True)
            
            # Construct Command
            abs_data_path = os.path.abspath(data_path)
            abs_model_path = os.path.abspath(model_path)
            
            log_file = "test.log" if self.backend == "local" else None
            
            cmd = DPCommandBuilder.test(
                model=abs_model_path,
                system=abs_data_path,
                prefix=results_prefix,
                head=head,
                log_file=log_file
            )
            
            cmd = guarded_command(
                command=cmd,
                artifact_checks=[
                    "\n".join(
                        [
                            "check_nonempty_artifact() {",
                            "    local artifact",
                            "    while IFS= read -r artifact; do",
                            '        if test -s "$artifact"; then return 0; fi',
                            f"    done < <(compgen -G {shlex.quote(results_prefix + '.*.out')})",
                            "    return 1",
                            "}",
                            "check_nonempty_artifact",
                        ]
                    ),
                ],
            )
            
            # Create JobConfig
            job_name = f"dp_test_{i}"
            
            # Use task-specific Slurm config if available, otherwise default
            task_slurm_config = self.slurm_config.copy()
            # Remove keys that shouldn't be passed via **kwargs if they are explicitly handled or invalid
            task_slurm_config.pop("job_name", None)
            task_slurm_config.pop("output_log", None)
            task_slurm_config.pop("error_log", None)
            
            job_config = JobConfig(
                job_name=job_name,
                command=cmd,
                env_setup=final_env_setup,
                output_log="test_job.out", # Changed to .out for consistency
                error_log="test_job.err",
                # Pass slurm config
                **task_slurm_config
            )
            
            # Generate Script
            script_name = "run_test.slurm" if self.backend == "slurm" else "run_test.sh"
            script_path = os.path.join(job_work_dir, script_name)
            
            self.job_manager.generate_script(job_config, script_path)
            
            try:
                submission_output = self.job_manager.submit(
                    script_path, working_dir=job_work_dir
                )
                if self.backend == "slurm":
                    if not isinstance(submission_output, str):
                        raise TypeError("Slurm submission did not return text output")
                    job_id = self.job_manager.parse_sbatch_job_id(submission_output)
                    if not job_id:
                        raise ValueError("Slurm submission returned no job id")
                    records.append(
                        JobRecord(
                            name=f"model-{i}",
                            backend="slurm",
                            job_id=job_id,
                            status=RunState.SUBMITTED,
                        )
                    )
                    continue

                outputs = validate_inference_outputs(
                    Path(job_work_dir).expanduser().resolve(), results_prefix
                )
                self.last_artifacts[f"model-{i}"] = [str(path) for path in outputs]
                records.append(
                    JobRecord(
                        name=f"model-{i}",
                        backend="local",
                        status=RunState.FINISHED,
                    )
                )
            except ArtifactValidationError as exc:
                self.logger.error(f"Inference job {i} failed: {exc}")
                records.append(
                    JobRecord(
                        name=f"model-{i}",
                        backend=self.backend,
                        status=RunState.FAILED,
                        failure=str(exc),
                        failure_category="ARTIFACT",
                    )
                )
            except Exception as exc:
                self.logger.error(f"Inference job {i} failed: {exc}")
                records.append(
                    JobRecord(
                        name=f"model-{i}",
                        backend=self.backend,
                        status=RunState.FAILED,
                        failure=str(exc),
                        failure_category="EXECUTION",
                    )
                )

        self.logger.info("Inference Workflow Submission Completed.")
        return records
