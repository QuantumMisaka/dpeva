import os
import logging
from dpeva.submission import JobManager, JobConfig

class InferenceWorkflow:
    """
    Workflow for running inference using an ensemble of DPA models.
    Supports both Local and Slurm backends via JobManager.
    """
    
    def __init__(self, config):
        self.config = config
        self._setup_logger()
        
        # Data and Model Configuration
        self.test_data_path = config.get("test_data_path")
        self.output_basedir = config.get("output_basedir", "./")
        
        # Auto-infer models_paths from output_basedir structure
        # Assumes structure: output_basedir/[0,1,2,3...]/model.ckpt.pt
        self.models_paths = []
        if os.path.exists(self.output_basedir):
            i = 0
            while True:
                possible_model = os.path.join(self.output_basedir, str(i), "model.ckpt.pt")
                if os.path.exists(possible_model):
                    self.models_paths.append(possible_model)
                    i += 1
                else:
                    break
        
        self.task_name = config.get("task_name", "test")
        self.head = config.get("head", "Hybrid_Perovskite")
        
        # Submission Configuration
        self.submission_config = config.get("submission", {})
        self.backend = self.submission_config.get("backend", "local")
        
        # Handle env_setup: support string or list of strings
        raw_env_setup = self.submission_config.get("env_setup", "")
        if isinstance(raw_env_setup, list):
            self.env_setup = "\n".join(raw_env_setup)
        else:
            self.env_setup = raw_env_setup
            
        self.slurm_config = self.submission_config.get("slurm_config", {})
        
        # Parallelism (for OMP settings if not in env_setup)
        self.omp_threads = config.get("omp_threads", 2)
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def _get_default_env_setup(self):
        """Provide default environment variables if user didn't specify any."""
        return f"""
export OMP_NUM_THREADS={self.omp_threads}
"""

    def run(self):
        self.logger.info(f"Initializing Inference Workflow (Backend: {self.backend})")
        
        if not self.test_data_path or not os.path.exists(self.test_data_path):
            self.logger.error(f"Test data path not found: {self.test_data_path}")
            return

        if not self.models_paths:
            self.logger.error("No models provided for inference.")
            return

        # Initialize Job Manager
        try:
            manager = JobManager(mode=self.backend)
        except ValueError as e:
            self.logger.error(str(e))
            return

        # Determine Environment Setup
        # User provided env_setup takes precedence. 
        # If empty, fallback to default OMP settings for Local, or minimal for Slurm.
        final_env_setup = self.env_setup if self.env_setup.strip() else self._get_default_env_setup()

        self.logger.info(f"Submitting {len(self.models_paths)} inference jobs...")
        
        for i, model_path in enumerate(self.models_paths):
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}, skipping.")
                continue
                
            # Define output directory structure: output_basedir/i/task_name
            if self.task_name:
                work_dir = os.path.join(self.output_basedir, str(i), self.task_name)
            else:
                work_dir = os.path.join(self.output_basedir, str(i))
                
            os.makedirs(work_dir, exist_ok=True)
            
            # Construct Command
            # We run the command from work_dir, so paths should be absolute
            abs_data_path = os.path.abspath(self.test_data_path)
            abs_model_path = os.path.abspath(model_path)
            
            # Command: dp --pt test ...
            # Note: We pipe output to test.log inside the command string for shell execution
            # -d results means the output will be in detail like results.[e,e_peratom,f,v,v_peratom].out
            cmd = (
                f"dp --pt test "
                f"-s {abs_data_path} "
                f"-m {abs_model_path} "
                f"-d results "
                f"--head {self.head} "
                f"2>&1 | tee test.log"
            )
            
            # Create JobConfig
            job_name = f"dp_test_{i}"
            job_config = JobConfig(
                job_name=job_name,
                command=cmd,
                env_setup=final_env_setup,
                output_log="test_job.out",
                error_log="test_job.err",
                # Slurm specific params from config
                partition=self.slurm_config.get("partition", "partition"),
                nodes=self.slurm_config.get("nodes", 1),
                ntasks=self.slurm_config.get("ntasks", 1),
                gpus_per_node=self.slurm_config.get("gpus_per_node", 0),
                qos=self.slurm_config.get("qos"),
                nodelist=self.slurm_config.get("nodelist"),
                walltime=self.slurm_config.get("walltime", "24:00:00")
            )
            
            # Generate Script
            script_name = "run_test.slurm" if self.backend == "slurm" else "run_test.sh"
            script_path = os.path.join(work_dir, script_name)
            
            manager.generate_script(job_config, script_path)
            
            # Submit Job
            # For local backend, this runs the script.
            # For slurm backend, this submits via sbatch.
            manager.submit(script_path, working_dir=work_dir)
            
        self.logger.info("Inference Workflow Submission Completed.")
