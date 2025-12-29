import os
import logging
import subprocess
import multiprocessing

class ModelEvaluator:
    """
    Evaluates DeepMD models on datasets using `dp test`.
    """
    
    def __init__(self, omp_threads=24):
        """
        Initialize the ModelEvaluator.
        
        Args:
            omp_threads (int): Number of OMP threads for inference (default: 24).
        """
        self.omp_threads = omp_threads
        self.logger = logging.getLogger(__name__)

    def _run_single_inference(self, data_path, model_path, output_dir, head="Target_FTS"):
        """Run inference for a single model."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare environment variables
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.omp_threads)
        env["DP_INTER_OP_PARALLELISM_THREADS"] = str(self.omp_threads // 2)
        env["DP_INTRA_OP_PARALLELISM_THREADS"] = str(self.omp_threads)
        
        cmd = [
            "dp", "--pt", "test",
            "-s", data_path,
            "-m", model_path,
            "-d", output_dir,
            "--head", head
        ]
        
        self.logger.info(f"Running inference: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            log_file = os.path.join(output_dir, "test.log")
            with open(log_file, "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)
            
            if result.returncode == 0:
                self.logger.info(f"Inference successful. Results saved to {output_dir}")
                return True
            else:
                self.logger.error(f"Inference failed. Check {log_file}. Error: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception during inference: {e}")
            return False

    def predict_ensemble(self, data_path, models_dict, head="Target_FTS", blocking=True):
        """
        Run inference for an ensemble of models in parallel.
        
        Args:
            data_path (str): Path to the test dataset.
            models_dict (dict): Dictionary mapping model index/name to (model_path, output_dir).
            head (str): The head to use for prediction (default: "Target_FTS").
            blocking (bool): If True, wait for all tasks to complete.
        """
        processes = []
        for name, (model_path, output_dir) in models_dict.items():
            p = multiprocessing.Process(
                target=self._run_single_inference,
                args=(data_path, model_path, output_dir, head)
            )
            processes.append(p)
            p.start()
        
        if blocking:
            for p in processes:
                p.join()
            self.logger.info("All inference tasks finished.")
        else:
            return processes
