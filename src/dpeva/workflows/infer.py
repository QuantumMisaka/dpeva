import os
import logging
from dpeva.inference.evaluator import ModelEvaluator

class InferenceWorkflow:
    """
    Workflow for running inference using an ensemble of DeepMD models.
    """
    
    def __init__(self, config):
        self.config = config
        self._setup_logger()
        
        self.test_data_path = config.get("test_data_path")
        self.models_paths = config.get("models_paths", []) # List of model paths
        self.output_basedir = config.get("output_basedir", "results")
        self.head = config.get("head", "Target_FTS")
        self.omp_threads = config.get("omp_threads", 24)
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("Initializing Inference Workflow")
        
        if not self.test_data_path or not os.path.exists(self.test_data_path):
            self.logger.error(f"Test data path not found: {self.test_data_path}")
            return

        if not self.models_paths:
            self.logger.error("No models provided for inference.")
            return

        evaluator = ModelEvaluator(omp_threads=self.omp_threads)
        
        # Prepare models dictionary: {index: (model_path, output_dir)}
        models_dict = {}
        for i, model_path in enumerate(self.models_paths):
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}, skipping.")
                continue
                
            # Output structure: output_basedir/0/results, output_basedir/1/results...
            # This matches the structure expected by collect workflow (project/0/test-val-npy/results)
            # Users should configure output_basedir to point to the project root or relevant subfolder
            
            # Assuming output_basedir is like "project_dir", we want results in "project_dir/i/test-val-npy/results"
            # Or if output_basedir is just a flat dir for results.
            # Let's align with the existing structure: 
            # DPEVA structure: project/0/test-val-npy/results
            
            # Let's allow flexible configuration. If output_basedir has a format string, use it.
            # Otherwise, just append index.
            
            # Default behavior for now: output_basedir/{i}/results
            output_dir = os.path.join(self.output_basedir, str(i), "results")
            models_dict[i] = (model_path, output_dir)
            
        self.logger.info(f"Starting ensemble inference for {len(models_dict)} models...")
        evaluator.predict_ensemble(self.test_data_path, models_dict, head=self.head, blocking=True)
        self.logger.info("Inference Workflow Completed.")
