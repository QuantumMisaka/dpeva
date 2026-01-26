import os
import sys

# Ensure dpeva is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from dpeva.workflows.train import TrainingWorkflow

def run_dpeva_train_workflow():
    """
    Run the DPEVA training workflow with production configuration.
    This script serves as a reusable runner for the training pipeline.
    """
    
    # Define paths relative to the project structure or absolute paths
    # Assuming test data is in dpeva/test for this example template
    test_dir = os.path.join(project_root, "test")
    
    config = {
        "work_dir": os.path.join(test_dir, "verification_test_run"),
        "input_json_path": os.path.join(test_dir, "input.json"),
        "num_models": 4,
        "mode": "init",
        "seeds": [19090, 42, 10032, 2933],
        "training_seeds": [19090, 42, 10032, 2933],
        "finetune_head_name": "Hybrid_Perovskite",
        "backend": "slurm",
        "base_model_path": os.path.join(test_dir, "DPA-3.1-3M.pt"),
        "slurm_config": {
            "partition": "4V100",
            "gpus_per_node": 4,
            "qos": "huge-gpu",
            "env_setup": """
export OMP_NUM_THREADS=2
export BASE_MODEL=DPA-3.1-3M.pt
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-${SLURM_JOB_ID}.log &
source /opt/envs/deepmd3.1.2.env
export DP_INTERFACE_PREC=high
"""
        }
    }

    print("Starting dpeva training workflow...")
    TrainingWorkflow(config).run()
    print("Workflow completed successfully.")

if __name__ == "__main__":
    run_dpeva_train_workflow()
