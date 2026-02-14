import os
import shutil
import json
import logging
from typing import Dict, Any

from dpeva.constants import DEFAULT_LOG_FILE

class TrainingIOManager:
    """
    Handles Data I/O for Training Workflow:
    - Workspace setup
    - File operations (copying models, saving configs)
    - Logging setup
    """
    
    def __init__(self, work_dir: str):
        self.work_dir = os.path.abspath(work_dir)
        self.logger = logging.getLogger(__name__)
        
    def configure_logging(self):
        """Configures file logging."""
        os.makedirs(self.work_dir, exist_ok=True)
        log_file = os.path.join(self.work_dir, DEFAULT_LOG_FILE)
        
        # Check for duplicate handlers
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file):
                return

        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.info(f"Logging configured to file: {log_file}")

    def create_task_dir(self, task_idx: int) -> str:
        """Creates directory for a specific training task."""
        folder_name = os.path.join(self.work_dir, str(task_idx))
        os.makedirs(folder_name, exist_ok=True)
        return folder_name

    def save_task_config(self, task_dir: str, config: Dict[str, Any]):
        """Saves input.json to task directory."""
        with open(os.path.join(task_dir, "input.json"), "w") as f:
            json.dump(config, f, indent=4)

    def copy_base_model(self, src_path: str, task_dir: str) -> str:
        """
        Copies base model to task directory.
        
        Returns:
            str: The filename of the copied model.
        """
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Base model {src_path} not found")
            
        shutil.copy(src_path, task_dir)
        return os.path.basename(src_path)
