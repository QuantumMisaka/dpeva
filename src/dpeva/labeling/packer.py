"""
Task Packer
===========

Handles the distribution of tasks into subdirectories for efficient submission.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

class TaskPacker:
    """
    Groups task directories into sub-jobs (e.g., N_50_0, N_50_1).
    """

    def __init__(self, tasks_per_job: int = 50, job_prefix: str = "N"):
        self.tasks_per_job = tasks_per_job
        self.job_prefix = job_prefix

    def pack(self, root_dir: Path, exclude_patterns: List[str] = None) -> List[Path]:
        """
        Move task directories into packed subdirectories.
        
        Args:
            root_dir: Directory containing task folders.
            exclude_patterns: Glob patterns to exclude.
            
        Returns:
            List[Path]: List of packed job directories created.
        """
        root_dir = Path(root_dir)
        exclude_patterns = exclude_patterns or ["CONVERGED", f"{self.job_prefix}_*"]
        
        # Identify task dirs
        task_dirs = []
        for item in root_dir.iterdir():
            if not item.is_dir():
                continue
            
            # Check exclusions
            excluded = False
            for pat in exclude_patterns:
                if item.match(pat):
                    excluded = True
                    break
            if excluded:
                continue
            
            task_dirs.append(item)
        
        task_dirs.sort()
        logger.info(f"Found {len(task_dirs)} tasks to pack.")
        
        if not task_dirs:
            return []

        main_count = 0
        sub_count = 0
        
        job_dirs = []
        current_job_dir = root_dir / f"{self.job_prefix}_{self.tasks_per_job}_{main_count}"
        current_job_dir.mkdir(exist_ok=True)
        job_dirs.append(current_job_dir)
        
        for task_dir in task_dirs:
            if sub_count >= self.tasks_per_job:
                main_count += 1
                sub_count = 0
                current_job_dir = root_dir / f"{self.job_prefix}_{self.tasks_per_job}_{main_count}"
                current_job_dir.mkdir(exist_ok=True)
                job_dirs.append(current_job_dir)
            
            try:
                shutil.move(str(task_dir), str(current_job_dir))
                sub_count += 1
            except Exception as e:
                logger.error(f"Failed to move {task_dir} to {current_job_dir}: {e}")
        
        logger.info(f"Packed tasks into {len(job_dirs)} job directories.")
        return job_dirs
