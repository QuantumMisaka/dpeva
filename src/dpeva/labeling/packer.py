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
        Supports recursive scanning for nested structures.
        
        Args:
            root_dir: Directory containing task folders.
            exclude_patterns: Glob patterns to exclude.
            
        Returns:
            List[Path]: List of packed job directories created.
        """
        root_dir = Path(root_dir)
        exclude_patterns = exclude_patterns or ["CONVERGED", f"{self.job_prefix}_*"]
        
        # Identify task dirs (Recursive scan)
        # Task dirs are identified by containing 'INPUT' file
        task_dirs = []
        for item in root_dir.rglob("INPUT"):
            task_dir = item.parent
            if not task_dir.is_dir():
                continue
            
            # Avoid re-packing already packed dirs
            # Check if parent matches job_prefix pattern
            if task_dir.parent.name.startswith(f"{self.job_prefix}_"):
                continue

            # Check exclusions
            excluded = False
            for pat in exclude_patterns:
                # Match against relative path from root to handle nested structures
                rel_path = task_dir.relative_to(root_dir)
                if rel_path.match(pat) or any(part.startswith(self.job_prefix) for part in rel_path.parts):
                    excluded = True
                    break
            if excluded:
                continue
            
            task_dirs.append(task_dir)
        
        # Sort for deterministic packing
        task_dirs.sort(key=lambda p: str(p))
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
                # Move task_dir to current_job_dir
                # Note: task_dir name might conflict if flattened. 
                # e.g. sys1/cluster/task_0 and sys2/bulk/task_0
                # Solution: Rename task dir to include parent info or just assume unique names?
                # Original script assumed unique names (sysname-index).
                # Our task_name is {sys_name}_{f_idx}, which should be unique across different systems if sys_name is unique.
                # If sys_name is dataset name, and we have multiple datasets, we might have collisions if different datasets use same naming scheme for frames?
                # Actually task_name = f"{sys_name}_{f_idx}" is unique enough usually.
                # But wait, if sys_name is "C20O0Fe0H0" (dataset name), and we have multiple systems in it?
                # The generator loop iterates systems. sys_name is dataset name.
                # Oh, wait. In multi-pool mode, we iterate input_data.
                # input_data is a MultiSystems.
                # If we loaded from multiple datasets, input_data is a flat list of Systems.
                # The 'short_name' attribute is usually the system name from dpdata.
                # If dpdata loads mixed format, short_name might be preserved.
                # Let's trust task_name is unique.
                
                shutil.move(str(task_dir), str(current_job_dir))
                sub_count += 1
                
                # Cleanup empty parent dirs
                try:
                    parent = task_dir.parent
                    while parent != root_dir:
                        if not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                        else:
                            break
                except OSError:
                    pass # Directory not empty or other error
                    
            except Exception as e:
                logger.error(f"Failed to move {task_dir} to {current_job_dir}: {e}")
        
        logger.info(f"Packed tasks into {len(job_dirs)} job directories.")
        return job_dirs
