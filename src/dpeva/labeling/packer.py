"""
Task Packer
===========

Handles the distribution of tasks into subdirectories for efficient submission.
"""

import logging
import shutil
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
                
                destination = self.resolve_destination(task_dir, current_job_dir, root_dir)
                shutil.move(str(task_dir), str(destination))
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

    @classmethod
    def resolve_destination(cls, task_dir: Path, job_dir: Path, root_dir: Path) -> Path:
        destination = job_dir / task_dir.name
        if not destination.exists():
            return destination

        qualified_name = cls._qualified_task_name(task_dir, root_dir)
        destination = job_dir / qualified_name
        if not destination.exists():
            return destination

        idx = 1
        while True:
            candidate = job_dir / f"{qualified_name}_dup{idx}"
            if not candidate.exists():
                return candidate
            idx += 1

    @classmethod
    def _qualified_task_name(cls, task_dir: Path, root_dir: Path) -> str:
        meta = cls._read_json_file(task_dir / "task_meta.json") or {}
        dataset = cls._safe_name_part(meta.get("dataset_name"))
        stru_type = cls._safe_name_part(meta.get("stru_type"))
        task_name = cls._safe_name_part(meta.get("task_name"), task_dir.name)

        if dataset is None or stru_type is None:
            inferred_dataset, inferred_type = cls._infer_dataset_type(task_dir, root_dir)
            dataset = dataset or inferred_dataset or "unknown"
            stru_type = stru_type or inferred_type or "unknown"
        return f"{dataset}__{stru_type}__{task_name}"

    @staticmethod
    def _safe_name_part(value: Any, fallback: Optional[str] = None) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            text = value.strip()
        elif fallback is not None:
            text = fallback
        else:
            return None
        return text.replace("/", "_").replace("\\", "_")

    @staticmethod
    def _infer_dataset_type(task_dir: Path, root_dir: Path) -> tuple[Optional[str], Optional[str]]:
        try:
            rel_path = task_dir.relative_to(root_dir)
        except ValueError:
            return None, None
        if len(rel_path.parts) >= 3:
            return rel_path.parts[0], rel_path.parts[1]
        return None, None

    @staticmethod
    def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    return loaded
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        return None
