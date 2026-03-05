"""
Resubmission Strategy
=====================

Handles the strategy for resubmitting failed calculations by modifying input parameters.
"""

import os
import logging
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class ResubmissionStrategy:
    """
    Manages parameters for each calculation attempt.
    
    This class handles the retrieval and application of parameters for different
    attempts in the calculation workflow. It allows for modifying input files
    based on the attempt index.
    """

    def __init__(self, attempt_params: List[Dict[str, Any]] = None):
        """
        Initialize with a list of parameter sets for each attempt.
        
        Args:
            attempt_params: List of dicts, where index i contains params for attempt i.
                            Example:
                            [
                                {"mixing_beta": 0.4},  # Attempt 0 (Initial)
                                {"mixing_beta": 0.1},  # Attempt 1 (Retry 1)
                            ]
        """
        self.attempt_params = attempt_params or [
            {"mixing_beta": 0.4, "mixing_ndim": 8},
            {"mixing_beta": 0.1, "mixing_ndim": 20},
            {"mixing_beta": 0.025, "mixing_ndim": 20}
        ]

    def get_params(self, attempt: int) -> Dict[str, Any]:
        """Get parameters for a specific attempt index."""
        if attempt < 0 or attempt >= len(self.attempt_params):
            return None
        return self.attempt_params[attempt]

    def apply(self, task_dir: str, attempt: int) -> bool:
        """
        Apply the parameters for the given attempt to the INPUT file.
        
        Args:
            task_dir: Directory containing the INPUT file.
            attempt: The attempt index (0-based).
            
        Returns:
            True if applied successfully, False if no params for this attempt.
        """
        params = self.get_params(attempt)
        if params is None:
            logger.warning(f"No parameters defined for attempt {attempt}")
            return False
            
        input_file = Path(task_dir) / "INPUT"
        if not input_file.exists():
            logger.error(f"INPUT file not found: {input_file}")
            return False
            
        return self._modify_input_file(input_file, params)

    def _modify_input_file(self, filepath: Path, changes: Dict[str, Any]) -> bool:
        """
        Modify specific keys in the INPUT file.
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return False

        new_lines = []
        modified_keys = set()
        
        for line in lines:
            stripped = line.strip()
            # Skip empty or comments
            if not stripped or stripped.startswith('#') or stripped.startswith('INPUT_PARAMETERS'):
                new_lines.append(line)
                continue
            
            parts = stripped.split()
            if len(parts) >= 2:
                key = parts[0]
                if key in changes:
                    # Replace value
                    # Try to preserve formatting (indentation)
                    original_key_end = line.find(key) + len(key)
                    # Find value start
                    value_start = -1
                    for i in range(original_key_end, len(line)):
                        if not line[i].isspace():
                            value_start = i
                            break
                    
                    if value_start != -1:
                        prefix = line[:value_start]
                        new_line = f"{prefix}{changes[key]}\n"
                    else:
                        new_line = f"{key:<20} {changes[key]}\n"
                        
                    new_lines.append(new_line)
                    modified_keys.add(key)
                    logger.debug(f"Modified {key} -> {changes[key]}")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Add missing keys
        for key, value in changes.items():
            if key not in modified_keys:
                new_lines.append(f"{key:<20} {value}\n")
                logger.debug(f"Added {key} -> {value}")
        
        try:
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            return True
        except Exception as e:
            logger.error(f"Error writing {filepath}: {e}")
            return False
