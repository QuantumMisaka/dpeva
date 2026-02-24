import string
import os
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict

# ==========================================
# 默认模板 (Default Templates)
# ==========================================

DEFAULT_SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -N ${nodes}
#SBATCH -n ${ntasks}
#SBATCH -t ${walltime}
${optional_slurm_params}
${custom_headers}

# Environment Setup
${env_setup}

# Command
${command}
"""

DEFAULT_LOCAL_TEMPLATE = """#!/bin/bash
# Job Name: ${job_name}
# Created by DPEVA

# Environment Setup
${env_setup}

# Command
${command}
"""

# ==========================================
# 模板配置数据类 (Config Dataclass)
# ==========================================

from dpeva.constants import (
    DEFAULT_SLURM_PARTITION, 
    DEFAULT_WALLTIME,
    DEFAULT_SLURM_GPUS_PER_NODE,
    DEFAULT_SLURM_CPUS_PER_TASK,
    DEFAULT_SLURM_QOS,
    DEFAULT_SLURM_NODELIST,
    DEFAULT_SLURM_NODES,
    DEFAULT_SLURM_NTASKS,
    DEFAULT_SLURM_CUSTOM_HEADERS
)

@dataclass
class JobConfig:
    """
    Unified Job Configuration Data Class.

    Contains all possible fields for job submission.
    Follows 'Explicit is better than implicit' principle; core Slurm parameters should be explicitly defined.
    """
    command: str
    job_name: str = "dpeva_job"
    
    # Slurm Specific
    partition: Optional[str] = DEFAULT_SLURM_PARTITION
    nodes: int = DEFAULT_SLURM_NODES
    ntasks: int = DEFAULT_SLURM_NTASKS
    
    # Advanced Slurm Options (Adapts to different cluster environments)
    gpus_per_node: Optional[int] = DEFAULT_SLURM_GPUS_PER_NODE      # Maps to #SBATCH --gpus-per-node
    cpus_per_task: Optional[int] = DEFAULT_SLURM_CPUS_PER_TASK      # Maps to #SBATCH --cpus-per-task
    qos: Optional[str] = DEFAULT_SLURM_QOS   # Maps to #SBATCH --qos
    nodelist: Optional[str] = DEFAULT_SLURM_NODELIST # Maps to #SBATCH -w
    
    walltime: str = DEFAULT_WALLTIME
    output_log: Optional[str] = None
    error_log: Optional[str] = None
    custom_headers: Union[str, List[str]] = DEFAULT_SLURM_CUSTOM_HEADERS
    
    # Common
    env_setup: Union[str, List[str]] = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to dictionary with all values as strings.
        
        Handles optional fields formatting to keep the template clean.
        """
        # Manual conversion to ensure lists (like custom_headers) are joined correctly
        d = {}
        for k, v in asdict(self).items():
            # Handle list types specifically
            if k == "custom_headers" and isinstance(v, list):
                d[k] = "\n".join(v)
            elif k == "env_setup" and isinstance(v, list):
                d[k] = "\n".join(v)
            elif v is None:
                d[k] = ""
            else:
                d[k] = str(v)
        
        # Aggregate optional Slurm parameters to keep the template clean
        optional_params = []
        
        # Dynamically generate Slurm parameter lines
        if self.partition:
            optional_params.append(f"#SBATCH -p {self.partition}")

        if self.output_log:
            optional_params.append(f"#SBATCH -o {self.output_log}")

        if self.error_log:
            optional_params.append(f"#SBATCH -e {self.error_log}")

        if self.gpus_per_node:
            optional_params.append(f"#SBATCH --gpus-per-node={self.gpus_per_node}")

        if self.cpus_per_task:
            optional_params.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")

        if self.qos:
            optional_params.append(f"#SBATCH --qos={self.qos}")

        if self.nodelist:
            optional_params.append(f"#SBATCH -w {self.nodelist}")
            
        d['optional_slurm_params'] = "\n".join(optional_params)
            
        return d

# ==========================================
# Template Engine
# ==========================================

class TemplateEngine:
    """
    Responsible for loading and rendering templates.
    """
    def __init__(self, template_content: str):
        """
        Initialize the TemplateEngine.

        Args:
            template_content (str): The raw template string.
        """
        self.template = string.Template(template_content)

    @classmethod
    def from_file(cls, filepath: str) -> 'TemplateEngine':
        """
        Create a TemplateEngine from a file.

        Args:
            filepath (str): Path to the template file.

        Returns:
            TemplateEngine: Initialized engine.
        """
        with open(filepath, 'r') as f:
            return cls(f.read())

    @classmethod
    def from_default(cls, mode: str) -> 'TemplateEngine':
        """
        Create a TemplateEngine from default templates.

        Args:
            mode (str): "slurm" or "local".

        Returns:
            TemplateEngine: Initialized engine.
        
        Raises:
            ValueError: If mode is unknown.
        """
        if mode == "slurm":
            return cls(DEFAULT_SLURM_TEMPLATE)
        elif mode == "local":
            return cls(DEFAULT_LOCAL_TEMPLATE)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'slurm' or 'local'.")

    def render(self, config: JobConfig) -> str:
        """
        Render the template with configuration values.

        Args:
            config (JobConfig): Configuration object.

        Returns:
            str: Rendered script content.
        """
        # safe_substitute allows variables in template that are not in config without error
        # But for Explicit, it is recommended that config covers all needed variables
        return self.template.safe_substitute(config.to_dict())
