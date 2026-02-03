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
    统一的作业配置类，包含所有可能用到的字段。
    遵循 'Explicit is better than implicit' 原则，核心 Slurm 参数应显式定义。
    """
    command: str
    job_name: str = "dpeva_job"
    
    # Slurm Specific
    partition: Optional[str] = DEFAULT_SLURM_PARTITION
    nodes: int = DEFAULT_SLURM_NODES
    ntasks: int = DEFAULT_SLURM_NTASKS
    
    # Advanced Slurm Options (适配不同集群环境)
    gpus_per_node: Optional[int] = DEFAULT_SLURM_GPUS_PER_NODE      # 对应 #SBATCH --gpus-per-node
    cpus_per_task: Optional[int] = DEFAULT_SLURM_CPUS_PER_TASK      # 对应 #SBATCH --cpus-per-task
    qos: Optional[str] = DEFAULT_SLURM_QOS   # 对应 #SBATCH --qos
    nodelist: Optional[str] = DEFAULT_SLURM_NODELIST # 对应 #SBATCH -w
    
    walltime: str = DEFAULT_WALLTIME
    output_log: Optional[str] = None
    error_log: Optional[str] = None
    custom_headers: Union[str, List[str]] = DEFAULT_SLURM_CUSTOM_HEADERS
    
    # Common
    env_setup: Union[str, List[str]] = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to dictionary with all values as strings.
        处理可选字段的格式化，保持模板简洁。
        """
        # Manual conversion to ensure lists (like custom_headers) are joined correctly
        d = {}
        for k, v in asdict(self).items():
            # 专门处理列表类型的 custom_headers
            if k == "custom_headers" and isinstance(v, list):
                d[k] = "\n".join(v)
            elif k == "env_setup" and isinstance(v, list):
                d[k] = "\n".join(v)
            elif v is None:
                d[k] = ""
            else:
                d[k] = str(v)
        
        # 聚合可选的 Slurm 参数，保持模板整洁
        optional_params = []
        
        # 动态生成 Slurm 参数行
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
# 模板引擎 (Template Engine)
# ==========================================

class TemplateEngine:
    """
    负责加载和渲染模板。
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
        # safe_substitute 允许模板中存在 config 中未定义的变量而不报错
        # 但为了 Explicit，建议 config 覆盖所有需要的变量
        return self.template.safe_substitute(config.to_dict())
