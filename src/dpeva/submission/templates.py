import string
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# ==========================================
# 默认模板 (Default Templates)
# ==========================================

DEFAULT_SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -p ${partition}
#SBATCH -N ${nodes}
#SBATCH -n ${ntasks}
#SBATCH -t ${walltime}
#SBATCH -o ${output_log}
#SBATCH -e ${error_log}
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

@dataclass
class JobConfig:
    """
    统一的作业配置类，包含所有可能用到的字段。
    用户只需按需填充，未使用的字段将在渲染时使用默认值或空字符串。
    """
    command: str
    job_name: str = "dpeva_job"
    
    # Slurm Specific
    partition: str = "partition"
    nodes: int = 1
    ntasks: int = 1
    walltime: str = "24:00:00"
    output_log: str = "job.out"
    error_log: str = "job.err"
    custom_headers: str = ""
    
    # Common
    env_setup: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary with all values as strings."""
        return {k: str(v) for k, v in asdict(self).items()}

# ==========================================
# 模板引擎 (Template Engine)
# ==========================================

class TemplateEngine:
    """
    负责加载和渲染模板。
    """
    def __init__(self, template_content: str):
        self.template = string.Template(template_content)

    @classmethod
    def from_file(cls, filepath: str) -> 'TemplateEngine':
        with open(filepath, 'r') as f:
            return cls(f.read())

    @classmethod
    def from_default(cls, mode: str) -> 'TemplateEngine':
        if mode == "slurm":
            return cls(DEFAULT_SLURM_TEMPLATE)
        elif mode == "local":
            return cls(DEFAULT_LOCAL_TEMPLATE)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'slurm' or 'local'.")

    def render(self, config: JobConfig) -> str:
        # safe_substitute 允许模板中存在 config 中未定义的变量而不报错
        # 但为了 Explicit，建议 config 覆盖所有需要的变量
        return self.template.safe_substitute(config.to_dict())
