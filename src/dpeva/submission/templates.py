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

@dataclass
class JobConfig:
    """
    统一的作业配置类，包含所有可能用到的字段。
    遵循 'Explicit is better than implicit' 原则，核心 Slurm 参数应显式定义。
    """
    command: str
    job_name: str = "dpeva_job"
    
    # Slurm Specific
    partition: str = "partition"
    nodes: int = 1
    ntasks: int = 1
    
    # Advanced Slurm Options (适配不同集群环境)
    gpus_per_node: int = 0      # 对应 #SBATCH --gpus-per-node
    cpus_per_task: int = 1      # 对应 #SBATCH --cpus-per-task
    qos: Optional[str] = None   # 对应 #SBATCH --qos
    nodelist: Optional[str] = None # 对应 #SBATCH -w
    
    walltime: str = "24:00:00"
    output_log: str = "job.out"
    error_log: str = "job.err"
    custom_headers: str = ""
    
    # Common
    env_setup: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to dictionary with all values as strings.
        处理可选字段的格式化，保持模板简洁。
        """
        d = {k: str(v) for k, v in asdict(self).items()}
        
        # 聚合可选的 Slurm 参数，保持模板整洁
        optional_params = []
        
        # 动态生成 Slurm 参数行
        if self.gpus_per_node > 0:
            optional_params.append(f"#SBATCH --gpus-per-node={self.gpus_per_node}")

        if self.cpus_per_task > 1:
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
