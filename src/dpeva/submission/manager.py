import os
import subprocess
import logging
from typing import Literal, Optional, Union
from .templates import TemplateEngine, JobConfig

logger = logging.getLogger(__name__)

class JobManager:
    """
    作业提交管理器。
    """
    def __init__(self, 
                 mode: Literal["local", "slurm"] = "local", 
                 custom_template_path: Optional[str] = None):
        self.mode = mode
        
        # 初始化模板引擎
        if custom_template_path:
            if os.path.exists(custom_template_path):
                self.engine = TemplateEngine.from_file(custom_template_path)
            else:
                raise FileNotFoundError(f"Custom template not found: {custom_template_path}")
        else:
            self.engine = TemplateEngine.from_default(mode)

    def generate_script(self, config: JobConfig, output_path: str) -> str:
        """生成作业脚本并写入文件"""
        content = self.engine.render(config)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        # 赋予执行权限 (Readability counts: 显式设置权限)
        os.chmod(output_path, 0o755)
        
        logger.info(f"Generated job script: {output_path}")
        return output_path

    def submit(self, script_path: str, working_dir: str = ".") -> str:
        """
        提交作业。
        Local 模式下直接运行 (bash)。
        Slurm 模式下使用 sbatch 提交。
        """
        script_path = os.path.abspath(script_path)
        
        if self.mode == "slurm":
            cmd = ["sbatch", script_path]
        else:
            # Local mode: 使用 nohup 后台运行，或者直接运行
            # 这里选择直接运行 bash，用户可以在 command 中决定是否 nohup
            cmd = ["bash", script_path]

        try:
            logger.info(f"Submitting job: {' '.join(cmd)} in {working_dir}")
            result = subprocess.run(
                cmd, 
                cwd=working_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"Submission result: {result.stdout.strip()}")
            return result.stdout.strip()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Submission failed: {e.stderr}")
            raise e
