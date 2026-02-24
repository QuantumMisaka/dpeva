import os
import subprocess
import logging
from typing import Literal, Optional, Union
from .templates import TemplateEngine, JobConfig

logger = logging.getLogger(__name__)

class JobManager:
    """
    Job Submission Manager.
    
    Handles the generation of submission scripts and execution of jobs
    for both local and Slurm environments.
    """
    def __init__(self, 
                 mode: Literal["local", "slurm"] = "local", 
                 custom_template_path: Optional[str] = None):
        """
        Initialize the JobManager.

        Args:
            mode (Literal["local", "slurm"], optional): Submission mode. Defaults to "local".
            custom_template_path (str, optional): Path to a custom template file. Defaults to None.
        """
        self.mode = mode
        
        # Initialize Template Engine
        if custom_template_path:
            if os.path.exists(custom_template_path):
                self.engine = TemplateEngine.from_file(custom_template_path)
            else:
                raise FileNotFoundError(f"Custom template not found: {custom_template_path}")
        else:
            self.engine = TemplateEngine.from_default(mode)

    def generate_script(self, config: JobConfig, output_path: str) -> str:
        """
        Generate a job script from configuration and write it to a file.

        Args:
            config (JobConfig): The job configuration object.
            output_path (str): The path to save the generated script.

        Returns:
            str: The absolute path to the generated script.
        """
        content = self.engine.render(config)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        # Grant execution permissions (Readability counts: Explicit permission setting)
        os.chmod(output_path, 0o755)
        
        logger.info(f"Generated job script: {output_path}")
        return output_path

    def submit(self, script_path: str, working_dir: str = ".") -> str:
        """
        Submit the job script.
        
        Args:
            script_path (str): Path to the submission script.
            working_dir (str, optional): Directory to execute the submission from. Defaults to ".".

        Returns:
            str: Output from the submission command (e.g. job ID).
        
        Raises:
            subprocess.CalledProcessError: If submission fails.
        """
        script_path = os.path.abspath(script_path)
        
        if self.mode == "slurm":
            cmd = ["sbatch", script_path]
        else:
            # Local mode: Run directly with bash
            # User can decide whether to use nohup in the command itself if needed
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

    def submit_python_script(self, script_content: str, script_name: str, job_config: JobConfig, working_dir: str = ".") -> str:
        """
        Helper to write a python script and submit it as a job.
        
        Args:
            script_content (str): The Python code to run.
            script_name (str): Name of the python file (e.g. "run_task.py").
            job_config (JobConfig): Configuration for the job. 
                                    The 'command' field will be overridden to run the python script.
            working_dir (str): Directory to write script and submit job.
            
        Returns:
            str: Submission result.
        """
        import sys
        
        # Ensure working dir exists
        os.makedirs(working_dir, exist_ok=True)
        
        script_path = os.path.join(working_dir, script_name)
        with open(script_path, "w") as f:
            f.write(script_content)
            
        # Update command to run the python script
        # Use sys.executable for safety and -u for unbuffered output
        cmd = f"{sys.executable} -u {os.path.abspath(script_path)}"
        job_config.command = cmd
        
        # Generate submission script (bash/slurm)
        submit_script_name = "submit_" + os.path.splitext(script_name)[0]
        if self.mode == "slurm":
            submit_script_name += ".slurm"
        else:
            submit_script_name += ".sh"
            
        submit_script_path = os.path.join(working_dir, submit_script_name)
        
        self.generate_script(job_config, submit_script_path)
        
        return self.submit(submit_script_path, working_dir=working_dir)
