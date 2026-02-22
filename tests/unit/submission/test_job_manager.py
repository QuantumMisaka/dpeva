import pytest
import os
import subprocess
from unittest.mock import MagicMock, patch
from dpeva.submission.manager import JobManager
from dpeva.submission.templates import JobConfig

@pytest.fixture
def basic_job_config():
    return JobConfig(
        job_name="test_job",
        command="echo hello",
        partition="debug",
        gpus_per_node=1
    )

class TestJobManager:
    
    def test_generate_script_local(self, tmp_path, basic_job_config):
        """Verify local script generation."""
        manager = JobManager(mode="local")
        output_path = tmp_path / "run.sh"
        
        manager.generate_script(basic_job_config, str(output_path))
        
        assert output_path.exists()
        content = output_path.read_text()
        
        # Local template should be simple bash
        assert "#!/bin/bash" in content
        assert "echo hello" in content
        # Should NOT contain SBATCH directives
        assert "#SBATCH" not in content

    def test_generate_script_slurm(self, tmp_path, basic_job_config):
        """Verify Slurm script generation."""
        manager = JobManager(mode="slurm")
        output_path = tmp_path / "run.slurm"
        
        manager.generate_script(basic_job_config, str(output_path))
        
        assert output_path.exists()
        content = output_path.read_text()
        
        # Check Slurm directives
        assert "#SBATCH -J test_job" in content
        assert "#SBATCH -p debug" in content
        assert "#SBATCH --gpus-per-node=1" in content
        assert "echo hello" in content

    @patch("subprocess.run")
    def test_submit_local(self, mock_run, tmp_path):
        """Verify local submission uses bash."""
        manager = JobManager(mode="local")
        script_path = tmp_path / "run.sh"
        script_path.touch()
        
        manager.submit(str(script_path))
        
        # Verify subprocess call
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "bash"
        assert cmd[1] == str(script_path)

    @patch("subprocess.run")
    def test_submit_slurm(self, mock_run, tmp_path):
        """Verify Slurm submission uses sbatch."""
        manager = JobManager(mode="slurm")
        script_path = tmp_path / "run.slurm"
        script_path.touch()
        
        # Mock success output
        mock_run.return_value.stdout = "Submitted batch job 12345"
        
        manager.submit(str(script_path))
        
        # Verify subprocess call
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "sbatch"
        assert cmd[1] == str(script_path)

    def test_env_setup_handling(self, tmp_path):
        """Verify env_setup is correctly inserted."""
        # Test List format
        config_list = JobConfig(
            command="echo hi", 
            env_setup=["module load python", "export VAR=1"]
        )
        
        manager = JobManager(mode="local")
        out_list = tmp_path / "env_list.sh"
        manager.generate_script(config_list, str(out_list))
        
        content = out_list.read_text()
        assert "module load python" in content
        assert "export VAR=1" in content
        
        # Test String format (backward compatibility)
        config_str = JobConfig(
            command="echo hi",
            env_setup="source activate myenv"
        )
        out_str = tmp_path / "env_str.sh"
        manager.generate_script(config_str, str(out_str))
        
        content = out_str.read_text()
        assert "source activate myenv" in content

    def test_custom_template_error(self):
        """Verify error when custom template missing."""
        with pytest.raises(FileNotFoundError):
            JobManager(custom_template_path="non_existent.jinja2")

    @patch("subprocess.run")
    def test_submit_error(self, mock_run, tmp_path):
        """Verify submission error handling."""
        manager = JobManager(mode="local")
        script_path = tmp_path / "run.sh"
        script_path.touch()
        
        # Mock failure
        mock_run.side_effect = subprocess.CalledProcessError(1, "bash", stderr="Error message")
        
        with pytest.raises(subprocess.CalledProcessError):
            manager.submit(str(script_path))

    @patch("dpeva.submission.manager.JobManager.submit")
    @patch("dpeva.submission.manager.JobManager.generate_script")
    def test_submit_python_script(self, mock_gen, mock_submit, tmp_path):
        """Verify python script submission helper."""
        manager = JobManager(mode="local")
        job_config = JobConfig(job_name="test", command="placeholder")
        
        manager.submit_python_script("print('hello')", "script.py", job_config, str(tmp_path))
        
        # Verify script creation
        py_script = tmp_path / "script.py"
        assert py_script.exists()
        assert "print('hello')" in py_script.read_text()
        
        # Verify command update
        assert "python" in job_config.command or "py" in job_config.command # sys.executable
        assert "script.py" in job_config.command
        
        # Verify generate and submit called
        mock_gen.assert_called()
        mock_submit.assert_called()
