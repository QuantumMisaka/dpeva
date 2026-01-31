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

    def test_custom_template_error(self):
        """Verify error when custom template missing."""
        with pytest.raises(FileNotFoundError):
            JobManager(custom_template_path="non_existent.jinja2")
