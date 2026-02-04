import pytest
import os
import numpy as np
from unittest.mock import MagicMock, patch, call
from dpeva.feature.generator import DescriptorGenerator
from dpeva.submission import JobConfig

@pytest.fixture
def mock_job_manager():
    with patch("dpeva.feature.generator.JobManager") as mock_cls:
        yield mock_cls

@pytest.fixture
def mock_deep_pot():
    # Use create=True because DeepPot might not exist if import failed
    with patch("dpeva.feature.generator.DeepPot", create=True) as mock_cls:
        yield mock_cls

class TestDescriptorGenerator:
    
    def test_init_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            DescriptorGenerator(model_path="model.pt", mode="invalid")

    def test_cli_generation_command_construction(self, mock_job_manager, tmp_path):
        """Verify CLI command construction and job submission."""
        # Setup
        model_path = tmp_path / "model.pt"
        data_path = tmp_path / "data"
        output_dir = tmp_path / "output"
        
        # Initialize generator
        generator = DescriptorGenerator(
            model_path=str(model_path),
            head="MyHead",
            mode="cli",
            backend="slurm",
            slurm_config={"partition": "gpu"}
        )
        
        # Run generation
        generator.run_cli_generation(str(data_path), str(output_dir))
        
        # Verify JobManager interaction
        manager_instance = mock_job_manager.return_value
        assert manager_instance.generate_script.called
        assert manager_instance.submit.called
        
        # Verify generated config
        call_args = manager_instance.generate_script.call_args
        job_config = call_args[0][0] # First arg is job_config
        
        assert isinstance(job_config, JobConfig)
        # Check command content
        assert "dp --pt eval-desc" in job_config.command
        assert f"-m {os.path.abspath(model_path)}" in job_config.command
        assert f"-s {os.path.abspath(data_path)}" in job_config.command
        assert "--head MyHead" in job_config.command
        
        # Check Slurm config passthrough
        assert job_config.partition == "gpu"

    @patch("dpeva.feature.generator._DEEPMD_AVAILABLE", True)
    def test_python_generation_recursion(self, mock_deep_pot, tmp_path):
        """Verify recursion logic in python mode."""
        # Setup directory structure:
        # data/
        #   sys1/ (leaf)
        #     type.raw
        #   group1/
        #     sys2/ (leaf)
        #       type.raw
        
        data_root = tmp_path / "data"
        (data_root / "sys1").mkdir(parents=True)
        (data_root / "sys1" / "type.raw").touch()
        
        (data_root / "group1" / "sys2").mkdir(parents=True)
        (data_root / "group1" / "sys2" / "type.raw").touch()
        
        output_root = tmp_path / "output"
        
        # Initialize generator (Python mode)
        generator = DescriptorGenerator(
            model_path="model.pt",
            mode="python",
            backend="local"
        )
        
        # Mock actual computation to avoid DP dependency and file writing
        with patch.object(generator, "compute_descriptors_python") as mock_compute:
            mock_compute.return_value = MagicMock() # Return a mock array-like
            
            # We also need to mock np.save to avoid error on saving MagicMock
            with patch("numpy.save") as mock_save:
                generator.run_python_generation(str(data_root), str(output_root))
                
                # Check that compute was called for both systems
                assert mock_compute.call_count == 2
                
                # Check paths passed to compute
                called_paths = []
                for call_args in mock_compute.call_args_list:
                    args, kwargs = call_args
                    if args:
                        called_paths.append(args[0])
                    elif 'data_path' in kwargs:
                        called_paths.append(kwargs['data_path'])
                    else:
                        # Should not happen given the call signature
                        pass
                
                expected_paths = [
                    str(data_root / "sys1"),
                    str(data_root / "group1" / "sys2")
                ]
                # Sort to ignore order
                assert sorted(called_paths) == sorted(expected_paths)
                
                # Check save locations
                save_locs = [args[0] for args, _ in mock_save.call_args_list]
                expected_saves = [
                    str(output_root / "sys1.npy"),
                    str(output_root / "group1" / "sys2.npy")
                ]
                assert sorted(save_locs) == sorted(expected_saves)

    def test_python_generation_slurm_wrapper(self, mock_job_manager, tmp_path):
        """Verify that python mode with slurm backend generates a worker script."""
        generator = DescriptorGenerator(
            model_path="model.pt",
            mode="python",
            backend="slurm"
        )
        
        output_dir = tmp_path / "output"
        generator.run_python_generation("data", str(output_dir))
        
        # Verify JobManager interaction
        manager_instance = mock_job_manager.return_value
        assert manager_instance.submit_python_script.called
        
        # Check arguments passed to submit_python_script
        # signature: (script_content, script_name, job_config, working_dir=...)
        call_args = manager_instance.submit_python_script.call_args
        args, kwargs = call_args
        
        content = args[0]
        script_name = args[1]
        
        assert script_name == "run_desc_worker.py"
        assert "from dpeva.feature.generator import DescriptorGenerator" in content
        assert 'mode="python"' in content
        assert 'backend="local"' in content # The worker runs locally on the node

    @patch("dpeva.feature.generator.load_systems")
    @patch("dpeva.feature.generator._DEEPMD_AVAILABLE", True)
    def test_compute_descriptors_python_auto(self, mock_load_systems, mock_deep_pot, tmp_path):
        """Verify compute_descriptors_python uses load_systems and handles list."""
        generator = DescriptorGenerator(
            model_path="model.pt",
            mode="python",
            backend="local"
        )
        generator.model = MagicMock()
        
        # Mock load_systems return
        mock_sys1 = MagicMock()
        mock_sys1.data = {'nopbc': False}
        mock_sys1.__len__.return_value = 10
        
        mock_sys2 = MagicMock()
        mock_sys2.data = {'nopbc': False}
        mock_sys2.__len__.return_value = 5
        
        mock_load_systems.return_value = [mock_sys1, mock_sys2]
        
        # Mock _get_desc_by_batch
        with patch.object(generator, "_get_desc_by_batch") as mock_get_batch:
            # Return list of arrays
            mock_get_batch.side_effect = [
                [np.zeros((10, 32))], # sys1
                [np.zeros((5, 32))]   # sys2
            ]
            
            result = generator.compute_descriptors_python("dummy_path", data_format="auto")
            
            # Verification
            mock_load_systems.assert_called_with("dummy_path", fmt="auto")
            assert mock_get_batch.call_count == 2
            assert result.shape == (15, 32)
