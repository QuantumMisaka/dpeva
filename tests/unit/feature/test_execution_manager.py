import pytest
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dpeva.feature.managers import FeatureExecutionManager, FeatureIOManager
from dpeva.submission import JobConfig

@pytest.fixture
def mock_job_manager():
    with patch("dpeva.feature.managers.JobManager") as mock_cls:
        yield mock_cls

class TestFeatureExecutionManager:
    
    def test_submit_cli_job_single_pool(self, mock_job_manager, tmp_path):
        """Test CLI job submission for single pool."""
        manager = FeatureExecutionManager(
            backend="slurm",
            slurm_config={"partition": "gpu"},
            env_setup="module load deepmd",
            dp_backend="pt",
            omp_threads=4
        )
        
        manager.submit_cli_job(
            data_path="data",
            output_dir="output",
            model_path="model.pt",
            head="OC20M",
            sub_pools=[]
        )
        
        # Verify
        jm = mock_job_manager.return_value
        assert jm.generate_script.called
        
        job_config = jm.generate_script.call_args[0][0]
        assert "dp --pt eval-desc" in job_config.command
        assert "-m " in job_config.command
        assert "module load deepmd" in job_config.env_setup
        assert job_config.partition == "gpu"

    def test_submit_cli_job_multi_pool(self, mock_job_manager, tmp_path):
        """Test CLI job submission for multi-pool."""
        manager = FeatureExecutionManager(
            backend="local",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            omp_threads=2
        )
        
        manager.submit_cli_job(
            data_path="data",
            output_dir="output",
            model_path="model.pt",
            head="OC20M",
            sub_pools=["pool1", "pool2"]
        )
        
        # Verify
        jm = mock_job_manager.return_value
        job_config = jm.generate_script.call_args[0][0]
        
        assert "Processing pool: pool1" in job_config.command
        assert "Processing pool: pool2" in job_config.command
        assert "mkdir -p" in job_config.command

    def test_submit_python_slurm_job(self, mock_job_manager):
        """Test Python Slurm job submission."""
        manager = FeatureExecutionManager(
            backend="slurm",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            omp_threads=4
        )
        
        manager.submit_python_slurm_job(
            data_path="data",
            output_dir="output",
            model_path="model.pt",
            head="OC20M",
            batch_size=100,
            output_mode="atomic"
        )
        
        jm = mock_job_manager.return_value
        assert jm.submit_python_script.called
        
        args = jm.submit_python_script.call_args[0]
        content = args[0]
        
        assert "from dpeva.feature.generator import DescriptorGenerator" in content
        assert "FeatureExecutionManager" in content
        assert "run_local_python_recursion" in content

    @patch("dpeva.feature.managers.FeatureIOManager")
    def test_run_local_python_recursion(self, MockIO, tmp_path):
        """Test local recursion logic."""
        # Setup directories
        # data/
        #   sys1/ (leaf)
        #   group/
        #     sys2/ (leaf)
        
        data_root = tmp_path / "data"
        (data_root / "sys1").mkdir(parents=True)
        (data_root / "group" / "sys2").mkdir(parents=True)
        
        output_root = tmp_path / "output"
        
        # Mock IO Manager to identify leafs
        io_instance = MockIO.return_value
        def is_leaf(path):
            p = str(path)
            return p.endswith("sys1") or p.endswith("sys2")
        io_instance.is_leaf_system.side_effect = is_leaf
        
        # Mock Generator
        mock_generator = MagicMock()
        mock_generator.compute_descriptors.return_value = np.zeros((10, 4))
        
        manager = FeatureExecutionManager("local", {}, "", "pt", 1)
        
        with patch("numpy.save") as mock_save:
            manager.run_local_python_recursion(
                mock_generator,
                str(data_root),
                str(output_root)
            )
            
            # Check calls
            assert mock_generator.compute_descriptors.call_count == 2
            
            # Check save paths
            save_paths = [args[0] for args, _ in mock_save.call_args_list]
            expected = [
                str(output_root / "sys1.npy"),
                str(output_root / "group" / "sys2.npy")
            ]
            assert sorted(save_paths) == sorted(expected)

class TestFeatureIOManager:
    def test_detect_multi_pool_structure(self, tmp_path):
        io = FeatureIOManager()
        
        root = tmp_path / "data"
        root.mkdir()
        
        # Pool 1
        (root / "pool1").mkdir()
        (root / "pool1" / "type.raw").touch() # Is system
        
        # Pool 2 (container)
        (root / "pool2").mkdir()
        (root / "pool2" / "sys1").mkdir()
        (root / "pool2" / "sys1" / "type.raw").touch()
        
        # Detect
        # pool1 is system -> not sub-pool (wait, logic says: if is_system, then NOT sub_pool)
        # The logic is: "A sub-pool is a directory that is NOT a system itself but contains systems."
        
        # pool1 has type.raw -> is_system -> not sub_pool.
        # pool2 has no type.raw -> not system -> is sub_pool.
        
        sub_pools = io.detect_multi_pool_structure(str(root))
        assert "pool2" in sub_pools
        assert "pool1" not in sub_pools
