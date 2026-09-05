import pytest
import numpy as np
import subprocess
from unittest.mock import MagicMock, patch
from dpeva.constants import WORKFLOW_FINISHED_TAG
from dpeva.feature.managers import FeatureExecutionManager, FeatureIOManager
from dpeva.compatibility import DeepMDAdapter
from dpeva.utils.command import DPCommandBuilder
from dpeva.utils.exceptions import WorkflowError

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
            output_dir=str(tmp_path / "output"),
            model_path="model.pt",
            head="OC20M",
            sub_pools=[]
        )
        
        # Verify
        jm = mock_job_manager.return_value
        assert jm.generate_script.called
        
        job_config = jm.generate_script.call_args[0][0]
        assert "dp --pt eval-desc" in job_config.command
        assert "find " in job_config.command
        assert job_config.command.index("find ") < job_config.command.index("DPEVA_TAG: WORKFLOW_FINISHED")
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
            output_dir=str(tmp_path / "output"),
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

    def test_empty_descriptor_artifact_does_not_emit_finished(self, mock_job_manager, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        manager = FeatureExecutionManager(
            backend="local",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            omp_threads=1,
        )

        manager.submit_cli_job(
            data_path=str(tmp_path / "data"),
            output_dir=str(output_dir),
            model_path="model.pt",
            head="head",
            sub_pools=[],
        )

        (output_dir / "empty.npy").touch()
        generated = mock_job_manager.return_value.generate_script.call_args[0][0].command
        command = generated.replace(generated.splitlines()[0], "true", 1)
        result = subprocess.run(
            ["bash", "-c", "set -Eeuo pipefail\n" + command],
            cwd=output_dir,
            text=True,
            capture_output=True,
        )

        assert result.returncode != 0
        assert "DPEVA_TAG: WORKFLOW_FINISHED" not in result.stdout

    def test_submit_cli_job_embed_keeps_hdf5_for_last_layer(self, mock_job_manager, tmp_path):
        """Embed CLI should support fitting-last-layer features through HDF5 atomic_feature."""
        manager = FeatureExecutionManager(
            backend="slurm",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            omp_threads=1,
        )

        manager.submit_cli_job(
            data_path="data",
            output_dir=str(tmp_path / "output"),
            model_path="model.pt",
            head="OC20M",
            sub_pools=[],
            feature_exporter="embed",
            feature_kind="fitting_last_layer",
            embedding_dtype="native",
        )

        jm = mock_job_manager.return_value
        job_config = jm.generate_script.call_args[0][0]
        script_path = jm.generate_script.call_args[0][1]

        assert "dp --pt embed" in job_config.command
        assert "eval-desc" not in job_config.command
        assert "--dtype native" in job_config.command
        assert "embedding.hdf5" in job_config.command
        assert "test -s " in job_config.command
        assert job_config.command.index("test -s ") < job_config.command.index("DPEVA_TAG: WORKFLOW_FINISHED")
        assert script_path.endswith("run_embed.slurm")

    def test_submit_cli_job_embed_multi_pool_writes_one_hdf5_per_pool(self, mock_job_manager, tmp_path):
        """Embed CLI should preserve DP-EVA's multi-pool output layout."""
        manager = FeatureExecutionManager(
            backend="local",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            omp_threads=1,
        )

        manager.submit_cli_job(
            data_path="data",
            output_dir=str(tmp_path / "output"),
            model_path="model.pt",
            head="OC20M",
            sub_pools=["pool1", "pool2"],
            feature_exporter="embed",
            feature_kind="descriptor",
            embedding_dtype="fp32",
        )

        jm = mock_job_manager.return_value
        job_config = jm.generate_script.call_args[0][0]

        assert "Processing pool: pool1" in job_config.command
        assert "Processing pool: pool2" in job_config.command
        assert "pool1/embedding.hdf5" in job_config.command
        assert "pool2/embedding.hdf5" in job_config.command
        assert "/output/embedding.hdf5" not in job_config.command

    def test_command_builder_embed_quotes_dtype_and_head(self):
        cmd = DPCommandBuilder.embed(
            model="model path.pt",
            system="data path",
            output="out/embedding.hdf5",
            head="OC20M",
            dtype="fp64",
            backend="pt",
        )

        assert cmd == (
            "dp --pt embed -s 'data path' -m 'model path.pt' "
            "-o out/embedding.hdf5 --dtype fp64 --head OC20M"
        )

    def test_injected_adapter_owns_command_backend(self, mock_job_manager, tmp_path):
        manager = FeatureExecutionManager(
            backend="slurm",
            slurm_config={},
            env_setup="",
            dp_backend="pt",
            omp_threads=1,
            adapter=DeepMDAdapter.for_legacy_unchecked("tf"),
        )
        manager.submit_cli_job(
            data_path="data",
            output_dir=str(tmp_path / "output"),
            model_path="model.pt",
            head=None,
            sub_pools=[],
        )
        command = mock_job_manager.return_value.generate_script.call_args[0][0].command
        assert "dp --tf eval-desc" in command

    def test_submit_python_slurm_job(self, mock_job_manager, tmp_path):
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
            output_dir=str(tmp_path / "output"),
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
        
        manager.run_local_python_recursion(
            mock_generator,
            str(data_root),
            str(output_root)
        )

        # Check calls
        assert mock_generator.compute_descriptors.call_count == 2

        # Check save paths
        expected = [
            output_root / "sys1.npy",
            output_root / "group" / "sys2.npy"
        ]
        assert all(path.is_file() and path.stat().st_size > 0 for path in expected)

    @patch("dpeva.feature.managers.FeatureIOManager")
    def test_run_local_python_recursion_aggregates_leaf_failure(self, MockIO, tmp_path, caplog):
        data_root = tmp_path / "data"
        (data_root / "sys1").mkdir(parents=True)
        (data_root / "group" / "sys2").mkdir(parents=True)
        output_root = tmp_path / "output"
        failed_path = data_root / "group" / "sys2"

        io_instance = MockIO.return_value
        io_instance.is_leaf_system.side_effect = lambda path: str(path).endswith(("sys1", "sys2"))
        mock_generator = MagicMock()

        def compute_descriptors(path, output_mode):
            if str(path) == str(failed_path):
                raise RuntimeError("compute failed")
            return np.ones((2, 3))

        mock_generator.compute_descriptors.side_effect = compute_descriptors

        manager = FeatureExecutionManager("local", {}, "", "pt", 1)

        with pytest.raises(WorkflowError, match=str(failed_path)):
            manager.run_local_python_recursion(mock_generator, str(data_root), str(output_root))

        successful_output = output_root / "sys1.npy"
        assert successful_output.is_file()
        assert successful_output.stat().st_size > 0
        assert WORKFLOW_FINISHED_TAG not in caplog.text

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
