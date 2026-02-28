
import os
import pytest
from unittest.mock import MagicMock, patch
from dpeva.training.managers import TrainingConfigManager, TrainingExecutionManager

class TestTrainingConfigManager:

    @pytest.fixture
    def base_config(self):
        return {
            "model": {
                "type_map": ["Fe"],
                "descriptor": {"type": "se_e2_a"},
                "fitting_net": {"neuron": [240, 240, 240]}
            },
            "training": {
                "training_data": {
                    "systems": "data/system1",
                    "batch_size": 1
                },
                "numb_steps": 100
            }
        }

    @pytest.fixture
    def config_manager(self, base_config, tmp_path):
        config_path = str(tmp_path / "config.json")
        return TrainingConfigManager(base_config, config_path)

    def test_resolve_data_path_absolute(self, config_manager, tmp_path):
        """Test if relative path is resolved to absolute path based on config location."""
        # Setup
        base_dir = tmp_path
        sys_dir = base_dir / "data" / "system1"
        sys_dir.mkdir(parents=True)
        
        config = config_manager.base_config.copy()
        
        # Test
        config_manager.resolve_data_path(config, 0)
        
        # Verify
        resolved_path = config["training"]["training_data"]["systems"]
        assert os.path.isabs(resolved_path)
        assert resolved_path == str(sys_dir)

    def test_resolve_data_path_container_expansion(self, config_manager, tmp_path):
        """Test auto-expansion of container directory."""
        # Setup container structure
        container = tmp_path / "data" / "container"
        container.mkdir(parents=True)
        (container / "sys1").mkdir()
        (container / "sys2").mkdir()
        (container / "not_a_sys.txt").touch()
        
        config = config_manager.base_config.copy()
        config["training"]["training_data"]["systems"] = str(container)
        
        # Test
        config_manager.resolve_data_path(config, 0)
        
        # Verify
        systems = config["training"]["training_data"]["systems"]
        assert isinstance(systems, list)
        assert len(systems) == 2
        assert str(container / "sys1") in systems
        assert str(container / "sys2") in systems

    def test_resolve_data_path_override(self, config_manager):
        """Test override mechanism."""
        config = config_manager.base_config.copy()
        override = "/abs/path/to/override"
        
        config_manager.resolve_data_path(config, 0, override_path=override)
        
        assert config["training"]["training_data"]["systems"] == override

    def test_prepare_task_configs(self, config_manager):
        """Test task config generation."""
        num_models = 2
        seeds = [123, 456]
        train_seeds = [111, 222]
        heads = ["head1", "head2"]
        
        configs = config_manager.prepare_task_configs(num_models, seeds, train_seeds, heads)
        
        assert len(configs) == 2
        assert configs[0]["model"]["fitting_net"]["seed"] == 123
        assert configs[0]["training"]["seed"] == 111
        assert configs[0]["model"]["finetune_head"] == "head1"
        
        assert configs[1]["model"]["fitting_net"]["seed"] == 456
        assert configs[1]["training"]["seed"] == 222
        assert configs[1]["model"]["finetune_head"] == "head2"


class TestTrainingExecutionManager:
    
    @pytest.fixture
    def manager(self):
        return TrainingExecutionManager(
            backend="local",
            slurm_config={},
            env_setup="export FOO=bar",
            dp_backend="pt",
            template_path=None
        )

    @patch("dpeva.submission.manager.JobManager.generate_script")
    def test_generate_script_single_gpu(self, mock_gen, manager, tmp_path):
        """Test script generation for single GPU/CPU."""
        task_dir = str(tmp_path)
        script_path = manager.generate_script(0, task_dir, "base.ckpt", omp_threads=4)
        
        assert script_path.endswith("train.sh")
        assert mock_gen.called
        
        job_config = mock_gen.call_args[0][0]
        assert "export OMP_NUM_THREADS=4" in job_config.command
        assert "dp --pt train" in job_config.command
        assert "base.ckpt" in job_config.command

    @patch("dpeva.submission.manager.JobManager.generate_script")
    def test_generate_script_multi_gpu(self, mock_gen, manager, tmp_path):
        """Test script generation for multi-GPU."""
        manager.slurm_config = {"gpus_per_node": 4}
        
        task_dir = str(tmp_path)
        manager.generate_script(0, task_dir, "base.ckpt", omp_threads=4)
        
        job_config = mock_gen.call_args[0][0]
        assert "torchrun" in job_config.command
        assert "--nproc_per_node" in job_config.command

    @patch("dpeva.submission.manager.JobManager")
    def test_submit_jobs_slurm_parallel(self, mock_job_cls, manager, tmp_path):
        """
        Verify that Slurm backend submits jobs in parallel and sets up correct dp train command.
        """
        # Setup manager with slurm backend
        manager.backend = "slurm"
        # Since we patched JobManager class, we need to set manager.job_manager to the instance mock
        manager.job_manager = mock_job_cls.return_value
        
        scripts = ["script1.slurm", "script2.slurm"]
        task_dirs = ["dir1", "dir2"]
        
        manager.submit_jobs(scripts, task_dirs)
        
        # Verify submit was called 2 times (loop)
        assert manager.job_manager.submit.call_count == 2
        calls = manager.job_manager.submit.call_args_list
        assert calls[0][0][0] == "script1.slurm"
        assert calls[1][0][0] == "script2.slurm"

    def test_generate_script_correct_dp_train_command(self, manager, tmp_path):
        """
        Verify that the generated script contains the correct 'dp train' command.
        """
        task_dir = str(tmp_path)
        with patch("dpeva.submission.manager.JobManager.generate_script") as mock_gen:
            manager.generate_script(0, task_dir, "base.ckpt", omp_threads=4)
            
            job_config = mock_gen.call_args[0][0]
            # Verify command content
            assert "dp --pt train" in job_config.command
            assert "input.json" in job_config.command
            assert "--finetune base.ckpt" in job_config.command
            assert "--skip-neighbor-stat" not in job_config.command # Single GPU shouldn't skip neighbor stat by default unless specified or multi-gpu
            # Wait, TrainingExecutionManager logic:
            # Single GPU: skip_neighbor_stat=False (default in DPCommandBuilder.train)
            # Multi GPU: skip_neighbor_stat=True

    def test_generate_script_correct_dp_train_command_multi_gpu(self, manager, tmp_path):
        """
        Verify that the generated script contains the correct 'dp train' command for multi-GPU.
        """
        manager.slurm_config = {"gpus_per_node": 4}
        task_dir = str(tmp_path)
        with patch("dpeva.submission.manager.JobManager.generate_script") as mock_gen:
            manager.generate_script(0, task_dir, "base.ckpt", omp_threads=4)
            
            job_config = mock_gen.call_args[0][0]
            # Verify command content
            assert "torchrun" in job_config.command
            assert "dp --pt train" in job_config.command
            assert "--skip-neighbor-stat" in job_config.command

    @patch("dpeva.training.managers.multiprocessing.Process")
    def test_submit_jobs_local_parallel_multiprocessing(self, mock_proc, manager, tmp_path):
        """
        Verify that Local backend uses multiprocessing for parallel execution.
        """
        # Setup manager with local backend
        manager.backend = "local"
        
        scripts = ["script1.sh", "script2.sh"]
        task_dirs = ["dir1", "dir2"]
        
        manager.submit_jobs(scripts, task_dirs, blocking=True)
        
        # Verify Process was instantiated 2 times
        assert mock_proc.call_count == 2
        
        # Verify start() was called 2 times
        assert mock_proc.return_value.start.call_count == 2
        
        # Verify join() was called 2 times (since blocking=True)
        assert mock_proc.return_value.join.call_count == 2
