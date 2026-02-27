
import os
import logging
import json
import pytest
from unittest.mock import MagicMock, patch
from dpeva.io.training import TrainingIOManager

class TestTrainingIOManager:
    
    @pytest.fixture
    def manager(self, tmp_path):
        return TrainingIOManager(str(tmp_path))

    def test_create_task_dir(self, manager, tmp_path):
        """Test task directory creation."""
        task_dir = manager.create_task_dir(0)
        
        assert os.path.exists(task_dir)
        assert task_dir == str(tmp_path / "0")

    def test_save_task_config(self, manager, tmp_path):
        """Test saving task config."""
        task_dir = manager.create_task_dir(0)
        config = {"foo": "bar"}
        
        manager.save_task_config(task_dir, config)
        
        config_file = os.path.join(task_dir, "input.json")
        assert os.path.exists(config_file)
        
        with open(config_file, "r") as f:
            loaded = json.load(f)
            assert loaded == config

    def test_copy_base_model(self, manager, tmp_path):
        """Test copying base model."""
        task_dir = manager.create_task_dir(0)
        
        # Create dummy model
        model_path = tmp_path / "model.ckpt"
        model_path.touch()
        
        copied_name = manager.copy_base_model(str(model_path), task_dir)
        
        assert copied_name == "model.ckpt"
        assert os.path.exists(os.path.join(task_dir, "model.ckpt"))

    def test_copy_base_model_not_found(self, manager, tmp_path):
        """Test copying non-existent model."""
        task_dir = manager.create_task_dir(0)
        
        with pytest.raises(FileNotFoundError):
            manager.copy_base_model("non_existent.ckpt", task_dir)
