import pytest
import json
import os
from unittest.mock import MagicMock, patch
from dpeva.training.trainer import ParallelTrainer

def test_prepare_input_from_real_file(real_config_loader, tmp_path, mock_job_manager_train):
    """
    Test ParallelTrainer's ability to modify input parameters using a real input.json template.
    """
    # 1. Setup real input file template
    input_template_path = tmp_path / "input_template.json"
    
    # Mimic a real DeepMD input structure
    input_content = {
        "model": {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [46, 92]
            },
            "fitting_net": {
                "neuron": [240, 240, 240]
            }
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 5000
        },
        "loss": {
            "start_pref_e": 0.02,
            "limit_pref_e": 1
        },
        "training": {
            "training_data": {
                "systems": ["data/system1", "data/system2"],
                "batch_size": "auto"
            },
            "validation_data": {
                "systems": ["data/system1"],
                "batch_size": 1
            },
            "numb_steps": 100000,
            "seed": 12345,
            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "save_freq": 1000
        }
    }
    
    with open(input_template_path, "w") as f:
        json.dump(input_content, f)
        
    # 2. Initialize Trainer
    config = {
        "work_dir": str(tmp_path),
        "input_json_path": str(input_template_path.resolve()), # Absolute path
        "num_models": 2,
        "base_model_path": "model.pt", 
        "training_data_path": str(tmp_path / "data"),
        "backend": "slurm",
        "seeds": [111, 222]
    }
    
    # Mock data existence
    (tmp_path / "data").mkdir()
    
    trainer = ParallelTrainer(
        base_config_path=config["input_json_path"],
        work_dir=config["work_dir"],
        num_models=config["num_models"],
        backend=config["backend"],
        training_data_path=config.get("training_data_path")
    )
    
    # 3. Test prepare_configs and setup_workdirs
    seeds = [111, 222]
    training_seeds = [111, 222]
    heads = ["head1", "head2"]
    
    trainer.prepare_configs(seeds, training_seeds, heads)
    
    # Setup dummy base models
    base_models = [tmp_path / "model1.pt", tmp_path / "model2.pt"]
    for p in base_models:
        p.touch()
        
    trainer.setup_workdirs([str(p) for p in base_models])
    
    # Check 0/input.json
    p0 = tmp_path / "0" / "input.json"
    assert p0.exists()
    
    with open(p0) as f:
        new_input = json.load(f)
        # Verify seed was updated
        assert new_input["training"]["seed"] == 111
        # Verify structure preserved
        assert new_input["model"]["descriptor"]["type"] == "se_e2_a"

def test_submit_script_content(tmp_path, mock_job_manager_train):
    """
    Verify Slurm script generation.
    """
    # Create dummy input.json
    input_path = tmp_path / "input.json"
    with open(input_path, "w") as f:
        json.dump({
            "model": {"fitting_net": {}},
            "training": {"seed": 0}
        }, f)
        
    config = {
        "work_dir": str(tmp_path),
        "input_json_path": str(input_path.resolve()),
        "num_models": 1,
        "backend": "slurm",
        "slurm_config": {
            "partition": "test_part",
            "qos": "test_qos",
            "env_setup": ["export TEST=1"]
        }
    }
    
    # Let's create dirs manually or let setup_workdirs do it
    # We need to run setup_workdirs to populate script_paths
    
    trainer = ParallelTrainer(
        base_config_path=config["input_json_path"],
        work_dir=config["work_dir"],
        num_models=config["num_models"],
        backend=config["backend"],
        slurm_config=config.get("slurm_config")
    )
    
    # Prepare
    trainer.prepare_configs([1], [1], ["head"])
    
    base_model = tmp_path / "model.pt"
    base_model.touch()
    
    trainer.setup_workdirs([str(base_model)])
    
    # Mock JobManager.generate_script and submit
    # mock_job_manager_train is already active
    
    trainer.train(blocking=False)
    
    # Verify submit called
    assert mock_job_manager_train.submit.called
