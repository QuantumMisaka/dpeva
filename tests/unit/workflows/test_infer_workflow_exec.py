import pytest
import os
import hashlib
import json
from unittest.mock import patch
from dpeva.workflows.infer import InferenceWorkflow

def create_mock_models(base_dir, num_models=3):
    """Helper to create dummy model files."""
    for i in range(num_models):
        model_dir = base_dir / str(i)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.ckpt.pt").touch()

def test_init_model_discovery(tmp_path):
    """Verify workflow finds models in 0/, 1/ subdirectories."""
    create_mock_models(tmp_path, num_models=3)
    
    config = {
        "work_dir": str(tmp_path),
        "data_path": str(tmp_path / "data"), # Dummy
        "task_name": "test_task"
    }
    
    workflow = InferenceWorkflow(config)
    assert len(workflow.models_paths) == 3
    # Check if paths are correct (order might vary so check set or existence)
    expected_paths = {str(tmp_path / str(i) / "model.ckpt.pt") for i in range(3)}
    assert set(workflow.models_paths) == expected_paths

def test_run_command_generation(tmp_path, mock_job_manager):
    """Verify dp test command construction."""
    create_mock_models(tmp_path, num_models=1)
    data_path = tmp_path / "test_data"
    data_path.mkdir()
    
    config = {
        "work_dir": str(tmp_path),
        "data_path": str(data_path),
        "task_name": "test_val",
        "model_head": "MyHead",
        "submission": {"backend": "slurm"}
    }
    
    workflow = InferenceWorkflow(config)
    mock_job_manager.submit.return_value = "Submitted batch job 109"
    mock_job_manager.parse_sbatch_job_id.return_value = "109"
    
    # Simulate running inside slurm to skip self-submission
    with patch.dict(os.environ, {"DPEVA_INTERNAL_BACKEND": "slurm"}):
        workflow.run()
    
    # Verify JobManager.submit was called
    assert mock_job_manager.generate_script.called
    assert mock_job_manager.submit.called
    
    # Inspect the JobConfig passed to generate_script
    call_args = mock_job_manager.generate_script.call_args
    job_config = call_args[0][0] # First arg is job_config
    
    # Command path checks
    assert str(data_path) in job_config.command
    assert str(tmp_path / "0" / "model.ckpt.pt") in job_config.command
    assert "--head MyHead" in job_config.command
    
    assert "-d results" in job_config.command

def test_no_models_found(tmp_path, caplog):
    """Verify error logging when no models are found."""
    import logging
    caplog.set_level(logging.ERROR)

    data_path = tmp_path / "data"
    data_path.mkdir(exist_ok=True)
    
    config = {
        "work_dir": str(tmp_path),
        "data_path": str(data_path)
    }
    
    workflow = InferenceWorkflow(config)
    
    from dpeva.utils.exceptions import WorkflowError

    # Use patch to verify logging call, avoiding caplog issues if propagation is disabled
    with patch.object(workflow.logger, 'error') as mock_error:
        with pytest.raises(WorkflowError, match="No models provided"):
            workflow.run()
        mock_error.assert_called_with("No models provided for inference.")

def test_auto_analysis_runs_only_for_local(tmp_path, mock_job_manager):
    create_mock_models(tmp_path, num_models=1)
    data_path = tmp_path / "test_data"
    data_path.mkdir()

    config = {
        "work_dir": str(tmp_path),
        "data_path": str(data_path),
        "submission": {"backend": "local"},
        "auto_analysis": True,
    }
    workflow = InferenceWorkflow(config)

    def submit_with_result(script, working_dir="."):
        os.makedirs(working_dir, exist_ok=True)
        with open(os.path.join(working_dir, "results.e.out"), "w", encoding="utf-8") as handle:
            handle.write("prediction\n")
        return ""

    mock_job_manager.submit.side_effect = submit_with_result
    with patch.object(workflow, "analyze_results") as mock_analyze:
        workflow.run()
        mock_analyze.assert_called_once()

def test_auto_analysis_ignored_for_non_local(tmp_path, mock_job_manager):
    create_mock_models(tmp_path, num_models=1)
    data_path = tmp_path / "test_data"
    data_path.mkdir()

    config = {
        "work_dir": str(tmp_path),
        "data_path": str(data_path),
        "submission": {"backend": "slurm"},
        "auto_analysis": True,
    }
    workflow = InferenceWorkflow(config)
    mock_job_manager.submit.return_value = "Submitted batch job 123"
    mock_job_manager.parse_sbatch_job_id.return_value = "123"
    with patch.object(workflow, "analyze_results") as mock_analyze, patch.object(workflow.logger, "warning") as mock_warning:
        with patch.dict(os.environ, {"DPEVA_INTERNAL_BACKEND": "slurm"}):
            workflow.run()
        mock_analyze.assert_not_called()
        mock_warning.assert_called_with("auto_analysis=true is ignored when backend is not local.")


def _write_model_ref(ref_path, model_path, *, backend="pt-expt", operations=None):
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(
        json.dumps(
            {
                "kind": "checkpoint",
                "family": "DPA4C",
                "backend": backend,
                "path": os.path.relpath(model_path, ref_path.parent),
                "checksum": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "supported_operations": operations or ["test"],
            }
        ),
        encoding="utf-8",
    )


def test_explicit_model_refs_suppress_legacy_scan_and_resolve_relative_paths(tmp_path):
    legacy = tmp_path / "legacy" / "0" / "model.ckpt.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    explicit = tmp_path / "models" / "explicit.pt"
    explicit.parent.mkdir()
    explicit.write_bytes(b"explicit")
    ref_path = tmp_path / "refs" / "model.json"
    _write_model_ref(ref_path, explicit)
    config_path = tmp_path / "configs" / "infer.json"
    config_path.parent.mkdir()
    config_path.write_text("{}", encoding="utf-8")

    workflow = InferenceWorkflow(
        {
            "work_dir": str(legacy.parent.parent),
            "data_path": str(tmp_path / "data"),
            "dp_backend": "pt-expt",
            "model_ref_paths": [os.path.relpath(ref_path, config_path.parent)],
        },
        config_path=str(config_path),
    )

    assert workflow.models_paths == [str(explicit)]
    assert workflow.model_refs[0].path == str(explicit)


def test_explicit_backend_mismatch_fails_before_execution(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    ref = tmp_path / "model.json"
    _write_model_ref(ref, model, backend="pt")

    with pytest.raises(ValueError, match="backend mismatch"):
        InferenceWorkflow(
            {
                "work_dir": str(tmp_path),
                "data_path": str(tmp_path / "data"),
                "dp_backend": "pt-expt",
                "model_ref_paths": [ref],
            }
        )


def test_legacy_bridge_emits_one_migration_warning(tmp_path, caplog):
    import logging

    loggers = [logging.getLogger(name) for name in ("dpeva", "dpeva.workflows.infer")]
    original_state = [
        (logger, logger.propagate, logger.handlers[:], logger.level)
        for logger in loggers
    ]
    try:
        # Other workflow tests configure this module logger for file capture;
        # this constructor-only assertion must be independent of test order.
        for logger in loggers:
            logger.handlers.clear()
            logger.propagate = True
            logger.setLevel(logging.WARNING)
        caplog.set_level(logging.WARNING)
        InferenceWorkflow(
            {
                "work_dir": str(tmp_path),
                "data_path": str(tmp_path / "data"),
            }
        )

        messages = [
            record.message
            for record in caplog.records
            if "using legacy numeric-directory model discovery" in record.message
        ]
        assert len(messages) == 1
    finally:
        for logger, propagate, handlers, level in original_state:
            logger.handlers[:] = handlers
            logger.propagate = propagate
            logger.setLevel(level)
