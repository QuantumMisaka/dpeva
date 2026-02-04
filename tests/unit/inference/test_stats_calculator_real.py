import pytest
import numpy as np
import os
import shutil
import json
from unittest.mock import patch
from dpeva.inference.stats import StatsCalculator
from dpeva.io.dataproc import DPTestResultParser
from dpeva.workflows.infer import InferenceWorkflow

def test_analyze_real_results(mock_dptest_output_dir, tmp_path):
    """
    Test full analysis flow using real output files.
    """
    # Create a mock work_dir that mimics the structure InferenceWorkflow expects
    work_dir = tmp_path / "0" / "test_task"
    work_dir.mkdir(parents=True)
    
    # Copy mock results to work_dir
    # Note: DPTestResultParser looks for {head}.e.out etc.
    # mock_dptest_output_dir has results.e.out
    for fname in os.listdir(mock_dptest_output_dir):
        shutil.copy(mock_dptest_output_dir / fname, work_dir / fname)
        
    # Test Parser directly first
    parser = DPTestResultParser(str(work_dir), head="results")
    data = parser.parse()
    
    assert data["energy"] is not None
    # Check field names for structured array
    assert "pred_e" in data["energy"].dtype.names
    assert "data_e" in data["energy"].dtype.names
    assert len(data["energy"]["pred_e"]) > 0
    
    # Test StatsCalculator
    # We construct it manually to verify logic
    f_pred = None
    f_true = None
    if data["force"] is not None:
        f_pred = np.column_stack((
            data["force"]["pred_fx"], 
            data["force"]["pred_fy"], 
            data["force"]["pred_fz"]
        )).flatten()
        
        if data["has_ground_truth"]:
             f_true = np.column_stack((
                data["force"]["data_fx"], 
                data["force"]["data_fy"], 
                data["force"]["data_fz"]
            )).flatten()
    
    calc = StatsCalculator(
        energy_per_atom=data["energy"]["pred_e"],
        force_flat=f_pred,
        energy_true=data["energy"]["data_e"] if data["has_ground_truth"] else None,
        force_true=f_true,
    )
    
    metrics = calc.compute_metrics()
    
    # Assertions
    assert "e_rmse" in metrics
    assert metrics["e_rmse"] >= 0
    if f_pred is not None and f_true is not None:
        assert "f_rmse" in metrics
        assert metrics["f_rmse"] >= 0

def test_inference_workflow_analyze(mock_dptest_output_dir, tmp_path):
    """
    Test InferenceWorkflow.analyze_results method.
    """
    # Setup structure: basedir/0/task_name
    work_dir = tmp_path / "0" / "test_val"
    work_dir.mkdir(parents=True)
    
    for fname in os.listdir(mock_dptest_output_dir):
        shutil.copy(mock_dptest_output_dir / fname, work_dir / fname)
        
    config = {
        "work_dir": str(tmp_path),
        "data_path": str(tmp_path / "dummy_data"),
        "task_name": "test_val",
        "head": "results" # Matches filename prefix results.*.out
    }
    
    workflow = InferenceWorkflow(config)
    # Mock models_paths to include this directory
    workflow.models_paths = [str(tmp_path / "0" / "model.ckpt.pt")] 
    
    # Mock matplotlib in visualizer
    with patch("dpeva.inference.visualizer.plt") as mock_plt:
         workflow.analyze_results()
         
    # Check if analysis output exists
    analysis_dir = work_dir / "analysis"
    assert analysis_dir.exists()
    assert (analysis_dir / "statistics.json").exists()
    
    # Verify statistics content
    with open(analysis_dir / "statistics.json", "r") as f:
        stats = json.load(f)
        assert "energy" in stats
