
import os
import shutil
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path
from dpeva.workflows.analysis import AnalysisWorkflow

class TestAnalysisWorkflow:
    
    @pytest.fixture
    def config(self, tmp_path):
        return {
            "result_dir": str(tmp_path / "results"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "ref_energies": {"O": -10.0, "H": -5.0}
        }

    @patch("dpeva.workflows.analysis.DPTestResultParser")
    @patch("dpeva.workflows.analysis.StatsCalculator")
    @patch("dpeva.workflows.analysis.InferenceVisualizer")
    def test_run_success(self, mock_vis, mock_calc, mock_parser, config, tmp_path):
        """Test successful execution of analysis workflow."""
        # Setup mocks
        mock_parser_instance = mock_parser.return_value
        mock_parser_instance.parse.return_value = {
            "energy": {"pred_e": np.array([1.0]), "data_e": np.array([1.1])},
            "force": {"pred_fx": np.array([0.1]), "pred_fy": np.array([0.1]), "pred_fz": np.array([0.1]),
                      "data_fx": np.array([0.1]), "data_fy": np.array([0.1]), "data_fz": np.array([0.1])},
            "virial": None,
            "has_ground_truth": True
        }
        mock_parser_instance.get_composition_list.return_value = ([{"O": 1}], [1])
        
        mock_calc_instance = mock_calc.return_value
        mock_calc_instance.compute_metrics.return_value = {"e_mae": 0.1, "e_rmse": 0.1}
        mock_calc_instance.compute_relative_energy.return_value = np.array([-0.1])
        mock_calc_instance.e_pred = np.array([1.0])
        mock_calc_instance.e_true = np.array([1.1])
        mock_calc_instance.f_pred = np.array([0.1, 0.1, 0.1])
        mock_calc_instance.f_true = np.array([0.1, 0.1, 0.1])
        mock_calc_instance.v_pred = None
        mock_calc_instance.v_true = None
        
        # Initialize Workflow
        workflow = AnalysisWorkflow(config)
        
        # Run
        workflow.run()
        
        # Verify
        mock_parser.assert_called_with(result_dir=config["result_dir"], head="results", type_map=config["type_map"])
        mock_calc.assert_called()
        mock_vis.assert_called_with(config["output_dir"])
        
        # Verify Visualization Calls
        mock_vis_instance = mock_vis.return_value
        mock_vis_instance.plot_distribution.assert_called()
        mock_vis_instance.plot_parity.assert_called()
        
        # Verify File Creation (Metrics)
        output_dir = Path(config["output_dir"])
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "metrics_summary.csv").exists()
        assert (output_dir / "cohesive_energy_pred_stats.json").exists()
        assert (output_dir / "analysis.log").exists()

    @patch("dpeva.workflows.analysis.DPTestResultParser")
    def test_run_failure(self, mock_parser, config):
        """Test failure handling."""
        mock_parser.side_effect = Exception("Parsing failed")
        
        workflow = AnalysisWorkflow(config)
        
        with pytest.raises(Exception, match="Parsing failed"):
            workflow.run()
