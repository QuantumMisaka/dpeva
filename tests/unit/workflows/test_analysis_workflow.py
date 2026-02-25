import os
import shutil
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path
from dpeva.workflows.analysis import AnalysisWorkflow
from dpeva.analysis.managers import AnalysisIOManager, UnifiedAnalysisManager

class TestAnalysisWorkflow:
    
    @pytest.fixture
    def config(self, tmp_path):
        return {
            "result_dir": str(tmp_path / "results"),
            "output_dir": str(tmp_path / "analysis"),
            "type_map": ["O", "H"],
            "ref_energies": {"O": -10.0, "H": -5.0}
        }

    @patch("dpeva.workflows.analysis.UnifiedAnalysisManager")
    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    def test_run_success(self, MockIOManager, MockManager, config):
        """Test successful execution of analysis workflow."""
        # Setup mocks
        mock_io = MockIOManager.return_value
        mock_manager = MockManager.return_value
        
        # Mock load_data return
        mock_data = {"energy": {"pred_e": np.array([1.0])}}
        mock_parser = MagicMock()
        mock_parser.get_composition_list.return_value = (None, None)
        mock_io.load_data.return_value = (mock_data, mock_parser)
        
        # Mock compute_metrics return
        mock_stats_export = {"energy": {}}
        mock_metrics = {"e_mae": 0.1}
        mock_stats_calc = MagicMock()
        mock_e_rel_pred = np.array([-0.1])
        mock_e_rel_true = np.array([-0.11])
        mock_manager.analyze_model.return_value = (
            mock_stats_export, 
            mock_metrics, 
            mock_stats_calc, 
            mock_e_rel_pred, 
            mock_e_rel_true
        )
        
        # Initialize Workflow
        workflow = AnalysisWorkflow(config)
        
        # Run
        workflow.run()
        
        # Verify Interactions
        mock_io.configure_logging.assert_called_once()
        mock_io.load_data.assert_called_with(config["result_dir"], config["type_map"])
        
        mock_manager.analyze_model.assert_called()
        
        mock_io.save_metrics.assert_called_with(mock_metrics)
        mock_io.save_summary_csv.assert_called_with(mock_metrics)
        mock_io.save_stats_desc.assert_called() # cohesive stats
        
        mock_io.close_logging.assert_called_once()

    @patch("dpeva.workflows.analysis.AnalysisIOManager")
    def test_run_failure(self, MockIOManager, config):
        """Test failure handling."""
        mock_io = MockIOManager.return_value
        mock_io.load_data.side_effect = Exception("Parsing failed")
        
        workflow = AnalysisWorkflow(config)
        
        with pytest.raises(Exception, match="Parsing failed"):
            workflow.run()
            
        mock_io.close_logging.assert_called_once()
