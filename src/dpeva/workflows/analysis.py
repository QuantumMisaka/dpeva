import os
import sys
import logging
import pandas as pd
from typing import Optional, Dict, List, Any, Union

from dpeva.config import AnalysisConfig
from dpeva.analysis.managers import AnalysisIOManager, AnalysisManager
from dpeva.constants import WORKFLOW_FINISHED_TAG

class AnalysisWorkflow:
    """
    Workflow for analyzing DeepMD test results.
    Refactored using DDD Managers.
    """
    
    def __init__(self, config: Union[AnalysisConfig, Dict[str, Any]]):
        """
        Initialize AnalysisWorkflow.
        
        Args:
            config: AnalysisConfig object or dictionary.
        """
        if isinstance(config, dict):
            self.config = AnalysisConfig(**config)
        else:
            self.config = config
            
        self.logger = logging.getLogger(__name__)
        
        # Initialize Managers
        self.io_manager = AnalysisIOManager(str(self.config.output_dir))
        self.analysis_manager = AnalysisManager(self.config)
        
    def run(self):
        """Execute the analysis workflow."""
        result_dir = str(self.config.result_dir)
        output_dir = str(self.config.output_dir)
        
        self.logger.info(f"Starting Analysis in {output_dir}")
        
        # Configure logging via IO Manager
        self.io_manager.configure_logging()
        
        try:
            # 1. Load Data
            # Note: DPTestResultParser expects result_dir to contain .e.out/.f.out files directly.
            # However, InferenceWorkflow creates subdirectories 0/task, 1/task etc.
            # AnalysisWorkflow seems designed to analyze ONE specific result folder, 
            # OR aggregate multiple if designed so. 
            # Based on current implementation of DPTestResultParser, it looks for ONE set of files.
            # But the integration test points result_dir to 'work_dir' which contains 0/, 1/ subdirs.
            # This is a mismatch. 
            # If AnalysisWorkflow is meant to analyze aggregated results, it should iterate.
            # If it's meant to analyze a single model's test result, config should point to it.
            # The integration test calls dpeva analysis with result_dir=work_dir.
            # But work_dir contains folders 0, 1, 2.
            # And InferenceWorkflow already does analysis for each model!
            # The new 'AnalysisWorkflow' seems redundant if it just repeats what InferenceWorkflow.analyze_results does,
            # UNLESS it is meant to aggregate?
            # Looking at AnalysisWorkflow code: it calls DPTestResultParser(result_dir).
            # DPTestResultParser looks for {head}.e_peratom.out in result_dir.
            # So AnalysisWorkflow expects to run on a directory WITH results.
            
            # Integration test failure: FileNotFoundError: Energy file not found: .../work/results.e_peratom.out
            # Because results are in work/0/test_val/results.e_peratom.out
            
            # FIX: We should probably analyze one of the sub-results in the integration test 
            # to verify AnalysisWorkflow works on A result.
            # Or AnalysisWorkflow should be smart enough to find results? 
            # Current implementation is simple: analyze ONE directory.
            
            data, parser = self.io_manager.load_data(result_dir, self.config.type_map)
            
            # 2. Compute Metrics
            metrics, stats_calc, e_rel_pred, e_rel_true = self.analysis_manager.compute_metrics(data, parser)
            
            # 3. Visualization
            self.analysis_manager.visualize(stats_calc, e_rel_pred, e_rel_true, output_dir)
            
            # 4. Save Results
            if metrics:
                self.io_manager.save_metrics(metrics)
                self.io_manager.save_summary_csv(metrics)
                
            if e_rel_pred is not None:
                stats_desc = pd.Series(e_rel_pred).describe().to_dict()
                self.io_manager.save_stats_desc(stats_desc, "cohesive_energy_pred_stats.json")

            self.logger.info("Analysis completed successfully.")
            self.logger.info(WORKFLOW_FINISHED_TAG)
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            raise e
        finally:
            self.io_manager.close_logging()
