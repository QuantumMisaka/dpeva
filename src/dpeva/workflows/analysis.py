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
