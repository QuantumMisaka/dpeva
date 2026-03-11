import os
import sys
import logging
import pandas as pd
from typing import Optional, Dict, List, Any, Union

from dpeva.config import AnalysisConfig
from dpeva.analysis.managers import AnalysisIOManager, UnifiedAnalysisManager
from dpeva.analysis.dataset import DatasetAnalysisManager
from dpeva.constants import WORKFLOW_FINISHED_TAG, LOG_FILE_ANALYSIS
from dpeva.utils.logs import setup_workflow_logger, close_workflow_logger

class AnalysisWorkflow:
    """
    Workflow for analyzing DeepMD test results.
    Refactored using UnifiedAnalysisManager.
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
        # Use UnifiedAnalysisManager
        self.analysis_manager = UnifiedAnalysisManager(ref_energies=self.config.ref_energies)
        self.dataset_analysis_manager = DatasetAnalysisManager()
        
    def run(self):
        output_dir = str(self.config.output_dir)
        self.logger.info(f"Starting Analysis in {output_dir}")
        setup_workflow_logger(
            logger_name="dpeva",
            work_dir=output_dir,
            filename=LOG_FILE_ANALYSIS,
            capture_stdout=False
        )
        try:
            if self.config.mode == "dataset":
                self._run_dataset_mode()
            else:
                self._run_model_mode(output_dir)
            self.logger.info("Analysis completed successfully.")
            self.logger.info(WORKFLOW_FINISHED_TAG)
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            raise
        finally:
            close_workflow_logger("dpeva", os.path.join(output_dir, LOG_FILE_ANALYSIS))

    def _run_dataset_mode(self):
        dataset_dir = str(self.config.dataset_dir) if self.config.dataset_dir else ""
        self.logger.info(f"Running dataset analysis mode for {dataset_dir}")
        self.dataset_analysis_manager.analyze(self.config.dataset_dir, self.config.output_dir)

    def _run_model_mode(self, output_dir: str):
        result_dir = str(self.config.result_dir) if self.config.result_dir else ""
        data, parser = self.io_manager.load_data(result_dir, self.config.type_map)
        atom_counts_list, atom_num_list = self._resolve_composition_info(parser)
        _, metrics, _, e_rel_pred, _ = self.analysis_manager.analyze_model(
            data=data,
            output_dir=output_dir,
            atom_counts_list=atom_counts_list,
            atom_num_list=atom_num_list
        )
        if metrics:
            self.io_manager.save_metrics(metrics)
            self.io_manager.save_summary_csv(metrics)
        if e_rel_pred is not None:
            stats_desc = pd.Series(e_rel_pred).describe().to_dict()
            self.io_manager.save_stats_desc(stats_desc, "cohesive_energy_pred_stats.json")

    def _resolve_composition_info(self, parser):
        if self.config.data_path:
            self.logger.info(f"Loading composition info from {self.config.data_path}...")
            return self.io_manager.load_composition_info(str(self.config.data_path))
        self.logger.info("Extracting composition info from filenames (Legacy Mode)...")
        return parser.get_composition_list()
