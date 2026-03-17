import os
import logging
import pandas as pd
from typing import Dict, Any, Union

from dpeva.config import AnalysisConfig
from dpeva.analysis.managers import AnalysisIOManager, UnifiedAnalysisManager
from dpeva.analysis.dataset import DatasetAnalysisManager
from dpeva.constants import FILENAME_COHESIVE_ENERGY_PRED_STATS_JSON, WORKFLOW_FINISHED_TAG, LOG_FILE_ANALYSIS
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
        self.io_manager = AnalysisIOManager(str(self.config.output_dir))
        self.analysis_manager = UnifiedAnalysisManager(
            ref_energies=self.config.ref_energies,
            enable_cohesive_energy=self.config.enable_cohesive_energy,
            allow_ref_energy_lstsq_completion=self.config.allow_ref_energy_lstsq_completion
        )
        self.dataset_analysis_manager = DatasetAnalysisManager(
            ref_energies=self.config.ref_energies,
            enable_cohesive_energy=self.config.enable_cohesive_energy,
            allow_ref_energy_lstsq_completion=self.config.allow_ref_energy_lstsq_completion
        )
        
    def run(self):
        """Run analysis workflow in dataset or model_test mode."""
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
        """Run dataset-only statistics and plotting pipeline."""
        dataset_dir = str(self.config.dataset_dir) if self.config.dataset_dir else ""
        self.logger.info(f"Running dataset analysis mode for {dataset_dir}")
        self.dataset_analysis_manager.analyze(self.config.dataset_dir, self.config.output_dir)

    def _run_model_mode(self, output_dir: str):
        """Run model_test result analysis and export metrics/statistics."""
        result_dir = str(self.config.result_dir) if self.config.result_dir else ""
        data, parser = self.io_manager.load_data(
            result_dir,
            self.config.type_map,
            self.config.results_prefix,
        )
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
            self.io_manager.save_stats_desc(stats_desc, FILENAME_COHESIVE_ENERGY_PRED_STATS_JSON)

    def _resolve_composition_info(self, parser):
        """Resolve composition source from data_path first, then legacy parser fallback."""
        if self.config.data_path:
            self.logger.info(f"Loading composition info from {self.config.data_path}...")
            return self.io_manager.load_composition_info(str(self.config.data_path))
        self.logger.info("Extracting composition info from filenames (Legacy Mode)...")
        return parser.get_composition_list()
