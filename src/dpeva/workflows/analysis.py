import os
import sys
import logging
import time
import pandas as pd
from typing import Dict, Any, Union, List, Optional

from dpeva.config import AnalysisConfig
from dpeva.analysis.managers import (
    AnalysisIOManager,
    UnifiedAnalysisManager,
    describe_analysis_plot_level,
)
from dpeva.analysis.dataset import DatasetAnalysisManager
from dpeva.constants import FILENAME_COHESIVE_ENERGY_PRED_STATS_JSON, WORKFLOW_FINISHED_TAG, LOG_FILE_ANALYSIS
from dpeva.submission.manager import JobManager
from dpeva.submission.templates import JobConfig
from dpeva.utils.logs import setup_workflow_logger, close_workflow_logger

class AnalysisWorkflow:
    """
    Workflow for analyzing DeepMD test results.
    Refactored using UnifiedAnalysisManager.
    """
    
    def __init__(self, config: Union[AnalysisConfig, Dict[str, Any]], config_path: str = None):
        """
        Initialize AnalysisWorkflow.
        
        Args:
            config: AnalysisConfig object or dictionary.
        """
        if isinstance(config, dict):
            if config_path and "config_path" not in config:
                config["config_path"] = config_path
            self.config = AnalysisConfig(**config)
        else:
            self.config = config
            if config_path and self.config.config_path is None:
                try:
                    self.config.config_path = config_path
                except Exception:
                    pass

        self.config_path = str(self.config.config_path) if self.config.config_path else None
        env_backend = os.environ.get("DPEVA_INTERNAL_BACKEND")
        if env_backend:
            self.backend = env_backend
        else:
            self.backend = self.config.submission.backend
        self.slurm_config = self.config.submission.slurm_config
            
        self.logger = logging.getLogger(__name__)
        self.io_manager = AnalysisIOManager(str(self.config.output_dir))
        self.analysis_manager = UnifiedAnalysisManager(
            ref_energies=self.config.ref_energies,
            enable_cohesive_energy=self.config.enable_cohesive_energy,
            allow_ref_energy_lstsq_completion=self.config.allow_ref_energy_lstsq_completion,
            slow_plot_threshold_seconds=self.config.slow_plot_threshold_seconds,
            enhanced_parity_renderer=self.config.enhanced_parity_renderer,
        )
        self.dataset_analysis_manager = DatasetAnalysisManager(
            ref_energies=self.config.ref_energies,
            enable_cohesive_energy=self.config.enable_cohesive_energy,
            allow_ref_energy_lstsq_completion=self.config.allow_ref_energy_lstsq_completion
        )
        
    def run(self):
        """Run analysis workflow in dataset or model_test mode."""
        if self.backend == "slurm":
            self._submit_to_slurm()
            return
        output_dir = str(self.config.output_dir)
        self.logger.info(f"Starting Analysis in {output_dir}")
        setup_workflow_logger(
            logger_name="dpeva",
            work_dir=output_dir,
            filename=LOG_FILE_ANALYSIS,
            capture_stdout=False
        )
        workflow_start = time.perf_counter()
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
            total_elapsed = time.perf_counter() - workflow_start
            self.logger.info(f"Analysis total elapsed time: {total_elapsed:.3f}s")
            close_workflow_logger("dpeva", os.path.join(output_dir, LOG_FILE_ANALYSIS))

    def _run_dataset_mode(self):
        """Run dataset-only statistics and plotting pipeline."""
        dataset_dir = str(self.config.dataset_dir) if self.config.dataset_dir else ""
        self.logger.info(f"Running dataset analysis mode for {dataset_dir}")
        self.logger.info(
            f"Plot level '{self.config.plot_level}': "
            f"{describe_analysis_plot_level(self.config.plot_level, mode='dataset')}"
        )
        self.dataset_analysis_manager.analyze(
            self.config.dataset_dir,
            self.config.output_dir,
            plot_level=self.config.plot_level,
        )

    def _run_model_mode(self, output_dir: str):
        """Run model_test result analysis and export metrics/statistics."""
        result_dir = str(self.config.result_dir) if self.config.result_dir else ""
        parse_start = time.perf_counter()
        self.logger.info(f"Stage[parse] Start parsing results from {result_dir}")
        data, parser = self.io_manager.load_data(
            result_dir,
            self.config.type_map,
            self.config.results_prefix,
        )
        parse_elapsed = time.perf_counter() - parse_start
        energy_data = data.get("energy") if isinstance(data, dict) else None
        energy_field_names = getattr(getattr(energy_data, "dtype", None), "names", None)
        frame_count = len(energy_data["pred_e"]) if energy_field_names and "pred_e" in energy_field_names else 0
        self.logger.info(f"Stage[parse] Finished in {parse_elapsed:.3f}s with {frame_count} frames")

        composition_start = time.perf_counter()
        self.logger.info("Stage[composition] Start loading composition information")
        atom_counts_list, atom_num_list = self._resolve_composition_info(parser)
        composition_elapsed = time.perf_counter() - composition_start
        composition_frames = len(atom_num_list) if atom_num_list is not None else 0
        self.logger.info(
            f"Stage[composition] Finished in {composition_elapsed:.3f}s with {composition_frames} composition frames"
        )

        stats_plot_start = time.perf_counter()
        self.logger.info("Stage[statistics+plot] Start statistics calculation and plotting")
        self.logger.info(
            f"Plot level '{self.config.plot_level}': "
            f"{describe_analysis_plot_level(self.config.plot_level, mode='model_test')}"
        )
        _, metrics, _, e_rel_pred, _ = self.analysis_manager.analyze_model(
            data=data,
            output_dir=output_dir,
            atom_counts_list=atom_counts_list,
            atom_num_list=atom_num_list,
            plot_level=self.config.plot_level,
        )
        stats_plot_elapsed = time.perf_counter() - stats_plot_start
        self.logger.info(f"Stage[statistics+plot] Finished in {stats_plot_elapsed:.3f}s")
        if metrics:
            self.io_manager.save_metrics(metrics)
            self.io_manager.save_summary_csv(metrics)
        if e_rel_pred is not None:
            stats_desc = pd.Series(e_rel_pred).describe().to_dict()
            self.io_manager.save_stats_desc(stats_desc, FILENAME_COHESIVE_ENERGY_PRED_STATS_JSON)

    def _has_valid_composition_info(
        self,
        atom_counts_list: Optional[List[Dict[str, int]]],
        atom_num_list: Optional[List[int]],
    ) -> bool:
        if not atom_counts_list or not atom_num_list:
            return False
        if len(atom_counts_list) != len(atom_num_list):
            return False
        return all(atom_num > 0 for atom_num in atom_num_list)

    def _composition_lists_match(
        self,
        left_counts: List[Dict[str, int]],
        left_nums: List[int],
        right_counts: List[Dict[str, int]],
        right_nums: List[int],
    ) -> bool:
        if len(left_counts) != len(right_counts) or len(left_nums) != len(right_nums):
            return False
        if left_nums != right_nums:
            return False
        return all(dict(left) == dict(right) for left, right in zip(left_counts, right_counts))

    def _resolve_composition_info(self, parser):
        """Resolve composition info with parser order priority for model_test outputs."""
        parser_counts, parser_nums = parser.get_composition_list()
        parser_valid = self._has_valid_composition_info(parser_counts, parser_nums)

        if self.config.data_path:
            self.logger.info(f"Stage[composition] Source data_path: {self.config.data_path}")
            data_counts, data_nums = self.io_manager.load_composition_info(str(self.config.data_path))
            data_valid = self._has_valid_composition_info(data_counts, data_nums)

            if parser_valid:
                if data_valid:
                    if self._composition_lists_match(parser_counts, parser_nums, data_counts, data_nums):
                        self.logger.info(
                            "Stage[composition] data_path composition validated against parser order."
                        )
                    else:
                        self.logger.warning(
                            "Stage[composition] data_path composition order does not match parser output order. "
                            "Using parser-aligned composition for cohesive energy."
                        )
                else:
                    self.logger.warning(
                        "Stage[composition] data_path composition is unavailable or invalid. "
                        "Using parser-aligned composition for cohesive energy."
                    )
                return parser_counts, parser_nums

            if data_valid:
                self.logger.info(
                    "Stage[composition] Parser composition is unavailable; falling back to data_path composition."
                )
                return data_counts, data_nums

            self.logger.warning(
                "Stage[composition] No valid composition info found from parser or data_path."
            )
            return parser_counts, parser_nums

        self.logger.info("Stage[composition] Source parser fallback: filenames (legacy mode)")
        return parser_counts, parser_nums

    def _submit_to_slurm(self):
        if not self.config_path:
            raise ValueError("Config path required for Slurm.")
        output_dir = os.path.abspath(str(self.config.output_dir))
        os.makedirs(output_dir, exist_ok=True)
        cmd = f"{sys.executable} -m dpeva.cli analysis {os.path.abspath(self.config_path)}"
        user_env_setup = self.config.submission.env_setup
        if isinstance(user_env_setup, list):
            user_env_setup = "\n".join(user_env_setup)
        elif user_env_setup is None:
            user_env_setup = ""
        final_env_setup = f"{user_env_setup}\nexport DPEVA_INTERNAL_BACKEND=local"
        job_conf = JobConfig(
            command=cmd,
            job_name=self.slurm_config.get("job_name", "dpeva_analysis"),
            partition=self.slurm_config.get("partition", "CPU-MISC"),
            qos=self.slurm_config.get("qos"),
            ntasks=self.slurm_config.get("ntasks", 1),
            output_log=os.path.join(output_dir, "analysis_slurm.out"),
            error_log=os.path.join(output_dir, "analysis_slurm.err"),
            env_setup=final_env_setup
        )
        manager = JobManager(mode="slurm")
        script_path = os.path.join(output_dir, "submit_analysis.slurm")
        manager.generate_script(job_conf, script_path)
        manager.submit(script_path, working_dir=output_dir)
        self.logger.info(f"AnalysisWorkflow submitted successfully to Slurm. Job script: {script_path}")
