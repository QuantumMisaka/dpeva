import json
from unittest.mock import MagicMock, patch

import dpdata

from dpeva.config import LabelingConfig
from dpeva.workflows.analysis import AnalysisWorkflow
from dpeva.workflows.labeling import LabelingWorkflow


@patch("dpeva.workflows.analysis.DatasetAnalysisManager")
@patch("dpeva.workflows.labeling.DataIntegrationManager")
@patch("dpeva.workflows.labeling.load_systems")
@patch("dpeva.workflows.labeling.LabelingManager")
def test_e2e_cycle_label_integration_analysis(
    MockLabelingManager,
    mock_load_systems,
    MockIntegrationManager,
    MockDatasetAnalysisManager,
    tmp_path,
):
    data_dir = tmp_path / "sampled_dpdata"
    data_dir.mkdir()
    (data_dir / "type.raw").touch()

    mock_sys = MagicMock(spec=dpdata.System)
    mock_load_systems.return_value = [mock_sys]

    label_manager = MockLabelingManager.return_value
    job_bundle = tmp_path / "bundle_0"
    job_bundle.mkdir()
    label_manager.prepare_tasks.return_value = [job_bundle]
    label_manager.process_results.return_value = ([], [])

    def _collect_and_export():
        cleaned = tmp_path / "outputs" / "cleaned"
        cleaned.mkdir(parents=True, exist_ok=True)

    label_manager.collect_and_export.side_effect = _collect_and_export

    def _integrate(**kwargs):
        merged = tmp_path / "outputs" / "merged_training_data"
        merged.mkdir(parents=True, exist_ok=True)
        summary = {"output_path": str(merged), "merged_system_count_after_dedup": 1}
        with open(merged / "integration_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        return summary

    MockIntegrationManager.return_value.integrate.side_effect = _integrate

    label_config = LabelingConfig(
        work_dir=str(tmp_path),
        input_data_path=str(data_dir),
        submission={"backend": "local"},
        dft_params={},
        attempt_params=[],
        pp_dir="/tmp/pp",
        orb_dir="/tmp/orb",
        integration_enabled=True,
    )
    LabelingWorkflow(label_config).run()

    merged_path = tmp_path / "outputs" / "merged_training_data"
    assert (merged_path / "integration_summary.json").exists()
    MockIntegrationManager.return_value.integrate.assert_called_once()

    analysis_config = {
        "mode": "dataset",
        "dataset_dir": str(merged_path),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["Fe", "C"],
    }
    AnalysisWorkflow(analysis_config).run()
    MockDatasetAnalysisManager.return_value.analyze.assert_called_once()
