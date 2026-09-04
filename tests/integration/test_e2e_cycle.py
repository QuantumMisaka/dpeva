import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import dpdata
import pytest

from dpeva.config import LabelingConfig
from dpeva.labeling.integration import DataIntegrationManager
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
    label_manager.extract_results.return_value = ([], [], [])

    def _collect_and_export():
        cleaned = tmp_path / "outputs" / "cleaned"
        cleaned.mkdir(parents=True, exist_ok=True)

    label_manager.collect_and_export.side_effect = _collect_and_export

    def _integrate(**kwargs):
        merged = tmp_path / "outputs" / "merged_training_data"
        merged.mkdir(parents=True, exist_ok=True)
        manifest_path = merged / "dataset-manifest.json"
        manifest = {
            "schema_version": "1.0",
            "dataset_id": "integration-e2e",
            "parents": [{"dataset_id": "new-labeled", "frame_count": 1}],
            "transformation": "merge",
            "frame_count": 1,
            "removed_frame_count": 0,
            "system_count": 1,
            "type_map": ["Fe", "C"],
            "format": "deepmd/npy/mixed",
            "source_entries": ["new-labeled"],
            "intersection_summary": {},
            "content_identity": None,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
        with open(merged / "dataset-manifest-e2e.json", "w") as f:
            json.dump(manifest, f, indent=4)
        summary = {
            "output_path": str(merged),
            "merged_system_count_after_dedup": 1,
            "dataset_manifest_path": "dataset-manifest-e2e.json",
        }
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
    assert (merged_path / "dataset-manifest.json").exists()
    summary = json.loads((merged_path / "integration_summary.json").read_text())
    assert not Path(summary["dataset_manifest_path"]).is_absolute()
    MockIntegrationManager.return_value.integrate.assert_called_once()

    analysis_config = {
        "mode": "dataset",
        "dataset_dir": str(merged_path),
        "output_dir": str(tmp_path / "analysis"),
        "type_map": ["Fe", "C"],
    }
    AnalysisWorkflow(analysis_config).run()
    MockDatasetAnalysisManager.return_value.analyze.assert_called_once()


def test_real_label_integration_emits_immutable_lineage(tmp_path):
    source = Path(__file__).parent / "data" / "sampled_dpdata" / "122"
    if not source.is_dir():
        pytest.skip(f"integration fixture not found: {source}")

    summary = DataIntegrationManager(
        deduplicate=True, output_format="deepmd/npy"
    ).integrate(
        new_labeled_data_path=source,
        merged_output_path=tmp_path / "merged_training_data",
        existing_training_data_path=source,
    )

    output = tmp_path / "merged_training_data"
    manifest_path = output / summary["dataset_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["content_identity"].startswith("sha256:")
    assert manifest["content_identity_strength"] == "exported-files-sha256"
    assert summary["dataset_manifest_path"] == manifest_path.name
    assert not Path(summary["dataset_manifest_path"]).is_absolute()
    assert summary["dataset_manifest_sha256"] == hashlib.sha256(
        (json.dumps(manifest, indent=4, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
