import pandas as pd
import numpy as np
from unittest.mock import patch

from dpeva.workflows.collect import CollectionWorkflow


def _base_config(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "desc").mkdir()
    (tmp_path / "testdata").mkdir()
    return {
        "project": str(tmp_path / "project"),
        "desc_dir": str(tmp_path / "desc"),
        "testdata_dir": str(tmp_path / "testdata"),
        "root_savedir": str(tmp_path / "savedir"),
        "uq_trust_mode": "no_filter",
        "backend": "local",
    }


def test_collect_run_orchestrates_three_phases(tmp_path):
    config = _base_config(tmp_path)
    with patch("dpeva.workflows.collect.setup_workflow_logger"), \
         patch.object(CollectionWorkflow, "_run_uq_phase", return_value=(pd.DataFrame(), pd.DataFrame(), ["s1"])) as mock_uq, \
         patch.object(CollectionWorkflow, "_run_sampling_phase", return_value=pd.DataFrame()) as mock_sampling, \
         patch.object(CollectionWorkflow, "_run_export_phase") as mock_export:
        workflow = CollectionWorkflow(config)
        workflow.run()
    mock_uq.assert_called_once()
    mock_sampling.assert_called_once()
    mock_export.assert_called_once()


def test_collect_run_no_filter_phase_outputs_candidate_identity(tmp_path):
    config = _base_config(tmp_path)
    with patch("dpeva.workflows.collect.setup_workflow_logger"), \
         patch("dpeva.io.collection.CollectionIOManager.load_descriptors", return_value=(["a-0", "b-0"], np.array([[1.0, 2.0], [3.0, 4.0]]))):
        workflow = CollectionWorkflow(config)
        df_desc, df_candidate, unique_system_names = workflow._run_no_filter_uq_phase()
    assert list(df_desc["dataname"]) == ["a-0", "b-0"]
    assert set(df_candidate["uq_identity"].unique()) == {"candidate"}
    assert unique_system_names == ["a", "b"]


def test_collect_extract_unique_system_names(tmp_path):
    config = _base_config(tmp_path)
    with patch("dpeva.workflows.collect.setup_workflow_logger"):
        workflow = CollectionWorkflow(config)
    names = workflow._extract_unique_system_names([["a", 0], ["a", 1], ["b", 0]])
    assert names == ["a", "b"]
