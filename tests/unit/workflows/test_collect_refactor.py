import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

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


def test_run_export_phase_logs_export_paths_and_summary(tmp_path):
    config = _base_config(tmp_path)
    with patch("dpeva.workflows.collect.setup_workflow_logger"):
        workflow = CollectionWorkflow(config)
    workflow.io_manager.export_dpdata = MagicMock(return_value=(1, 2, 3, 4))
    workflow.io_manager.last_export_paths = {
        "sampled_dpdata": "/tmp/sample_out",
        "other_dpdata": "/tmp/other_out",
    }
    workflow.logger = MagicMock()
    workflow._run_export_phase(pd.DataFrame({"dataname": ["sys-0"]}), ["sys"])
    log_lines = [str(call.args[0]) for call in workflow.logger.info.call_args_list]
    assert any("Export path (sampled_dpdata): /tmp/sample_out" in line for line in log_lines)
    assert any("Export path (other_dpdata): /tmp/other_out" in line for line in log_lines)
    assert any("Export summary: sampled=1 systems/3 frames, other=2 systems/4 frames." in line for line in log_lines)


def _filtered_config(tmp_path):
    config = _base_config(tmp_path)
    config.update(
        {
            "uq_trust_mode": "manual",
            "uq_qbc_trust_lo": 0.1,
            "uq_qbc_trust_hi": 0.3,
            "uq_rnd_rescaled_trust_lo": 0.2,
            "uq_rnd_rescaled_trust_hi": 0.4,
            "enable_diagnostic_plots": False,
            "select_n": 1,
        }
    )
    return config


def _setup_filtered_phase_mocks(workflow, has_gt):
    pred = MagicMock()
    pred.dataname_list = [["pool/sys1", 0, 2]]
    pred.datanames_nframe = {"pool/sys1": 1}
    workflow.uq_manager.load_predictions = MagicMock(return_value=([pred], has_gt))
    workflow.uq_manager.run_analysis = MagicMock(
        return_value=(
            {
                "uq_qbc_for": np.array([0.2]),
                "uq_rnd_for": np.array([0.3]),
                "diff_maxf_0_frame": np.array([0.1]),
            },
            np.array([0.25]),
        )
    )
    workflow.uq_manager.log_uq_statistics = MagicMock()
    workflow.uq_manager.run_auto_threshold = MagicMock()
    workflow.uq_manager.qbc_params = {"lo": 0.1, "hi": 0.3}
    workflow.uq_manager.rnd_params = {"lo": 0.2, "hi": 0.4}
    uq_filter = MagicMock()
    uq_filter.get_identity_labels.return_value = pd.DataFrame(
        {
            "dataname": ["pool/sys1-0"],
            "uq_qbc_for": [0.2],
            "uq_rnd_for_rescaled": [0.25],
            "uq_identity": ["candidate"],
        }
    )
    workflow.uq_manager.run_filtering = MagicMock(
        return_value=(
            pd.DataFrame(
                {
                    "dataname": ["pool/sys1-0"],
                    "uq_qbc_for": [0.2],
                    "uq_rnd_for_rescaled": [0.25],
                    "desc_stru_0": [1.0],
                    "desc_stru_1": [2.0],
                }
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            uq_filter,
        )
    )
    workflow.io_manager.load_descriptors = MagicMock(
        return_value=(["pool/sys1-0"], np.array([[1.0, 2.0]]))
    )
    workflow.io_manager.save_dataframe = MagicMock()


def test_filtered_phase_always_plots_rnd_and_qbc_trust_ranges(tmp_path):
    config = _filtered_config(tmp_path)
    with patch("dpeva.workflows.collect.setup_workflow_logger"):
        workflow = CollectionWorkflow(config)
    _setup_filtered_phase_mocks(workflow, has_gt=False)
    vis = MagicMock()
    workflow._run_filtered_uq_phase(vis)
    assert vis.plot_uq_with_trust_range.call_count == 2
    qbc_call = vis.plot_uq_with_trust_range.call_args_list[0]
    rnd_call = vis.plot_uq_with_trust_range.call_args_list[1]
    assert qbc_call.args[1] == "UQ-QbC-force"
    assert rnd_call.args[1] == "UQ-RND-force"
    vis.plot_uq_fdiff_scatter.assert_not_called()
    vis.plot_uq_vs_error.assert_not_called()
    vis.plot_uq_diff_parity.assert_not_called()
    vis.plot_candidate_vs_error.assert_not_called()


def test_filtered_phase_defaults_to_core_only_when_ground_truth_available(tmp_path):
    config = _filtered_config(tmp_path)
    with patch("dpeva.workflows.collect.setup_workflow_logger"):
        workflow = CollectionWorkflow(config)
    _setup_filtered_phase_mocks(workflow, has_gt=True)
    vis = MagicMock()
    workflow._run_filtered_uq_phase(vis)
    vis.plot_uq_fdiff_scatter.assert_called_once()
    vis.plot_uq_vs_error.assert_not_called()
    vis.plot_uq_diff_parity.assert_not_called()
    vis.plot_candidate_vs_error.assert_not_called()
    assert any(
        entry["reason"] == "diagnostic_plots_disabled" and entry["layer"] == "diagnostic"
        for entry in workflow._plot_audit_entries
    )


def test_filtered_phase_plots_diagnostic_layer_when_enabled(tmp_path):
    config = _filtered_config(tmp_path)
    config["enable_diagnostic_plots"] = True
    with patch("dpeva.workflows.collect.setup_workflow_logger"):
        workflow = CollectionWorkflow(config)
    _setup_filtered_phase_mocks(workflow, has_gt=True)
    vis = MagicMock()
    workflow._run_filtered_uq_phase(vis)
    vis.plot_uq_fdiff_scatter.assert_called_once()
    assert vis.plot_uq_vs_error.call_count == 2
    vis.plot_uq_diff_parity.assert_called_once()
    vis.plot_candidate_vs_error.assert_called_once()


def test_filtered_phase_skips_force_error_plot_when_diff_invalid(tmp_path):
    config = _filtered_config(tmp_path)
    with patch("dpeva.workflows.collect.setup_workflow_logger"):
        workflow = CollectionWorkflow(config)
    _setup_filtered_phase_mocks(workflow, has_gt=True)
    workflow.uq_manager.run_analysis = MagicMock(
        return_value=(
            {
                "uq_qbc_for": np.array([0.2]),
                "uq_rnd_for": np.array([0.3]),
                "diff_maxf_0_frame": np.array([np.nan]),
            },
            np.array([0.25]),
        )
    )
    vis = MagicMock()
    workflow._run_filtered_uq_phase(vis)
    vis.plot_uq_fdiff_scatter.assert_not_called()
    vis.plot_uq_vs_error.assert_not_called()
    vis.plot_uq_diff_parity.assert_not_called()
    vis.plot_candidate_vs_error.assert_not_called()


def test_filtered_phase_logs_skip_message_with_candidate_plot(tmp_path):
    config = _filtered_config(tmp_path)
    with patch("dpeva.workflows.collect.setup_workflow_logger"):
        workflow = CollectionWorkflow(config)
    _setup_filtered_phase_mocks(workflow, has_gt=False)
    vis = MagicMock()
    workflow._run_filtered_uq_phase(vis)
    vis.plot_candidate_vs_error.assert_not_called()
    assert any(
        entry["reason"] == "invalid_or_missing_ground_truth" and "candidate-vs-error" in entry["plots"]
        for entry in workflow._plot_audit_entries
    )


def test_filtered_phase_skips_rescaled_dependent_plots_when_rescaled_rnd_missing(tmp_path):
    config = _filtered_config(tmp_path)
    with patch("dpeva.workflows.collect.setup_workflow_logger"):
        workflow = CollectionWorkflow(config)
    _setup_filtered_phase_mocks(workflow, has_gt=True)
    workflow.uq_manager.run_analysis = MagicMock(
        return_value=(
            {
                "uq_qbc_for": np.array([0.2]),
                "uq_rnd_for": np.array([0.3]),
                "diff_maxf_0_frame": np.array([0.1]),
            },
            None,
        )
    )
    vis = MagicMock()
    workflow._run_filtered_uq_phase(vis)
    assert vis.plot_uq_with_trust_range.call_count == 1
    vis.plot_uq_identity_scatter.assert_not_called()
    vis.plot_uq_fdiff_scatter.assert_not_called()
    vis.plot_uq_vs_error.assert_not_called()
    vis.plot_uq_diff_parity.assert_not_called()
    vis.plot_candidate_vs_error.assert_not_called()
    assert any(
        entry["reason"] == "diagnostic_plots_disabled" and "candidate-vs-error" in entry["plots"]
        for entry in workflow._plot_audit_entries
    )


def test_run_sampling_phase_logs_multipool_summary_generated_in_joint_mode(tmp_path):
    config = _base_config(tmp_path)
    config["training_desc_dir"] = str(tmp_path / "train_desc")
    (tmp_path / "train_desc").mkdir()
    with patch("dpeva.workflows.collect.setup_workflow_logger"):
        workflow = CollectionWorkflow(config)
    workflow.io_manager.load_descriptors = MagicMock(return_value=(["train-0"], np.array([[0.5, 0.6]])))
    workflow.io_manager.save_dataframe = MagicMock()
    workflow.sampling_manager.prepare_features = MagicMock(
        return_value=(np.array([[1.0, 2.0], [0.5, 0.6]]), True, 1)
    )
    workflow.sampling_manager.execute_sampling = MagicMock(
        return_value={
            "selected_indices": [0],
            "pca_features": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "explained_variance": np.array([0.7, 0.3]),
            "random_indices": [1],
            "scores_direct": np.array([0.8, 0.6]),
            "scores_random": np.array([0.5, 0.4]),
            "full_pca_features": np.array([[0.0, 0.1], [0.2, 0.3]]),
        }
    )
    vis = MagicMock()
    df_candidate = pd.DataFrame(
        {"dataname": ["pool/sys1-0"], "desc_stru_0": [1.0], "desc_stru_1": [2.0]}
    )
    df_desc = df_candidate.copy()
    workflow._run_sampling_phase(df_candidate, df_desc, vis)
    assert any(
        entry["status"] == "generated" and "Final_sampled_PCAview_by_pool" in entry["plots"]
        for entry in workflow._plot_audit_entries
    )


def test_run_sampling_phase_logs_multipool_summary_skipped_in_normal_mode(tmp_path):
    config = _base_config(tmp_path)
    with patch("dpeva.workflows.collect.setup_workflow_logger"):
        workflow = CollectionWorkflow(config)
    workflow.io_manager.save_dataframe = MagicMock()
    workflow.sampling_manager.prepare_features = MagicMock(
        return_value=(np.array([[1.0, 2.0], [0.5, 0.6]]), False, None)
    )
    workflow.sampling_manager.execute_sampling = MagicMock(
        return_value={
            "selected_indices": [0],
            "pca_features": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "explained_variance": np.array([0.7, 0.3]),
            "random_indices": [1],
            "scores_direct": np.array([0.8, 0.6]),
            "scores_random": np.array([0.5, 0.4]),
            "full_pca_features": np.array([[0.0, 0.1], [0.2, 0.3]]),
        }
    )
    vis = MagicMock()
    df_candidate = pd.DataFrame(
        {
            "dataname": ["sys1-0", "sys2-0"],
            "desc_stru_0": [1.0, 0.5],
            "desc_stru_1": [2.0, 0.6],
        }
    )
    df_desc = df_candidate.copy()
    workflow._run_sampling_phase(df_candidate, df_desc, vis)
    assert any(
        entry["status"] == "skipped"
        and entry["reason"] == "joint_mode_disabled"
        and "Final_sampled_PCAview_by_pool" in entry["plots"]
        for entry in workflow._plot_audit_entries
    )
