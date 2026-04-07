
import pytest
from unittest.mock import MagicMock, patch
import json
import pandas as pd
from dpeva.labeling.manager import LabelingManager

class TestLabelingManager:
    
    @pytest.fixture
    def manager(self, tmp_path):
        config = {
            "work_dir": str(tmp_path),
            "input_data_path": "dummy",
            "submission": {"backend": "local"},
            "tasks_per_job": 50,
            "dft_params": {},
            "attempt_params": []
        }
        return LabelingManager(config)

    def test_mgr_001_path_generation_multipool(self, manager, mock_multisystems):
        """
        MGR-001: Verify hierarchical path generation for Multi-Pool.
        """
        # Mock generator to avoid real file generation
        manager.generator = MagicMock()
        # Mock analyzer
        manager.generator.analyzer.analyze.return_value = (MagicMock(), "cluster", [True, True, True])
        
        # Setup dataset map
        dataset_map = {"DS1": mock_multisystems}
        
        # Mock packer to avoid error on non-existent files
        manager.packer.pack = MagicMock(return_value=[])
        
        # Act
        manager.prepare_tasks(dataset_map)
        
        # Assert
        # Check generate call args
        # Expected path: inputs/DS1/cluster/sys_test_0
        manager.generator.generate.assert_called()
        call_args = manager.generator.generate.call_args
        # args: (atoms, output_dir, task_name, stru_type, vacuum_status, dataset_name, system_name)
        # We check kwargs or positionals
        kwargs = call_args[1]
        
        assert kwargs["dataset_name"] == "DS1"
        assert kwargs["system_name"] == "sys_test"
        
        task_dir = call_args[0][1] # 2nd pos arg is task_dir
        assert "inputs/DS1/cluster/sys_test" in str(task_dir)

    def test_mgr_002_task_packing(self, manager, tmp_path):
        """
        MGR-002: Verify task packing logic.
        """
        # Setup mock tasks
        input_dir = manager.input_dir
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 100 task directories
        task_dirs = []
        for i in range(100):
            d = input_dir / "DS1" / "cluster" / f"task_{i}"
            d.mkdir(parents=True)
            (d / "INPUT").touch()
            task_dirs.append(d)
            
        # Act
        # Manager calls packer.pack
        # But here we want to test manager's integration with packer
        # Let's use real packer (manager.packer is initialized)
        packed_dirs = manager.packer.pack(input_dir)
        
        # Assert
        # Should have 2 packed dirs (50 tasks each)
        assert len(packed_dirs) == 2
        assert (input_dir / "N_50_0").exists()
        assert (input_dir / "N_50_1").exists()
        
        # Check if tasks are moved
        assert len(list((input_dir / "N_50_0").iterdir())) == 50
        assert len(list((input_dir / "N_50_1").iterdir())) == 50

    @patch("dpeva.labeling.manager.dpdata")
    def test_mgr_003_statistics_logic(self, mock_dpdata, manager, tmp_path):
        """
        MGR-003: Verify statistics aggregation logic.
        """
        # Setup directories
        (manager.input_dir / "DS1" / "cluster" / "Failed_Task").mkdir(parents=True)
        (manager.input_dir / "DS1" / "cluster" / "Failed_Task" / "INPUT").touch()
        # Mock failed task meta
        with open(manager.input_dir / "DS1" / "cluster" / "Failed_Task" / "task_meta.json", "w") as f:
            json.dump({"dataset_name": "DS1", "stru_type": "cluster"}, f)
    
        (manager.converged_dir / "DS1" / "cluster" / "Conv_Task").mkdir(parents=True)
        (manager.converged_dir / "DS1" / "cluster" / "Conv_Task" / "INPUT").touch()
        (manager.converged_dir / "DS1" / "cluster" / "Conv_Task" / "STRU").touch()
        # Mock conv task meta
        with open(manager.converged_dir / "DS1" / "cluster" / "Conv_Task" / "task_meta.json", "w") as f:
            json.dump({"dataset_name": "DS1", "stru_type": "cluster"}, f)
    
        # Mock postprocessor
        mock_sys = MagicMock() # Don't need spec if dpdata is mocked
        manager.postprocessor.load_data = MagicMock(return_value=mock_sys)
    
        # Mock DataFrame return
        # Create a real DataFrame for logic test
        import pandas as pd
        df_real = pd.DataFrame({
            "sys_idx": [0],
            "frame_idx": [0],
            "max_force": [0.1],
            # dataset/type will be merged
        })
        manager.postprocessor.compute_metrics = MagicMock(return_value=df_real)
        manager.postprocessor.filter_data = MagicMock(return_value=df_real)
        # Also mock export_data to avoid real IO if possible, or let it fail gracefully if we provide enough data
        # export_data uses dpdata systems. Mocked systems are MagicMock.
        # Calling export_data with MagicMock systems might crash inside dpdata calls.
        # It is better to mock export_data as well.
        manager.postprocessor.export_data = MagicMock()
    
        # Act
        manager.collect_and_export()
    
        # Assert
        # We can't easily assert log output without caplog fixture, but we can verify execution flow
        manager.postprocessor.load_data.assert_called_once()
        assert manager.postprocessor.compute_metrics.call_count >= 1
        manager.postprocessor.export_data.assert_called()

    def test_mgr_004_failed_task_scanning(self, manager):
        """
        MGR-004: Verify scanning of failed tasks in inputs directory.
        """
        # Setup
        (manager.input_dir / "DS1" / "cluster" / "Failed").mkdir(parents=True)
        (manager.input_dir / "DS1" / "cluster" / "Failed" / "INPUT").touch()
        
        # Exclude OUT.ABACUS
        (manager.input_dir / "DS1" / "cluster" / "Running" / "OUT.ABACUS").mkdir(parents=True)
        (manager.input_dir / "DS1" / "cluster" / "Running" / "OUT.ABACUS" / "INPUT").touch()
        
        failed_tasks_info = manager._collect_failed_tasks_info()
        assert failed_tasks_info == [{"dataset": "DS1", "type": "cluster"}]

    def test_mgr_005_runner_script_avoids_shell_true(self, manager, tmp_path):
        script = manager.generate_runner_script(tmp_path)
        assert "shell=True" not in script
        assert "subprocess.run(cmd, check=True" in script
        assert "import shlex" in script
        assert "shlex.split(abacus_cmd)" in script

    @patch("dpeva.labeling.manager.dpdata")
    def test_mgr_006_invalid_task_meta_is_handled(self, mock_dpdata, manager):
        failed_task = manager.input_dir / "DS1" / "cluster" / "Failed_Task"
        failed_task.mkdir(parents=True)
        (failed_task / "INPUT").touch()
        (failed_task / "task_meta.json").write_text("{invalid json")

        (manager.converged_dir / "DS1" / "cluster" / "Conv_Task").mkdir(parents=True)
        (manager.converged_dir / "DS1" / "cluster" / "Conv_Task" / "INPUT").touch()
        (manager.converged_dir / "DS1" / "cluster" / "Conv_Task" / "STRU").touch()
        with open(manager.converged_dir / "DS1" / "cluster" / "Conv_Task" / "task_meta.json", "w") as f:
            json.dump({"dataset_name": "DS1", "stru_type": "cluster"}, f)

        mock_system = MagicMock()
        manager.postprocessor.load_data = MagicMock(return_value=mock_system)
        import pandas as pd
        df_real = pd.DataFrame({"sys_idx": [0], "frame_idx": [0], "max_force": [0.1]})
        manager.postprocessor.compute_metrics = MagicMock(return_value=df_real)
        manager.postprocessor.filter_data = MagicMock(return_value=df_real)
        manager.postprocessor.export_data = MagicMock()

        manager.collect_and_export()

        manager.postprocessor.load_data.assert_called_once()
        assert manager.postprocessor.compute_metrics.call_count >= 1

    def test_mgr_007_aggregate_stats(self, manager):
        df = pd.DataFrame(
            [
                {"dataset": "DS1", "type": "cluster"},
                {"dataset": "DS1", "type": "cluster"},
                {"dataset": "DS1", "type": "surface"},
            ]
        )
        df_clean = pd.DataFrame(
            [
                {"dataset": "DS1", "type": "cluster"},
                {"dataset": "DS1", "type": "surface"},
            ]
        )
        failed = [{"dataset": "DS1", "type": "cluster"}, {"dataset": "DS2", "type": "bulk"}]
        stats = manager._aggregate_stats(df, df_clean, failed)
        assert stats["DS1"]["cluster"]["total"] == 3
        assert stats["DS1"]["cluster"]["conv"] == 2
        assert stats["DS1"]["cluster"]["fail"] == 1
        assert stats["DS1"]["cluster"]["clean"] == 1
        assert stats["DS2"]["bulk"]["total"] == 1

    def test_mgr_008_prepare_is_idempotent(self, manager, mock_multisystems):
        stale_task = manager.input_dir / "N_50_0" / "stale_task"
        stale_task.mkdir(parents=True, exist_ok=True)
        (stale_task / "INPUT").touch()

        manager.generator = MagicMock()
        manager.generator.analyzer.analyze.return_value = (MagicMock(), "cluster", [True, True, True])
        manager.packer.pack = MagicMock(return_value=[])

        dataset_map = {"DS1": mock_multisystems}
        manager.prepare_tasks(dataset_map)

        assert not stale_task.exists()
        manager.packer.pack.assert_called_once_with(manager.input_dir)

    def test_mgr_009_prepare_keeps_outputs_and_converged(self, manager, mock_multisystems):
        output_marker = manager.output_dir / "keep.txt"
        converged_marker = manager.converged_dir / "keep.txt"
        output_marker.parent.mkdir(parents=True, exist_ok=True)
        converged_marker.parent.mkdir(parents=True, exist_ok=True)
        output_marker.write_text("output")
        converged_marker.write_text("converged")

        manager.generator = MagicMock()
        manager.generator.analyzer.analyze.return_value = (MagicMock(), "cluster", [True, True, True])
        manager.packer.pack = MagicMock(return_value=[])

        dataset_map = {"DS1": mock_multisystems}
        manager.prepare_tasks(dataset_map)

        assert output_marker.exists()
        assert converged_marker.exists()

    @patch("dpeva.labeling.manager.logger.warning")
    @patch("dpeva.labeling.manager.dpdata")
    def test_mgr_010_skip_incomplete_converged_systems(self, mock_dpdata, mock_warning, manager, caplog):
        for name in ("Conv_Task_0", "Conv_Task_1"):
            task_dir = manager.converged_dir / "DS1" / "cluster" / name
            task_dir.mkdir(parents=True)
            (task_dir / "INPUT").touch()
            (task_dir / "STRU").touch()
            with open(task_dir / "task_meta.json", "w") as f:
                json.dump({"dataset_name": "DS1", "stru_type": "cluster"}, f)

        valid_system = MagicMock()
        manager.postprocessor.load_data = MagicMock(side_effect=[valid_system, None])
        df_real = pd.DataFrame({"sys_idx": [0], "frame_idx": [0], "max_force": [0.1]})
        manager.postprocessor.compute_metrics = MagicMock(return_value=df_real)
        manager.postprocessor.filter_data = MagicMock(return_value=df_real)
        manager.postprocessor.export_data = MagicMock()

        manager.collect_and_export()

        assert manager.postprocessor.load_data.call_count == 2
        mock_warning.assert_any_call("Skipped 1 converged directories due to incomplete parsed data.")

    def test_mgr_011_extract_results_routes_bad_converged(self, manager):
        job_dir = manager.input_dir / "N_50_0"
        for name in ("ok_0", "bad_0", "fail_0"):
            task_dir = job_dir / name
            task_dir.mkdir(parents=True)
            (task_dir / "INPUT").touch()

        manager.postprocessor.classify_task_status = MagicMock(
            side_effect=[
                ("converged", "ok"),
                ("bad_converged", "missing_total_force_block"),
                ("failed", "scf_not_converged"),
            ]
        )
        manager._move_task_dir = MagicMock()

        converged, bad_converged, failed = manager.extract_results([job_dir])

        assert len(converged) == 1
        assert len(bad_converged) == 1
        assert len(failed) == 1
        assert manager._move_task_dir.call_count == 2
        manager._move_task_dir.assert_any_call(converged[0], manager.converged_dir, "CONVERGED")
        manager._move_task_dir.assert_any_call(bad_converged[0], manager.bad_converged_dir, "BAD_CONVERGED")

    def test_mgr_012_metadata_missing_in_packed_dir_falls_to_unknown(self, manager):
        task_dir = manager.input_dir / "N_50_0" / "task_x"
        task_dir.mkdir(parents=True)
        (task_dir / "INPUT").touch()
        info = manager._collect_failed_tasks_info()
        assert info == [{"dataset": "unknown", "type": "unknown"}]

    def test_mgr_013_metadata_damaged_falls_back_to_sidecar(self, manager):
        task_dir = manager.converged_dir / "N_50_0" / "task_x"
        task_dir.mkdir(parents=True)
        (task_dir / "task_meta.json").write_text("{bad json")
        (task_dir / "task_identity.json").write_text(
            json.dumps({"dataset": "DSX", "type": "cluster", "task_name": "task_x"})
        )
        meta = manager._read_task_meta(task_dir, manager.converged_dir)
        assert meta["dataset"] == "DSX"
        assert meta["type"] == "cluster"

    def test_mgr_014_extract_results_persists_identity_mapping(self, manager):
        job_dir = manager.input_dir / "N_50_0"
        task_dir = job_dir / "ok_0"
        task_dir.mkdir(parents=True)
        (task_dir / "INPUT").touch()
        (task_dir / "task_meta.json").write_text(
            json.dumps({"dataset_name": "DS1", "stru_type": "cluster", "task_name": "ok_0"})
        )
        manager.postprocessor.classify_task_status = MagicMock(return_value=("converged", "ok"))

        manager.extract_results([job_dir])

        moved_task = manager.converged_dir / "DS1" / "cluster" / "ok_0"
        assert moved_task.exists()
        assert (moved_task / "task_identity.json").exists()
        identity_map = json.loads(manager.identity_map_path.read_text())
        assert identity_map["by_task_name"]["ok_0"]["dataset"] == "DS1"

    def test_mgr_015_consistency_check_detects_mismatch(self, manager):
        stats = {"DS1": {"cluster": {"total": 3, "conv": 2, "fail": 1, "clean": 2, "filt": 0}}}
        df = pd.DataFrame([{"dataset": "DS1", "type": "cluster"}])
        df_clean = pd.DataFrame([{"dataset": "DS1", "type": "cluster"}])
        failed = [{"dataset": "DS1", "type": "cluster"}]
        consistency = manager._validate_stats_consistency(stats, df, df_clean, failed)
        assert consistency["trusted"] is False
        assert len(consistency["errors"]) > 0

    @patch("dpeva.labeling.manager.dpdata")
    def test_mgr_016_rebuild_statistics_report_outputs_repaired_file(self, mock_dpdata, manager):
        task_dir = manager.converged_dir / "DS1" / "cluster" / "Conv_Task"
        task_dir.mkdir(parents=True)
        (task_dir / "INPUT").touch()
        (task_dir / "STRU").touch()
        (task_dir / "task_meta.json").write_text(
            json.dumps({"dataset_name": "DS1", "stru_type": "cluster", "task_name": "Conv_Task"})
        )
        manager.postprocessor.load_data = MagicMock(return_value=MagicMock())
        df = pd.DataFrame([{"sys_idx": 0, "frame_idx": 0, "max_force": 0.1}])
        manager.postprocessor.compute_metrics = MagicMock(return_value=df)
        manager.postprocessor.filter_data = MagicMock(side_effect=lambda x: x.copy())

        report = manager.rebuild_statistics_report()

        assert report["consistency"]["trusted"] is True
        assert (manager.output_dir / "labeling_stats_report.repaired.json").exists()

    @patch("dpeva.labeling.manager.dpdata")
    def test_mgr_017_fp7_style_branch_alignment_stays_stable(self, mock_dpdata, manager):
        task_a = manager.converged_dir / "DeepCNT" / "cluster" / "C108Fe55_0"
        task_b = manager.converged_dir / "oc22-FeCOH" / "bulk" / "C20Fe20_0"
        for task_dir, dataset, stru_type, task_name in (
            (task_a, "DeepCNT", "cluster", "C108Fe55_0"),
            (task_b, "oc22-FeCOH", "bulk", "C20Fe20_0"),
        ):
            task_dir.mkdir(parents=True)
            (task_dir / "INPUT").touch()
            (task_dir / "STRU").touch()
            (task_dir / "task_meta.json").write_text(
                json.dumps(
                    {
                        "dataset_name": dataset,
                        "stru_type": stru_type,
                        "task_name": task_name,
                    }
                )
            )

        manager.postprocessor.load_data = MagicMock(side_effect=[MagicMock(), MagicMock()])
        manager.postprocessor.compute_metrics = MagicMock(
            return_value=pd.DataFrame([{"sys_idx": 99, "frame_idx": 0, "max_force": 0.1}])
        )
        manager.postprocessor.filter_data = MagicMock(side_effect=lambda x: x.copy())
        manager.postprocessor.export_data = MagicMock()

        report = manager.rebuild_statistics_report()

        assert report["consistency"]["trusted"] is True
        assert report["branches"]["DeepCNT"]["cluster"]["conv"] == 1
        assert report["branches"]["oc22-FeCOH"]["bulk"]["conv"] == 1
        assert report["global"]["conv"] == 2
        assert manager.postprocessor.compute_metrics.call_count == 2
