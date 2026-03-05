import os
from unittest.mock import MagicMock, patch

import pandas as pd

from dpeva.io.collection import CollectionIOManager


@patch("dpeva.io.collection.load_systems")
def test_export_path_traversal_is_blocked(mock_load_systems, tmp_path):
    io_manager = CollectionIOManager(str(tmp_path / "project"), "dpeva_results")
    io_manager.ensure_dirs()

    mock_sys = MagicMock()
    mock_sys.target_name = "../../sensitive.txt"
    mock_sys.short_name = "../../sensitive.txt"
    mock_sys.__len__.return_value = 10
    sub_sys = MagicMock()
    mock_sys.sub_system.return_value = sub_sys
    mock_load_systems.return_value = [mock_sys]

    df_final = pd.DataFrame({"dataname": ["../../sensitive.txt-0", "../../sensitive.txt-1"]})
    counts = io_manager.export_dpdata(
        testdata_dir="dummy",
        df_final=df_final,
        unique_system_names=["../../sensitive.txt"],
    )

    assert sub_sys.to_deepmd_npy.call_count == 0
    assert counts == (0, 0, 0, 0)


@patch("dpeva.io.collection.load_systems")
def test_export_valid_nested_name_is_allowed(mock_load_systems, tmp_path):
    io_manager = CollectionIOManager(str(tmp_path / "project"), "dpeva_results")
    io_manager.ensure_dirs()

    sys_name = "datasetA/systemB"
    mock_sys = MagicMock()
    mock_sys.target_name = sys_name
    mock_sys.short_name = sys_name
    mock_sys.__len__.return_value = 2
    sub_sys = MagicMock()
    mock_sys.sub_system.return_value = sub_sys
    mock_load_systems.return_value = [mock_sys]

    df_final = pd.DataFrame({"dataname": [f"{sys_name}-0"]})
    io_manager.export_dpdata(
        testdata_dir="dummy",
        df_final=df_final,
        unique_system_names=[sys_name],
    )

    out_paths = [c.args[0] for c in sub_sys.to_deepmd_npy.call_args_list]
    expected_sampled = os.path.join(io_manager.dpdata_savedir, "sampled_dpdata", "datasetA", "systemB")
    expected_other = os.path.join(io_manager.dpdata_savedir, "other_dpdata", "datasetA", "systemB")
    assert expected_sampled in out_paths
    assert expected_other in out_paths
