from pathlib import Path
from unittest.mock import patch

from dpeva.labeling.integration import DataIntegrationManager


class _FakeMultiSystems(list):
    def to(self, fmt, output):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "export.ok").write_text(fmt)


class _FakeSystem:
    def __init__(self, coords, atom_names=None):
        self.data = {"coords": coords, "atom_names": atom_names or ["Fe", "C"]}


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_export(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    old_dir = tmp_path / "old_train"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    old_dir.mkdir()

    mock_load_systems.side_effect = [
        [_FakeSystem([[[0.0, 0.0, 0.0]]])],
        [_FakeSystem([[[1.0, 0.0, 0.0]]])],
    ]

    manager = DataIntegrationManager(deduplicate=False)
    result = manager.integrate(
        new_labeled_data_path=new_dir,
        merged_output_path=out_dir,
        existing_training_data_path=old_dir,
    )

    assert result["output_path"] == str(out_dir)
    assert result["existing_system_count"] == 1
    assert result["new_system_count"] == 1
    assert result["filtered_system_count"] == 0
    assert (out_dir / "export.ok").exists()
    assert (out_dir / "integration_summary.json").exists()


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_deduplicate(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()

    dup_coords = [[[0.0, 0.0, 0.0]]]
    mock_load_systems.side_effect = [[_FakeSystem(dup_coords), _FakeSystem(dup_coords)]]

    manager = DataIntegrationManager(deduplicate=True)
    result = manager.integrate(
        new_labeled_data_path=new_dir,
        merged_output_path=out_dir,
    )

    assert result["output_path"] == str(out_dir)
    assert result["deduplicate_enabled"] is True
    assert result["merged_system_count_after_dedup"] == 1
    assert result["filtered_system_count"] == 1
    assert (out_dir / "export.ok").exists()
    assert (out_dir / "integration_summary.json").exists()


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_incompatible_atom_names(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    old_dir = tmp_path / "old_train"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    old_dir.mkdir()

    mock_load_systems.side_effect = [
        [_FakeSystem([[[0.0, 0.0, 0.0]]], atom_names=["Fe", "C"])],
        [_FakeSystem([[[1.0, 0.0, 0.0]]], atom_names=["Fe", "O"])],
    ]

    manager = DataIntegrationManager(deduplicate=False)
    import pytest
    with pytest.raises(ValueError, match="Incompatible atom_names"):
        manager.integrate(
            new_labeled_data_path=new_dir,
            merged_output_path=out_dir,
            existing_training_data_path=old_dir,
        )
