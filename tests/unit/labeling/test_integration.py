from pathlib import Path
from unittest.mock import patch

from dpeva.labeling.integration import DataIntegrationManager


class _FakeMultiSystems(list):
    def to(self, fmt, output):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "export.ok").write_text(fmt)


class _FakeSystem:
    def __init__(self, coords, atom_names=None, atom_types=None, atom_numbs=None):
        names = atom_names or ["Fe", "C"]
        self.data = {
            "coords": coords,
            "atom_names": names,
            "atom_types": atom_types if atom_types is not None else [0],
            "atom_numbs": atom_numbs if atom_numbs is not None else [1 for _ in names],
            "type_map": list(names),
        }


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
    assert result["output_format"] == "deepmd/npy/mixed"
    assert (out_dir / "export.ok").read_text() == "deepmd/npy/mixed"
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
    assert result["output_format"] == "deepmd/npy/mixed"
    assert (out_dir / "export.ok").exists()
    assert (out_dir / "integration_summary.json").exists()


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_custom_output_format(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    mock_load_systems.side_effect = [[_FakeSystem([[[0.0, 0.0, 0.0]]])]]

    manager = DataIntegrationManager(deduplicate=False, output_format="deepmd/npy")
    result = manager.integrate(
        new_labeled_data_path=new_dir,
        merged_output_path=out_dir,
    )

    assert result["output_format"] == "deepmd/npy"
    assert (out_dir / "export.ok").read_text() == "deepmd/npy"


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


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_reorders_compatible_atom_names(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    old_dir = tmp_path / "old_train"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    old_dir.mkdir()

    existing = _FakeSystem(
        [[[0.0, 0.0, 0.0]]],
        atom_names=["H", "C", "O", "Fe"],
        atom_types=[0, 1, 2, 3],
        atom_numbs=[1, 1, 1, 1],
    )
    new = _FakeSystem(
        [[[1.0, 0.0, 0.0]]],
        atom_names=["C", "Fe", "H", "O"],
        atom_types=[2, 0, 3, 1],
        atom_numbs=[1, 1, 1, 1],
    )
    mock_load_systems.side_effect = [[existing], [new]]

    manager = DataIntegrationManager(deduplicate=False)
    result = manager.integrate(
        new_labeled_data_path=new_dir,
        merged_output_path=out_dir,
        existing_training_data_path=old_dir,
    )

    assert result["compatibility_issues"] == 0
    assert new.data["atom_names"] == ["H", "C", "O", "Fe"]
    assert new.data["type_map"] == ["H", "C", "O", "Fe"]
    assert new.data["atom_types"] == [0, 1, 2, 3]
