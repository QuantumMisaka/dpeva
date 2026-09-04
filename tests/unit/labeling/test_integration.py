import errno
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import dpdata
import numpy as np
import pytest

from dpeva.labeling.integration import (
    DataIntegrationManager,
    PublicationError,
)


class _FakeMultiSystems(list):
    def to(self, fmt, output):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "export.ok").write_text(fmt)
        payload = "|".join(repr(system.data.get("coords")) for system in self)
        (Path(output) / "frames.payload").write_text(payload)


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


def _real_system(
    coords, *, energies=None, cells=None, nopbc=False, atom_names=("H", "O"), atom_types=(0, 1)
):
    coords = np.asarray(coords, dtype=float)
    nframes = coords.shape[0]
    data = {
        "atom_names": list(atom_names),
        "atom_numbs": [1 for _ in atom_names],
        "atom_types": np.asarray(atom_types, dtype=int),
        "orig": np.zeros(3),
        "cells": np.repeat(
            np.asarray(cells if cells is not None else np.eye(3), dtype=float)[None, :, :],
            nframes,
            axis=0,
        ),
        "coords": coords,
    }
    if nopbc:
        data["nopbc"] = True
    if energies is not None:
        data["energies"] = np.asarray(energies, dtype=float)
    return dpdata.System(data=data)


def test_real_dpdata_internal_duplicate_frames_are_detected_and_deduplicated():
    system = _real_system(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        energies=[1.0, 1.0],
    )
    frames = list(DataIntegrationManager._iter_frames(system))
    kept, overlap = DataIntegrationManager._analyze_frames(frames)
    assert len(frames) == 2
    assert len(kept) == 1
    assert overlap == 1


def test_real_dpdata_partial_intersection_across_multi_frame_systems():
    first = _real_system(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        energies=[1.0, 2.0],
    )
    second = _real_system(
        [
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ],
        energies=[2.0, 3.0],
    )
    frames = [
        frame
        for system in (first, second)
        for frame in DataIntegrationManager._iter_frames(system)
    ]
    kept, overlap = DataIntegrationManager._analyze_frames(frames)
    assert len(kept) == 3
    assert overlap == 1


def test_real_dpdata_identity_includes_cell_pbc_and_atom_type_map():
    base = _real_system([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    other_cell = _real_system(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], cells=np.eye(3) * 2
    )
    nonperiodic = _real_system(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], nopbc=True
    )
    other_types = _real_system(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], atom_names=("O", "H"), atom_types=(1, 0)
    )
    identities = {
        DataIntegrationManager._frame_identity(frame)
        for frame in (base, other_cell, nonperiodic, other_types)
    }
    assert len(identities) == 4


def test_real_dpdata_duplicate_structure_with_conflicting_labels_fails_closed():
    first = _real_system([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], energies=[1.0])
    second = _real_system([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], energies=[2.0])
    with pytest.raises(ValueError, match="conflicting labels"):
        DataIntegrationManager._analyze_frames([first, second])


@patch("dpeva.labeling.integration.load_systems")
def test_real_dpdata_integration_deduplicates_before_export(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    mock_load_systems.return_value = [
        _real_system(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            energies=[1.0, 1.0],
        )
    ]

    result = DataIntegrationManager(deduplicate=True, output_format="deepmd/npy").integrate(
        new_labeled_data_path=new_dir, merged_output_path=out_dir
    )

    assert result["merged_system_count_before_dedup"] == 1
    assert result["merged_system_count_after_dedup"] == 1
    assert result["filtered_system_count"] == 0
    assert result["merged_frame_count_before_dedup"] == 2
    assert result["merged_frame_count_after_dedup"] == 1
    manifest = json.loads((out_dir / "dataset-manifest.json").read_text())
    assert manifest["intersection_summary"] == {
        "method": "frame-identity-v1",
        "overlap_frame_count": 1,
        "removed_frame_count": 1,
        "evidence_ref": "in-memory:frame-identity-v1",
    }


@patch("dpeva.labeling.integration.load_systems")
def test_real_dpdata_label_conflict_fails_before_export(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    frame = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    mock_load_systems.return_value = [
        _real_system(frame, energies=[1.0]),
        _real_system(frame, energies=[2.0]),
    ]

    with pytest.raises(ValueError, match="conflicting labels"):
        DataIntegrationManager(deduplicate=True).integrate(
            new_labeled_data_path=new_dir, merged_output_path=out_dir
        )
    assert not out_dir.exists()


@patch("dpeva.labeling.integration.load_systems")
def test_real_dpdata_system_and_frame_counts_are_distinct(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    mock_load_systems.return_value = [
        _real_system(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ],
            energies=[1.0, 2.0],
        ),
        _real_system(
            [[[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]],
            energies=[3.0],
        ),
    ]

    result = DataIntegrationManager(deduplicate=False, output_format="deepmd/npy").integrate(
        new_labeled_data_path=new_dir, merged_output_path=out_dir
    )

    # Both input systems have the same formula and coalesce in dpdata, while
    # their three frames remain independently accounted for.
    assert result["merged_system_count_before_dedup"] == 1
    assert result["merged_system_count_after_dedup"] == 1
    assert result["filtered_system_count"] == 0
    assert result["merged_frame_count_before_dedup"] == 3
    assert result["merged_frame_count_after_dedup"] == 3


@patch("dpeva.labeling.integration.load_systems")
def test_real_dpdata_frame_dedup_does_not_report_system_removal(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    mock_load_systems.return_value = [
        _real_system(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            energies=[1.0, 1.0],
        )
    ]

    result = DataIntegrationManager(deduplicate=True, output_format="deepmd/npy").integrate(
        new_labeled_data_path=new_dir, merged_output_path=out_dir
    )

    assert result["filtered_frame_count"] == 1
    assert result["filtered_system_count"] == 0
    manifest = json.loads((out_dir / "dataset-manifest.json").read_text())
    assert manifest["system_count"] == 1


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

    manifest = json.loads((out_dir / "dataset-manifest.json").read_text())
    assert manifest["frame_count"] == result["merged_frame_count_after_dedup"]
    assert [parent["frame_count"] for parent in manifest["parents"]] == [
        result["existing_frame_count"],
        result["new_frame_count"],
    ]
    assert result["dataset_manifest_path"].startswith("dataset-manifest-")
    assert (out_dir / result["dataset_manifest_path"]).exists()
    generation_bytes = (out_dir / result["dataset_manifest_path"]).read_bytes()
    assert generation_bytes == (out_dir / "dataset-manifest.json").read_bytes()
    assert result["dataset_manifest_sha256"] == hashlib.sha256(generation_bytes).hexdigest()
    assert manifest["source_entries"] == ["existing-training", "new-labeled"]
    assert manifest["validation_result"]["status"] == "passed"
    assert manifest["intersection_summary"]["overlap_frame_count"] == 0
    assert all(not Path(entry).is_absolute() for entry in manifest["source_entries"])
    relocated = tmp_path / "relocated"
    out_dir.rename(relocated)
    assert (relocated / result["dataset_manifest_path"]).exists()


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
    manifest = json.loads((out_dir / "dataset-manifest.json").read_text())
    assert manifest["frame_count"] == 1
    assert manifest["removed_frame_count"] == 1
    assert manifest["intersection_summary"]["removed_frame_count"] == 1
    assert manifest["intersection_summary"]["evidence_ref"] == "in-memory:frame-identity-v1"
    assert [parent["dataset_id"] for parent in manifest["parents"]] == ["new-labeled"]
    assert manifest["parents"][0]["manifest_ref"] is None
    assert result["dataset_manifest_path"].startswith("dataset-manifest-")
    assert (out_dir / result["dataset_manifest_path"]).exists()


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_unexplained_overlap_fails_when_deduplication_is_disabled(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    duplicate = [[[0.0, 0.0, 0.0]]]
    mock_load_systems.return_value = [_FakeSystem(duplicate), _FakeSystem(duplicate)]

    with pytest.raises(ValueError, match="unexplained duplicate/intersection"):
        DataIntegrationManager(deduplicate=False).integrate(
            new_labeled_data_path=new_dir, merged_output_path=out_dir
        )
    assert not out_dir.exists()


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_deduplicate_drops_empty_coords(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()

    mock_load_systems.side_effect = [[_FakeSystem([])]]

    manager = DataIntegrationManager(deduplicate=True)
    result = manager.integrate(
        new_labeled_data_path=new_dir,
        merged_output_path=out_dir,
    )

    assert result["merged_system_count_before_dedup"] == 1
    assert result["merged_system_count_after_dedup"] == 0
    assert result["filtered_system_count"] == 1


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


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_rejects_type_map_conflict_before_export(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    old_dir = tmp_path / "old_train"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    old_dir.mkdir()

    existing = _FakeSystem([[[0.0, 0.0, 0.0]]])
    conflicting = _FakeSystem([[[1.0, 0.0, 0.0]]])
    conflicting.data["type_map"] = ["Fe", "O"]
    mock_load_systems.side_effect = [[existing], [conflicting]]

    with pytest.raises(ValueError, match="Incompatible type_map"):
        DataIntegrationManager().integrate(
            new_labeled_data_path=new_dir,
            merged_output_path=out_dir,
            existing_training_data_path=old_dir,
        )

    assert not (out_dir / "export.ok").exists()
    assert not (out_dir / "integration_summary.json").exists()
    assert not (out_dir / "dataset-manifest.json").exists()


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_rejects_count_conflict_before_export(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    mock_load_systems.return_value = [_FakeSystem([[[0.0, 0.0, 0.0]]])]

    manager = DataIntegrationManager()
    with patch.object(manager, "_count_total_frames", side_effect=[1, 2]):
        with pytest.raises(ValueError, match="frame count conflict"):
            manager.integrate(new_labeled_data_path=new_dir, merged_output_path=out_dir)

    assert not (out_dir / "export.ok").exists()
    assert not (out_dir / "integration_summary.json").exists()
    assert not (out_dir / "dataset-manifest.json").exists()


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_missing_existing_path_fails_before_export(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    missing = tmp_path / "missing-training"

    with pytest.raises(FileNotFoundError, match="Existing training data path not found"):
        DataIntegrationManager().integrate(
            new_labeled_data_path=new_dir,
            merged_output_path=out_dir,
            existing_training_data_path=missing,
        )

    mock_load_systems.assert_not_called()
    assert not out_dir.exists()


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_summary_replace_failure_keeps_previous_generation_reference(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged-v1"
    retry_dir = tmp_path / "merged-v2"
    new_dir.mkdir()
    first = _FakeSystem([[[0.0, 0.0, 0.0]]])
    second = _FakeSystem([[[2.0, 0.0, 0.0]]])
    mock_load_systems.side_effect = [[first], [second], [second]]
    manager = DataIntegrationManager()

    first_summary = manager.integrate(new_labeled_data_path=new_dir, merged_output_path=out_dir)
    old_summary_bytes = (out_dir / "integration_summary.json").read_bytes()
    old_manifest_name = first_summary["dataset_manifest_path"]
    old_root_manifest_bytes = (out_dir / "dataset-manifest.json").read_bytes()

    original_write = manager._write_json_atomic

    def fail_summary(path, payload):
        if path.name == "integration_summary.json":
            raise OSError("injected summary publication failure")
        return original_write(path, payload)

    with patch.object(manager, "_write_json_atomic", side_effect=fail_summary):
        with pytest.raises(OSError, match="summary publication failure"):
            manager.integrate(new_labeled_data_path=new_dir, merged_output_path=retry_dir)

    assert (out_dir / "integration_summary.json").read_bytes() == old_summary_bytes
    assert (out_dir / "dataset-manifest.json").read_bytes() == old_root_manifest_bytes
    assert (out_dir / old_manifest_name).exists()
    assert not retry_dir.exists()

    retried_summary = manager.integrate(new_labeled_data_path=new_dir, merged_output_path=retry_dir)
    assert retried_summary["dataset_manifest_path"] != old_manifest_name
    assert (retry_dir / retried_summary["dataset_manifest_path"]).exists()


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_refuses_existing_output(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    out_dir.mkdir()
    marker = out_dir / "keep.txt"
    marker.write_text("old evidence")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        DataIntegrationManager().integrate(new_labeled_data_path=new_dir, merged_output_path=out_dir)

    mock_load_systems.assert_not_called()
    assert marker.read_text() == "old evidence"


@pytest.mark.parametrize("failure_point", ["export", "manifest", "pointer", "summary"])
@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_integration_manager_publication_failures_leave_no_final_bundle(
    mock_load_systems, failure_point, tmp_path
):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    mock_load_systems.return_value = [_FakeSystem([[[0.0, 0.0, 0.0]]])]
    manager = DataIntegrationManager()

    if failure_point == "export":
        patcher = patch.object(_FakeMultiSystems, "to", side_effect=OSError("export failure"))
    elif failure_point == "manifest":
        patcher = patch.object(
            manager, "_write_immutable_json", side_effect=OSError("manifest failure")
        )
    elif failure_point == "pointer":
        original = manager._write_bytes_atomic

        def fail_pointer(path, payload):
            if path.name == "dataset-manifest.json":
                raise OSError("pointer failure")
            return original(path, payload)

        patcher = patch.object(manager, "_write_bytes_atomic", side_effect=fail_pointer)
    else:
        patcher = patch.object(
            manager, "_write_json_atomic", side_effect=OSError("summary failure")
        )

    with patcher:
        with pytest.raises(OSError, match=f"{failure_point} failure"):
            manager.integrate(new_labeled_data_path=new_dir, merged_output_path=out_dir)

    assert not out_dir.exists()
    assert not list(tmp_path.glob(".merged.staging-*"))


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_competing_final_target_is_never_overwritten(mock_load_systems, tmp_path):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    mock_load_systems.return_value = [_FakeSystem([[[0.0, 0.0, 0.0]]])]
    manager = DataIntegrationManager()
    original_rename = manager._rename_noreplace

    def competitor(source, target):
        target.mkdir()
        (target / "competitor.marker").write_text("keep")
        return original_rename(source, target)

    with patch.object(manager, "_rename_noreplace", side_effect=competitor):
        with pytest.raises(FileExistsError, match="appeared during publication"):
            manager.integrate(new_labeled_data_path=new_dir, merged_output_path=out_dir)

    assert (out_dir / "competitor.marker").read_text() == "keep"
    assert not list(tmp_path.glob(".merged.staging-*"))


@patch("dpeva.labeling.integration.dpdata.MultiSystems", _FakeMultiSystems)
@patch("dpeva.labeling.integration.load_systems")
def test_bundle_publication_does_not_claim_recursive_crash_durability(
    mock_load_systems, tmp_path
):
    new_dir = tmp_path / "new_cleaned"
    out_dir = tmp_path / "merged"
    new_dir.mkdir()
    mock_load_systems.return_value = [_FakeSystem([[[0.0, 0.0, 0.0]]])]
    manager = DataIntegrationManager()

    manager.integrate(new_labeled_data_path=new_dir, merged_output_path=out_dir)
    assert out_dir.is_dir()


def test_rename_noreplace_unsupported_fails_closed_without_fallback(tmp_path):
    source = tmp_path / "staging"
    target = tmp_path / "merged"
    source.mkdir()
    (source / "payload").write_text("source")

    with patch("dpeva.labeling.integration.ctypes.CDLL", return_value=object()):
        with patch("dpeva.labeling.integration.os.rename") as plain_rename:
            with pytest.raises(PublicationError, match="renameat2"):
                DataIntegrationManager._rename_noreplace(source, target)

    plain_rename.assert_not_called()
    assert source.exists()
    assert not target.exists()


class _FakeRenameAt2:
    def __call__(self, *args):
        return -1


class _FakeLibc:
    def __init__(self):
        self.renameat2 = _FakeRenameAt2()


@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EINVAL, errno.ENOTSUP])
def test_rename_noreplace_syscall_failure_fails_closed(error, tmp_path):
    source = tmp_path / "staging"
    target = tmp_path / "merged"
    source.mkdir()
    (source / "payload").write_text("source")

    with patch("dpeva.labeling.integration.ctypes.CDLL", return_value=_FakeLibc()):
        with patch("dpeva.labeling.integration.ctypes.get_errno", return_value=error):
            with patch("dpeva.labeling.integration.os.rename") as plain_rename:
                with pytest.raises(PublicationError, match="renameat2"):
                    DataIntegrationManager._rename_noreplace(source, target)

    plain_rename.assert_not_called()
    assert source.exists()
    assert not target.exists()
