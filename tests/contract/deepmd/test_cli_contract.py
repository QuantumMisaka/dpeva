"""Real DeepMD-kit 3.2.0 PT CPU CLI contracts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from dpeva.compatibility import CapabilityKey, CapabilityMatrix, CapabilityUnavailable
from dpeva.compatibility.adapter import DeepMDAdapter

from conftest import classify_contract_result, frame_count, run_contract


def _numeric_table(path: Path) -> np.ndarray:
    table = np.loadtxt(path)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    assert table.ndim == 2
    assert table.shape[1] >= 2
    assert np.isfinite(table).all()
    return table


@pytest.mark.deepmd_contract
def test_deepmd_version_is_exact_stable_release(dp_executable: str) -> None:
    result = run_contract([dp_executable, "--version"], [])
    assert result.stdout.strip() == "DeePMD-kit v3.2.0"


@pytest.mark.deepmd_contract
def test_pt_test_requires_numeric_output(
    dp_executable: str, pt_model: Path, periodic_data: Path, tmp_path: Path
) -> None:
    prefix = tmp_path / "results"
    energy_output = prefix.with_suffix(".e.out")
    result = run_contract(
        [
            dp_executable,
            "--pt",
            "test",
            "-s",
            str(periodic_data),
            "-m",
            str(pt_model),
            "-d",
            str(prefix),
        ],
        [energy_output],
    )
    assert result.returncode == 0
    table = _numeric_table(energy_output)
    assert table.shape[0] == frame_count(periodic_data)


@pytest.mark.deepmd_contract
def test_pt_eval_desc_has_one_descriptor_per_frame(
    dp_executable: str, pt_model: Path, periodic_data: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "descriptors"
    run_contract(
        [
            dp_executable,
            "--pt",
            "eval-desc",
            "-s",
            str(periodic_data),
            "-m",
            str(pt_model),
            "-o",
            str(output_dir),
        ],
        [output_dir],
    )
    files = sorted(output_dir.rglob("*.npy"))
    assert files
    arrays = [np.load(path) for path in files]
    assert all(array.ndim == 3 for array in arrays)
    assert all(np.isfinite(array).all() for array in arrays)
    assert sum(array.shape[0] for array in arrays) == frame_count(periodic_data)


@pytest.mark.deepmd_contract
def test_pt_embed_has_required_hdf5_datasets(
    dp_executable: str, pt_model: Path, periodic_data: Path, tmp_path: Path
) -> None:
    output = tmp_path / "embedding.hdf5"
    run_contract(
        [
            dp_executable,
            "--pt",
            "embed",
            "-s",
            str(periodic_data),
            "-m",
            str(pt_model),
            "-o",
            str(output),
        ],
        [output],
    )
    required = {"descriptor", "atomic_feature", "structural_feature", "atom_types"}
    with h5py.File(output, "r") as handle:
        groups: list[h5py.Group] = []

        def collect(_name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Group):
                groups.append(obj)

        handle.visititems(collect)
        assert groups
        data_groups = [group for group in groups if required <= set(group.keys())]
        assert data_groups
        for group in data_groups:
            descriptor = np.asarray(group["descriptor"])
            atomic = np.asarray(group["atomic_feature"])
            structural = np.asarray(group["structural_feature"])
            atom_types = np.asarray(group["atom_types"])
            assert descriptor.ndim >= 3
            assert atomic.ndim >= 3
            assert structural.ndim >= 2
            assert atom_types.ndim == 2
            assert descriptor.shape[0] == structural.shape[0]
            assert atom_types.shape[0] == descriptor.shape[0]
            assert descriptor.size > 0
            assert atomic.size > 0
            assert structural.size > 0
            assert atom_types.size > 0
            assert np.isfinite(descriptor).all()
            assert np.isfinite(atomic).all()
            assert np.isfinite(structural).all()


@pytest.mark.deepmd_contract
def test_contract_failure_categories_are_fail_closed(tmp_path: Path) -> None:
    zero_exit = subprocess.CompletedProcess(["fake"], 0, "", "")
    nonzero = subprocess.CompletedProcess(["fake"], 7, "", "failure")
    assert classify_contract_result(zero_exit, [tmp_path / "missing.out"]) == "ARTIFACT"
    assert classify_contract_result(nonzero, []) == "EXECUTION"


@pytest.mark.deepmd_contract
def test_pt_expt_embed_is_rejected_before_subprocess_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = DeepMDAdapter(
        "pt-expt", CapabilityMatrix.load_default(), allow_experimental=True
    )
    key = CapabilityKey(
        operation="embed",
        backend="pt-expt",
        model_family="DPA4C",
        artifact="exportable",
        data_format="deepmd/npy",
        environment="cpu-periodic",
    )
    with pytest.raises(CapabilityUnavailable, match="unsupported"):
        command = adapter.embed(
            "model.pt2", str(tmp_path / "data"), str(tmp_path / "embedding.hdf5"), capability_key=key
        )
        subprocess.run(command.split(), check=False, capture_output=True, text=True)
    assert calls == []
