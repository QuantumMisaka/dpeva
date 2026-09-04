"""Periodic DPA4C ``pt-expt eval-desc`` contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dpeva.compatibility import CapabilityKey, CapabilityMatrix
from dpeva.compatibility.adapter import DeepMDAdapter
from dpeva.io.collection import CollectionIOManager

from conftest import frame_count, run_contract, write_cpu_attestation


@pytest.mark.deepmd_contract
def test_periodic_pt_expt_eval_desc_is_consumable_by_collection(
    dp_executable: str, dpa4c_model: Path, periodic_data: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "dpa4c-descriptors"
    key = CapabilityKey(
        operation="eval-desc",
        backend="pt-expt",
        model_family="DPA4C",
        artifact="exportable",
        data_format="deepmd/npy",
        environment="cpu-periodic",
    )
    adapter = DeepMDAdapter(
        "pt-expt", CapabilityMatrix.load_default(), allow_experimental=True
    )
    adapter.preflight(key)
    result = run_contract(
        [
            dp_executable,
            "--pt-expt",
            "eval-desc",
            "-s",
            str(periodic_data),
            "-m",
            str(dpa4c_model),
            "-o",
            str(output_dir),
        ],
        [output_dir],
    )
    names, descriptors = CollectionIOManager(str(tmp_path), ".").load_descriptors(
        str(output_dir)
    )
    assert names
    assert descriptors.ndim == 2
    assert descriptors.shape[0] == frame_count(periodic_data)
    assert np.isfinite(descriptors).all()
    write_cpu_attestation("eval-desc", dp_executable, result, case="dpa4c-periodic-eval-desc")
