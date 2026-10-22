"""Real DeepMD-kit 3.2.0 PT CPU CLI contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from dpeva.compatibility import (
    CapabilityEvidence,
    CapabilityKey,
    CapabilityMatrix,
    CapabilityUnavailable,
    validate_promotion_evidence,
)
from dpeva.compatibility.adapter import DeepMDAdapter
from dpeva.io.dataproc import DPTestResultParser
from dpeva.io.lmdb_index import read_lmdb_frame_index

from conftest import classify_contract_result, frame_count, head_args, run_contract, write_cpu_attestation


def _numeric_table(path: Path) -> np.ndarray:
    table = np.loadtxt(path)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    assert table.ndim == 2
    assert table.shape[1] >= 2
    assert np.isfinite(table).all()
    return table


def _compact_dpa4_train_config(
    root: Path, lmdb_path: Path, type_map: list[str]
) -> dict[str, object]:
    """Build a two-step DPA4 training contract with explicit LMDB inputs."""

    return {
        "_comment": "DP-EVA LMDB CPU contract fixture; execution only.",
        "model": {
            "type": "dpa4",
            "type_map": type_map,
            "descriptor": {
                "type": "dpa4",
                "sel": 120,
                "rcut": 6.0,
                "env_exp": [7, 5],
                "channels": 16,
                "n_radial": 6,
                "radial_mlp": [0],
                "use_env_seed": True,
                "random_gamma": True,
                "l_schedule": [1, 1],
                "mmax": 1,
                "so2_norm": False,
                "so2_layers": 2,
                "so2_attn_res": "none",
                "n_focus": 1,
                "focus_dim": 0,
                "n_atten_head": 1,
                "ffn_neurons": 0,
                "grid_mlp": False,
                "ffn_blocks": 1,
                "sandwich_norm": [True, False, True, False],
                "mlp_bias": False,
                "layer_scale": False,
                "full_attn_res": "none",
                "block_attn_res": "none",
                "s2_activation": [False, True],
                "activation_function": "silu",
                "glu_activation": True,
                "use_amp": False,
                "precision": "float32",
                "seed": 42,
            },
            "fitting_net": {
                "neuron": [0],
                "activation_function": "silu",
                "precision": "float32",
                "seed": 42,
            },
            "use_compile": False,
            "enable_tf32": False,
        },
        "learning_rate": {
            "type": "wsd",
            "start_lr": 0.0005,
            "stop_lr": 1e-06,
            "warmup_steps": 1,
            "warmup_start_factor": 0.2,
            "decay_phase_ratio": 0.65,
            "decay_type": "cosine",
        },
        "loss": {
            "type": "ener",
            "loss_func": "mae",
            "f_use_norm": True,
            "start_pref_e": 20,
            "limit_pref_e": 20,
            "start_pref_f": 20,
            "limit_pref_f": 20,
            "start_pref_v": 0,
            "limit_pref_v": 0,
        },
        "optimizer": {
            "type": "HybridMuon",
            "muon_mode": "slice",
            "enable_gram": True,
            "magma_muon": True,
            "lr_adjust": 0.0,
            "weight_decay": 0.001,
        },
        "training": {
            "training_data": {
                "systems": str(lmdb_path),
                "batch_size": 1,
            },
            "validation_data": {
                "systems": str(lmdb_path),
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": 2,
            "gradient_max_norm": 5.0,
            "save_freq": 1,
            "max_ckpt_keep": 1,
            "enable_ema": False,
            "disp_file": str(root / "lcurve.out"),
            "disp_freq": 1,
            "disp_avg": True,
            "disp_training": True,
            "time_training": True,
            "save_dir": str(root / "models"),
            "seed": 7,
        },
        "validating": {
            "full_validation": False,
            "ema_full_validation": False,
            "validation_freq": 1,
        },
    }


@pytest.mark.deepmd_contract
def test_deepmd_version_is_exact_stable_release(dp_executable: str) -> None:
    result = run_contract([dp_executable, "--version"], [])
    assert result.stdout.strip() == "DeePMD-kit v3.2.0"


@pytest.mark.deepmd_contract
def test_pt_test_requires_numeric_output(
    dp_executable: str, pt_model: Path, pt_head: str | None, periodic_data: Path, tmp_path: Path
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
            *head_args(pt_head),
            "-d",
            str(prefix),
        ],
        [energy_output],
    )
    assert result.returncode == 0
    table = _numeric_table(energy_output)
    assert table.shape[0] == frame_count(periodic_data)
    attestation_path = write_cpu_attestation("test", dp_executable, result, case="pt-test")
    manifest_record = next(
        record for record in CapabilityMatrix.load_default().records
        if record.key.operation == "test"
        and record.key.backend == "pt"
        and record.key.data_format == "deepmd/npy"
    )
    promoted_shape = manifest_record.model_copy(update={
        "status": "supported",
        "required_evidence": ("cpu-contract",),
        "sai_verification_cases": None,
        "evidence_ref": CapabilityEvidence(cpu_contract=str(attestation_path)),
    })
    assert validate_promotion_evidence(promoted_shape, Path.cwd())


@pytest.mark.deepmd_contract
def test_pt_test_reads_lmdb_and_attributes_detail(
    dp_executable: str,
    pt_model: Path,
    pt_head: str | None,
    lmdb_periodic_data: Path,
    tmp_path: Path,
) -> None:
    index = read_lmdb_frame_index(lmdb_periodic_data)
    prefix = tmp_path / "results"
    energy_output = prefix.with_suffix(".e_peratom.out")
    result = run_contract(
        [
            dp_executable,
            "--pt",
            "test",
            "-s",
            str(lmdb_periodic_data),
            "-m",
            str(pt_model),
            *head_args(pt_head),
            "-d",
            str(prefix),
        ],
        [energy_output],
    )
    table = _numeric_table(energy_output)
    assert table.shape[0] == len(index.frames)
    parsed = DPTestResultParser(
        str(tmp_path),
        head="results",
        testdata_dir=str(lmdb_periodic_data),
    ).parse()
    assert parsed["dataname_list"] == index.dataname_list
    assert parsed["datanames_nframe"] == index.datanames_nframe
    attestation_path = write_cpu_attestation(
        "test",
        dp_executable,
        result,
        case="pt-test-lmdb",
        data_format="deepmd/lmdb",
    )
    manifest_record = next(
        record
        for record in CapabilityMatrix.load_default().records
        if record.key.operation == "test"
        and record.key.data_format == "deepmd/lmdb"
    )
    promoted_shape = manifest_record.model_copy(
        update={
            "status": "supported",
            "required_evidence": ("cpu-contract",),
            "sai_verification_cases": None,
            "evidence_ref": CapabilityEvidence(cpu_contract=str(attestation_path)),
        }
    )
    assert validate_promotion_evidence(promoted_shape, Path.cwd())


@pytest.mark.deepmd_contract
def test_pt_train_consumes_lmdb(
    dp_executable: str,
    lmdb_periodic_data: Path,
    periodic_type_map: list[str],
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "input.json"
    config_path.write_text(
        json.dumps(
            _compact_dpa4_train_config(
                tmp_path, lmdb_periodic_data, periodic_type_map
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "models" / "model.ckpt-2.pt"
    result = run_contract(
        [
            dp_executable,
            "--pt",
            "train",
            str(config_path),
        ],
        [checkpoint],
    )
    assert checkpoint.is_file()
    assert (tmp_path / "lcurve.out").is_file()
    write_cpu_attestation(
        "train",
        dp_executable,
        result,
        case="pt-train-lmdb",
        data_format="deepmd/lmdb",
    )


@pytest.mark.deepmd_contract
def test_pt_eval_desc_has_one_descriptor_per_frame(
    dp_executable: str, pt_model: Path, pt_head: str | None, periodic_data: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "descriptors"
    result = run_contract(
        [
            dp_executable,
            "--pt",
            "eval-desc",
            "-s",
            str(periodic_data),
            "-m",
            str(pt_model),
            *head_args(pt_head),
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
    write_cpu_attestation("eval-desc", dp_executable, result, case="pt-eval-desc")


@pytest.mark.deepmd_contract
def test_pt_embed_has_required_hdf5_datasets(
    dp_executable: str, pt_model: Path, pt_head: str | None, periodic_data: Path, tmp_path: Path
) -> None:
    output = tmp_path / "embedding.hdf5"
    result = run_contract(
        [
            dp_executable,
            "--pt",
            "embed",
            "-s",
            str(periodic_data),
            "-m",
            str(pt_model),
            *head_args(pt_head),
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
    write_cpu_attestation("embed", dp_executable, result, case="pt-embed")


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
