"""Contract tests for the packaged DeepMD capability declaration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dpeva.compatibility.deepmd import (
    CapabilityEvidence,
    CapabilityKey,
    CapabilityMatrix,
    CapabilityUnavailable,
)


def _key(**updates: str) -> CapabilityKey:
    values = {
        "operation": "eval-desc",
        "backend": "pt-expt",
        "model_family": "DPA4C",
        "artifact": "exportable",
        "data_format": "deepmd/npy",
        "environment": "cpu-periodic",
    }
    values.update(updates)
    return CapabilityKey(**values)


def test_manifest_uses_only_canonical_states() -> None:
    matrix = CapabilityMatrix.load_default()
    assert len(matrix.records) == 15
    assert {record.status for record in matrix.records} <= {
        "supported",
        "experimental",
        "unsupported",
        "blocked-upstream",
    }


def test_periodic_pt_expt_is_experimental_and_non_pbc_is_blocked() -> None:
    matrix = CapabilityMatrix.load_default()
    periodic = matrix.get(_key())
    assert periodic.status == "experimental"
    assert periodic.version_range == ">=3.2,<3.3"
    with pytest.raises(CapabilityUnavailable, match="blocked-upstream"):
        matrix.require(_key(environment="cpu-non-pbc"))
    with pytest.raises(CapabilityUnavailable, match="blocked-upstream"):
        matrix.require(_key(operation="inference", environment="cpu-non-pbc"))


def test_experimental_requires_explicit_opt_in() -> None:
    matrix = CapabilityMatrix.load_default()
    with pytest.raises(CapabilityUnavailable, match="experimental"):
        matrix.require(_key())
    assert matrix.require(_key(), allow_experimental=True).status == "experimental"


def test_pt_expt_embed_is_unsupported() -> None:
    matrix = CapabilityMatrix.load_default()
    key = _key(operation="embed")
    with pytest.raises(CapabilityUnavailable, match="unsupported"):
        matrix.require(key, allow_experimental=True)


def test_get_requires_one_exact_key() -> None:
    matrix = CapabilityMatrix.load_default()
    with pytest.raises(CapabilityUnavailable, match="no capability record"):
        matrix.get(_key(operation="does-not-exist"))


def test_key_is_strict_and_immutable() -> None:
    with pytest.raises(ValidationError):
        CapabilityKey(
            operation=1,
            backend="pt",
            model_family="DPA4",
            artifact="checkpoint",
            data_format="deepmd/npy",
            environment="cpu",
        )
    key = _key()
    with pytest.raises(ValidationError):
        key.operation = "train"
    assert hash(key)


def test_record_and_evidence_are_immutable_and_hashable() -> None:
    record = CapabilityMatrix.load_default().get(_key())
    with pytest.raises(ValidationError):
        record.status = "supported"
    assert hash(record)
    evidence = CapabilityEvidence(cpu_contract="tests/contract/deepmd/test.py")
    with pytest.raises(ValidationError):
        evidence.cpu_contract = "tampered"
    assert hash(evidence)


def test_candidate_evaluation_declares_both_model_roles() -> None:
    record = next(
        item
        for item in CapabilityMatrix.load_default().records
        if item.key.operation == "candidate-evaluation"
    )
    assert record.covered_roles == ("regular", "ema")
    with pytest.raises(TypeError):
        record.covered_roles[0] = "ema"  # type: ignore[index]


def test_malformed_manifest_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(
        Path("src/dpeva/compatibility/deepmd-3.2.json").read_text(encoding="utf-8")
    )
    payload["records"][0]["unknown"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        CapabilityMatrix.load(path)


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    payload = json.loads(
        Path("src/dpeva/compatibility/deepmd-3.2.json").read_text(encoding="utf-8")
    )
    payload["records"].append(payload["records"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate capability key"):
        CapabilityMatrix.load(path)


def test_resource_load_works_outside_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    matrix = CapabilityMatrix.load_default()
    assert matrix.records


def test_manifest_has_no_unapproved_status_or_issue() -> None:
    records = CapabilityMatrix.load_default().records
    non_pbc = [record for record in records if record.key.environment == "cpu-non-pbc"]
    assert {record.key.operation for record in non_pbc} == {"eval-desc", "inference"}
    assert {record.status for record in non_pbc} == {"blocked-upstream"}
    assert {
        record.upstream_issue
        for record in non_pbc
    } == {"https://github.com/deepmodeling/deepmd-kit/issues/6002"}


def test_new_routes_are_explicit_policy_unsupported_records() -> None:
    """These rows are policy declarations, not executable backend routing."""

    records = CapabilityMatrix.load_default().records
    policy = {
        ("train", "dpa-adapt", "DPA4"),
        ("train", "jax", "DPA4"),
        ("train", "tf2", "DPA4"),
    }
    observed = {
        (item.key.operation, item.key.backend, item.key.model_family): item.status
        for item in records
        if (item.key.operation, item.key.backend, item.key.model_family) in policy
    }
    assert set(observed) == policy
    assert set(observed.values()) == {"unsupported"}
    assert "supported" not in {item.status for item in records}


def test_unknown_status_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(
        Path("src/dpeva/compatibility/deepmd-3.2.json").read_text(encoding="utf-8")
    )
    payload["records"][0]["status"] = "maybe"
    path = tmp_path / "unknown-status.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        CapabilityMatrix.load(path)
