"""Capability declarations for the DeepMD LMDB data format."""

from pathlib import Path

import pytest

from dpeva.compatibility import (
    CapabilityKey,
    CapabilityMatrix,
    CapabilityUnavailable,
    validate_promotion_evidence,
)


def _lmdb_key(operation: str, **updates: str) -> CapabilityKey:
    values = {
        "operation": operation,
        "backend": "pt",
        "model_family": "DPA4",
        "artifact": "checkpoint",
        "data_format": "deepmd/lmdb",
        "environment": "cpu",
    }
    values.update(updates)
    return CapabilityKey(**values)


def test_lmdb_train_has_cpu_contract_but_remains_experimental() -> None:
    matrix = CapabilityMatrix.load_default()
    record = matrix.get(_lmdb_key("train"))

    assert record.status == "experimental"
    assert record.verification_status == "implemented"
    assert record.verification_command is not None
    assert record.evidence_ref is not None
    assert record.evidence_ref.cpu_contract.endswith("pt-train-lmdb.json")
    assert validate_promotion_evidence(record, Path.cwd())
    with pytest.raises(CapabilityUnavailable, match="experimental"):
        matrix.require(_lmdb_key("train"))
    assert (
        matrix.require(_lmdb_key("train"), allow_experimental=True).status
        == "experimental"
    )


def test_lmdb_fine_tune_remains_planned() -> None:
    matrix = CapabilityMatrix.load_default()
    record = matrix.get(_lmdb_key("fine-tune"))

    assert record.status == "experimental"
    assert record.verification_status == "planned"
    with pytest.raises(CapabilityUnavailable, match="experimental"):
        matrix.require(_lmdb_key("fine-tune"))


def test_lmdb_test_is_promoted_by_cpu_and_sai_evidence() -> None:
    matrix = CapabilityMatrix.load_default()
    record = matrix.get(_lmdb_key("test"))

    assert record.status == "supported"
    assert record.verification_status == "implemented"
    assert record.verification_command is not None
    assert record.evidence_ref is not None
    assert record.evidence_ref.cpu_contract.endswith("pt-test-lmdb.json")
    assert record.evidence_ref.sai_qualification is not None
    assert record.evidence_ref.sai_qualification.endswith(
        "sai-lmdb-ab-job-1893546-attestation.json"
    )
    assert validate_promotion_evidence(record, Path.cwd())
    assert matrix.require(_lmdb_key("test")).status == "supported"


@pytest.mark.parametrize("operation", ["eval-desc", "embed"])
def test_lmdb_cli_feature_exporters_are_blocked_upstream(operation: str) -> None:
    matrix = CapabilityMatrix.load_default()
    record = matrix.get(_lmdb_key(operation))

    assert record.status == "blocked-upstream"
    assert record.verification_status == "blocked"
    assert record.upstream_issue == "https://github.com/deepmodeling/deepmd-kit/issues/6034"
    with pytest.raises(CapabilityUnavailable, match="blocked-upstream"):
        matrix.require(_lmdb_key(operation), allow_experimental=True)
