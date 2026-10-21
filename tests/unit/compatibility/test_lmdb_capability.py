"""Capability declarations for the DeepMD LMDB data format."""

import pytest

from dpeva.compatibility import (
    CapabilityKey,
    CapabilityMatrix,
    CapabilityUnavailable,
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


@pytest.mark.parametrize("operation", ["train", "fine-tune"])
def test_lmdb_training_is_experimental_until_evidence_exists(operation: str) -> None:
    matrix = CapabilityMatrix.load_default()
    record = matrix.get(_lmdb_key(operation))

    assert record.status == "experimental"
    assert record.verification_status == "planned"
    with pytest.raises(CapabilityUnavailable, match="experimental"):
        matrix.require(_lmdb_key(operation))
    assert (
        matrix.require(_lmdb_key(operation), allow_experimental=True).status
        == "experimental"
    )


def test_lmdb_test_is_experimental_until_contract_evidence_exists() -> None:
    matrix = CapabilityMatrix.load_default()
    record = matrix.get(_lmdb_key("test"))

    assert record.status == "experimental"
    assert record.verification_status == "planned"
    with pytest.raises(CapabilityUnavailable, match="experimental"):
        matrix.require(_lmdb_key("test"))


@pytest.mark.parametrize("operation", ["eval-desc", "embed"])
def test_lmdb_cli_feature_exporters_are_blocked_upstream(operation: str) -> None:
    matrix = CapabilityMatrix.load_default()
    record = matrix.get(_lmdb_key(operation))

    assert record.status == "blocked-upstream"
    assert record.verification_status == "blocked"
    assert record.upstream_issue == "https://github.com/deepmodeling/deepmd-kit/issues/6034"
    with pytest.raises(CapabilityUnavailable, match="blocked-upstream"):
        matrix.require(_lmdb_key(operation), allow_experimental=True)
