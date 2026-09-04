from dataclasses import FrozenInstanceError

import pytest

from dpeva.compatibility import CapabilityKey, CapabilityMatrix, CapabilityUnavailable
from dpeva.compatibility.adapter import DeepMDAdapter


@pytest.fixture
def matrix() -> CapabilityMatrix:
    return CapabilityMatrix.load_default()


def test_adapter_backend_state_is_instance_local(matrix: CapabilityMatrix) -> None:
    pt = DeepMDAdapter("pt", matrix, allow_experimental=True)
    expt = DeepMDAdapter("pt-expt", matrix, allow_experimental=True)

    assert pt.base_command == ("dp", "--pt")
    assert expt.base_command == ("dp", "--pt-expt")
    assert pt.base_command == ("dp", "--pt")


def test_adapter_is_immutable(matrix: CapabilityMatrix) -> None:
    adapter = DeepMDAdapter("pt", matrix)
    with pytest.raises(FrozenInstanceError):
        adapter.backend = "tf"  # type: ignore[misc]


def test_preflight_rejects_mismatched_backend(matrix: CapabilityMatrix) -> None:
    adapter = DeepMDAdapter("pt", matrix)
    key = CapabilityKey(
        operation="eval-desc",
        backend="pt-expt",
        model_family="DPA4C",
        artifact="exportable",
        data_format="deepmd/npy",
        environment="cpu-periodic",
    )
    with pytest.raises(CapabilityUnavailable, match="does not match"):
        adapter.preflight(key)


def test_pt_expt_embed_is_rejected_before_command_build(matrix: CapabilityMatrix) -> None:
    adapter = DeepMDAdapter("pt-expt", matrix)
    key = CapabilityKey(
        operation="embed",
        backend="pt-expt",
        model_family="DPA4C",
        artifact="exportable",
        data_format="deepmd/npy",
        environment="cpu-periodic",
    )
    with pytest.raises(CapabilityUnavailable):
        adapter.preflight(key)


def test_adapter_preserves_command_quoting(matrix: CapabilityMatrix) -> None:
    adapter = DeepMDAdapter.for_legacy_unchecked("pt", matrix)
    command = adapter.test("model path.pt", "data path", "result prefix")
    assert command == "dp --pt test -s 'data path' -m 'model path.pt' -d 'result prefix'"


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


def test_strict_command_requires_complete_capability_key(matrix: CapabilityMatrix) -> None:
    with pytest.raises(CapabilityUnavailable, match="capability_key"):
        DeepMDAdapter("pt-expt", matrix).embed("model.pt", "data", "out.hdf5")


def test_strict_command_rejects_unsupported_capability(matrix: CapabilityMatrix) -> None:
    adapter = DeepMDAdapter("pt-expt", matrix, allow_experimental=True)
    with pytest.raises(CapabilityUnavailable, match="unsupported"):
        adapter.embed("model.pt", "data", "out.hdf5", capability_key=_key(operation="embed"))


def test_strict_command_rejects_operation_mismatch(matrix: CapabilityMatrix) -> None:
    adapter = DeepMDAdapter("pt-expt", matrix, allow_experimental=True)
    with pytest.raises(CapabilityUnavailable, match="operation"):
        adapter.train("input.json", capability_key=_key())


def test_strict_command_allows_explicit_experimental_capability(matrix: CapabilityMatrix) -> None:
    adapter = DeepMDAdapter("pt-expt", matrix, allow_experimental=True)
    command = adapter.eval_desc("model.pt", "data", "desc", capability_key=_key())
    assert command.startswith("dp --pt-expt eval-desc")


def test_legacy_adapter_is_explicit_unchecked_bridge(matrix: CapabilityMatrix) -> None:
    adapter = DeepMDAdapter.for_legacy_unchecked("pt-expt", matrix)
    assert adapter.embed("model.pt", "data", "out.hdf5").startswith("dp --pt-expt embed")
