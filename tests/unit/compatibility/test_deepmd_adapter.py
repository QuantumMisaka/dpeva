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
    adapter = DeepMDAdapter("pt", matrix)
    command = adapter.test("model path.pt", "data path", "result prefix")
    assert command == "dp --pt test -s 'data path' -m 'model path.pt' -d 'result prefix'"

