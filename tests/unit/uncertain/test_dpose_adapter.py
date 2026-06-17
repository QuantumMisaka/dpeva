import numpy as np
import pytest

from dpeva.uncertain.dpose import DeepMDTorchDPOSEAdapter, DPOSEEnsemble, resolve_last_layer_weights
from dpeva.uncertain.llpr import LLPRState


class FakeDifferentiableDeepMDAdapter(DeepMDTorchDPOSEAdapter):
    def __init__(self):
        pass

    @property
    def supports_force_dpose(self):
        return True

    def evaluate_last_layer_graph(self, coords):
        features = coords[..., :2]
        base_energy = features.sum(dim=(1, 2))
        return features, base_energy


class FakeEnergyOnlyDeepMDAdapter(DeepMDTorchDPOSEAdapter):
    def __init__(self):
        pass

    @property
    def supports_force_dpose(self):
        return False


def test_dpose_adapter_force_ensemble_matches_autograd_for_fake_model():
    torch = pytest.importorskip("torch")
    coords = torch.tensor(
        [
            [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
        ],
        requires_grad=True,
    )
    state = LLPRState.from_training_features(
        np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        regularizer=1.0,
    )
    dpose = DPOSEEnsemble(
        state=state,
        weights=np.array([1.0, 1.0]),
        n_members=4,
        random_seed=11,
    )

    result = FakeDifferentiableDeepMDAdapter().evaluate_dpose(
        coords,
        dpose,
        targets="energy_force",
    )

    assert result.energy_ensemble.shape == (1, 4)
    assert result.force_ensemble.shape == (1, 4, 2, 3)
    np.testing.assert_allclose(
        result.force_uncertainty,
        np.std(result.force_ensemble, axis=1, ddof=1),
    )


def test_dpose_adapter_rejects_force_for_non_differentiable_model():
    torch = pytest.importorskip("torch")
    state = LLPRState.from_training_features(
        np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        regularizer=1.0,
    )
    dpose = DPOSEEnsemble(
        state=state,
        weights=np.array([1.0, 1.0]),
        n_members=2,
    )

    with pytest.raises(RuntimeError, match="does not support force DPOSE"):
        FakeEnergyOnlyDeepMDAdapter().evaluate_dpose(
            torch.zeros((1, 2, 3)),
            dpose,
            targets="force",
        )


def test_resolve_last_layer_weights_loads_explicit_npy(tmp_path):
    path = tmp_path / "weights.npy"
    np.save(path, np.array([[1.0, 2.0]]))

    weights = resolve_last_layer_weights(
        feature_dimension=2,
        last_layer_weights_path=path,
    )

    np.testing.assert_allclose(weights, np.array([1.0, 2.0]))


def test_resolve_last_layer_weights_matches_checkpoint_by_feature_dimension(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": {
                "fitting.other.weight": torch.ones((4, 5)),
                "fitting.energy.weight": torch.tensor([[0.5, 1.5, 2.5]]),
            }
        },
        path,
    )

    weights = resolve_last_layer_weights(
        feature_dimension=3,
        model_path=path,
        model_head="energy",
    )

    np.testing.assert_allclose(weights, np.array([0.5, 1.5, 2.5]))


def test_resolve_last_layer_weights_falls_back_when_head_key_is_absent(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": {
                "model.Default.atomic_model.descriptor.blocks.0.weight": torch.ones((3, 3)),
                "model.Default.atomic_model.fitting_net.output_layer.matrix": torch.tensor(
                    [[0.5], [1.5], [2.5]]
                ),
            }
        },
        path,
    )

    weights = resolve_last_layer_weights(
        feature_dimension=3,
        model_path=path,
        model_head="RANDOM",
    )

    np.testing.assert_allclose(weights, np.array([0.5, 1.5, 2.5]))


def test_resolve_last_layer_weights_reports_candidates_when_missing(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {"fitting.energy.weight": torch.ones((1, 4))}}, path)

    with pytest.raises(RuntimeError, match=r"candidate 2D weight shapes.*fitting.energy.weight.*\(1, 4\)"):
        resolve_last_layer_weights(
            feature_dimension=3,
            model_path=path,
            model_head="energy",
        )
