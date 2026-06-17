from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from dpeva.uncertain.llpr import LLPRState


DPOSETargets = Literal["energy", "force", "energy_force"]


@dataclass(frozen=True)
class DPOSEPrediction:
    energy_ensemble: np.ndarray | None = None
    energy_uncertainty: np.ndarray | None = None
    force_ensemble: np.ndarray | None = None
    force_uncertainty: np.ndarray | None = None
    force_uncertainty_max: np.ndarray | None = None


@dataclass(frozen=True)
class DPOSEEnsemble:
    """Last-layer shallow ensemble sampled from an LLPR posterior."""

    state: LLPRState
    weights: np.ndarray
    n_members: int
    random_seed: int | None = None

    def sampled_weights(self) -> np.ndarray:
        if self.n_members < 2:
            raise ValueError("n_members must be at least 2.")
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if weights.shape[0] != self.state.feature_dimension:
            raise ValueError(
                "last-layer weights feature dimension does not match LLPR state: "
                f"weights={weights.shape[0]}, state={self.state.feature_dimension}."
            )
        covariance = (np.asarray(self.state.alpha, dtype=float) ** 2) * self.state.inverse_covariance
        rng = np.random.default_rng(self.random_seed)
        displacements = rng.multivariate_normal(
            np.zeros(covariance.shape[0]),
            0.5 * (covariance + covariance.T),
            size=self.n_members,
        )
        return weights[None, :] + displacements

    def sample_energy_ensemble(self, features: np.ndarray, mean_energy: np.ndarray) -> np.ndarray:
        frame_features = np.asarray(features, dtype=float)
        if frame_features.ndim == 3:
            if self.state.feature_normalization == "mean":
                frame_features = frame_features.mean(axis=1)
            else:
                frame_features = frame_features.sum(axis=1)
        mean_energy = np.asarray(mean_energy, dtype=float).reshape(-1)
        if frame_features.shape[0] != mean_energy.shape[0]:
            raise ValueError("mean_energy length must match number of feature frames.")
        raw = frame_features @ self.sampled_weights().T
        return raw - raw.mean(axis=1, keepdims=True) + mean_energy[:, None]


class DeepMDTorchDPOSEAdapter:
    """
    Thin adapter contract for differentiable DeepMD DPOSE.

    The public DeepPot.eval_fitting_last_layer API detaches middle outputs, so
    force-level DPOSE must use a subclass that evaluates the PyTorch graph and
    returns differentiable last-layer features.
    """

    @property
    def supports_force_dpose(self) -> bool:
        return False

    def evaluate_last_layer_graph(self, coords):
        raise NotImplementedError

    def evaluate_dpose(
        self,
        coords,
        ensemble: DPOSEEnsemble,
        targets: DPOSETargets = "energy",
    ) -> DPOSEPrediction:
        needs_force = targets in {"force", "energy_force"}
        if needs_force and not self.supports_force_dpose:
            raise RuntimeError(
                "This DeepMD model does not support force DPOSE. "
                "Use llpr_targets='energy' or a differentiable energy-gradient model."
            )

        torch = _require_torch()
        features, base_energy = self.evaluate_last_layer_graph(coords)
        sampled_weights = torch.as_tensor(
            ensemble.sampled_weights(),
            dtype=features.dtype,
            device=features.device,
        )
        frame_features = features.sum(dim=1)
        raw_energy = frame_features @ sampled_weights.T
        energy_ensemble = raw_energy - raw_energy.mean(dim=1, keepdim=True) + base_energy[:, None]
        energy_np = energy_ensemble.detach().cpu().numpy()
        energy_unc = energy_np.std(axis=1, ddof=1)

        if not needs_force:
            return DPOSEPrediction(
                energy_ensemble=energy_np,
                energy_uncertainty=energy_unc,
            )

        force_members = []
        for member in range(energy_ensemble.shape[1]):
            grad = torch.autograd.grad(
                energy_ensemble[:, member].sum(),
                coords,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            force_members.append((-grad).detach().cpu().numpy())
        force_ensemble = np.stack(force_members, axis=1)
        force_unc = force_ensemble.std(axis=1, ddof=1)
        return DPOSEPrediction(
            energy_ensemble=energy_np,
            energy_uncertainty=energy_unc,
            force_ensemble=force_ensemble,
            force_uncertainty=force_unc,
            force_uncertainty_max=np.linalg.norm(force_unc, axis=-1).max(axis=1),
        )


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for force-level DPOSE.") from exc
    return torch


def resolve_last_layer_weights(
    feature_dimension: int,
    last_layer_weights_path: str | Path | None = None,
    model_path: str | Path | None = None,
    model_head: str | None = None,
) -> np.ndarray:
    """Resolve a DeepMD energy fitting last-layer weight vector."""
    if last_layer_weights_path is not None:
        weights = np.asarray(np.load(last_layer_weights_path), dtype=float)
        return _coerce_weight_vector(weights, feature_dimension, str(last_layer_weights_path))
    if model_path is None:
        raise RuntimeError(
            "DPOSE energy_ensemble requires llpr_last_layer_weights_path or llpr_model_path."
        )

    torch = _require_torch()
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    candidates: list[tuple[str, tuple[int, ...], np.ndarray]] = []
    head_matches: list[tuple[str, np.ndarray]] = []
    fallback_matches: list[tuple[str, np.ndarray]] = []
    for key, value in state_dict.items():
        if hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if arr.ndim != 2:
            continue
        candidates.append((key, tuple(arr.shape), arr))
        try:
            match = (key, _coerce_weight_vector(arr, feature_dimension, key))
        except ValueError:
            continue
        fallback_matches.append(match)
        if model_head and model_head in key:
            head_matches.append(match)

    matches = head_matches if head_matches else fallback_matches
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        keys = ", ".join(key for key, _ in matches)
        raise RuntimeError(
            "Multiple last-layer weight candidates match feature dimension "
            f"{feature_dimension}: {keys}. Set llpr_last_layer_weights_path."
        )

    candidate_text = ", ".join(f"{key}: {shape}" for key, shape, _ in candidates) or "none"
    raise RuntimeError(
        "Could not resolve DeepMD energy last-layer weights for feature dimension "
        f"{feature_dimension}; candidate 2D weight shapes: {candidate_text}."
    )


def _coerce_weight_vector(weights: np.ndarray, feature_dimension: int, source: str) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == feature_dimension:
        return arr
    if arr.ndim == 2:
        if arr.shape == (1, feature_dimension):
            return arr.reshape(feature_dimension)
        if arr.shape == (feature_dimension, 1):
            return arr.reshape(feature_dimension)
    raise ValueError(
        f"Last-layer weights from {source} must be shape ({feature_dimension},), "
        f"(1, {feature_dimension}), or ({feature_dimension}, 1); got {arr.shape}."
    )


def _extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise RuntimeError("Unsupported checkpoint format for last-layer weight resolution.")
