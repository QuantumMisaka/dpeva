from __future__ import annotations

from dataclasses import dataclass
from math import erf, pi, sqrt
from pathlib import Path
from typing import Literal

import numpy as np


CalibrationMethod = Literal["squared_residuals", "absolute_residuals", "crps"]
FeatureNormalization = Literal["sum", "mean", "none"]


@dataclass(frozen=True)
class LLPRPrediction:
    """Frame-level LLPR uncertainty values."""

    total: np.ndarray
    per_atom: np.ndarray
    alpha: float | np.ndarray
    calibrated: bool


@dataclass(frozen=True)
class LLPRState:
    """Reusable LLPR covariance state for DeepMD last-layer features."""

    covariance: np.ndarray
    cholesky: np.ndarray
    regularizer: float
    alpha: float | np.ndarray = 1.0
    calibrated: bool = False
    feature_normalization: FeatureNormalization = "sum"

    @classmethod
    def from_training_features(
        cls,
        features: np.ndarray,
        regularizer: float = 1e-8,
        alpha: float | np.ndarray = 1.0,
        calibrated: bool | None = None,
        feature_normalization: FeatureNormalization = "sum",
    ) -> "LLPRState":
        frame_features = _frame_features(features, feature_normalization)
        if frame_features.size == 0:
            raise ValueError("LLPR training features cannot be empty.")
        if regularizer <= 0:
            raise ValueError("regularizer must be positive.")
        covariance = frame_features.T @ frame_features
        covariance = covariance + regularizer * np.eye(covariance.shape[0])
        cholesky = np.linalg.cholesky(_symmetrize(covariance))
        return cls(
            covariance=covariance,
            cholesky=cholesky,
            regularizer=regularizer,
            alpha=_normalize_alpha(alpha),
            calibrated=_is_calibrated(alpha) if calibrated is None else calibrated,
            feature_normalization=feature_normalization,
        )

    def predict_uncertainty(self, features: np.ndarray) -> LLPRPrediction:
        frame_features = _frame_features(features, self.feature_normalization)
        solved = np.linalg.solve(self.cholesky, frame_features.T)
        variance = np.sum(solved * solved, axis=0)
        raw_total = np.sqrt(np.maximum(variance, 0.0))
        total = _apply_alpha(raw_total, self.alpha)
        natoms = _natoms_per_frame(features)
        return LLPRPrediction(
            total=total,
            per_atom=total / natoms,
            alpha=self.alpha,
            calibrated=self.calibrated,
        )

    @property
    def feature_dimension(self) -> int:
        return int(self.covariance.shape[0])

    @property
    def inverse_covariance(self) -> np.ndarray:
        identity = np.eye(self.covariance.shape[0])
        y = np.linalg.solve(self.cholesky, identity)
        return np.linalg.solve(self.cholesky.T, y)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            covariance=self.covariance,
            cholesky=self.cholesky,
            regularizer=np.asarray(self.regularizer, dtype=float),
            alpha=np.asarray(self.alpha, dtype=float),
            calibrated=np.asarray(self.calibrated, dtype=bool),
            feature_normalization=np.asarray(self.feature_normalization),
            feature_dimension=np.asarray(self.feature_dimension, dtype=int),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "LLPRState":
        with np.load(path, allow_pickle=False) as data:
            covariance = np.asarray(data["covariance"], dtype=float)
            cholesky = np.asarray(data["cholesky"], dtype=float)
            feature_dimension = int(np.asarray(data["feature_dimension"]).item())
            if covariance.shape != (feature_dimension, feature_dimension):
                raise ValueError(
                    "LLPR state covariance shape does not match stored feature_dimension."
                )
            return cls(
                covariance=covariance,
                cholesky=cholesky,
                regularizer=float(np.asarray(data["regularizer"]).item()),
                alpha=_normalize_alpha(data["alpha"]),
                calibrated=bool(np.asarray(data["calibrated"]).item()),
                feature_normalization=str(np.asarray(data["feature_normalization"]).item()),
            )


class LLPRCalibrator:
    """Global post-hoc scale calibration for LLPR uncertainties."""

    def __init__(self, method: CalibrationMethod = "squared_residuals") -> None:
        self.method = method

    def fit_alpha(self, residuals: np.ndarray, uncertainties: np.ndarray) -> float:
        residuals = np.asarray(residuals, dtype=float)
        uncertainties = np.asarray(uncertainties, dtype=float)
        if residuals.shape != uncertainties.shape:
            raise ValueError("residuals and uncertainties must have the same shape.")
        if residuals.ndim == 0:
            residuals = residuals.reshape(1, 1)
            uncertainties = uncertainties.reshape(1, 1)
        elif residuals.ndim == 1:
            residuals = residuals.reshape(-1, 1)
            uncertainties = uncertainties.reshape(-1, 1)
        else:
            residuals = residuals.reshape(-1, residuals.shape[-1])
            uncertainties = uncertainties.reshape(-1, uncertainties.shape[-1])

        mask = np.isfinite(residuals) & np.isfinite(uncertainties) & (uncertainties > 0)
        if not np.any(mask):
            raise ValueError("No finite positive uncertainties available for calibration.")

        alpha_values = []
        for channel in range(residuals.shape[-1]):
            channel_mask = mask[:, channel]
            if not np.any(channel_mask):
                alpha_values.append(np.nan)
                continue
            r = residuals[channel_mask, channel]
            u = uncertainties[channel_mask, channel]
            alpha_values.append(self._fit_channel_alpha(r, u))
        alpha = np.asarray(alpha_values, dtype=float)
        if np.any(~np.isfinite(alpha)):
            raise ValueError("No finite positive uncertainties available for one or more channels.")
        return float(alpha[0]) if alpha.shape == (1,) else alpha

    def _fit_channel_alpha(self, residuals: np.ndarray, uncertainties: np.ndarray) -> float:
        if self.method == "squared_residuals":
            return float(np.sqrt(np.mean((residuals / uncertainties) ** 2)))
        if self.method == "absolute_residuals":
            return float(np.mean(np.abs(residuals) / uncertainties) * np.sqrt(np.pi / 2.0))
        if self.method == "crps":
            return _fit_crps_alpha(residuals, uncertainties)
        raise ValueError(f"Unsupported LLPR calibration method: {self.method}")


class ShallowEnsembleSampler:
    """Sample cheap last-layer energy ensembles from an LLPR state."""

    def __init__(
        self,
        state: LLPRState,
        n_members: int,
        random_seed: int | None = None,
    ) -> None:
        if n_members < 2:
            raise ValueError("n_members must be at least 2.")
        self.state = state
        self.n_members = n_members
        self.random_seed = random_seed

    def sample_energy_ensemble(
        self,
        features: np.ndarray,
        mean_energy: np.ndarray,
    ) -> np.ndarray:
        frame_features = _frame_features(features, self.state.feature_normalization)
        mean_energy = np.asarray(mean_energy, dtype=float).reshape(-1)
        if frame_features.shape[0] != mean_energy.shape[0]:
            raise ValueError("mean_energy length must match number of feature frames.")
        rng = np.random.default_rng(self.random_seed)
        covariance = (self.state.alpha**2) * self.state.inverse_covariance
        weights = rng.multivariate_normal(
            np.zeros(covariance.shape[0]),
            _symmetrize(covariance),
            size=self.n_members,
        )
        offsets = frame_features @ weights.T
        offsets = offsets - offsets.mean(axis=1, keepdims=True)
        return mean_energy[:, None] + offsets


def _frame_features(features: np.ndarray, normalization: FeatureNormalization = "sum") -> np.ndarray:
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if normalization == "sum":
            return arr.sum(axis=1)
        if normalization == "mean":
            return arr.mean(axis=1)
        if normalization == "none":
            raise ValueError("normalization='none' is only valid for 2D frame features.")
        raise ValueError(f"Unsupported feature normalization: {normalization}")
    raise ValueError("features must have shape (nframes, nfeat) or (nframes, natoms, nfeat).")


def _natoms_per_frame(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features)
    if arr.ndim == 3:
        return np.full(arr.shape[0], arr.shape[1], dtype=float)
    if arr.ndim == 2:
        return np.ones(arr.shape[0], dtype=float)
    raise ValueError("features must have shape (nframes, nfeat) or (nframes, natoms, nfeat).")


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _fit_crps_alpha(residuals: np.ndarray, uncertainties: np.ndarray) -> float:
    def objective(alpha: float) -> float:
        return _crps_derivative(alpha, residuals, uncertainties)

    lo, hi = _bracket_root(objective, 1e-10, 50.0)
    try:
        from scipy.optimize import root_scalar

        result = root_scalar(objective, bracket=(lo, hi), method="brentq")
        if result.converged:
            return float(result.root)
    except ImportError:
        pass
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_lo = objective(lo)
        f_mid = objective(mid)
        if abs(f_mid) < 1e-12 or abs(hi - lo) < 1e-12:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def _gaussian_crps(abs_residuals: np.ndarray, sigma: np.ndarray) -> float:
    # Closed-form Gaussian CRPS for zero-mean residuals.
    z = abs_residuals / sigma
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
    pdf = np.exp(-0.5 * z**2) / sqrt(2.0 * pi)
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / sqrt(pi))
    return float(np.mean(crps))


def calibration_benchmark(
    residuals: np.ndarray,
    uncertainties: np.ndarray,
    method: CalibrationMethod = "crps",
) -> dict[str, float]:
    residuals = np.asarray(residuals, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    if residuals.shape != uncertainties.shape:
        raise ValueError("residuals and uncertainties must have the same shape.")
    mask = np.isfinite(residuals) & np.isfinite(uncertainties) & (uncertainties > 0)
    if not np.any(mask):
        raise ValueError("No finite positive uncertainties available for calibration.")
    r = residuals[mask]
    u = uncertainties[mask]
    alpha = float(LLPRCalibrator(method=method).fit_alpha(r, u))
    sigma = np.maximum(alpha * u, 1e-20)
    nll = 0.5 * np.log(2.0 * pi * sigma**2) + 0.5 * (r / sigma) ** 2
    abs_r = np.abs(r)
    return {
        "alpha": alpha,
        "nll": float(np.mean(nll)),
        "gaussian_crps": _gaussian_crps(abs_r, sigma),
        "coverage_1sigma": float(np.mean(abs_r <= sigma)),
        "coverage_2sigma": float(np.mean(abs_r <= 2.0 * sigma)),
        "coverage_3sigma": float(np.mean(abs_r <= 3.0 * sigma)),
        "n_samples": float(r.size),
    }


def _crps_derivative(alpha: float, residuals: np.ndarray, uncertainties: np.ndarray) -> float:
    alpha = max(float(alpha), 1e-20)
    u = residuals / (alpha * uncertainties)
    phi = np.exp(-0.5 * u * u) / sqrt(2.0 * pi)
    cdf = 0.5 * (1.0 + np.vectorize(erf)(u / sqrt(2.0)))
    inv_sqrt_pi = 1.0 / sqrt(pi)
    f_u = inv_sqrt_pi - 2.0 * phi - u * (2.0 * cdf - 1.0)
    return float(np.sum(uncertainties * (f_u - u * (1.0 - 2.0 * cdf))))


def _bracket_root(func, lo: float, hi: float) -> tuple[float, float]:
    f_lo = func(lo)
    f_hi = func(hi)
    if abs(f_lo) < 1e-12:
        lo *= 10.0
        f_lo = func(lo)
    if f_lo * f_hi <= 0.0:
        return lo, hi
    cur_hi = hi
    for _ in range(12):
        cur_hi *= 10.0
        f_cur = func(cur_hi)
        if f_lo * f_cur <= 0.0:
            return lo, cur_hi
    cur_lo = lo
    for _ in range(12):
        cur_lo /= 10.0
        f_cur = func(cur_lo)
        if f_cur * f_hi <= 0.0:
            return cur_lo, hi
    raise RuntimeError("CRPS calibration failed to bracket alpha root.")


def _normalize_alpha(alpha: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(alpha, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _is_calibrated(alpha: float | np.ndarray) -> bool:
    return bool(np.any(np.asarray(alpha, dtype=float) != 1.0))


def _apply_alpha(values: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    alpha_arr = np.asarray(alpha, dtype=float)
    if alpha_arr.ndim == 0:
        return values * float(alpha_arr)
    if alpha_arr.shape == values.shape:
        return values * alpha_arr
    if alpha_arr.shape == (1,):
        return values * float(alpha_arr[0])
    raise ValueError("Per-channel alpha cannot be applied to scalar frame uncertainty.")
