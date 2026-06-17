import numpy as np

from dpeva.uncertain.llpr import (
    calibration_benchmark,
    LLPRCalibrator,
    LLPRState,
    ShallowEnsembleSampler,
)


def test_llpr_state_computes_system_and_per_atom_uncertainty():
    features = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
        ]
    )
    train_features = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ]
    )

    state = LLPRState.from_training_features(
        train_features,
        regularizer=0.5,
        alpha=2.0,
    )
    result = state.predict_uncertainty(features)

    assert result.total.shape == (2,)
    assert result.per_atom.shape == (2,)
    assert result.calibrated is True
    np.testing.assert_allclose(result.per_atom, result.total / np.array([2, 2]))


def test_llpr_calibrator_squared_residuals_scales_uncertainty():
    residuals = np.array([2.0, 2.0])
    uncertainties = np.array([1.0, 2.0])

    alpha = LLPRCalibrator(method="squared_residuals").fit_alpha(
        residuals,
        uncertainties,
    )

    np.testing.assert_allclose(alpha, np.sqrt(np.mean((residuals / uncertainties) ** 2)))


def test_llpr_calibrator_absolute_residuals_uses_gaussian_sigma_units():
    residuals = np.array([2.0, 2.0])
    uncertainties = np.array([1.0, 2.0])

    alpha = LLPRCalibrator(method="absolute_residuals").fit_alpha(
        residuals,
        uncertainties,
    )

    np.testing.assert_allclose(
        alpha,
        np.mean(np.abs(residuals) / uncertainties) * np.sqrt(np.pi / 2.0),
    )


def test_llpr_calibrator_preserves_per_channel_multipliers():
    residuals = np.array([[2.0, 3.0], [4.0, 3.0]])
    uncertainties = np.array([[1.0, 3.0], [2.0, 1.5]])

    alpha = LLPRCalibrator(method="squared_residuals").fit_alpha(
        residuals,
        uncertainties,
    )

    np.testing.assert_allclose(
        alpha,
        np.sqrt(np.mean((residuals / uncertainties) ** 2, axis=0)),
    )


def test_llpr_state_can_use_metatrain_system_feature_normalization():
    features = np.array(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[4.0, 0.0], [0.0, 4.0]],
        ]
    )

    state = LLPRState.from_training_features(
        features,
        regularizer=0.5,
        feature_normalization="mean",
    )

    expected_frame_features = features.mean(axis=1)
    expected_cov = expected_frame_features.T @ expected_frame_features + 0.5 * np.eye(2)
    np.testing.assert_allclose(state.covariance, expected_cov)


def test_llpr_state_save_load_npz_roundtrip(tmp_path):
    features = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 1.0], [1.0, 3.0]],
        ]
    )
    state = LLPRState.from_training_features(
        features,
        regularizer=0.25,
        alpha=np.array([1.5, 2.5]),
        calibrated=True,
        feature_normalization="mean",
    )

    path = tmp_path / "llpr_state.npz"
    state.save_npz(path)
    loaded = LLPRState.load_npz(path)

    np.testing.assert_allclose(loaded.covariance, state.covariance)
    np.testing.assert_allclose(loaded.cholesky, state.cholesky)
    np.testing.assert_allclose(loaded.inverse_covariance, state.inverse_covariance)
    np.testing.assert_allclose(loaded.alpha, state.alpha)
    assert loaded.regularizer == state.regularizer
    assert loaded.calibrated is True
    assert loaded.feature_normalization == "mean"
    assert loaded.feature_dimension == 2


def test_calibration_benchmark_reports_scoring_and_coverage():
    residuals = np.array([-1.0, 0.0, 1.0, 2.0])
    uncertainties = np.array([1.0, 1.0, 2.0, 2.0])

    report = calibration_benchmark(residuals, uncertainties, method="squared_residuals")

    assert set(report) >= {
        "alpha",
        "nll",
        "gaussian_crps",
        "coverage_1sigma",
        "coverage_2sigma",
        "coverage_3sigma",
    }
    assert report["alpha"] > 0
    assert report["nll"] < float("inf")
    assert 0.0 <= report["coverage_1sigma"] <= 1.0


def test_shallow_ensemble_sampler_returns_reproducible_energy_samples():
    features = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    state = LLPRState.from_training_features(features, regularizer=1.0, alpha=1.0)
    sampler = ShallowEnsembleSampler(state, n_members=4, random_seed=7)

    ensemble_a = sampler.sample_energy_ensemble(features, mean_energy=np.array([5.0]))
    ensemble_b = sampler.sample_energy_ensemble(features, mean_energy=np.array([5.0]))

    assert ensemble_a.shape == (1, 4)
    np.testing.assert_allclose(ensemble_a, ensemble_b)
    np.testing.assert_allclose(ensemble_a.mean(axis=1), np.array([5.0]), atol=1.0)
