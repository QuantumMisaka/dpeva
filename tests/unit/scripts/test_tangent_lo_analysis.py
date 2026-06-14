from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "practices"
    / "uq_correlation"
    / "scripts"
    / "analyze_tangent_lo.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_tangent_lo", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manual_tangent_lo_summary_preserves_frames_and_high_error_enrichment():
    mod = load_module()
    frame = pd.DataFrame(
        {
            "dataname": [f"f{i}" for i in range(6)],
            "uq_qbc_for": [0.04, 0.08, 0.14, 0.18, 0.24, 0.30],
            "uq_rnd_rescaled": [0.04, 0.14, 0.08, 0.18, 0.14, 0.30],
            "force_error_max": [0.1, 0.2, 0.8, 1.0, 2.5, 3.0],
            "force_error_rms": [0.01, 0.02, 0.08, 0.10, 0.25, 0.30],
        }
    )

    classified, summary, thresholds = mod.analyze_tangent_lo(
        frame,
        qbc_lo=0.10,
        qbc_hi=0.20,
        rnd_lo=0.10,
        rnd_hi=0.20,
        high_error_fraction=1 / 3,
    )

    assert thresholds["threshold_source"] == "manual"
    assert classified["uq_identity"].tolist() == [
        "accurate",
        "candidate",
        "candidate",
        "candidate",
        "failed",
        "failed",
    ]
    assert summary.groupby("error_metric")["n_frames"].sum().to_dict() == {
        "force_error_max": len(frame),
        "force_error_rms": len(frame),
    }

    force_max = summary[summary["error_metric"] == "force_error_max"]
    by_identity = {row.uq_identity: row for row in force_max.itertuples()}
    assert by_identity["failed"].high_error_rate == 1.0
    assert by_identity["failed"].high_error_enrichment == 3.0
    assert by_identity["candidate"].force_error_max_p95 > by_identity["accurate"].force_error_max_p95


def test_auto_threshold_falls_back_to_quantiles_when_kde_returns_none(monkeypatch):
    mod = load_module()
    frame = pd.DataFrame(
        {
            "dataname": [f"f{i}" for i in range(5)],
            "uq_qbc_for": np.linspace(0.1, 0.5, 5),
            "uq_rnd_rescaled": np.linspace(0.2, 0.6, 5),
            "force_error_max": np.linspace(1.0, 5.0, 5),
            "force_error_rms": np.linspace(0.1, 0.5, 5),
        }
    )

    monkeypatch.setattr(mod.UQCalculator, "calculate_trust_lo", lambda *args, **kwargs: None)

    _classified, _summary, thresholds = mod.analyze_tangent_lo(frame, ratio=0.33, width=0.25)

    assert thresholds["threshold_source"] == "auto-derived-fallback-quantile"
    np.testing.assert_allclose(thresholds["qbc_lo"], frame["uq_qbc_for"].quantile(0.67))
    np.testing.assert_allclose(thresholds["rnd_lo"], frame["uq_rnd_rescaled"].quantile(0.67))
    np.testing.assert_allclose(thresholds["qbc_hi"], thresholds["qbc_lo"] + 0.25)
    np.testing.assert_allclose(thresholds["rnd_hi"], thresholds["rnd_lo"] + 0.25)
