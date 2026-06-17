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
    / "uq_correlation_report.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("uq_correlation_report", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_energy_per_atom_to_total_uses_frame_atom_counts():
    mod = load_module()

    total = mod.energy_per_atom_to_total([1.5, -2.0, 0.25], [2, 4, 8])

    np.testing.assert_allclose(total, [3.0, -8.0, 2.0])


def test_build_frame_table_merges_uq_llpr_and_energy_errors():
    mod = load_module()
    qbc = pd.DataFrame(
        {
            "dataname": ["sys-0", "sys-1"],
            "natoms": [2, 4],
            "data_e_per_atom": [1.0, 2.0],
            "pred_e_per_atom": [1.4, 1.75],
            "uq_qbc_for": [0.2, 0.3],
            "uq_rnd_for": [0.4, 0.6],
            "uq_rnd_rescaled": [0.5, 0.7],
            "diff_maxf_0_frame": [1.1, 2.2],
            "diff_rmsf_0_frame": [0.11, 0.22],
        }
    )
    llpr = pd.DataFrame(
        {
            "dataname": ["sys-0", "sys-1"],
            "uq_llpr_energy_per_atom": [0.01, 0.02],
            "uq_dpose_energy_ensemble_std_per_atom": [0.03, 0.04],
        }
    )

    table = mod.build_frame_table(qbc, llpr)

    assert table["dataname"].tolist() == ["sys-0", "sys-1"]
    np.testing.assert_allclose(table["energy_error_per_atom_abs"], [0.4, 0.25])
    np.testing.assert_allclose(table["pred_e_total"], [2.8, 7.0])
    np.testing.assert_allclose(table["data_e_total"], [2.0, 8.0])
    np.testing.assert_allclose(table["energy_error_total_abs"], [0.8, 1.0])
    assert "force_error_max" in table
    assert "force_error_rms" in table


def test_build_frame_table_merges_llpr_by_basename_when_pool_prefix_differs():
    mod = load_module()
    qbc = pd.DataFrame(
        {
            "dataname": ["other_dpdata/sys-0"],
            "natoms": [2],
            "data_e_per_atom": [1.0],
            "pred_e_per_atom": [1.4],
            "uq_qbc_for": [0.2],
            "uq_rnd_for": [0.4],
            "uq_rnd_rescaled": [0.5],
            "diff_maxf_0_frame": [1.1],
            "diff_rmsf_0_frame": [0.11],
        }
    )
    llpr = pd.DataFrame(
        {
            "dataname": ["sys-0"],
            "uq_llpr_energy_per_atom": [0.01],
            "uq_dpose_energy_ensemble_std_per_atom": [0.03],
        }
    )

    table = mod.build_frame_table(qbc, llpr)

    assert table.loc[0, "uq_llpr_energy_per_atom"] == 0.01
    assert table.loc[0, "uq_dpose_energy_ensemble_std_per_atom"] == 0.03


def test_correlation_summary_drops_invalid_pairs_per_metric():
    mod = load_module()
    df = pd.DataFrame(
        {
            "uq_a": [1.0, 2.0, np.nan, 4.0],
            "err_a": [1.0, 3.0, 5.0, np.inf],
            "err_b": [4.0, 3.0, 2.0, 1.0],
        }
    )

    summary = mod.correlation_summary(df, [("uq_a", "err_a"), ("uq_a", "err_b")])

    by_pair = {(row.uq_metric, row.error_metric): row for row in summary.itertuples()}
    assert by_pair[("uq_a", "err_a")].n_frames == 2
    assert by_pair[("uq_a", "err_b")].n_frames == 3
    assert np.isfinite(by_pair[("uq_a", "err_b")].spearman_rho)


def test_enrichment_table_reports_top_uncertainty_recall_of_high_error():
    mod = load_module()
    df = pd.DataFrame(
        {
            "uq": [0.99, 0.10, 0.80, 0.20, 0.70],
            "err": [9.0, 1.0, 8.0, 2.0, 7.0],
        }
    )

    enrichment = mod.enrichment_table(df, "uq", "err", top_fractions=[0.4], high_error_fraction=0.4)

    row = enrichment.iloc[0]
    assert row["n_selected"] == 2
    assert row["n_high_error"] == 2
    assert row["n_high_error_selected"] == 2
    assert row["high_error_recall"] == 1.0
    assert row["enrichment"] == 2.5


def test_markdown_table_does_not_require_optional_tabulate():
    mod = load_module()
    df = pd.DataFrame({"metric": ["uq"], "value": [1.25]})

    rendered = mod.markdown_table(df)

    assert "| metric | value |" in rendered
    assert "| uq | 1.25 |" in rendered
