from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "practices"
    / "uq_correlation"
    / "scripts"
    / "build_html_dashboard.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("build_html_dashboard", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_uq_efficiency_rows_compare_qbc_llpr_and_dpose_costs():
    mod = load_module()
    runtime = pd.DataFrame(
        {
            "stage": ["Candidate inference", "Train feature extraction", "Candidate feature extraction"],
            "n_jobs": [4, 1, 1],
            "max_elapsed_min": [1.9, 0.5, 2.3],
            "gpu_minutes": [7.5, 0.5, 2.3],
        }
    )
    energy_ensemble = np.zeros((10, 8))

    rows = mod.build_uq_efficiency_rows(runtime, energy_ensemble)

    by_method = {row[0]: row for row in rows}
    assert set(by_method) == {"QbC/RND force UQ", "LLPR analytic energy UQ", "DPOSE energy ensemble UQ"}
    assert "4 model inference" in by_method["QbC/RND force UQ"][1]
    assert "2.8" in by_method["LLPR analytic energy UQ"][2]
    assert "(10, 8)" in by_method["DPOSE energy ensemble UQ"][4]
