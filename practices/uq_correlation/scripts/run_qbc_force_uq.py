#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpeva.uncertain.manager import UQManager  # noqa: E402


def run(
    project_dir: Path,
    testdata_dir: Path,
    output_csv: Path,
    testing_dir: str = "test_val",
    results_prefix: str = "results",
    num_models: int = 4,
) -> None:
    manager = UQManager(
        project_dir=str(project_dir),
        testing_dir=testing_dir,
        testing_head=results_prefix,
        uq_config={"trust_mode": "auto", "scheme": "tangent_lo"},
        num_models=num_models,
        testdata_dir=str(testdata_dir),
    )
    preds, _ = manager.load_predictions()
    uq_results, uq_rnd_rescaled = manager.run_analysis(preds)
    baseline = preds[0]
    datanames = [f"{name}-{frame_idx}" for name, frame_idx, _ in baseline.dataname_list]
    natoms = [int(natom) for _, _, natom in baseline.dataname_list]
    df = pd.DataFrame(
        {
            "dataname": datanames,
            "system": [name for name, _, _ in baseline.dataname_list],
            "frame_idx": [int(frame_idx) for _, frame_idx, _ in baseline.dataname_list],
            "natoms": natoms,
            "data_e_per_atom": baseline.energy["data_e"],
            "pred_e_per_atom": baseline.energy["pred_e"],
            "uq_qbc_for": uq_results["uq_qbc_for"],
            "uq_rnd_for": uq_results["uq_rnd_for"],
            "uq_rnd_rescaled": uq_rnd_rescaled,
            "diff_maxf_0_frame": uq_results["diff_maxf_0_frame"],
            "diff_rmsf_0_frame": uq_results["diff_rmsf_0_frame"],
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=ROOT / "work" / "dpa4_mini_ensemble")
    parser.add_argument("--testdata-dir", type=Path, default=ROOT.parent / "other_dpdata")
    parser.add_argument("--testing-dir", default="test_val")
    parser.add_argument("--results-prefix", default="results")
    parser.add_argument("--num-models", type=int, default=4)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs" / "qbc_force_frame_table.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        project_dir=args.project_dir,
        testdata_dir=args.testdata_dir,
        output_csv=args.output_csv,
        testing_dir=args.testing_dir,
        results_prefix=args.results_prefix,
        num_models=args.num_models,
    )


if __name__ == "__main__":
    main()
