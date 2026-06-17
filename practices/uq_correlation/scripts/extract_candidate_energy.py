#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from uq_correlation_report import energy_per_atom_to_total


ROOT = Path(__file__).resolve().parents[1]


def run(qbc_csv: Path, output_npy: Path, output_csv: Path) -> None:
    df = pd.read_csv(qbc_csv)
    required = {"dataname", "natoms", "pred_e_per_atom"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"qbc table is missing columns: {missing}")
    total_energy = energy_per_atom_to_total(df["pred_e_per_atom"], df["natoms"])
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, total_energy)
    pd.DataFrame({"dataname": df["dataname"], "energy": total_energy}).to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qbc-csv", type=Path, default=ROOT / "outputs" / "qbc_force_frame_table.csv")
    parser.add_argument("--output-npy", type=Path, default=ROOT / "outputs" / "candidate_energy.npy")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs" / "candidate_energy.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.qbc_csv, args.output_npy, args.output_csv)


if __name__ == "__main__":
    main()
