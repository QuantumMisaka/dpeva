#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QBC_CSV = ROOT / "outputs" / "qbc_force_frame_table.csv"
DEFAULT_LLPR_CSV = ROOT / "outputs" / "llpr_energy_uq.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

PRIMARY_PAIRS = [
    ("uq_qbc_for", "force_error_max"),
    ("uq_qbc_for", "force_error_rms"),
    ("uq_rnd_for", "force_error_max"),
    ("uq_rnd_for", "force_error_rms"),
    ("uq_rnd_rescaled", "force_error_max"),
    ("uq_rnd_rescaled", "force_error_rms"),
    ("uq_llpr_energy_per_atom", "energy_error_per_atom_abs"),
    ("uq_dpose_energy_ensemble_std_per_atom", "energy_error_per_atom_abs"),
]

EXPLORATORY_PAIRS = [
    ("uq_llpr_energy_per_atom", "force_error_max"),
    ("uq_dpose_energy_ensemble_std_per_atom", "force_error_max"),
]


def energy_per_atom_to_total(values: Sequence[float], natoms: Sequence[float]) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    natoms_arr = np.asarray(natoms, dtype=float).reshape(-1)
    if values_arr.shape != natoms_arr.shape:
        raise ValueError("values and natoms must have the same length")
    if np.any(natoms_arr <= 0):
        raise ValueError("natoms must be positive")
    return values_arr * natoms_arr


def build_frame_table(qbc: pd.DataFrame, llpr: pd.DataFrame | None = None) -> pd.DataFrame:
    required = {
        "dataname",
        "natoms",
        "data_e_per_atom",
        "pred_e_per_atom",
        "uq_qbc_for",
        "uq_rnd_for",
        "uq_rnd_rescaled",
        "diff_maxf_0_frame",
        "diff_rmsf_0_frame",
    }
    missing = sorted(required - set(qbc.columns))
    if missing:
        raise ValueError(f"qbc table is missing columns: {missing}")

    table = qbc.copy()
    if llpr is not None and not llpr.empty:
        if "dataname" not in llpr.columns:
            raise ValueError("llpr table must contain dataname")
        table = _merge_llpr(table, llpr)

    table["pred_e_total"] = energy_per_atom_to_total(table["pred_e_per_atom"], table["natoms"])
    table["data_e_total"] = energy_per_atom_to_total(table["data_e_per_atom"], table["natoms"])
    table["energy_error_per_atom_abs"] = np.abs(
        table["pred_e_per_atom"].to_numpy(dtype=float) - table["data_e_per_atom"].to_numpy(dtype=float)
    )
    table["energy_error_total_abs"] = np.abs(
        table["pred_e_total"].to_numpy(dtype=float) - table["data_e_total"].to_numpy(dtype=float)
    )
    table["force_error_max"] = table["diff_maxf_0_frame"]
    table["force_error_rms"] = table["diff_rmsf_0_frame"]
    return table


def _merge_llpr(qbc: pd.DataFrame, llpr: pd.DataFrame) -> pd.DataFrame:
    table = qbc.merge(llpr, on="dataname", how="left", validate="one_to_one")
    llpr_value_columns = [col for col in llpr.columns if col != "dataname"]
    if not llpr_value_columns or not table[llpr_value_columns].isna().any().any():
        return table

    qbc_keyed = qbc.copy()
    llpr_keyed = llpr.copy()
    qbc_keyed["_llpr_merge_key"] = qbc_keyed["dataname"].map(lambda value: str(value).split("/")[-1])
    llpr_keyed["_llpr_merge_key"] = llpr_keyed["dataname"].map(lambda value: str(value).split("/")[-1])
    fallback = qbc_keyed.merge(
        llpr_keyed.drop(columns=["dataname"]),
        on="_llpr_merge_key",
        how="left",
        validate="one_to_one",
    ).drop(columns=["_llpr_merge_key"])

    for column in llpr_value_columns:
        if column in table.columns and column in fallback.columns:
            missing = table[column].isna()
            if missing.any():
                table.loc[missing, column] = fallback.loc[missing, column].to_numpy()
    return table


def _valid_pair(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _corr_or_nan(func, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    result = func(x, y)
    return float(result.statistic), float(result.pvalue)


def correlation_summary(df: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for uq_metric, error_metric in pairs:
        if uq_metric not in df.columns or error_metric not in df.columns:
            continue
        x, y = _valid_pair(df, uq_metric, error_metric)
        pearson_r, pearson_p = _corr_or_nan(stats.pearsonr, x, y)
        spearman_rho, spearman_p = _corr_or_nan(stats.spearmanr, x, y)
        kendall_tau, kendall_p = _corr_or_nan(stats.kendalltau, x, y)
        rows.append(
            {
                "uq_metric": uq_metric,
                "error_metric": error_metric,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "kendall_tau": kendall_tau,
                "kendall_p": kendall_p,
                "n_frames": int(len(x)),
            }
        )
    return pd.DataFrame(rows)


def enrichment_table(
    df: pd.DataFrame,
    uq_metric: str,
    error_metric: str,
    top_fractions: Sequence[float] = (0.01, 0.05, 0.10),
    high_error_fraction: float = 0.05,
) -> pd.DataFrame:
    x, y = _valid_pair(df, uq_metric, error_metric)
    n = len(x)
    rows = []
    if n == 0:
        return pd.DataFrame(rows)

    n_high = max(1, int(math.ceil(n * high_error_fraction)))
    high_indices = set(np.argsort(y)[-n_high:])
    baseline = n_high / n
    order = np.argsort(x)[::-1]
    for fraction in top_fractions:
        n_selected = max(1, int(math.ceil(n * fraction)))
        selected = set(order[:n_selected])
        hits = len(high_indices & selected)
        precision = hits / n_selected
        recall = hits / n_high
        rows.append(
            {
                "uq_metric": uq_metric,
                "error_metric": error_metric,
                "top_fraction": float(fraction),
                "high_error_fraction": float(high_error_fraction),
                "n_frames": int(n),
                "n_selected": int(n_selected),
                "n_high_error": int(n_high),
                "n_high_error_selected": int(hits),
                "high_error_precision": float(precision),
                "high_error_recall": float(recall),
                "enrichment": float(precision / baseline) if baseline > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def all_enrichment_tables(df: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    frames = [
        enrichment_table(df, uq_metric, error_metric)
        for uq_metric, error_metric in pairs
        if uq_metric in df.columns and error_metric in df.columns
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _safe_name(*parts: str) -> str:
    return "_".join(part.replace("/", "_").replace(" ", "_") for part in parts)


def write_plots(df: pd.DataFrame, pairs: Iterable[tuple[str, str]], figures_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for uq_metric, error_metric in pairs:
        if uq_metric not in df.columns or error_metric not in df.columns:
            continue
        x, y = _valid_pair(df, uq_metric, error_metric)
        if len(x) == 0:
            continue
        stem = _safe_name(uq_metric, "vs", error_metric)

        scatter_path = figures_dir / f"uq_vs_error_scatter_{stem}.png"
        plt.figure(figsize=(5.5, 4.2))
        plt.scatter(x, y, s=10, alpha=0.55, edgecolors="none")
        plt.xlabel(uq_metric)
        plt.ylabel(error_metric)
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=200)
        plt.close()
        paths.append(scatter_path)

        if len(x) >= 10:
            hexbin_path = figures_dir / f"uq_vs_error_hexbin_{stem}.png"
            plt.figure(figsize=(5.5, 4.2))
            plt.hexbin(x, y, gridsize=35, mincnt=1, cmap="viridis")
            plt.colorbar(label="frames")
            plt.xlabel(uq_metric)
            plt.ylabel(error_metric)
            plt.tight_layout()
            plt.savefig(hexbin_path, dpi=200)
            plt.close()
            paths.append(hexbin_path)

        rank_path = figures_dir / f"uncertainty_rank_error_curve_{stem}.png"
        order = np.argsort(x)[::-1]
        y_ranked = y[order]
        top_fraction = np.arange(1, len(y_ranked) + 1, dtype=float) / len(y_ranked)
        cumulative_mean = np.cumsum(y_ranked) / np.arange(1, len(y_ranked) + 1)
        plt.figure(figsize=(5.5, 4.2))
        plt.plot(top_fraction, cumulative_mean)
        plt.xlabel("top uncertainty fraction")
        plt.ylabel(f"cumulative mean {error_metric}")
        plt.tight_layout()
        plt.savefig(rank_path, dpi=200)
        plt.close()
        paths.append(rank_path)
    return paths


def write_report(
    output_path: Path,
    frame_table: pd.DataFrame,
    corr: pd.DataFrame,
    enrichment: pd.DataFrame,
    figures: Sequence[Path],
) -> None:
    lines = [
        "# DPA4 Mini UQ-Error Correlation Report",
        "",
        f"- Frames: {len(frame_table)}",
        "- Baseline model: ensemble member 0",
        "- QbC committee: ensemble members 1..3",
        "- Energy error: absolute per-atom prediction error from model 0",
        "- Force errors: model-0 frame max and RMS force errors",
        "- Force DPOSE is not evaluated because current detached-feature implementation only supports energy-level DPOSE/LLPR.",
        "",
        "## Correlation Summary",
        "",
    ]
    lines.append(markdown_table(corr) if not corr.empty else "No valid correlation pairs.")
    lines.extend(["", "## Enrichment Summary", ""])
    lines.append(markdown_table(enrichment) if not enrichment.empty else "No valid enrichment pairs.")
    lines.extend(["", "## Figures", ""])
    if figures:
        for path in figures:
            rel = path.relative_to(output_path.parent)
            lines.append(f"- `{rel}`")
    else:
        lines.append("No figures generated.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_markdown_cell(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_markdown_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def run(
    qbc_csv: Path,
    llpr_csv: Path | None,
    output_dir: Path,
    primary_pairs: Sequence[tuple[str, str]] = PRIMARY_PAIRS,
    exploratory_pairs: Sequence[tuple[str, str]] = EXPLORATORY_PAIRS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    qbc = pd.read_csv(qbc_csv)
    llpr = pd.read_csv(llpr_csv) if llpr_csv and llpr_csv.exists() else None
    frame_table = build_frame_table(qbc, llpr)
    frame_table_path = output_dir / "uq_error_frame_table.csv"
    frame_table.to_csv(frame_table_path, index=False)

    pairs = list(primary_pairs) + list(exploratory_pairs)
    corr = correlation_summary(frame_table, pairs)
    corr.to_csv(output_dir / "correlation_summary.csv", index=False)

    enrichment = all_enrichment_tables(frame_table, pairs)
    enrichment.to_csv(output_dir / "enrichment_summary.csv", index=False)

    figures = write_plots(frame_table, pairs, output_dir / "figures")
    write_report(output_dir / "report.md", frame_table, corr, enrichment, figures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--qbc-csv", type=Path, default=DEFAULT_QBC_CSV)
    parser.add_argument("--llpr-csv", type=Path, default=DEFAULT_LLPR_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        base = args.config.parent
        qbc_csv = (base / raw["qbc_csv"]).resolve()
        llpr_csv = (base / raw["llpr_csv"]).resolve() if raw.get("llpr_csv") else None
        output_dir = (base / raw["output_dir"]).resolve()
        primary_pairs = [tuple(item) for item in raw.get("primary_pairs", PRIMARY_PAIRS)]
        exploratory_pairs = [tuple(item) for item in raw.get("exploratory_pairs", EXPLORATORY_PAIRS)]
        run(qbc_csv, llpr_csv, output_dir, primary_pairs, exploratory_pairs)
        return
    run(args.qbc_csv, args.llpr_csv, args.output_dir)


if __name__ == "__main__":
    main()
