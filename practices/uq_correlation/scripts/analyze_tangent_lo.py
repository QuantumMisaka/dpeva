#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from dpeva.uncertain.calculator import UQCalculator
from dpeva.uncertain.filter import UQFilter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_FRAME_TABLE = OUTPUT_DIR / "uq_error_frame_table.csv"
DEFAULT_CLASSIFIED = OUTPUT_DIR / "tangent_lo_frame_table.csv"
DEFAULT_SUMMARY = OUTPUT_DIR / "tangent_lo_force_error_summary.csv"
DEFAULT_THRESHOLDS = OUTPUT_DIR / "tangent_lo_thresholds.json"
DEFAULT_FIGURES_DIR = OUTPUT_DIR / "figures"
DEFAULT_MARKDOWN = OUTPUT_DIR / "uq_method_deep_dive.md"

IDENTITY_ORDER = ["accurate", "candidate", "failed"]
SOURCE_URLS = {
    "atomistic_cookbook": "https://atomistic-cookbook.org/examples/pet-mad-uq/pet-mad-uq.html",
    "metatrain_llpr": "https://docs.metatensor.org/metatrain/latest/architectures/generated/llpr.html",
    "dpose": "https://arxiv.org/html/2602.15747v1",
}


def _finite_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def derive_thresholds(
    df: pd.DataFrame,
    qbc_lo: float | None = None,
    qbc_hi: float | None = None,
    rnd_lo: float | None = None,
    rnd_hi: float | None = None,
    ratio: float = 0.33,
    width: float = 0.25,
) -> dict[str, float | str]:
    manual_values = [qbc_lo, qbc_hi, rnd_lo, rnd_hi]
    if all(value is not None for value in manual_values):
        return {
            "threshold_source": "manual",
            "qbc_lo": float(qbc_lo),
            "qbc_hi": float(qbc_hi),
            "rnd_lo": float(rnd_lo),
            "rnd_hi": float(rnd_hi),
            "ratio": float(ratio),
            "width": float(width),
        }
    if any(value is not None for value in manual_values):
        raise ValueError("Manual tangent_lo thresholds require qbc_lo, qbc_hi, rnd_lo, and rnd_hi together.")

    calculator = UQCalculator()
    qbc_values = _finite_values(df["uq_qbc_for"])
    rnd_values = _finite_values(df["uq_rnd_rescaled"])
    qbc_auto = calculator.calculate_trust_lo(qbc_values, ratio=ratio)
    rnd_auto = calculator.calculate_trust_lo(rnd_values, ratio=ratio)
    source = "auto-derived"
    if qbc_auto is None or rnd_auto is None:
        qbc_auto = float(pd.Series(qbc_values).quantile(1.0 - ratio))
        rnd_auto = float(pd.Series(rnd_values).quantile(1.0 - ratio))
        source = "auto-derived-fallback-quantile"

    return {
        "threshold_source": source,
        "qbc_lo": float(qbc_auto),
        "qbc_hi": float(qbc_auto + width),
        "rnd_lo": float(rnd_auto),
        "rnd_hi": float(rnd_auto + width),
        "ratio": float(ratio),
        "width": float(width),
    }


def classify_tangent_lo(df: pd.DataFrame, thresholds: dict[str, float | str]) -> pd.DataFrame:
    filt = UQFilter(
        scheme="tangent_lo",
        trust_lo=float(thresholds["qbc_lo"]),
        trust_hi=float(thresholds["qbc_hi"]),
        rnd_trust_lo=float(thresholds["rnd_lo"]),
        rnd_trust_hi=float(thresholds["rnd_hi"]),
    )
    candidate, accurate, failed = filt.filter(df, qbc_col="uq_qbc_for", rnd_col="uq_rnd_rescaled")
    classified = df.copy()
    classified["uq_identity"] = "unclassified"
    classified.loc[accurate.index, "uq_identity"] = "accurate"
    classified.loc[candidate.index, "uq_identity"] = "candidate"
    classified.loc[failed.index, "uq_identity"] = "failed"
    return classified


def _top_error_mask(values: pd.Series, fraction: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    n_high = max(1, int(math.ceil(len(valid) * fraction)))
    threshold = valid.sort_values().iloc[-n_high]
    return numeric >= threshold


def summarize_tangent_lo(
    classified: pd.DataFrame,
    error_metrics: Sequence[str] = ("force_error_max", "force_error_rms"),
    high_error_fraction: float = 0.05,
) -> pd.DataFrame:
    rows = []
    n_total = len(classified)
    for error_metric in error_metrics:
        high_mask = _top_error_mask(classified[error_metric], high_error_fraction)
        n_high = int(high_mask.sum())
        baseline = n_high / n_total if n_total else float("nan")
        for identity in IDENTITY_ORDER:
            subset = classified[classified["uq_identity"] == identity]
            subset_high = high_mask.loc[subset.index] if not subset.empty else pd.Series(dtype=bool)
            n_frames = int(len(subset))
            n_high_selected = int(subset_high.sum()) if n_frames else 0
            values = pd.to_numeric(subset[error_metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            high_error_rate = n_high_selected / n_frames if n_frames else float("nan")
            rows.append(
                {
                    "uq_identity": identity,
                    "error_metric": error_metric,
                    "n_frames": n_frames,
                    "frame_fraction": n_frames / n_total if n_total else float("nan"),
                    "n_high_error": n_high,
                    "n_high_error_selected": n_high_selected,
                    "high_error_rate": high_error_rate,
                    "high_error_recall": n_high_selected / n_high if n_high else float("nan"),
                    "high_error_enrichment": high_error_rate / baseline if baseline else float("nan"),
                    f"{error_metric}_median": float(values.median()) if len(values) else float("nan"),
                    f"{error_metric}_p95": float(values.quantile(0.95)) if len(values) else float("nan"),
                    f"{error_metric}_max": float(values.max()) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def analyze_tangent_lo(
    df: pd.DataFrame,
    qbc_lo: float | None = None,
    qbc_hi: float | None = None,
    rnd_lo: float | None = None,
    rnd_hi: float | None = None,
    ratio: float = 0.33,
    width: float = 0.25,
    high_error_fraction: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    required = {"dataname", "uq_qbc_for", "uq_rnd_rescaled", "force_error_max", "force_error_rms"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"frame table is missing columns: {missing}")
    thresholds = derive_thresholds(df, qbc_lo, qbc_hi, rnd_lo, rnd_hi, ratio=ratio, width=width)
    classified = classify_tangent_lo(df, thresholds)
    summary = summarize_tangent_lo(classified, high_error_fraction=high_error_fraction)
    return classified, summary, thresholds


def _tangent_y(x_values: np.ndarray, thresholds: dict[str, float | str]) -> np.ndarray:
    qbc_lo = float(thresholds["qbc_lo"])
    qbc_hi = float(thresholds["qbc_hi"])
    rnd_lo = float(thresholds["rnd_lo"])
    rnd_hi = float(thresholds["rnd_hi"])
    return -((rnd_hi - rnd_lo) / (qbc_hi - qbc_lo)) * (x_values - qbc_lo) + rnd_lo


def write_plots(classified: pd.DataFrame, thresholds: dict[str, float | str], figures_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    qbc_hi = float(thresholds["qbc_hi"])
    rnd_hi = float(thresholds["rnd_hi"])
    x_line = np.linspace(0, qbc_hi, 200)
    y_line = _tangent_y(x_line, thresholds)
    line_mask = y_line < rnd_hi

    paths = []
    colored = figures_dir / "tangent_lo_qbc_rnd_force_error.png"
    plt.figure(figsize=(6.2, 5.0))
    scatter = plt.scatter(
        classified["uq_qbc_for"],
        classified["uq_rnd_rescaled"],
        c=classified["force_error_max"],
        s=10,
        alpha=0.68,
        cmap="viridis",
        edgecolors="none",
    )
    plt.colorbar(scatter, label="force_error_max")
    plt.axvline(qbc_hi, color="black", linestyle="--", linewidth=1.2)
    plt.axhline(rnd_hi, color="black", linestyle="--", linewidth=1.2)
    plt.plot(x_line[line_mask], y_line[line_mask], color="#7c3aed", linestyle="--", linewidth=1.8)
    plt.xlabel("uq_qbc_for")
    plt.ylabel("uq_rnd_rescaled")
    plt.title("auto-derived tangent_lo vs force error")
    plt.tight_layout()
    plt.savefig(colored, dpi=180)
    plt.close()
    paths.append(colored)

    by_identity = figures_dir / "tangent_lo_qbc_rnd_identity.png"
    colors = {"accurate": "#2f855a", "candidate": "#2563eb", "failed": "#b42318", "unclassified": "#667085"}
    plt.figure(figsize=(6.2, 5.0))
    for identity, subset in classified.groupby("uq_identity"):
        plt.scatter(
            subset["uq_qbc_for"],
            subset["uq_rnd_rescaled"],
            s=10,
            alpha=0.62,
            edgecolors="none",
            color=colors.get(identity, "#667085"),
            label=identity,
        )
    plt.axvline(qbc_hi, color="black", linestyle="--", linewidth=1.2)
    plt.axhline(rnd_hi, color="black", linestyle="--", linewidth=1.2)
    plt.plot(x_line[line_mask], y_line[line_mask], color="#7c3aed", linestyle="--", linewidth=1.8)
    plt.xlabel("uq_qbc_for")
    plt.ylabel("uq_rnd_rescaled")
    plt.title("tangent_lo identity regions")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(by_identity, dpi=180)
    plt.close()
    paths.append(by_identity)

    box = figures_dir / "tangent_lo_force_error_by_identity.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharex=True)
    for ax, metric, title in [
        (axes[0], "force_error_max", "Max force error"),
        (axes[1], "force_error_rms", "RMS force error"),
    ]:
        data = [
            classified.loc[classified["uq_identity"] == identity, metric].to_numpy(dtype=float)
            for identity in IDENTITY_ORDER
        ]
        ax.boxplot(data, tick_labels=IDENTITY_ORDER, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(box, dpi=180)
    plt.close(fig)
    paths.append(box)
    return paths


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}g}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = list(df.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_fmt(row[col]) if isinstance(row[col], (float, int, np.floating, np.integer)) else str(row[col]) for col in headers) + " |")
    return "\n".join(rows)


def write_deep_dive(
    path: Path,
    summary: pd.DataFrame,
    thresholds: dict[str, float | str],
    correlation: pd.DataFrame,
) -> None:
    force_corr = correlation[correlation["error_metric"].isin(["force_error_max", "force_error_rms"])]
    energy_corr = correlation[correlation["error_metric"] == "energy_error_per_atom_abs"]
    best_force = force_corr.sort_values("spearman_rho", ascending=False).iloc[0]
    best_energy = energy_corr.sort_values("spearman_rho", ascending=False).iloc[0]
    summary_view = summary[
        [
            "uq_identity",
            "error_metric",
            "n_frames",
            "frame_fraction",
            "high_error_rate",
            "high_error_recall",
            "high_error_enrichment",
        ]
    ].copy()
    source = thresholds["threshold_source"]
    text = f"""# DP-EVA UQ Method Deep Dive

## tangent_lo 双维度划分

本实验按 DP-EVA `UQFilter(scheme="tangent_lo")` 重新划分候选池，输入维度为 `uq_qbc_for` 与 `uq_rnd_rescaled`。阈值来源为 `{source}`：`qbc_lo={_fmt(thresholds['qbc_lo'])}`、`qbc_hi={_fmt(thresholds['qbc_hi'])}`、`rnd_lo={_fmt(thresholds['rnd_lo'])}`、`rnd_hi={_fmt(thresholds['rnd_hi'])}`。当前实验没有人工阈值配置，因此 auto-derived 口径使用项目默认 `ratio=0.33`、`width=0.25`；若 KDE 无法给出 lo，则使用 `1-ratio` 分位数 fallback。

{markdown_table(summary_view)}

解释：`tangent_lo` 不是单一 UQ 排序，而是在 QbC 与 RND-rescaled 平面上把低不确定度、候选区和超高不确定度失败区分开。因此它更适合回答“DP-EVA collect 会把哪些结构送入候选集”，而不替代单指标 Spearman/Pearson 排序。本实验中 force 主比较的最佳 rank correlation 是 `{best_force.uq_metric}` vs `{best_force.error_metric}`，Spearman = `{_fmt(best_force.spearman_rho)}`；tangent_lo 汇总则检验这些几何分区是否富集真实 high-force-error frame。

## LLPR/DPOSE 偏差是否合理

现有结果显示，QbC/RND force UQ 与真实 force error 强正相关，而 `uq_llpr_energy_per_atom` / `uq_dpose_energy_ensemble_std_per_atom` 与 `energy_error_per_atom_abs` 只有弱正相关；最佳 energy 主比较是 `{best_energy.uq_metric}`，Spearman = `{_fmt(best_energy.spearman_rho)}`。这在当前实现下是合理的：

- DP-EVA QbC/RND 直接从 force 输出差异构造 frame-level force UQ，因此目标与 `force_error_max` / `force_error_rms` 对齐。
- 当前 DP-EVA LLPR/DPOSE 使用 detached fitting-last-layer features，输出 energy-level analytic uncertainty 与 energy ensemble std；它不是 force-aware DPOSE。
- force DPOSE 需要可微 energy ensemble 对坐标求导或等价 Jacobian 路径。detached feature 表无法恢复该路径，所以 LLPR/DPOSE vs force error 只能作为 exploratory。

## 外部资料依据

- Atomistic Cookbook PET-MAD UQ 教程展示 PET-MAD 内置 LLPR uncertainty，并把 ensemble/LLPR uncertainty 用于数据集、能量差、MD 平均量等 derived quantities：{SOURCE_URLS['atomistic_cookbook']}
- metatrain LLPR 文档把 LLPR 定义为 wrapper architecture，可输出 `{{target}}_uncertainty` standard deviation 和 `{{target}}_ensemble`，用于 cheap uncertainty quantification：{SOURCE_URLS['metatrain_llpr']}
- DPOSE/浅层系综资料强调 shallow ensemble 通过共享 backbone、只 ensemble last layer 来降低 full ensemble 成本，并指出 force uncertainty 的可靠校准需要 force-aware 目标或可微传播：{SOURCE_URLS['dpose']}

## 方法比较

| 方法 | 本实验信号 | 浅层含义 | 主要成本 | 主要限制 |
| --- | --- | --- | --- | --- |
| QbC force ensemble | force error 强相关 | 多个完整模型的 force 输出差异 | 训练/推理多个模型 | 成本高于单模型，依赖成员多样性 |
| RND force deviation | force error 强相关，本实验最高 | baseline 与 committee force 偏离 | 复用 ensemble 推理 | 仍是 force-output ensemble，不是 last-layer posterior |
| DP-EVA fitting-last-layer 后处理 | energy risk 补充 | frozen representation + last-layer feature covariance/weights | feature 导出 + 线性代数 | 当前 detached feature 只支持 energy-level |
| LLPR/DPOSE shallow ensemble | energy error 弱正相关 | last-layer posterior / last-layer sampled ensemble | 比 full ensemble 低，需 feature 和权重 | force UQ 需可微 graph adapter |

## 结论

在这个 DPA4 Mini 实验上，主动学习若以 force error 控制为目标，应优先使用 QbC/RND 及其 tangent_lo 双维度筛选；LLPR/DPOSE 更适合补充 energy-level 风险、组合量不确定度或低成本后验分析。若要把 DPOSE 用作 force-level 主采样信号，需要实现可微的 DeepMD PyTorch graph adapter 或显式 force-aware shallow ensemble。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DP-EVA tangent_lo identity regions against force errors.")
    parser.add_argument("--frame-table", type=Path, default=DEFAULT_FRAME_TABLE)
    parser.add_argument("--classified-output", type=Path, default=DEFAULT_CLASSIFIED)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--thresholds-output", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--qbc-lo", type=float)
    parser.add_argument("--qbc-hi", type=float)
    parser.add_argument("--rnd-lo", type=float)
    parser.add_argument("--rnd-hi", type=float)
    parser.add_argument("--ratio", type=float, default=0.33)
    parser.add_argument("--width", type=float, default=0.25)
    parser.add_argument("--high-error-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.frame_table)
    classified, summary, thresholds = analyze_tangent_lo(
        frame,
        qbc_lo=args.qbc_lo,
        qbc_hi=args.qbc_hi,
        rnd_lo=args.rnd_lo,
        rnd_hi=args.rnd_hi,
        ratio=args.ratio,
        width=args.width,
        high_error_fraction=args.high_error_fraction,
    )
    args.classified_output.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(args.classified_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    with args.thresholds_output.open("w") as handle:
        json.dump(thresholds, handle, indent=2)
    write_plots(classified, thresholds, args.figures_dir)
    correlation = pd.read_csv(OUTPUT_DIR / "correlation_summary.csv")
    write_deep_dive(args.markdown_output, summary, thresholds, correlation)
    print(args.summary_output)
    print(args.markdown_output)


if __name__ == "__main__":
    main()
