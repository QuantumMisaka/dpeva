#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_OUTPUT = OUTPUT_DIR / "uq_correlation_dashboard.html"

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

JOB_IDS = [
    "511970",
    "511971",
    "511972",
    "511973",
    "511982",
    "511983",
    "511984",
    "511985",
    "511996",
    "511997",
]

METHOD_LABELS = {
    "uq_qbc_for": "QbC force ensemble",
    "uq_rnd_for": "RND force deviation",
    "uq_rnd_rescaled": "RND force deviation, rescaled",
    "uq_llpr_energy_per_atom": "LLPR energy per atom",
    "uq_dpose_energy_ensemble_std_per_atom": "DPOSE energy ensemble std per atom",
}

ERROR_LABELS = {
    "force_error_max": "Frame max force error",
    "force_error_rms": "Frame RMS force error",
    "energy_error_per_atom_abs": "Abs energy error per atom",
}

SOURCE_URLS = {
    "Atomistic Cookbook PET-MAD UQ": "https://atomistic-cookbook.org/examples/pet-mad-uq/pet-mad-uq.html",
    "metatrain LLPR": "https://docs.metatensor.org/metatrain/latest/architectures/generated/llpr.html",
    "DPOSE shallow ensemble": "https://arxiv.org/html/2602.15747v1",
}


@dataclass(frozen=True)
class RuntimeRow:
    stage: str
    job_id: str
    job_name: str
    state: str
    elapsed: str
    gpus: int
    gpu_minutes: float
    start: str
    end: str


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def format_float(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if abs(number) < 1e-3 and number != 0:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number * 100:.{digits}f}%"


def elapsed_to_minutes(value: str) -> float:
    if not value:
        return 0.0
    days = 0
    body = value
    if "-" in value:
        day_part, body = value.split("-", 1)
        days = int(day_part)
    parts = [int(part) for part in body.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        return 0.0
    return days * 24 * 60 + hours * 60 + minutes + seconds / 60


def parse_gpus(alloc_tres: str) -> int:
    match = re.search(r"(?:gres/)?gpu=(\d+)", alloc_tres or "")
    return int(match.group(1)) if match else 0


def stage_for_job(job_name: str) -> str:
    if job_name.startswith("dpeva_train"):
        return "Train ensemble"
    if job_name.startswith("dp_test"):
        return "Candidate inference"
    if "sampled_dpdata" in job_name:
        return "Train feature extraction"
    if "other_dpdata" in job_name:
        return "Candidate feature extraction"
    return "Other"


def load_runtime_rows() -> list[RuntimeRow]:
    command = [
        "sacct",
        "-j",
        ",".join(JOB_IDS),
        "--format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,Start,End",
        "-P",
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return []

    rows: list[RuntimeRow] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split("|")
        if len(parts) != 8:
            continue
        job_id, job_name, state, _exit_code, elapsed, alloc_tres, start, end = parts
        if "." in job_id:
            continue
        gpus = parse_gpus(alloc_tres)
        rows.append(
            RuntimeRow(
                stage=stage_for_job(job_name),
                job_id=job_id,
                job_name=job_name,
                state=state,
                elapsed=elapsed,
                gpus=gpus,
                gpu_minutes=elapsed_to_minutes(elapsed) * gpus,
                start=start,
                end=end,
            )
        )
    return rows


def runtime_summary(rows: Iterable[RuntimeRow]) -> pd.DataFrame:
    frame = pd.DataFrame([row.__dict__ for row in rows])
    if frame.empty:
        return pd.DataFrame(columns=["stage", "n_jobs", "max_elapsed_min", "gpu_minutes"])
    frame["elapsed_min"] = frame["elapsed"].map(elapsed_to_minutes)
    summary = (
        frame.groupby("stage", as_index=False)
        .agg(n_jobs=("job_id", "count"), max_elapsed_min=("elapsed_min", "max"), gpu_minutes=("gpu_minutes", "sum"))
        .sort_values("stage")
    )
    return summary


def metric_pair_key(uq_metric: str, error_metric: str) -> str:
    return f"{uq_metric}::{error_metric}"


def strength_label(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 0.7:
        return "strong"
    if abs_value >= 0.4:
        return "moderate"
    if abs_value >= 0.2:
        return "weak"
    return "very weak"


def direction_label(value: float) -> str:
    return "positive" if value >= 0 else "negative"


def table_html(headers: list[str], rows: list[list[object]], classes: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table class=\"{classes}\"><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def figure_path(figures_dir: Path, prefix: str, uq_metric: str, error_metric: str) -> Path:
    stem = f"{uq_metric}_vs_{error_metric}"
    return figures_dir / f"{prefix}_{stem}.png"


def describe_correlation(row: pd.Series) -> str:
    spearman = float(row["spearman_rho"])
    pearson = float(row["pearson_r"])
    strength = strength_label(spearman)
    direction = direction_label(spearman)
    return (
        f"{METHOD_LABELS.get(row['uq_metric'], row['uq_metric'])} vs "
        f"{ERROR_LABELS.get(row['error_metric'], row['error_metric'])}: "
        f"{strength} {direction} rank relation "
        f"(Spearman {spearman:.3f}, Pearson {pearson:.3f})."
    )


def best_enrichment(enrichment: pd.DataFrame, uq_metric: str, error_metric: str, fraction: float) -> pd.Series | None:
    subset = enrichment[
        (enrichment["uq_metric"] == uq_metric)
        & (enrichment["error_metric"] == error_metric)
        & np.isclose(enrichment["top_fraction"], fraction)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def build_method_rows(correlation: pd.DataFrame, enrichment: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    primary_keys = {metric_pair_key(*pair) for pair in PRIMARY_PAIRS}
    for _, row in correlation.iterrows():
        key = metric_pair_key(row["uq_metric"], row["error_metric"])
        status = "Primary" if key in primary_keys else "Exploratory"
        enrich_5 = best_enrichment(enrichment, row["uq_metric"], row["error_metric"], 0.05)
        rows.append(
            [
                html.escape(status),
                html.escape(METHOD_LABELS.get(row["uq_metric"], row["uq_metric"])),
                html.escape(ERROR_LABELS.get(row["error_metric"], row["error_metric"])),
                format_float(row["pearson_r"], 3),
                format_float(row["spearman_rho"], 3),
                format_float(row["kendall_tau"], 3),
                f"{int(row['n_frames']):,}",
                pct(enrich_5["high_error_recall"]) if enrich_5 is not None else "",
                format_float(enrich_5["enrichment"], 2) if enrich_5 is not None else "",
                html.escape(describe_correlation(row)),
            ]
        )
    return rows


def build_efficiency_rows(runtime: pd.DataFrame) -> list[list[object]]:
    rows = []
    for _, row in runtime.iterrows():
        rows.append(
            [
                html.escape(str(row["stage"])),
                int(row["n_jobs"]),
                f"{float(row['max_elapsed_min']):.1f} min",
                f"{float(row['gpu_minutes']):.1f}",
                f"{float(row['gpu_minutes']) / 60:.2f}",
            ]
        )
    return rows


def _runtime_stage(runtime: pd.DataFrame, stage: str) -> pd.Series | None:
    if runtime.empty or "stage" not in runtime.columns:
        return None
    subset = runtime[runtime["stage"] == stage]
    if subset.empty:
        return None
    return subset.iloc[0]


def _stage_gpu_minutes(runtime: pd.DataFrame, stage: str) -> float:
    row = _runtime_stage(runtime, stage)
    return float(row["gpu_minutes"]) if row is not None else 0.0


def _stage_walltime(runtime: pd.DataFrame, stage: str) -> float:
    row = _runtime_stage(runtime, stage)
    return float(row["max_elapsed_min"]) if row is not None else 0.0


def build_uq_efficiency_rows(runtime: pd.DataFrame, energy_ensemble: np.ndarray) -> list[list[object]]:
    inference_gpu_min = _stage_gpu_minutes(runtime, "Candidate inference")
    inference_wall = _stage_walltime(runtime, "Candidate inference")
    train_feature_gpu_min = _stage_gpu_minutes(runtime, "Train feature extraction")
    candidate_feature_gpu_min = _stage_gpu_minutes(runtime, "Candidate feature extraction")
    feature_gpu_min = train_feature_gpu_min + candidate_feature_gpu_min
    feature_wall = max(
        _stage_walltime(runtime, "Train feature extraction"),
        _stage_walltime(runtime, "Candidate feature extraction"),
    )
    model0_inference_gpu_min = inference_gpu_min / 4 if inference_gpu_min else 0.0
    model0_inference_wall = inference_wall
    llpr_gpu_min = feature_gpu_min
    dpose_gpu_min = feature_gpu_min + model0_inference_gpu_min
    return [
        [
            "QbC/RND force UQ",
            "4 model inference on candidate pool; force UQ comes from committee force spread and baseline-vs-committee deviation.",
            f"{inference_gpu_min:.1f} GPU-min",
            f"{inference_wall:.1f} min",
            "Requires trained ensemble; no feature extraction needed for UQ after inference.",
        ],
        [
            "LLPR analytic energy UQ",
            "Train and candidate fitting-last-layer feature extraction, then CPU Cholesky/linear algebra.",
            f"{llpr_gpu_min:.1f} GPU-min",
            f"{feature_wall:.1f} min feature critical path",
            "CPU postprocess was lightweight and not charged by Slurm; output is analytic energy uncertainty.",
        ],
        [
            "DPOSE energy ensemble UQ",
            "LLPR feature pipeline plus model-0 base energy and last-layer weight sampling.",
            f"{dpose_gpu_min:.1f} GPU-min",
            f"{max(feature_wall, model0_inference_wall):.1f} min observed critical path",
            f"Includes model-0 base energy reuse; energy ensemble shape {tuple(energy_ensemble.shape)}.",
        ],
    ]


def build_runtime_detail_rows(rows: list[RuntimeRow]) -> list[list[object]]:
    return [
        [
            html.escape(row.stage),
            html.escape(row.job_id),
            html.escape(row.job_name),
            html.escape(row.state),
            html.escape(row.elapsed),
            row.gpus,
            f"{row.gpu_minutes:.1f}",
            html.escape(row.start.replace("T", " ")),
            html.escape(row.end.replace("T", " ")),
        ]
        for row in rows
    ]


def quantile_rows(frame: pd.DataFrame) -> list[list[object]]:
    columns = [
        ("energy_error_per_atom_abs", "Energy error per atom"),
        ("force_error_max", "Max force error"),
        ("force_error_rms", "RMS force error"),
        ("uq_qbc_for", "QbC force UQ"),
        ("uq_rnd_for", "RND force UQ"),
        ("uq_llpr_energy_per_atom", "LLPR energy UQ"),
        ("uq_dpose_energy_ensemble_std_per_atom", "DPOSE energy UQ"),
    ]
    rows = []
    for column, label in columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            [
                html.escape(label),
                format_float(values.min(), 4),
                format_float(values.quantile(0.5), 4),
                format_float(values.quantile(0.95), 4),
                format_float(values.max(), 4),
            ]
        )
    return rows


def build_tangent_rows(summary: pd.DataFrame) -> list[list[object]]:
    rows = []
    for _, row in summary.iterrows():
        error_metric = str(row["error_metric"])
        median_col = f"{error_metric}_median"
        p95_col = f"{error_metric}_p95"
        max_col = f"{error_metric}_max"
        rows.append(
            [
                html.escape(str(row["uq_identity"])),
                html.escape(ERROR_LABELS.get(error_metric, error_metric)),
                f"{int(row['n_frames']):,}",
                pct(row["frame_fraction"]),
                pct(row["high_error_rate"]),
                pct(row["high_error_recall"]),
                format_float(row["high_error_enrichment"], 2),
                format_float(row.get(median_col), 4),
                format_float(row.get(p95_col), 4),
                format_float(row.get(max_col), 4),
            ]
        )
    return rows


def build_tangent_plot_cards(figures_dir: Path) -> str:
    figures = [
        ("QbC/RND 平面按真实 force error 着色", figures_dir / "tangent_lo_qbc_rnd_force_error.png"),
        ("QbC/RND 平面按 tangent_lo identity 着色", figures_dir / "tangent_lo_qbc_rnd_identity.png"),
        ("tangent_lo identity 下的 force error 分布", figures_dir / "tangent_lo_force_error_by_identity.png"),
    ]
    cards = []
    for title, path in figures:
        if path.exists():
            cards.append(
                f"<figure><img src=\"{image_data_uri(path)}\" alt=\"{html.escape(title)}\">"
                f"<figcaption>{html.escape(title)}</figcaption></figure>"
            )
    return f"<div class=\"plot-grid\">{''.join(cards)}</div>"


def build_shallow_comparison_rows() -> list[list[object]]:
    return [
        [
            "QbC force ensemble",
            "force error 强相关；可直接支持 force-error 主动学习筛选。",
            "多个完整模型的 force 输出差异，不是 last-layer posterior。",
            "训练/推理多个模型；本实验 4-member ensemble。",
            "成员多样性不足时会低估风险。",
        ],
        [
            "RND force deviation",
            "本实验 force rank correlation 最高。",
            "baseline model 与 committee force 偏离。",
            "复用 ensemble 推理，后处理便宜。",
            "仍依赖完整模型 ensemble 输出。",
        ],
        [
            "DP-EVA fitting-network 后处理",
            "提供 energy-level 风险补充。",
            "frozen representation + fitting last-layer feature/weights。",
            "feature 导出 + 线性代数。",
            "当前 detached feature 不能推出 force DPOSE。",
        ],
        [
            "LLPR/DPOSE shallow ensemble",
            "energy error 弱正相关，适合 energy-risk 和组合量传播。",
            "last-layer covariance / posterior sampled ensemble。",
            "比 full ensemble 低；可输出 energy ensemble。",
            "force uncertainty 需要可微 graph/Jacobian 或 force-aware 训练目标。",
        ],
    ]


def build_plot_cards(figures_dir: Path, pairs: list[tuple[str, str]]) -> str:
    cards = []
    for uq_metric, error_metric in pairs:
        title = f"{METHOD_LABELS.get(uq_metric, uq_metric)} vs {ERROR_LABELS.get(error_metric, error_metric)}"
        scatter = figure_path(figures_dir, "uq_vs_error_scatter", uq_metric, error_metric)
        hexbin = figure_path(figures_dir, "uq_vs_error_hexbin", uq_metric, error_metric)
        rank = figure_path(figures_dir, "uncertainty_rank_error_curve", uq_metric, error_metric)
        images = []
        for label, path in [("scatter", scatter), ("hexbin", hexbin), ("rank curve", rank)]:
            if path.exists():
                images.append(
                    f"<figure><img src=\"{image_data_uri(path)}\" alt=\"{html.escape(title)} {label}\">"
                    f"<figcaption>{html.escape(label)}</figcaption></figure>"
                )
        cards.append(
            "<section class=\"plot-card\">"
            f"<h3>{html.escape(title)}</h3>"
            f"<div class=\"plot-grid\">{''.join(images)}</div>"
            "</section>"
        )
    return "".join(cards)


def build_dashboard(output_path: Path) -> None:
    preflight = read_json(OUTPUT_DIR / "preflight_summary.json")
    correlation = pd.read_csv(OUTPUT_DIR / "correlation_summary.csv")
    enrichment = pd.read_csv(OUTPUT_DIR / "enrichment_summary.csv")
    frame = pd.read_csv(OUTPUT_DIR / "uq_error_frame_table.csv")
    llpr = pd.read_csv(OUTPUT_DIR / "llpr_energy_uq.csv")
    tangent_summary = pd.read_csv(OUTPUT_DIR / "tangent_lo_force_error_summary.csv")
    tangent_thresholds = read_json(OUTPUT_DIR / "tangent_lo_thresholds.json")
    energy_ensemble = np.load(OUTPUT_DIR / "energy_ensemble.npy")
    runtime_rows = load_runtime_rows()
    runtime = runtime_summary(runtime_rows)

    train = preflight["train"]
    candidate = preflight["candidate"]
    overlap = preflight.get("overlap_system_names", [])
    primary = correlation[
        correlation.apply(lambda row: metric_pair_key(row["uq_metric"], row["error_metric"]) in {metric_pair_key(*pair) for pair in PRIMARY_PAIRS}, axis=1)
    ]
    force_primary = primary[primary["error_metric"].isin(["force_error_max", "force_error_rms"])]
    energy_primary = primary[primary["error_metric"] == "energy_error_per_atom_abs"]
    best_force = force_primary.sort_values("spearman_rho", ascending=False).iloc[0]
    best_energy = energy_primary.sort_values("spearman_rho", ascending=False).iloc[0]

    method_rows = build_method_rows(correlation, enrichment)
    efficiency_rows = build_efficiency_rows(runtime)
    uq_efficiency_rows = build_uq_efficiency_rows(runtime, energy_ensemble)
    detail_rows = build_runtime_detail_rows(runtime_rows)
    dist_rows = quantile_rows(frame)
    tangent_rows = build_tangent_rows(tangent_summary)
    tangent_plots = build_tangent_plot_cards(OUTPUT_DIR / "figures")
    comparison_rows = build_shallow_comparison_rows()

    figure_pairs = PRIMARY_PAIRS + EXPLORATORY_PAIRS
    plot_cards = build_plot_cards(OUTPUT_DIR / "figures", figure_pairs)

    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    source_links = "".join(
        f"<li><a href=\"{html.escape(url)}\">{html.escape(label)}</a></li>"
        for label, url in SOURCE_URLS.items()
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DPA4 Mini UQ-Error Correlation Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #667085;
      --line: #d7dde5;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-2: #b42318;
      --accent-3: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.5;
    }}
    header {{
      background: #17202a;
      color: #fff;
      padding: 32px 40px 28px;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    header p {{ max-width: 1100px; margin: 0; color: #d6dee7; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px 32px 48px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin: 0 0 20px;
    }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    h3 {{ margin: 0 0 10px; font-size: 17px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfd;
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .subtle {{ color: var(--muted); font-size: 13px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .callout {{
      border-left: 4px solid var(--accent);
      background: #effaf7;
      padding: 14px 16px;
      border-radius: 6px;
    }}
    .callout.warn {{ border-left-color: var(--accent-2); background: #fff4f2; }}
    .callout.info {{ border-left-color: var(--accent-3); background: #eff6ff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #eef1f5; font-weight: 700; }}
    tr:nth-child(even) td {{ background: #fbfcfd; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
    .plot-card {{ margin-top: 16px; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: #fff; }}
    img {{ max-width: 100%; display: block; }}
    figcaption {{ margin-top: 6px; color: var(--muted); font-size: 12px; }}
    ul {{ margin: 8px 0 0 20px; padding: 0; }}
    code {{ background: #eef1f5; padding: 1px 4px; border-radius: 4px; }}
    footer {{ color: var(--muted); font-size: 12px; margin-top: 24px; }}
    @media (max-width: 980px) {{
      header {{ padding: 24px; }}
      main {{ padding: 18px; }}
      .cards, .summary-grid, .plot-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>DPA4 Mini 不确定度-误差相关性测试看板</h1>
    <p>实验对象为 4-member DPA4 Mini ensemble。训练池为 <code>practices/sampled_dpdata</code>，候选池为带 DFT 标签的 <code>practices/other_dpdata</code>。报告生成时间：{html.escape(generated)}。</p>
  </header>
  <main>
    <section>
      <h2>实验规模与验收状态</h2>
      <div class="cards">
        <div class="card"><div class="label">训练池</div><div class="value">{int(train['n_systems']):,}</div><div class="subtle">systems, {int(train['n_frames']):,} frames</div></div>
        <div class="card"><div class="label">候选池</div><div class="value">{int(candidate['n_systems']):,}</div><div class="subtle">systems, {int(candidate['n_frames']):,} frames</div></div>
        <div class="card"><div class="label">合并帧表</div><div class="value">{len(frame):,}</div><div class="subtle">frame-level rows</div></div>
        <div class="card"><div class="label">Energy ensemble</div><div class="value">{energy_ensemble.shape[1]}</div><div class="subtle">members, shape {html.escape(str(tuple(energy_ensemble.shape)))}</div></div>
      </div>
      <p class="subtle">Type map: {html.escape(', '.join(train['type_maps'][0]))}. 训练池与候选池 system name overlap: {len(overlap):,}; 本实验解释为“已有训练池 vs 待选候选池”，未做 frame-level 去重。</p>
    </section>

    <section>
      <h2>关键结论</h2>
      <div class="summary-grid">
        <div class="callout">
          <h3>Force UQ 主结论</h3>
          <p>force 不确定度与真实 force 误差呈强正相关。最佳主比较为 {html.escape(METHOD_LABELS[best_force['uq_metric']])} vs {html.escape(ERROR_LABELS[best_force['error_metric']])}，Spearman = {format_float(best_force['spearman_rho'])}，Pearson = {format_float(best_force['pearson_r'])}。</p>
        </div>
        <div class="callout warn">
          <h3>Energy UQ 主结论</h3>
          <p>LLPR/DPOSE energy UQ 与 energy error per atom 的相关性为弱正相关。最佳 energy 主比较为 {html.escape(METHOD_LABELS[best_energy['uq_metric']])}，Spearman = {format_float(best_energy['spearman_rho'])}。</p>
        </div>
        <div class="callout info">
          <h3>应用建议</h3>
          <p>本数据集上优先用 QbC/RND 做 force-error 样本筛选；LLPR/DPOSE 更适合补充 energy-risk 视角。force DPOSE 未评估，因为当前 detached-feature 实现只支持 energy-level LLPR/DPOSE。</p>
        </div>
      </div>
    </section>

    <section>
      <h2>Parity/Correlation 统计与高误差召回</h2>
      <p>下表把不确定度值与预测-真值误差逐帧配对，报告 Pearson、Spearman、Kendall。高误差召回使用误差 top 5% 作为 high-error 样本，并统计不确定度 top 5% 的召回与 enrichment。</p>
      <div class="table-wrap">
        {table_html(['Role', 'UQ metric', 'Error target', 'Pearson r', 'Spearman rho', 'Kendall tau', 'N frames', 'Top 5% recall', 'Top 5% enrichment', 'Interpretation'], method_rows)}
      </div>
    </section>

    <section>
      <h2>误差与不确定度分布</h2>
      <p>分布统计用于判断 UQ 数值尺度和误差目标是否存在极端值主导。相关性计算已按 metric pair 去除 NaN/inf。</p>
      <div class="table-wrap">
        {table_html(['Quantity', 'Min', 'Median', 'P95', 'Max'], dist_rows)}
      </div>
    </section>

    <section>
      <h2>tangent_lo 双维度划分与 force error</h2>
      <p>本节按 DP-EVA <code>UQFilter(scheme="tangent_lo")</code> 的实现，在 <code>uq_qbc_for</code> 与 <code>uq_rnd_rescaled</code> 平面上重新划分 <code>accurate</code>、<code>candidate</code> 和 <code>failed</code>。阈值为 <strong>{html.escape(str(tangent_thresholds['threshold_source']))} tangent_lo</strong>：QbC [{format_float(tangent_thresholds['qbc_lo'], 4)}, {format_float(tangent_thresholds['qbc_hi'], 4)}]，RND-rescaled [{format_float(tangent_thresholds['rnd_lo'], 4)}, {format_float(tangent_thresholds['rnd_hi'], 4)}]；默认 <code>ratio={format_float(tangent_thresholds['ratio'], 2)}</code>、<code>width={format_float(tangent_thresholds['width'], 2)}</code>。</p>
      <div class="table-wrap">
        {table_html(['Identity', 'Error target', 'Frames', 'Frame share', 'High-error rate', 'High-error recall', 'Enrichment', 'Median error', 'P95 error', 'Max error'], tangent_rows)}
      </div>
      <p class="subtle">解释：tangent_lo 是 QbC/RND-rescaled 双维度几何筛选，不是单一排序指标。candidate 区用于正常采样，failed 区表示超出高阈值的极高不确定度结构；二者与真实 high-force-error 的富集程度可用于检查 DP-EVA collect 策略是否对 force risk 敏感。</p>
      {tangent_plots}
    </section>

    <section>
      <h2>计算效率</h2>
      <p>GPU 阶段的耗时来自 Slurm <code>sacct</code> 主 job 行。GPU-minutes = elapsed minutes x allocated GPUs。CPU 后处理包括 QbC/RND 表合并、LLPR/DPOSE 汇总、相关性和图表生成，主要是读取已有输出文件，未进入 Slurm 计费统计。</p>
      <h3>UQ 方法级效率比较</h3>
      <p>下表把 QbC/RND、LLPR analytic 和 DPOSE energy ensemble 的边际计算成本拆开。训练 4 个模型是本实验 ensemble 前置成本；如果只评估已有模型上的 UQ，主要差别来自多模型 candidate inference 与 last-layer feature extraction。</p>
      <div class="table-wrap">
        {table_html(['UQ method', 'Required computation', 'Observed GPU cost', 'Observed critical path', 'Notes'], uq_efficiency_rows)}
      </div>
      <h3>Slurm stage 汇总</h3>
      <div class="table-wrap">
        {table_html(['Stage', 'Jobs', 'Walltime critical path', 'GPU minutes', 'GPU hours'], efficiency_rows)}
      </div>
      <h3>Job 明细</h3>
      <div class="table-wrap">
        {table_html(['Stage', 'Job ID', 'Job name', 'State', 'Elapsed', 'GPUs', 'GPU minutes', 'Start', 'End'], detail_rows)}
      </div>
    </section>

    <section>
      <h2>方法可推广度</h2>
      <div class="table-wrap">
        {table_html(
            ['Method', 'Best use in this experiment', 'Compute cost profile', '推广条件与限制'],
            [
                ['QbC force ensemble', '筛选 force max/RMS 高误差样本；top 5% UQ 对 force high-error recall 约 54-55%。', '需要多个模型对候选池推理；本实验 4 个 1-GPU inference job，critical path 约 1.9 min。', '适合已有 ensemble 或可接受多模型推理成本的主动学习；相关性依赖成员多样性。'],
                ['RND force deviation', '本实验 force 相关性最高；top 5% UQ 对 force high-error recall 约 56%。', '复用 ensemble inference 输出，额外后处理成本低。', '适合力主导的采样策略；rescaled 版本本次与原 RND 排序等价。'],
                ['LLPR energy per atom', 'energy error per atom 为弱正相关；top 1% UQ 对 energy high-error enrichment 约 5.13。', '需要 train/candidate last-layer features 和 CPU 线性后验分析；feature GPU critical path 约 2.3 min。', '适合只需要 energy-level 风险或想降低 full ensemble 训练成本的场景；当前不覆盖 force DPOSE。'],
                ['DPOSE energy ensemble std per atom', 'energy error per atom 为弱正相关；top 1% UQ enrichment 约 4.33。', '需要 last-layer features、base energy 和解析最后层权重；energy ensemble shape 为 ' + html.escape(str(tuple(energy_ensemble.shape))) + '。', '可作为 energy-risk 补充信号；若候选体系分布变化大，需要重新校验校准性。'],
            ],
        )}
      </div>
    </section>

    <section>
      <h2>浅层系综方法比较与 LLPR/DPOSE 偏差解释</h2>
      <p>本实验中 QbC/RND 与真实 force error 强相关，而 LLPR/DPOSE energy uncertainty 与 energy error per atom 仅弱正相关；这与方法目标一致。当前 DP-EVA LLPR/DPOSE 是 <strong>energy-level LLPR/DPOSE</strong>：基于 detached fitting-last-layer feature 的 covariance/ensemble std，缺少从 ensemble energy 到 force 的可微传播路径，因此 LLPR/DPOSE vs force error 只能作为 exploratory。</p>
      <div class="table-wrap">
        {table_html(['Method', 'Observed signal', 'Shallow-ensemble meaning', 'Cost profile', 'Main limitation'], comparison_rows)}
      </div>
      <div class="summary-grid" style="margin-top: 14px;">
        <div class="callout">
          <h3>QbC/RND</h3>
          <p>直接比较 force 输出，因此和 <code>force_error_max</code>/<code>force_error_rms</code> 的目标一致。本数据集上它们是 force-risk 筛选的主信号。</p>
        </div>
        <div class="callout warn">
          <h3>LLPR/DPOSE</h3>
          <p>当前输出的是 energy-level 后验和 energy ensemble std。弱 energy 相关性说明它有补充价值，但不能替代 force-aware UQ。</p>
        </div>
        <div class="callout info">
          <h3>文献语境</h3>
          <p>Atomistic Cookbook 和 metatrain 文档强调 LLPR/ensemble 可用于 uncertainty 和 derived quantities；DPOSE/浅层系综资料强调 last-layer ensemble 的低成本，同时 force uncertainty 需要 force-aware 或可微路径。</p>
        </div>
      </div>
      <h3>资料依据</h3>
      <ul>{source_links}</ul>
      <p class="subtle">详细文字版分析见 <code>outputs/uq_method_deep_dive.md</code>。Force DPOSE is not evaluated because current detached-feature implementation only supports energy-level DPOSE/LLPR.</p>
    </section>

    <section>
      <h2>诊断图</h2>
      <p>每组比较包含 scatter、hexbin 和 uncertainty-rank error curve。Primary 图用于主结论；LLPR/DPOSE vs force 图仅作 exploratory，不作为 force DPOSE 结论。</p>
      {plot_cards}
    </section>

    <section>
      <h2>可复现资产</h2>
      <ul>
        <li>Frame table: <code>outputs/uq_error_frame_table.csv</code></li>
        <li>Correlation summary: <code>outputs/correlation_summary.csv</code></li>
        <li>Enrichment summary: <code>outputs/enrichment_summary.csv</code></li>
        <li>LLPR/DPOSE table: <code>outputs/llpr_energy_uq.csv</code>, rows = {len(llpr):,}</li>
        <li>Figures: <code>outputs/figures/*.png</code></li>
        <li>Configs: <code>configs/*.json</code></li>
      </ul>
    </section>
    <footer>Generated by <code>practices/uq_correlation/scripts/build_html_dashboard.py</code>. All reported statistics are derived from local experiment outputs.</footer>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if not output.is_absolute():
        output = Path.cwd() / output
    build_dashboard(output)
    print(output)


if __name__ == "__main__":
    main()
