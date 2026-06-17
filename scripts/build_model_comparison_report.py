#!/usr/bin/env python
"""Build a single-file HTML summary for DP-EVA model comparison reports."""

from __future__ import annotations

import argparse
import base64
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ModelMetric:
    name: str
    family: str
    descriptor_dim: int
    candidates: int
    selected_frames: int
    selected_systems: int
    important_pcs: int
    retained_variance: float
    pc1_dominance: float
    top3_dominance: float
    direct_coverage: float
    random_coverage: float
    novelty: float
    diversity: float
    caveat: str


REPORT_VIEW_PATHS = {
    "DPA3": "test/dpa3-dpeva-test/e2e_dpa3_filter128/dpeva_post_joint_dpa3_filter128/view/Final_sampled_PCAview.png",
    "DPA4 Neo": "test/dpa4-dpeva-test/e2e_dpa4_neo_filter128/dpeva_post_joint_dpa4_filter128/view/Final_sampled_PCAview.png",
    "DPA4 Air": "test/dpa4-dpeva-test/e2e_dpa4_air_filter128/dpeva_post_joint_dpa4_air_filter128/view/Final_sampled_PCAview.png",
    "DPA4 Plus": "test/dpa4-dpeva-test/e2e_dpa4_plus_filter128/dpeva_post_joint_dpa4_plus_filter128/view/Final_sampled_PCAview.png",
    "DPA4 Mini": "test/dpa4-dpeva-test/e2e_dpa4_mini_filter128/dpeva_post_joint_dpa4_mini_filter128/view/Final_sampled_PCAview.png",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def parse_percent(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100
    return float(text)


def parse_first_int(text: object) -> int:
    if pd.isna(text):
        return 0
    return int(str(text).split("/", 1)[0].strip())


def h(text: object) -> str:
    return html.escape(str(text), quote=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required report input not found: {path}")
    return pd.read_csv(path)


def top3_from_pca(pca: pd.DataFrame, run: str) -> float:
    col = "explained_ratio" if "explained_ratio" in pca.columns else "explained_variance_ratio"
    run_pca = pca[pca["run"] == run] if "run" in pca.columns else pca
    return float(run_pca.sort_values("pc")[col].astype(float).head(3).sum())


def load_dpa4_metrics(root: Path) -> tuple[list[ModelMetric], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "test" / "dpa4-dpeva-test"
    metrics = read_csv(base / "neo_air_plus_sampling_metrics.csv")
    pca = read_csv(base / "neo_air_plus_pca_table.csv")
    pools = read_csv(base / "neo_air_plus_pool_counts.csv")
    overlap = read_csv(base / "neo_air_plus_selection_overlap.csv")

    rows: list[ModelMetric] = []
    caveats = {
        "Neo": "紧凑 descriptor，计算链路完整。",
        "Air": "链路完整，规避 Plus train descriptor OOM，活跃维度更丰富。",
        "Plus": "train descriptor 173/177，可复现性受 OOM caveat 影响。",
    }
    for row in metrics.to_dict("records"):
        run = str(row["run"])
        rows.append(
            ModelMetric(
                name=f"DPA4 {run}",
                family="DPA4",
                descriptor_dim=int(row["descriptor_dim"]),
                candidates=int(row["uq_candidate_frames"]),
                selected_frames=int(row["final_selected_frames"]),
                selected_systems=int(row["sampled_systems"]),
                important_pcs=int(row["important_pcs"]),
                retained_variance=float(row["important_pc_cum_var"]),
                pc1_dominance=float(
                    pca[(pca["run"] == run) & (pca["pc"] == 1)]["explained_ratio"].iloc[0]
                ),
                top3_dominance=top3_from_pca(pca, run),
                direct_coverage=float(row["coverage_selected_mean"]),
                random_coverage=float(row["coverage_random_mean"]),
                novelty=float(row["novelty_to_train_median"]),
                diversity=float(row["selected_nn_diversity_median"]),
                caveat=caveats.get(run, ""),
            )
        )
    return rows, pca, pools, overlap


def load_dpa3_metrics(root: Path) -> tuple[list[ModelMetric], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "test" / "dpa3-dpeva-test"
    metrics = read_csv(base / "dpa3_dpa4_sampling_metrics.csv")
    pca = read_csv(base / "dpa3_dpa4_pca_table.csv")
    pools = read_csv(base / "dpa3_dpa4_pool_counts.csv")
    overlap = read_csv(base / "dpa3_dpa4_selection_overlap.csv")

    rows: list[ModelMetric] = []
    for row in metrics.to_dict("records"):
        run = str(row["run"])
        if run != "DPA3":
            continue
        rows.append(
            ModelMetric(
                name="DPA3 baseline",
                family="DPA3 baseline",
                descriptor_dim=int(row["descriptor_dim"]),
                candidates=int(row["uq_candidate_frames"]),
                selected_frames=int(row["final_selected_frames"]),
                selected_systems=int(row["sampled_systems"]),
                important_pcs=int(row["important_pcs"]),
                retained_variance=float(row["important_pc_cum_var"]),
                pc1_dominance=float(
                    pca[(pca["run"] == run) & (pca["pc"] == 1)]["explained_ratio"].iloc[0]
                ),
                top3_dominance=top3_from_pca(pca, run),
                direct_coverage=float(row["coverage_selected_mean"]),
                random_coverage=float(row["coverage_random_mean"]),
                novelty=float(row["novelty_to_train_median"]),
                diversity=float(row["selected_nn_diversity_median"]),
                caveat="DPA3 基线：128 维 descriptor、采样帧少且 novelty 高，但 DIRECT coverage 低于当前 DPA4/MACE 结果。",
            )
        )
    return rows, pca, pools, overlap


def load_mace_metrics(root: Path) -> tuple[list[ModelMetric], dict[str, pd.DataFrame], pd.DataFrame]:
    base = root / "test" / "mace-dpeva-test" / "mini_uq_mace_direct_filter128"
    metrics = read_csv(base / "mace_mini_sampling_metrics.csv")
    overlap = read_csv(base / "mace_mini_selection_overlap.csv")
    pca_by_model: dict[str, pd.DataFrame] = {}
    pca_files = {
        "MACE small": base
        / "dpeva_post_joint_mace_small_mini_uq_filter128"
        / "dataframe"
        / "pca_explained_variance.csv",
        "MACE medium": base
        / "dpeva_post_joint_mace_medium_mini_uq_filter128"
        / "dataframe"
        / "pca_explained_variance.csv",
    }
    for model, path in pca_files.items():
        if path.exists():
            pca_by_model[model] = pd.read_csv(path)

    rows: list[ModelMetric] = []
    for row in metrics.to_dict("records"):
        name = str(row["Model"])
        rows.append(
            ModelMetric(
                name=name,
                family="MACE comparison" if name.startswith("MACE") else "DPA4 baseline",
                descriptor_dim=int(row["Descriptor dim"]),
                candidates=int(row["Candidates"]),
                selected_frames=int(row["Selected frames"]),
                selected_systems=int(row["Selected systems"]),
                important_pcs=int(row["Important PCs"]),
                retained_variance=parse_percent(row["Retained variance"]),
                pc1_dominance=parse_percent(row["PC1 dominance"]),
                top3_dominance=parse_percent(row["Top3 dominance"]),
                direct_coverage=float(row["DIRECT coverage"]),
                random_coverage=float(row["Random coverage"]),
                novelty=float(row["Novelty to train"]),
                diversity=float(row["Diversity"]),
                caveat=(
                    "复用 DPA4 Mini UQ candidate，未重新计算 MACE 模型不确定度。"
                    if name.startswith("MACE")
                    else "DPA4 Mini baseline，与 MACE 使用同一 candidate 集合。"
                ),
            )
        )
    return rows, pca_by_model, overlap


def bar_svg(
    rows: Iterable[tuple[str, float]],
    *,
    title: str,
    unit: str = "",
    width: int = 760,
    bar_color: str = "#256f6c",
) -> str:
    data = list(rows)
    max_value = max((value for _, value in data), default=1) or 1
    row_h = 34
    left = 170
    right = 92
    top = 46
    height = top + row_h * len(data) + 18
    plot_w = width - left - right
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{h(title)}">',
        f'<text x="0" y="22" class="svg-title">{h(title)}</text>',
    ]
    for i, (label, value) in enumerate(data):
        y = top + i * row_h
        bar_w = max(2, plot_w * value / max_value)
        parts.append(f'<text x="0" y="{y + 17}" class="svg-label">{h(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="18" rx="5" class="svg-track" />')
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="18" rx="5" fill="{bar_color}" />'
        )
        parts.append(
            f'<text x="{left + plot_w + 10}" y="{y + 15}" class="svg-value">{value:.3g}{h(unit)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def grouped_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{h(item)}</th>" for item in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{h(item)}</td>" for item in row) + "</tr>")
    return f'<table class="data-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def image_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def embedded_png_gallery(root: Path, *, include_png: bool) -> str:
    if not include_png:
        return '<p class="muted">当前报告使用轻量模式，未嵌入原始 PNG 视图。</p>'
    cards = []
    for label, rel_path in REPORT_VIEW_PATHS.items():
        uri = image_data_uri(root / rel_path)
        if not uri:
            continue
        cards.append(
            f"""
            <figure class="figure-card">
              <img src="{uri}" alt="{h(label)} Final sampled PCA view">
              <figcaption>{h(label)} 原始 DP-EVA PCA 采样视图</figcaption>
            </figure>
            """
        )
    if not cards:
        return '<p class="muted">未找到可嵌入的 DPA3/DPA4 原始 PNG 视图。</p>'
    return f'<div class="figure-grid">{"".join(cards)}</div>'


def ranking_cards(dpa3: list[ModelMetric], dpa4: list[ModelMetric], mace: list[ModelMetric]) -> str:
    dpa3_base = dpa3[0] if dpa3 else None
    air = next(item for item in dpa4 if item.name == "DPA4 Air")
    plus = next(item for item in dpa4 if item.name == "DPA4 Plus")
    mace_small = next(item for item in mace if item.name == "MACE small")
    mini = next(item for item in mace if item.name == "DPA4 Mini")
    cards = [
        (
            "综合讨论首选",
            "DPA4 Air",
            f"{air.important_pcs} 个 important PCs，novelty median {air.novelty:.3f}，链路完整且无 Plus OOM caveat。",
        ),
        (
            "采样效率最佳",
            "MACE small",
            f"{mace_small.selected_frames} selected frames，DIRECT coverage {mace_small.direct_coverage:.4f}，但复用 Mini UQ。",
        ),
        (
            "训练集 novelty 最高",
            "DPA4 Mini",
            f"novelty {mini.novelty:.4f}，适合作为 MACE descriptor 对照 baseline。",
        ),
        (
            "最大模型 caveat",
            "DPA4 Plus",
            f"{plus.selected_frames} selected frames，4 PCs 解释 {pct(plus.retained_variance)}，但 train descriptor 仅 173/177。",
        ),
    ]
    if dpa3_base:
        cards.append(
            (
                "历史基线参照",
                "DPA3 baseline",
                f"128 维 descriptor，{dpa3_base.selected_frames} selected frames，novelty {dpa3_base.novelty:.4f}，coverage {dpa3_base.direct_coverage:.4f}。",
            )
        )
    return "".join(
        f"""
        <article class="rank-card">
          <span>{h(kicker)}</span>
          <strong>{h(model)}</strong>
          <p>{h(detail)}</p>
        </article>
        """
        for kicker, model, detail in cards
    )


def source_evidence_cards() -> str:
    evidence = [
        (
            "CLI 一次读取 system 全部测试帧",
            "deepmd/entrypoints/eval_desc.py:97-132",
            "data.get_test() 后取 nframes，把 coord / box / atype 一次性传入 dp.eval_descriptor()；这不是 MD 式逐帧 streaming。",
        ),
        (
            "descriptor 依赖完整 forward",
            "deepmd/pt/infer/deep_eval.py:875-884",
            "eval_descriptor() 先打开 descriptor hook，然后调用 self.eval(..., atomic=False)，再从 hook 缓存取 descriptor。",
        ),
        (
            "atomic=False 仍请求主输出",
            "deepmd/pt/infer/deep_eval.py:447-485",
            "atomic=False 不是 descriptor-only；源码注释和筛选逻辑显示仍请求 energy / force / virial 相关 output defs。",
        ),
        (
            "descriptor hook 后仍进入 fitting",
            "deepmd/pt/model/atomic_model/dp_atomic_model.py:293-298",
            "descriptor 被 append 到 eval_descriptor_list 后，forward 继续执行 fitting_net，因此峰值内存包含后续能量/力/virial路径。",
        ),
    ]
    return "".join(
        f"""
        <article class="evidence-card">
          <span>{h(title)}</span>
          <code>{h(path)}</code>
          <p>{h(detail)}</p>
        </article>
        """
        for title, path, detail in evidence
    )


def dpeva_algorithm_section() -> str:
    cards = [
        (
            "UQ 候选池",
            "先通过模型不确定度筛出值得关注的结构区域。DPA4 链路使用各模型自己的候选池；MACE small/medium 当前复用 DPA4 Mini UQ candidate，因此它们主要比较 descriptor 与采样阶段。",
        ),
        (
            "descriptor 表征空间",
            "把候选构型映射到模型内部表征流形，再用 standardized descriptor 做 PCA。important PCs、PC1 dominance 和 retained variance 用来判断表征是否过度集中，或是否保留了更丰富的结构差异。",
        ),
        (
            "DIRECT / BIRCH 多样性采样",
            "在 UQ 候选中用 descriptor 空间覆盖和聚类控制多样性。coverage 相对 random 的提升，是评价 DP-EVA 采样是否真的优于随机抽取的核心证据。",
        ),
        (
            "Kaiser rule / PCA 口径",
            "important PCs 按 eigenvalue > 1 的 Kaiser rule 统计。PC 数更多不必然代表模型更强，必须结合 coverage、novelty、采样帧数和 caveat 一起解释。",
        ),
    ]
    card_html = "".join(
        f"""
        <article class="algorithm-card">
          <span>{h(title)}</span>
          <p>{h(text)}</p>
        </article>
        """
        for title, text in cards
    )
    return f"""
  <h2 id="dpeva-algorithm">DP-EVA 算法讨论点</h2>
  <div class="panel algorithm-panel">
    <p><strong>本看板需要补充算法叙事：</strong>已有图表能展示结果，但组会讨论还需要说明 DP-EVA 如何把“不确定区域”和“descriptor 表征能力”连接到最终采样。下面把当前实验统一解释为一条主动学习链路，而不是孤立模型排名。</p>
    <div class="pipeline" aria-label="DP-EVA 主动学习流程">
      <span>Train / Fine-tune</span>
      <span>Infer + UQ</span>
      <span>Descriptor</span>
      <span>PCA / Kaiser</span>
      <span>DIRECT / BIRCH</span>
      <span>Label + Integrate</span>
    </div>
    <div class="algorithm-grid">{card_html}</div>
    <div class="discussion-box">
      <strong>导师讨论问题</strong>
      <ul>
        <li>是否把链路完整、novelty 较高且无 Plus OOM caveat 的 DPA4 Air 作为下一轮 DPA4 baseline？</li>
        <li>MACE small 的低采样帧高 coverage，是 descriptor 压缩更有效，还是复用 Mini UQ candidate 带来的口径限制？</li>
        <li>DPA4 Plus 是否值得先做 descriptor-only 或 per-system batching 工程优化后，再纳入公平比较？</li>
        <li>下一轮主动学习应固定 UQ source 比 descriptor，还是固定 descriptor 比 UQ source？</li>
      </ul>
    </div>
  </div>
"""


def metrics_table(rows: list[ModelMetric]) -> str:
    return grouped_table(
        [
            "模型",
            "候选帧",
            "采样帧",
            "采样体系",
            "Descriptor dim",
            "Important PCs",
            "Retained variance",
            "PC1 dominance",
            "DIRECT coverage",
            "Random coverage",
            "Novelty",
            "Diversity",
        ],
        [
            [
                row.name,
                row.candidates,
                row.selected_frames,
                row.selected_systems,
                row.descriptor_dim,
                row.important_pcs,
                pct(row.retained_variance),
                pct(row.pc1_dominance),
                f"{row.direct_coverage:.4f}",
                f"{row.random_coverage:.4f}",
                f"{row.novelty:.4f}",
                f"{row.diversity:.4f}",
            ]
            for row in rows
        ],
    )


def pca_svg_from_table(pca: pd.DataFrame, *, title: str) -> str:
    rows = []
    for item in pca.sort_values("pc").head(8).to_dict("records"):
        value = item.get("explained_ratio", item.get("explained_variance_ratio", 0.0))
        rows.append((f"PC{int(item['pc'])}", float(value) * 100))
    return bar_svg(rows, title=title, unit="%", bar_color="#8c5f20")


def render_html(
    *,
    root: Path,
    dpa3_rows: list[ModelMetric],
    dpa3_pca: pd.DataFrame,
    dpa3_pools: pd.DataFrame,
    dpa3_overlap: pd.DataFrame,
    dpa4_rows: list[ModelMetric],
    dpa4_pca: pd.DataFrame,
    dpa4_pools: pd.DataFrame,
    dpa4_overlap: pd.DataFrame,
    mace_rows: list[ModelMetric],
    mace_pca: dict[str, pd.DataFrame],
    mace_overlap: pd.DataFrame,
    include_png: bool,
) -> str:
    all_rows = dpa3_rows + dpa4_rows + mace_rows
    selected_svg = bar_svg(
        [(row.name, float(row.selected_frames)) for row in all_rows],
        title="最终采样帧数：越少代表压缩效率越高",
        unit="",
        bar_color="#1f6f8b",
    )
    pc_svg = bar_svg(
        [(row.name, float(row.important_pcs)) for row in all_rows],
        title="Important PCs：descriptor 活跃维度",
        unit="",
        bar_color="#6a6f2b",
    )
    coverage_svg = bar_svg(
        [(row.name, row.direct_coverage) for row in all_rows],
        title="DIRECT coverage：采样覆盖能力",
        unit="",
        bar_color="#2d7d46",
    )
    novelty_svg = bar_svg(
        [(row.name, row.novelty) for row in all_rows],
        title="Novelty to train：相对训练集新颖性",
        unit="",
        bar_color="#7b4f8f",
    )

    dpa4_pca_svgs = "".join(
        f'<div class="chart-card">{pca_svg_from_table(dpa4_pca[dpa4_pca["run"] == run], title=f"DPA4 {run} PCA 方差")}</div>'
        for run in ["Neo", "Air", "Plus"]
    )
    dpa3_pca_svgs = "".join(
        f'<div class="chart-card">{pca_svg_from_table(dpa3_pca[dpa3_pca["run"] == run], title=f"{run} PCA 方差")}</div>'
        for run in ["DPA3"]
        if not dpa3_pca[dpa3_pca["run"] == run].empty
    )
    mace_pca_svgs = "".join(
        f'<div class="chart-card">{pca_svg_from_table(frame, title=f"{name} PCA 方差")}</div>'
        for name, frame in mace_pca.items()
    )

    dpa3_pool_table = grouped_table(
        ["Run", "Pool", "Sampled frames"],
        [[row["run"], row["pool"], row["sampled_frames"]] for row in dpa3_pools.to_dict("records")],
    )
    dpa3_overlap_table = grouped_table(
        list(dpa3_overlap.columns),
        [list(row.values()) for row in dpa3_overlap.to_dict("records")],
    )
    dpa4_pool_table = grouped_table(
        ["Run", "Pool", "Sampled frames"],
        [[row["run"], row["pool"], row["sampled_frames"]] for row in dpa4_pools.to_dict("records")],
    )
    dpa4_overlap_table = grouped_table(
        list(dpa4_overlap.columns),
        [list(row.values()) for row in dpa4_overlap.to_dict("records")],
    )
    mace_overlap_table = grouped_table(
        list(mace_overlap.columns),
        [list(row.values()) for row in mace_overlap.to_dict("records")],
    )
    dpa3_baseline = dpa3_rows[0] if dpa3_rows else None
    dpa3_summary = (
        f"DPA3 baseline：128 维 descriptor，{dpa3_baseline.selected_frames} selected frames，"
        f"{dpa3_baseline.important_pcs} 个 important PCs，novelty median={dpa3_baseline.novelty:.4f}，"
        f"DIRECT coverage={dpa3_baseline.direct_coverage:.4f}。"
        if dpa3_baseline
        else "DPA3 baseline：未找到可用结果。"
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DP-EVA 模型表征与采样统一展示报告</title>
  <style>
    :root {{
      --ink: #17201d;
      --muted: #65706b;
      --line: #dbe3df;
      --paper: #fbfcfa;
      --panel: #ffffff;
      --green: #256f6c;
      --blue: #1f6f8b;
      --gold: #8c6a22;
      --warn: #9a5b17;
      --soft: #eef5f1;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: "Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 18% 0%, rgba(37, 111, 108, .15), transparent 28%),
        linear-gradient(180deg, #eef5f1 0, #fbfcfa 360px);
      line-height: 1.58;
    }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 34px 28px 56px; }}
    h1 {{ margin: 0; font-size: 34px; line-height: 1.18; letter-spacing: 0; }}
    h2 {{ margin: 42px 0 14px; font-size: 23px; border-bottom: 1px solid var(--line); padding-bottom: 8px; }}
    h3 {{ margin: 24px 0 10px; font-size: 17px; }}
    p {{ margin: 8px 0 12px; }}
    .eyebrow {{ color: var(--green); font-weight: 700; margin-bottom: 8px; }}
    .subtitle {{ max-width: 860px; color: #34413c; font-size: 16px; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(300px, .8fr);
      gap: 22px;
      align-items: stretch;
      margin-bottom: 24px;
    }}
    .top-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 18px;
    }}
    .top-nav a {{
      color: #24403a;
      text-decoration: none;
      border: 1px solid rgba(37,111,108,.24);
      background: rgba(255,255,255,.72);
      border-radius: 999px;
      padding: 5px 11px;
      font-size: 13px;
      font-weight: 700;
    }}
    .hero-panel, .panel, .chart-card, .rank-card, .figure-card {{
      background: rgba(255,255,255,.92);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 12px 34px rgba(23, 32, 29, .07);
    }}
    .hero-panel {{ padding: 24px; }}
    .summary-list {{ margin: 16px 0 0; padding-left: 20px; }}
    .summary-list li {{ margin: 7px 0; }}
    .rank-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .rank-card {{ padding: 16px; min-height: 150px; }}
    .rank-card span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .rank-card strong {{ display: block; margin-top: 8px; font-size: 22px; color: var(--green); }}
    .rank-card p {{ color: #33423d; font-size: 13px; }}
    .panel {{ padding: 18px; margin: 16px 0; overflow-x: auto; }}
    .algorithm-panel {{ overflow: visible; }}
    .pipeline {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 8px;
      margin: 16px 0;
    }}
    .pipeline span {{
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 54px;
      border: 1px solid rgba(37,111,108,.22);
      border-radius: 8px;
      background: linear-gradient(180deg, #f4faf7, #ffffff);
      color: #203e38;
      font-size: 13px;
      font-weight: 800;
      text-align: center;
      padding: 8px 10px;
    }}
    .algorithm-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .algorithm-card {{
      border: 1px solid #d7ded9;
      border-radius: 8px;
      background: #fbfdfb;
      padding: 14px;
      min-height: 190px;
    }}
    .algorithm-card span {{ display: block; color: var(--green); font-weight: 800; margin-bottom: 8px; }}
    .algorithm-card p {{ color: #34423e; font-size: 13px; }}
    .discussion-box {{
      margin-top: 14px;
      border: 1px solid rgba(31,111,139,.25);
      border-radius: 8px;
      background: linear-gradient(135deg, #f0f7fa, #ffffff);
      padding: 14px 16px;
    }}
    .discussion-box strong {{ color: var(--blue); }}
    .discussion-box ul {{ margin: 10px 0 0; padding-left: 20px; }}
    .discussion-box li {{ margin: 6px 0; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .chart-card {{ padding: 16px; overflow: hidden; }}
    .evidence-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }}
    .evidence-card {{
      border: 1px solid #d7ded9;
      border-radius: 8px;
      background: linear-gradient(180deg, #ffffff, #f7faf8);
      padding: 14px;
      min-height: 176px;
    }}
    .evidence-card span {{ display: block; font-weight: 800; color: #1f514e; margin-bottom: 8px; }}
    .evidence-card code {{
      display: block;
      white-space: normal;
      background: #eef4f1;
      color: #31423d;
      padding: 6px 7px;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.35;
    }}
    .evidence-card p {{ font-size: 13px; color: #3c4945; }}
    .svg-title {{ font-size: 18px; font-weight: 700; fill: var(--ink); }}
    .svg-label {{ font-size: 13px; fill: #35443f; }}
    .svg-value {{ font-size: 12px; fill: var(--muted); }}
    .svg-track {{ fill: #e7eeea; }}
    .data-table {{ border-collapse: collapse; width: 100%; min-width: 860px; font-size: 13px; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: right; vertical-align: top; }}
    .data-table th {{ background: #edf3f0; color: #24302c; font-weight: 700; position: sticky; top: 0; }}
    .data-table th:first-child, .data-table td:first-child,
    .data-table th:nth-child(2), .data-table td:nth-child(2) {{ text-align: left; }}
    .callout {{ border-left: 4px solid var(--warn); background: #fff7e8; padding: 13px 16px; margin: 14px 0; }}
    .source-callout {{
      border: 1px solid #ead2a9;
      background: linear-gradient(135deg, #fff8eb, #ffffff);
      border-radius: 8px;
      padding: 16px 18px;
      margin: 16px 0;
    }}
    .source-callout strong {{ color: #87510f; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .figure-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .figure-card {{ margin: 0; padding: 12px; }}
    .figure-card img {{ width: 100%; display: block; border-radius: 6px; border: 1px solid var(--line); }}
    .figure-card figcaption {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 900px) {{
      main {{ padding: 24px 16px 40px; }}
      .hero, .rank-grid, .chart-grid, .figure-grid, .evidence-grid, .algorithm-grid, .pipeline {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
<main>
  <nav class="top-nav" aria-label="报告章节">
    <a href="#summary">Executive summary</a>
    <a href="#ranking">模型排序</a>
    <a href="#dpeva-algorithm">DP-EVA 算法</a>
    <a href="#charts">关键图</a>
    <a href="#dpa3">DPA3 基线</a>
    <a href="#dpa4">DPA4</a>
    <a href="#mace">MACE</a>
    <a href="#deepmd-root-cause">DeepMD 根因</a>
    <a href="#limitations">限制说明</a>
  </nav>
  <section class="hero" id="summary">
    <div class="hero-panel">
      <div class="eyebrow">Executive Summary</div>
      <h1>DP-EVA 模型表征与采样统一展示报告</h1>
      <p class="subtitle">本页把 DPA3 baseline、DPA4 Neo/Air/Plus/Mini 与 MACE small/medium 的当前测试结果放入同一套叙事：descriptor 表征形态、DIRECT 采样效率、coverage、novelty 和可复现性限制。</p>
      <ul class="summary-list">
        <li><strong>综合讨论首选：</strong>DPA4 Air，在完整 DP-EVA 链路下兼顾 descriptor 活跃维度、novelty 与可复现性。</li>
        <li><strong>DPA3 基线定位：</strong>{h(dpa3_summary)}</li>
        <li><strong>采样效率最强：</strong>MACE small，仅 87 帧达到最高 DIRECT coverage，但它复用 DPA4 Mini UQ candidate。</li>
        <li><strong>最大限制：</strong>DPA4 Plus 出现 train descriptor 173/177 caveat；MACE small/medium 未重新计算模型不确定度。</li>
      </ul>
    </div>
    <div class="rank-grid">{ranking_cards(dpa3_rows, dpa4_rows, mace_rows)}</div>
  </section>

  <h2 id="ranking">模型综合排序</h2>
  <div class="panel">
    <p><strong>表征/采样综合口径：</strong>优先看链路完整性、important PCs、PCA dominance、DIRECT coverage、novelty、采样帧数和已知限制。该排序用于组会讨论，不是新的绝对评分体系。</p>
    <ol>
      <li><strong>DPA4 Air：</strong>链路完整，8 个 important PCs，selected frames=440，novelty median=1.6674，适合作为当前 DPA4 主讨论对象。</li>
      <li><strong>MACE small：</strong>selected frames=87 且 coverage=0.9070，descriptor 压缩效率突出；但仅代表 “Mini UQ + MACE descriptor + DIRECT”。</li>
      <li><strong>MACE medium：</strong>coverage 与 diversity 接近 small，采样帧数更高，descriptor 分布更平缓。</li>
      <li><strong>DPA3 baseline：</strong>提供历史基线参照；selected frames 少、novelty 高，但 DIRECT coverage 低于当前 DPA4/MACE 结果。</li>
      <li><strong>DPA4 Mini / Neo / Plus：</strong>Mini novelty 最高；Neo 最紧凑；Plus 规模最大但 PCA 更集中且有 descriptor OOM caveat。</li>
    </ol>
  </div>

  {dpeva_algorithm_section()}

  <h2 id="charts">关键摘要图</h2>
  <div class="chart-grid">
    <div class="chart-card">{selected_svg}</div>
    <div class="chart-card">{pc_svg}</div>
    <div class="chart-card">{coverage_svg}</div>
    <div class="chart-card">{novelty_svg}</div>
  </div>

  <h2 id="dpa3">DPA3 基线</h2>
  <div class="panel">
    <p><strong>DPA3 vs DPA4 baseline：</strong>DPA3 结果来自 <code>test/dpa3-dpeva-test/dpa3_dpa4_sampling_metrics.csv</code>。它适合作为历史基线和 descriptor 表征参照，而不应替代 DPA4/MACE 的当前主比较。</p>
    {metrics_table(dpa3_rows)}
  </div>
  <div class="chart-grid">{dpa3_pca_svgs}</div>
  <h3>DPA3 pool 分布</h3>
  <div class="panel">{dpa3_pool_table}</div>
  <h3>DPA3 / DPA4 selection overlap</h3>
  <div class="panel">{dpa3_overlap_table}</div>

  <h2 id="dpa4">DPA4 Neo / Air / Plus</h2>
  <div class="panel">
    <p>DPA4 三个模型使用同一 DP-EVA 设置完成 train -> infer -> feature -> collect 链路。Air 完整通过；Plus 的 train descriptor 存在 173/177 caveat。</p>
    {metrics_table(dpa4_rows)}
  </div>
  <div class="chart-grid">{dpa4_pca_svgs}</div>
  <h3>DPA4 pool 分布</h3>
  <div class="panel">{dpa4_pool_table}</div>
  <h3>DPA4 selection overlap</h3>
  <div class="panel">{dpa4_overlap_table}</div>

  <h2 id="mace">DPA4 Mini / MACE small / MACE medium</h2>
  <div class="callout">MACE small/medium 未重新计算模型不确定度；这里比较的是 “DPA4 Mini UQ candidate + 各模型 descriptor + DIRECT” 的采样行为。</div>
  <div class="panel">{metrics_table(mace_rows)}</div>
  <div class="chart-grid">{mace_pca_svgs}</div>
  <h3>MACE selection overlap</h3>
  <div class="panel">{mace_overlap_table}</div>

  <h2>原始 DP-EVA 视图证据</h2>
  <p class="muted">以下嵌入 DPA3/DPA4 各 run 的原始 Final sampled PCA view，用于和摘要指标交叉讨论。MACE 当前测试目录没有对应 PNG，已在上方用 CSV/JSON 生成 SVG 摘要图。</p>
  {embedded_png_gallery(root, include_png=include_png)}

  <h2 id="deepmd-root-cause">DeepMD eval-desc 源码根因</h2>
  <div class="source-callout">
    <strong>结论：</strong>DPA4 Plus 的 descriptor OOM 风险来自 DeepMD `dp eval-desc` 的实现路径：它对单个 system 采用整 system 全帧批量传入，`eval_descriptor()` 并不是纯 descriptor kernel，而是通过 descriptor hook 包住一次完整 `self.eval(... atomic=False)` forward；而 `atomic=False` 仍会请求 energy / force / virial 相关输出。因此峰值显存/内存会同时受系统帧数、原子数、descriptor 中间量和能量/力/virial forward 路径影响。
  </div>
  <div class="evidence-grid">{source_evidence_cards()}</div>
  <div class="panel">
    <p><strong>对本报告的解释影响：</strong>Plus 的 173/177 train descriptor caveat 不应简单理解为“descriptor 本身过大”。更准确的表述是：`eval-desc` 当前路径会先对一个 system 的全部测试帧做完整模型前向，再从 hook 中取 descriptor；这解释了为什么 DPA4 Plus 在大 system 上更容易触发内存峰值，而 Air/Neo 更稳定。</p>
    <p><strong>工程启示：</strong>后续如果要复现实验或向 DeepMD 反馈修复方向，应优先讨论 `eval-desc` 是否能提供 descriptor-only path，或至少支持按 frame/batch 分块处理单个 system。</p>
  </div>

  <h2 id="limitations">限制说明</h2>
  <div class="panel">
    <ul>
      <li><strong>不要过度外推到模型训练性能：</strong>本报告比较的是 DP-EVA 采样行为和 descriptor 表征，不是最终势函数精度排名。</li>
      <li><strong>DPA3 baseline caveat：</strong>DPA3 基线已纳入统一图表和基线章节，但它的角色是历史参照；建议优先用 coverage、novelty、selected frames 和 overlap 解释其差异。</li>
      <li><strong>Plus train descriptor caveat：</strong>DPA4 Plus 仅完成 173/177 train descriptor extraction，复现实验时必须优先解决内存问题。</li>
      <li><strong>MACE UQ caveat：</strong>MACE small/medium 复用 DPA4 Mini UQ candidate，未重新计算 MACE ensemble 或 MACE 不确定度。</li>
      <li><strong>BIRCH threshold caveat：</strong>MACE 256 维 descriptor 使用 effective threshold 预探测，和 DPA4 原始 threshold 不完全等价。</li>
      <li><strong>展示口径：</strong>综合排序服务于课题组讨论，建议结合图、表和 caveat 一起解释。</li>
    </ul>
  </div>
</main>
</body>
</html>
"""


def build_report(root: Path, output: Path, *, include_png: bool) -> Path:
    dpa3_rows, dpa3_pca, dpa3_pools, dpa3_overlap = load_dpa3_metrics(root)
    dpa4_rows, dpa4_pca, dpa4_pools, dpa4_overlap = load_dpa4_metrics(root)
    mace_rows, mace_pca, mace_overlap = load_mace_metrics(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_html(
            root=root,
            dpa3_rows=dpa3_rows,
            dpa3_pca=dpa3_pca,
            dpa3_pools=dpa3_pools,
            dpa3_overlap=dpa3_overlap,
            dpa4_rows=dpa4_rows,
            dpa4_pca=dpa4_pca,
            dpa4_pools=dpa4_pools,
            dpa4_overlap=dpa4_overlap,
            mace_rows=mace_rows,
            mace_pca=mace_pca,
            mace_overlap=mace_overlap,
            include_png=include_png,
        ),
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output", type=Path, default=Path("test/DP_EVA_MODEL_COMPARISON_SUMMARY.html"))
    parser.add_argument("--no-embed-png", action="store_true", help="Skip base64 PNG embedding and keep SVG summaries.")
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output
    built = build_report(root, output.resolve(), include_png=not args.no_embed_png)
    print(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
