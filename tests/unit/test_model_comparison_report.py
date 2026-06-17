import importlib.util
import sys
from pathlib import Path

import pandas as pd


def load_report_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_model_comparison_report.py"
    spec = importlib.util.spec_from_file_location("build_model_comparison_report", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xde\xfc\xfb"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def seed_report_inputs(root: Path) -> None:
    dpa4 = root / "test" / "dpa4-dpeva-test"
    dpa3 = root / "test" / "dpa3-dpeva-test"
    mace = root / "test" / "mace-dpeva-test" / "mini_uq_mace_direct_filter128"
    dpa4.mkdir(parents=True, exist_ok=True)
    dpa3.mkdir(parents=True, exist_ok=True)
    mace.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "run": "Neo",
                "base_model": "neo.pt",
                "params": 760000,
                "descriptor_dim": 32,
                "train_desc_usable": 177,
                "train_desc_expected": 177,
                "pool_frames": 68001,
                "uq_candidate_frames": 21254,
                "final_selected_frames": 354,
                "important_pcs": 6,
                "important_pc_cum_var": 0.8774,
                "coverage_selected_mean": 0.8081,
                "coverage_random_mean": 0.7945,
                "coverage_lift_mean": 1.0170,
                "novelty_to_train_mean": 1.1889,
                "novelty_to_train_median": 1.0350,
                "selected_nn_diversity_mean": 1.62,
                "selected_nn_diversity_median": 1.425,
                "workflow_finished": True,
                "descriptor_note": "compact descriptor",
                "sampled_systems": 176,
                "sampled_frames_exported": 354,
                "other_systems": 1095,
                "other_frames_exported": 67647,
            },
            {
                "run": "Air",
                "base_model": "air.pt",
                "params": 2760000,
                "descriptor_dim": 64,
                "train_desc_usable": 177,
                "train_desc_expected": 177,
                "pool_frames": 68001,
                "uq_candidate_frames": 19728,
                "final_selected_frames": 440,
                "important_pcs": 8,
                "important_pc_cum_var": 0.9157,
                "coverage_selected_mean": 0.7801,
                "coverage_random_mean": 0.7536,
                "coverage_lift_mean": 1.0352,
                "novelty_to_train_mean": 1.9681,
                "novelty_to_train_median": 1.6674,
                "selected_nn_diversity_mean": 2.05,
                "selected_nn_diversity_median": 1.8026,
                "workflow_finished": True,
                "descriptor_note": "richer active dimensions",
                "sampled_systems": 192,
                "sampled_frames_exported": 440,
                "other_systems": 1093,
                "other_frames_exported": 67561,
            },
            {
                "run": "Plus",
                "base_model": "plus.pt",
                "params": 16036686,
                "descriptor_dim": 64,
                "train_desc_usable": 173,
                "train_desc_expected": 177,
                "pool_frames": 68001,
                "uq_candidate_frames": 17852,
                "final_selected_frames": 523,
                "important_pcs": 4,
                "important_pc_cum_var": 0.9760,
                "coverage_selected_mean": 0.8000,
                "coverage_random_mean": 0.7276,
                "coverage_lift_mean": 1.0995,
                "novelty_to_train_mean": 1.1689,
                "novelty_to_train_median": 0.7986,
                "selected_nn_diversity_mean": 0.899,
                "selected_nn_diversity_median": 0.7074,
                "workflow_finished": True,
                "descriptor_note": "OOM caveat",
                "sampled_systems": 224,
                "sampled_frames_exported": 523,
                "other_systems": 1094,
                "other_frames_exported": 67478,
            },
        ]
    ).to_csv(dpa4 / "neo_air_plus_sampling_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"run": "Neo", "pc": 1, "eigenvalue": 13.2, "explained_ratio": 0.41, "cumulative_ratio": 0.41, "important": True},
            {"run": "Neo", "pc": 2, "eigenvalue": 6.4, "explained_ratio": 0.20, "cumulative_ratio": 0.61, "important": True},
            {"run": "Air", "pc": 1, "eigenvalue": 34.4, "explained_ratio": 0.54, "cumulative_ratio": 0.54, "important": True},
            {"run": "Air", "pc": 2, "eigenvalue": 9.0, "explained_ratio": 0.14, "cumulative_ratio": 0.68, "important": True},
            {"run": "Plus", "pc": 1, "eigenvalue": 46.6, "explained_ratio": 0.73, "cumulative_ratio": 0.73, "important": True},
            {"run": "Plus", "pc": 2, "eigenvalue": 12.7, "explained_ratio": 0.20, "cumulative_ratio": 0.93, "important": True},
        ]
    ).to_csv(dpa4 / "neo_air_plus_pca_table.csv", index=False)
    pd.DataFrame(
        [
            {"run": "Neo", "pool": "pool-a", "sampled_frames": 10},
            {"run": "Air", "pool": "pool-a", "sampled_frames": 12},
            {"run": "Plus", "pool": "pool-a", "sampled_frames": 15},
        ]
    ).to_csv(dpa4 / "neo_air_plus_pool_counts.csv", index=False)
    pd.DataFrame(
        [
            {"run_a": "Neo", "run_b": "Plus", "selected_a": 354, "selected_b": 523, "intersection": 16, "union": 861, "jaccard": 0.0186},
            {"run_a": "Air", "run_b": "Neo", "selected_a": 440, "selected_b": 354, "intersection": 24, "union": 770, "jaccard": 0.0312},
        ]
    ).to_csv(dpa4 / "neo_air_plus_selection_overlap.csv", index=False)

    pd.DataFrame(
        [
            {
                "run": "DPA3",
                "family": "DPA3",
                "base_model": "dpa3.pt",
                "params": 3000000,
                "descriptor_dim": 128,
                "train_desc_usable": 177,
                "train_desc_expected": 177,
                "pool_frames": 68001,
                "uq_candidate_frames": 20839,
                "final_selected_frames": 88,
                "important_pcs": 18,
                "important_pc_cum_var": 0.8620,
                "coverage_selected_mean": 0.6593,
                "coverage_random_mean": 0.3876,
                "coverage_lift_mean": 1.7660,
                "novelty_to_train_mean": 4.7454,
                "novelty_to_train_median": 4.7779,
                "selected_nn_diversity_mean": 2.1633,
                "selected_nn_diversity_median": 1.9648,
                "workflow_finished": True,
                "sampled_systems": 77,
                "sampled_frames_exported": 88,
                "other_systems": 1092,
                "other_frames_exported": 67913,
            },
            {
                "run": "Mini",
                "family": "DPA4",
                "base_model": "mini.pt",
                "params": 443812,
                "descriptor_dim": 32,
                "train_desc_usable": 177,
                "train_desc_expected": 177,
                "pool_frames": 68001,
                "uq_candidate_frames": 22605,
                "final_selected_frames": 359,
                "important_pcs": 6,
                "important_pc_cum_var": 0.8469,
                "coverage_selected_mean": 0.8927,
                "coverage_random_mean": 0.7087,
                "coverage_lift_mean": 1.2655,
                "novelty_to_train_mean": 2.8893,
                "novelty_to_train_median": 2.7309,
                "selected_nn_diversity_mean": 0.4450,
                "selected_nn_diversity_median": 0.3973,
                "workflow_finished": True,
                "sampled_systems": 181,
                "sampled_frames_exported": 359,
                "other_systems": 1093,
                "other_frames_exported": 67642,
            },
        ]
    ).to_csv(dpa3 / "dpa3_dpa4_sampling_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"run": "DPA3", "pc": 1, "eigenvalue": 61.7, "explained_ratio": 0.4818, "cumulative_ratio": 0.4818, "important": True},
            {"run": "DPA3", "pc": 2, "eigenvalue": 12.5, "explained_ratio": 0.0978, "cumulative_ratio": 0.5796, "important": True},
            {"run": "DPA3", "pc": 3, "eigenvalue": 9.6, "explained_ratio": 0.0752, "cumulative_ratio": 0.6548, "important": True},
            {"run": "Mini", "pc": 1, "eigenvalue": 12.7, "explained_ratio": 0.3955, "cumulative_ratio": 0.3955, "important": True},
        ]
    ).to_csv(dpa3 / "dpa3_dpa4_pca_table.csv", index=False)
    pd.DataFrame(
        [
            {"run": "DPA3", "pool": "pool-a", "sampled_frames": 5},
            {"run": "Mini", "pool": "pool-a", "sampled_frames": 20},
        ]
    ).to_csv(dpa3 / "dpa3_dpa4_pool_counts.csv", index=False)
    pd.DataFrame(
        [
            {"run_a": "DPA3", "run_b": "Mini", "selected_a": 88, "selected_b": 359, "intersection": 3, "union": 444, "jaccard": 0.0068},
        ]
    ).to_csv(dpa3 / "dpa3_dpa4_selection_overlap.csv", index=False)

    pd.DataFrame(
        [
            {
                "Model": "DPA4 Mini",
                "UQ source": "dpa4_mini",
                "Descriptor dim": 32,
                "Candidates": 22605,
                "Selected frames": 359,
                "Selected systems": 181,
                "Sample rate": "1.59%",
                "Important PCs": 6,
                "Retained variance": "84.69%",
                "PC1 dominance": "39.55%",
                "Top3 dominance": "71.22%",
                "DIRECT coverage": 0.8927,
                "Random coverage": 0.7087,
                "Novelty to train": 2.8893,
                "Diversity": 0.4450,
                "Effective threshold": "",
                "BIRCH clusters": "",
                "Export sampled": "181 / 359",
                "Export other": "1093 / 67642",
            },
            {
                "Model": "MACE small",
                "UQ source": "dpa4_mini",
                "Descriptor dim": 256,
                "Candidates": 22605,
                "Selected frames": 87,
                "Selected systems": 76,
                "Sample rate": "0.38%",
                "Important PCs": 18,
                "Retained variance": "96.33%",
                "PC1 dominance": "29.94%",
                "Top3 dominance": "65.86%",
                "DIRECT coverage": 0.9070,
                "Random coverage": 0.6692,
                "Novelty to train": 0.4567,
                "Diversity": 0.5818,
                "Effective threshold": 0.3105,
                "BIRCH clusters": 757,
                "Export sampled": "76 / 87",
                "Export other": "1094 / 67914",
            },
            {
                "Model": "MACE medium",
                "UQ source": "dpa4_mini",
                "Descriptor dim": 256,
                "Candidates": 22605,
                "Selected frames": 142,
                "Selected systems": 113,
                "Sample rate": "0.63%",
                "Important PCs": 16,
                "Retained variance": "96.31%",
                "PC1 dominance": "25.52%",
                "Top3 dominance": "62.49%",
                "DIRECT coverage": 0.9031,
                "Random coverage": 0.6677,
                "Novelty to train": 0.4746,
                "Diversity": 0.5980,
                "Effective threshold": 0.2967,
                "BIRCH clusters": 765,
                "Export sampled": "113 / 142",
                "Export other": "1092 / 67859",
            },
        ]
    ).to_csv(mace / "mace_mini_sampling_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"left": "DPA4 Mini", "right": "MACE small", "intersection": 7, "union": 439, "jaccard": 0.0159, "left_only": 352, "right_only": 80},
            {"left": "DPA4 Mini", "right": "MACE medium", "intersection": 11, "union": 490, "jaccard": 0.0224, "left_only": 348, "right_only": 131},
        ]
    ).to_csv(mace / "mace_mini_selection_overlap.csv", index=False)
    for model in ["mace_small", "mace_medium"]:
        model_dir = mace / f"dpeva_post_joint_{model}_mini_uq_filter128" / "dataframe"
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"pc": 1, "explained_variance_ratio": 0.30, "cumulative_explained_variance_ratio": 0.30, "retained_by_kaiser": True},
                {"pc": 2, "explained_variance_ratio": 0.20, "cumulative_explained_variance_ratio": 0.50, "retained_by_kaiser": True},
            ]
        ).to_csv(model_dir / "pca_explained_variance.csv", index=False)

    for slug in ["neo", "air", "plus", "mini"]:
        view_name = {
            "neo": "e2e_dpa4_neo_filter128/dpeva_post_joint_dpa4_filter128",
            "air": "e2e_dpa4_air_filter128/dpeva_post_joint_dpa4_air_filter128",
            "plus": "e2e_dpa4_plus_filter128/dpeva_post_joint_dpa4_plus_filter128",
            "mini": "e2e_dpa4_mini_filter128/dpeva_post_joint_dpa4_mini_filter128",
        }[slug]
        write_png(dpa4 / view_name / "view" / "Final_sampled_PCAview.png")
    write_png(
        dpa3
        / "e2e_dpa3_filter128"
        / "dpeva_post_joint_dpa3_filter128"
        / "view"
        / "Final_sampled_PCAview.png"
    )


def test_report_builder_generates_single_file_summary(tmp_path):
    seed_report_inputs(tmp_path)
    output = tmp_path / "test" / "DP_EVA_MODEL_COMPARISON_SUMMARY.html"

    module = load_report_module()
    assert module.main(["--repo-root", str(tmp_path), "--output", str(output)]) == 0

    html = output.read_text(encoding="utf-8")
    assert "DP-EVA 模型表征与采样统一展示报告" in html
    assert "Executive Summary" in html
    assert "模型综合排序" in html
    assert "DP-EVA 算法讨论点" in html
    assert "UQ 候选池" in html
    assert "descriptor 表征空间" in html
    assert "DIRECT / BIRCH 多样性采样" in html
    assert "Kaiser rule / PCA 口径" in html
    assert "导师讨论问题" in html
    assert "限制说明" in html
    assert "DPA4 Neo / Air / Plus" in html
    assert "DPA3 基线" in html
    assert "DPA3 baseline" in html
    assert "DPA3 vs DPA4 baseline" in html
    assert "128 维 descriptor" in html
    assert "88 selected frames" in html
    assert "DPA3 原始 DP-EVA PCA 采样视图" in html
    assert "DPA4 Mini / MACE small / MACE medium" in html
    assert "<svg" in html
    assert "data:image/png;base64" in html
    assert "MACE small/medium 未重新计算模型不确定度" in html
    assert "Plus train descriptor" in html
    assert "DeepMD eval-desc 源码根因" in html
    assert "deepmd/entrypoints/eval_desc.py:97-132" in html
    assert "deepmd/pt/infer/deep_eval.py:875-884" in html
    assert "整 system 全帧批量传入" in html
    assert "energy / force / virial" in html


def test_report_builder_can_skip_png_embedding(tmp_path):
    seed_report_inputs(tmp_path)
    output = tmp_path / "light.html"

    module = load_report_module()
    assert module.main(["--repo-root", str(tmp_path), "--output", str(output), "--no-embed-png"]) == 0

    html = output.read_text(encoding="utf-8")
    assert "<svg" in html
    assert "data:image/png;base64" not in html
