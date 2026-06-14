---
title: DPA4 Mini UQ-Error Correlation Experiment Plan
status: archived
audience: Developers
last-updated: 2026-06-14
owner: Maintainers
---

  # DPA4 Mini UQ-Error Correlation Experiment Plan

  ## Summary

  设计一条可复现实验链路：以 practices/sampled_dpdata 作为已有训练集，重新微调 DPA4 Mini ensemble；以
  practices/other_dpdata 作为带 DFT 标签的待选池，计算当前 uq_qbc_rnd force uncertainty、analytic LLPR
  energy uncertainty、DPOSE energy ensemble uncertainty，并与预测误差做相关性比较。

  默认选择：

  - 模型：DPA4 Mini 模板，4-member ensemble。
  - 环境：dpeva-dpa4-test。
  - 资源：训练优先 4V100PX 或 4V100；短测用 flood-gpu，正式长跑用 huge-gpu。
  - 误差指标：energy MAE per atom、force max error per frame、force RMS error per frame。
  - 相关性指标：Pearson、Spearman、Kendall，并输出 top-k 高不确定度召回高误差样本的 enrichment 表。

  ## Key Changes

  新增一套实验资产，不改动核心 DP-EVA workflow 语义：

  - 新增 practices/uq_correlation/ 作为实验根目录，包含：
      - configs/：训练、推理、feature、LLPR/DPOSE、分析配置。
      - scripts/：预检、提取 base energy、解析最后层权重、合并 UQ 与误差、生成报告。
      - outputs/：运行产物目录，不纳入核心源码契约。

  - 复用现有 DP-EVA 能力：
      - dpeva train：用 sampled_dpdata 训练 4 个 DPA4 Mini 模型，输出 work_dir/0..3/model.ckpt.pt。
      - dpeva infer：对 other_dpdata 生成每个模型的 results.e_peratom.out、results.f.out。
      - dpeva feature Python mode：分别为 sampled_dpdata 和 other_dpdata 导出
        feature_kind="fitting_last_layer"。

      - UQManager.run_analysis()：计算 uq_qbc_for、uq_rnd_for、uq_rnd_rescaled 和 force error。
      - UQManager.run_llpr_analysis()：计算 uq_llpr_energy_per_atom、uq_dpose_energy_ensemble_std_per_atom。

  ## Data Flow

  1. Preflight
      - 校验 sampled_dpdata 与 other_dpdata 均可用 dpdata.LabeledSystem(fmt="deepmd/npy") 读取。
      - 输出系统数、frame 数、type map、一致性摘要。
      - 记录 sampled_dpdata 与 other_dpdata 的 system 名重叠情况；不按 system 名去重，实验解释为“已有训练池
        vs 待选候选池”，不是严格 composition-OOD benchmark。

  2. Train Ensemble
      - 基于 examples/recipes/training/dpa4/mini 生成实验训练配置。
      - training_data_path = practices/sampled_dpdata。
      - num_models = 4，使用固定 seeds [19090, 42, 10032, 2933]。
      - 训练产物目录：practices/uq_correlation/work/dpa4_mini_ensemble/0..3/model.ckpt.pt。

  3. Predict Candidate Pool
      - 对 practices/other_dpdata 跑 dpeva infer。
      - work_dir 指向 ensemble 训练目录。
      - task_name = "test_val"，results_prefix = "results"。
      - 使用模型 0 作为 baseline prediction；模型 1..3 作为 QbC committee，符合现有
        UQCalculator.compute_qbc_rnd() 语义。

  4. Compute QbC/RND UQ + Force Error
      - 解析 0..3/test_val/results.*.out。
      - 输出 frame-level 表：
          - dataname
          - uq_qbc_for
          - uq_rnd_for
          - uq_rnd_rescaled
          - diff_maxf_0_frame
          - diff_rmsf_0_frame

      - force 相关性以 diff_maxf_0_frame 和 diff_rmsf_0_frame 为目标误差。

  5. Compute LLPR/DPOSE Energy UQ
      - 用模型 0 导出 last-layer features：
          - train features: sampled_dpdata
          - candidate features: other_dpdata

      - 用模型 0 对 other_dpdata 直接推理得到 frame total base energy，写入 candidate_energy.npy；不要直接把
        results.e_peratom.out 当作 DPOSE base energy，若使用该文件必须乘以对应 natoms 转回 total energy。

      - 解析模型 0 的真实最后层权重；优先用显式权重文件，若无则调用现有
        resolve_last_layer_weights(model_path=model0, model_head=...)。

      - 调用 run_llpr_analysis() 生成：
          - uq_llpr_energy_total
          - uq_llpr_energy_per_atom
          - uq_dpose_energy_ensemble_std
          - uq_dpose_energy_ensemble_std_per_atom
          - energy_ensemble.npy

  6. Energy and Force Error Targets
      - Energy error 使用模型 0 的预测与 DFT label：
          - energy_error_per_atom_abs = abs(pred_e_per_atom - data_e_per_atom)
          - 若需要 total energy error，使用 abs(pred_e_total - data_e_total)，其中 total 由 per-atom energy
            乘 natoms。

      - Force error 使用现有 parser 的 model 0 force diff：
          - force_error_max = diff_maxf_0_frame
          - force_error_rms = diff_rmsf_0_frame

      - LLPR/DPOSE 只和 energy error 做主比较；可额外报告其与 force error 的弱相关性，但明确标为
        exploratory。

  7. Correlation Report
      - 合并所有 frame-level 数据为 uq_error_frame_table.csv。
      - 生成 correlation_summary.csv，每行包含：
          - uq_metric
          - error_metric
          - pearson_r
          - pearson_p
          - spearman_rho
          - spearman_p
          - kendall_tau
          - kendall_p
          - n_frames

      - 生成排序诊断：
          - top 1%、5%、10% uncertainty 中的 high-error recall。
          - high-error 定义为对应误差的 top 5%。

      - 生成图：
          - uq_vs_error_scatter_*.png
          - uq_vs_error_hexbin_*.png
          - uncertainty_rank_error_curve_*.png

      - 生成 HTML 或 Markdown 报告，核心比较：
          - uq_qbc_for vs force max/RMS error
          - uq_rnd_for vs force max/RMS error
          - uq_rnd_rescaled vs force max/RMS error
          - uq_llpr_energy_per_atom vs energy error per atom
          - uq_dpose_energy_ensemble_std_per_atom vs energy error per atom

  ## Resource and Runtime Defaults
      - Preferred partition: 4V100PX if idle; otherwise 4V100.
      - Short validation: qos=flood-gpu, walltime=04:00:00.
      - Full run if expected longer: qos=huge-gpu, conservative walltime.
      - Use 1 node, 4 GPUs per model training job, matching current DPA4 Mini template.

  - Inference and feature generation:
      - Single-GPU jobs on 4V100PX/4V100.
      - Use rush-1o2gpu for short debug; flood-1o2gpu or improper-gpu for large full-pool jobs depending
        expected duration.

  - CPU-only correlation/report scripts:
      - Run on login node only if lightweight; otherwise submit to CPU-MISC with rush-cpu.

  ## Test Plan

  - Unit-level:
      - Test frame table merge with synthetic prediction/UQ arrays.
      - Test energy conversion: e_peratom * natoms -> total energy for DPOSE base energy.
      - Test correlation function handles NaN/inf by dropping invalid pairs per metric pair.
      - Test top-k enrichment calculation on deterministic small arrays.

  - Integration smoke:
      - Use a tiny subset: first 3 systems from sampled_dpdata, first 5 systems from other_dpdata.
      - Run one-model or two-model dry smoke only to validate scripts and paths.
      - Confirm output table row count equals candidate frame count for the subset.

  - Full acceptance:
      - 4 trained models exist at 0..3/model.ckpt.pt.
      - All inference result files exist for other_dpdata.
      - LLPR feature files exist for every train and candidate system.
      - energy_ensemble.npy shape is (n_candidate_frames, n_members).
      - Final correlation_summary.csv contains all planned UQ-error metric pairs.
      - Report states force DPOSE is not evaluated because current detached-feature implementation only
        supports energy-level DPOSE/LLPR.

  ## Assumptions

  - DPA4 Mini base model file is available in the environment or will be placed at the path used by the
    generated config.

  - model_head follows the DPA4 Mini template default, unless the base model requires a different head.
  - Model 0 is the baseline model for prediction error and RND; models 1..3 form the QbC committee.
  - other_dpdata labels are used only for evaluation of prediction error, not for UQ computation.
  - No frame-level deduplication is performed between train and candidate pools unless a later requirement
    explicitly asks for strict leakage removal.
