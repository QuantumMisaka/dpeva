---
title: DP-EVA DPOSE/Metatrain-Parity LLPR Implementation Plan
status: archived
audience: Developers
last-updated: 2026-06-14
owner: Maintainers
---

  # DP-EVA DPOSE/Metatrain-Parity LLPR Implementation Plan

  ## Summary

  目标是把当前 DP-EVA 的 LLPR 从“energy-level 后处理评分器”升级为“除输入来自 DeepMD last-layer
  feature 外，其余算法语义与 DPOSE 论文和 metatrain LLPR 一致”的实现。保留现有 qbc_rnd 不确定度作
  为可选 backend；LLPR/DPOSE 新实现支持用户只算 energy UQ，也支持在 DeepMD PyTorch 图可用时计算
  force UQ。

  关键可行性结论：DeepMD-kit 的公开 eval_fitting_last_layer(...) 返回 detached numpy/tensor，足够
  做 energy LLPR covariance 和 post-hoc energy uncertainty，但不足以做 force DPOSE。force UQ 需要
  新增 DP-EVA 内部 DeepMDTorchDPOSEAdapter，直接走 DeepMD PyTorch model graph，对 sampled last-
  layer ensemble energy 求坐标梯度。对 direct-force 非能量梯度模型，force DPOSE 不可严格成立，必须
  报错或降级 energy-only。

  当前实现状态（2026-06-11）：DP-EVA 已完成 energy-level LLPR、DPOSE ensemble 数据结构、
  DeepMDTorchDPOSEAdapter 抽象契约、fake differentiable adapter 单测，以及真实 DPA3/DPA4 GPU
  energy-only last-layer LLPR 验证。真实 DeepMD force-level DPOSE adapter 尚未实现；当前 collect/
  UQManager 对 detached feature array 请求 force 或 energy_force 会 fail-fast，不能输出真实 force
  DPOSE 列。

  force-level DPOSE 后续工程量评估：原型阶段约 3-5 个有效工作日，目标是只支持 PyTorch .pt backend、
  OMat24 head、DPA3/DPA4 energy-gradient 模型、小 batch GPU 验证；工程化接入约 1-2 周，覆盖
  deepmd-kit 版本 guard、真实 last-layer weight 提取、batching、日志、Slurm 回归测试和 collect 输出
  列。核心风险不是 DPOSE 公式，而是 deepmd-kit 内部 PyTorch backend 不是稳定公开 API，且公开
  eval_fitting_last_layer 路径及当前 hook 会 detach middle_output。

  ## Energy-Level Shallow Ensemble Feasibility

  基于 `test/lab-cosmo/metatrain/src/metatrain/llpr` 与 DPOSE 论文材料，shallow ensemble UQ 可以基于
  当前 DP-EVA 项目实现，但需要区分三个层级：

  1. analytic energy LLPR uncertainty：当前已经基本实现。输入是 DeepMD fitting last-layer
     features，核心输出是标准差形式的 `energy_uncertainty` / `uq_llpr_energy_per_atom`。该层级不需要
     last-layer weights，只需要训练集特征、候选集特征、正则化协方差和可选 calibration residuals。

  2. metatrain-parity energy shallow ensemble：可以在 DP-EVA 内实现，但还缺少生产级 DeepMD
     last-layer weight extractor。metatrain 的 `LLPRUncertaintyModel.generate_ensemble()` 从 wrapped
     model 的 `last_layer_parameter_names` 读取真实最后一层权重，以这些权重为均值、以
     `alpha^2 * (F.T F + regularizer I)^-1` 为协方差采样 ensemble weights，然后在 forward 中用
     last-layer features 得到 `energy_ensemble`，并把 ensemble mean 重新居中到 base model energy。
     DP-EVA 已有 `DPOSEEnsemble` 数据结构和重居中逻辑雏形，因此算法上可行；工程缺口是从
     DPA3/DPA4/SeZM DeepMD PyTorch checkpoint 中稳定定位并展开 energy fitting net 的真实最后一层权重。

  3. proper-scoring-trained shallow ensemble：也可以在 DP-EVA 内实现，但不建议作为第一阶段目标。
     metatrain 在初始 LLPR covariance、Cholesky、calibration、ensemble sampling 后，可以继续用
     `gaussian_nll_ensemble`、`gaussian_crps_ensemble` 或 `empirical_crps_ensemble` 训练 ensemble
     weights，并可选择冻结 backbone 或训练全部参数。DP-EVA 当前没有 metatensor/metatomic 数据流和
     trainer 抽象，若直接复刻会明显扩大项目边界。更合理的路线是先实现“冻结 DeepMD backbone，只训练
     shallow ensemble heads”的轻量 PyTorch trainer，并只支持 energy ensemble；force ensemble 仍依赖
     后续 differentiable DeepMD adapter。

  因此，推荐 DP-EVA 自身实现前两个层级，并把第三个层级作为可选增强：

  - 第一阶段：保留现有 analytic energy LLPR，补齐 calibration benchmark、状态持久化和 collect 输出说明。
  - 第二阶段：实现 DeepMD last-layer weight extractor，暴露 `energy_ensemble` 用户接口，并用单测证明
    ensemble mean 被重居中到 base energy、ensemble std 与 analytic LLPR uncertainty 在未训练条件下
    统计一致。
  - 第三阶段：实现可选的 frozen-backbone shallow ensemble trainer，最小支持 Gaussian NLL 和 Gaussian
    CRPS；empirical CRPS 可在 ensemble size 和性能评估成熟后再加入。

  这个方案不需要把 metatrain 作为运行时依赖并入 DP-EVA。metatrain 继续作为算法对齐基准；DP-EVA 只
  复用其 LLPR/DPOSE 语义，并围绕 DeepMD/DPA 工作流提供轻量实现。

  ## Key Changes

  - 更新 LLPR 数学实现，使校准与 metatrain 一致：
      - squared_residuals: alpha = sqrt(mean((residual / uncertainty)^2))
      - absolute_residuals: alpha = mean(abs(residual) / uncertainty) * sqrt(pi / 2)
      - crps: per-channel Brent root solve，语义对齐 metatrain GaussianCRPSCalibrator
      - 支持 per-target / per-channel multiplier，而不是只支持单个标量 alpha

  - 拆分 LLPR/DPOSE 核心：
      - LLPRState: covariance/cholesky/multiplier，只负责 analytic uncertainty
      - DPOSEEnsemble: 从真实 last-layer weights + inverse covariance 采样 ensemble weights
      - DPOSEPrediction: 输出 energy_uncertainty, energy_ensemble, 可选 force_uncertainty,
        force_ensemble

  - 新增 DeepMD 内部 adapter：
      - 继续保留 eval_fitting_last_layer(...) 路径用于 offline energy-only LLPR
      - 新增 PyTorch graph 路径，用于读取 differentiable last-layer features、真实最后一层权重、
        base energy，并生成 ensemble energy

      - 对 do_grad_r("energy") 可用的模型计算 force_ensemble = -grad(energy_ensemble_member,
        coords)

      - 对不支持 differentiable energy force 的模型，llpr_targets 包含 force 时抛出明确错误

  - 配置接口：
      - 保留 uq_backend="qbc_rnd" 原行为
      - uq_backend="llpr" 使用 metatrain-parity energy LLPR
      - 新增 llpr_targets: Literal["energy", "force", "energy_force"] = "energy"，允许只计算
        energy UQ

      - 新增 llpr_num_ensemble_members 用于 DPOSE ensemble；未设置时只输出 analytic uncertainty，
        不输出 ensemble

      - 新增 llpr_strict_metatrain_parity: bool = True，默认启用严格公式和 unsupported-model fail-
        fast

  - collect 行为：
      - energy-only：按 energy_uncertainty 或 energy_uncertainty_per_atom 进入筛选
        llpr_collect_score="force_uncertainty_max"

      - hybrid：允许保留 QbC/RND 与 LLPR/DPOSE 并列输出，不互相覆盖

  ## Implementation Steps

  1. 先修正 dpeva.uncertain.llpr 的 metatrain parity 数学：
      - 改 calibration 公式
      - 增加 per-channel alpha
      - 增加 CRPS root solver
      - 增加 covariance normalization mode，energy/system target 默认按 metatrain 方式处理

  2. 增加 DeepMD adapter：
      - 检测模型是否为 PyTorch backend
      - 检测是否可返回 middle_output 且未 detached
      - 提取 last-layer weights，先支持 DPA/SeZM energy fitting net 的常见路径
      - 检测 force 是否来自 energy gradient；direct-force 路径拒绝 force DPOSE

  3. 增加 DPOSE ensemble 计算：
      - 从真实 last-layer weights 采样 member weights
      - ensemble 输出重新居中到 base model energy，语义对齐 metatrain
      - energy UQ = ensemble std；force UQ = force ensemble std

  4. 接入 workflow：
      - feature workflow 保留 last-layer .npy 导出能力
      - collect workflow 根据 llpr_targets 分流 energy-only 或 torch-graph force DPOSE
      - 保留 qbc_rnd、当前 no_filter、DIRECT/2-DIRECT 行为

  5. 更新文档：
      - 明确 eval_fitting_last_layer 只保证 energy-only LLPR
      - 明确 force DPOSE 需要 DeepMD PyTorch graph adapter
      - 文档中禁止把 unsupported direct-force 模型标称为完整 DPOSE

  ## Test Plan

  - Unit tests:
      - calibration 三种方法逐项对齐 metatrain 公式
      - covariance/cholesky/uncertainty 对固定小矩阵给出解析结果
      - per-channel multiplier shape 与 broadcasting 正确
      - unsupported direct-force 模型请求 force UQ 时 fail-fast

  - Adapter tests:
      - 用 fake DeepMD PyTorch fitting net 验证 last-layer weight extraction
      - 验证 sampled energy ensemble mean 被重新居中到 base energy
      - 验证 force_ensemble = -grad(energy_member, coords) 与手写解析梯度一致
      - 验证 llpr_targets="energy" 不触发 force graph/autograd

  - Workflow tests:
      - uq_backend="qbc_rnd" 原测试保持通过
      - uq_backend="llpr", llpr_targets="energy" 只生成 energy columns
      - uq_backend="llpr", llpr_targets="force" 生成 force summary columns
      - collect 可选择 llpr_collect_score="energy_uncertainty_per_atom" 或 "force_uncertainty_max"

  - Verification commands:
      - pytest tests/unit/uncertain/test_llpr.py tests/unit/uncertain/test_llpr_manager.py -q
      - pytest tests/unit/feature/test_last_layer_generator.py tests/unit/workflows/
        test_collect_workflow_routing.py -q

      - pytest tests/unit -q
      - 全量 pytest tests -q，预期仍需单独说明当前 labeling integration mock 既有失败，除非同时修
        复该测试

  ## Assumptions And Defaults

  - 默认实现路径选择“内部 adapter”，不 fork DeepMD-kit。
  - 默认 llpr_targets="energy"，用户可以只算 energy UQ；force UQ 只有显式请求时才计算。
  - “完整 DPOSE”只对可通过 energy 对坐标求梯度得到 force 的 DeepMD PyTorch 模型成立。
  - 对 direct-force 或无法暴露 differentiable last-layer weights 的模型，不伪造 force DPOSE，不回
    退成 QbC/RND 后冒充 DPOSE。

  - metatrain 作为算法对齐基准，不作为运行时依赖并入 DP-EVA。
