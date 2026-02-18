# DP-EVA 集成测试开发计划

## 0. 背景与目标

本计划面向 DP-EVA 的生产级工作流（Train / Infer / Feature / Collect / Analysis），基于项目在真实问题上的运行产物目录 [test-for-multiple-datapool](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/test/test-for-multiple-datapool) 反推输入输出边界，并交付可在 Slurm 队列上运行的集成测试编排与验收准则。

目标：

- 在 Slurm 环境下对各 Workflow 的端到端关键路径做回归覆盖。
- 使用统一的完成标记 `DPEVA_TAG: WORKFLOW_FINISHED` 作为链式编排锚点。
- 对输入数据进行最小化裁剪，在保留核心语义的前提下降低计算与 IO 成本。
- 交付可复用的“测试资产”（最小数据集、最小配置模板、测试编排器、验收断言）与设计报告。

## 1. 范围与假设

范围：

- 覆盖 CLI 子命令：`dpeva train`、`dpeva infer`、`dpeva feature`、`dpeva collect`、`dpeva analysis`。
- 覆盖 Slurm 后端：`submission.backend="slurm"`。
- 覆盖多数据池（Multi Pool）数据结构：`Pool/System` 两级目录语义。

假设：

- 执行环境具备 `sbatch/squeue/scancel` 且可提交短作业。
- 执行环境可用 DeepMD-kit `dp` 命令，且后端由 `dp_backend`（默认 `pt`）控制。
- 允许将集成测试标记为“需 Slurm 环境”，不强制在本地 CI 默认跑全链路。

## 2. 交付物清单

- 集成测试输入输出分析报告（基于生产目录反推 I/O）。
- Workflow 最小化配置模板（对照 [examples/recipes](file:///home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/dpeva/examples/recipes)）。
- Slurm 集成测试编排器：
  - 负责提交作业、定位日志文件、轮询完成标记、按步骤串联。
  - 负责在失败时收集关键日志并给出可诊断的错误信息。
- 最小数据裁剪脚本：
  - 从生产级 dpdata/descriptor 资产裁剪生成小规模测试数据（可选择运行时生成到临时目录）。
- Pytest 集成测试用例：
  - 按 Workflow 分层（单工作流 E2E）+ 全链路（Feature→Train→Infer→Collect→Analysis）。

## 3. 里程碑拆分

### M1：I/O 反推与测试边界定义

- 从生产目录抽样关键文件，识别：
  - 输入资产：dpdata（训练/候选）、descriptor（训练/候选）、基础模型、各 Workflow config。
  - 输出资产：模型目录结构、dp test 结果、UQ/采样数据帧、可视化、导出的 dpdata。
- 明确每个 Workflow 的“最小可验收输出”（必须产物）与“非关键输出”（可选产物）。

### M2：最小化配置模板沉淀

- 从 `docs/api/INPUT_PARAMETERS.md` 与 `examples/recipes/*/config*.json` 提取每个 Workflow 的最小字段集。
- 统一约定集成测试中的路径解析策略（全部使用 config 相对路径 + `resolve_config_paths` 自动转绝对）。
- 统一 Slurm 配置字段（分区、CPU/GPU、walltime、日志文件名）。

### M3：Slurm 编排器与完成标记对齐

- 实现可复用的 Slurm 运行工具：
  - 提交：`dpeva <subcommand> <config>`。
  - 监控：轮询指定 `output_log`，直到出现 `DPEVA_TAG: WORKFLOW_FINISHED`。
  - 超时：给出最后 N 行日志摘要并失败。
- 修复/补齐工作流完成标记的一致性（若有 Workflow 未输出该标记，则补齐）。

### M4：最小数据裁剪与端到端用例实现

- 将生产级多数据池 dpdata 裁剪为：
  - 1–2 个 pool。
  - 每 pool 1–2 个 system。
  - 每 system 1–3 帧（尽量覆盖能量/力/应力字段）。
- 将 descriptor 资产裁剪为：
  - 与裁剪后 system 一一对应的 `.npy` 文件。
- 用裁剪数据跑通 Feature / Train / Infer / Collect / Analysis 的最小链路，并固化验收断言。

## 4. 验收标准（Definition of Done）

- 每个 Workflow 至少 1 个 Slurm 集成用例通过，且可在失败时快速定位原因（日志可读、断言明确）。
- 全链路用例可串联执行（Feature→Train→Infer→Collect→Analysis），每一步通过完成标记触发下一步。
- 裁剪后的测试输入规模显著小于生产目录，并能在合理资源配置下完成。
- 交付完整设计报告与使用说明，包含：
  - 运行前置条件（DeepMD/Slurm 环境）。
  - 如何指定 partition/qos/gpu 等参数。
  - 预期输出与断言说明。

