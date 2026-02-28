# DP-EVA 集成测试开发计划

- Status: active
- Audience: Developers / Infra
- Last-Updated: 2026-02-18

## 0.1 相关方

- 开发者：维护 tests/integration 与 workflow 输出约定
- 平台维护：提供 Slurm 队列与 DeepMD 环境
- 使用者：跑通端到端链路并根据断言定位问题

## 0. 背景与目标

本计划面向 DP-EVA 的生产级工作流（Train / Infer / Feature / Collect / Analysis），基于项目在真实问题上的运行产物目录 [test-for-multiple-datapool](/test/test-for-multiple-datapool) 反推输入输出边界，并交付可在 Slurm 队列上运行的集成测试编排与验收准则。

目标：

- 在 Slurm 环境下对各 Workflow 的端到端关键路径做回归覆盖。
- 使用统一的完成标记 `DPEVA_TAG: WORKFLOW_FINISHED` 作为链式编排锚点。
- 对输入数据进行最小化裁剪，在保留核心语义的前提下降低计算与 IO 成本。
- 交付可复用的测试资产（最小数据集、最小配置模板、测试编排器、验收断言）与设计报告。

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
- Workflow 最小化配置模板（对照 `examples/recipes`）。
- Slurm 集成测试编排器：
  - 提交作业、定位日志文件、轮询完成标记、按步骤串联。
  - 失败时收集关键日志并给出可诊断的错误信息。
- 最小数据裁剪脚本：
  - 从生产级 dpdata/descriptor 资产裁剪生成小规模测试数据。
- Pytest 集成测试用例：
  - 单工作流 E2E + 全链路（Feature→Train→Infer→Collect→Analysis）。

## 3. 里程碑拆分

### M1：I/O 反推与测试边界定义

- 从生产目录抽样关键文件，识别输入/输出资产。
- 明确每个 Workflow 的“最小可验收输出”（必须产物）与“非关键输出”（可选产物）。

### M2：最小化配置模板沉淀

- 从 `docs/reference/config-schema.md` 与 `examples/recipes/*/config*.json` 提取每个 Workflow 的最小字段集。
- 统一约定集成测试中的路径解析策略（相对路径 + 自动解析）。
- 统一 Slurm 配置字段（分区、CPU/GPU、walltime、日志文件名）。

### M3：Slurm 编排器与完成标记对齐

- 实现 Slurm 提交与监控工具（监控完成标记、超时给出日志尾部摘要）。
- 修复/补齐完成标记一致性（如发现缺失则补齐）。

### M4：最小数据裁剪与端到端用例实现

- 生产级 dpdata 裁剪为：少量 pool/system/frames。
- 描述符资产裁剪与 system 一一对应。
- 固化端到端最小链路验收断言。

## 4. 验收标准（Definition of Done）

- 每个 Workflow 至少 1 个 Slurm 集成用例通过，失败时可快速定位原因（日志可读、断言明确）。
- 全链路用例可串联执行，并以完成标记触发下一步。
- 裁剪后输入规模显著小于生产目录，并能在合理资源配置下完成。
- 交付使用说明：前置条件、partition/qos/gpu 指定方式、预期输出与断言。

## 5. 异常处理

- 无 Slurm 环境：集成测试默认跳过，仅执行 unit tests。
- DeepMD 环境不可用：优先补齐 `env_setup`，并在作业日志中确认 `dp --version`。
- 资源不足导致排队/超时：降低资源申请（walltime/gpu/cpu）并缩小裁剪数据规模。

## 6. 变更记录

- 2026-02-18：补齐计划的交付物、里程碑与验收标准，并明确异常处理策略。
