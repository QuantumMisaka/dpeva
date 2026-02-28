# DP-EVA 集成测试设计与交付报告（Slurm + Multi DataPool）

- Status: active
- Audience: Developers / Infra
- Last-Updated: 2026-02-18

本报告给出基于真实生产目录 [test-for-multiple-datapool](/test/test-for-multiple-datapool) 反推的集成测试设计，并交付可执行的 Slurm 编排用例与输入裁剪方案。

## 1. 交付物索引

- 开发计划文档：./integration-slurm-plan.md
- 生产目录 I/O 分析：./multi-datapool-artifacts.md
- Workflow 最小配置模板说明：./integration-config-templates.md
- Slurm 集成测试（代码）：
  - 用例：[test_slurm_multidatapool_e2e.py](/tests/integration/test_slurm_multidatapool_e2e.py)
  - 编排器：[orchestrator.py](/tests/integration/slurm_multidatapool/orchestrator.py)
  - Slurm 工具：[slurm_utils.py](/tests/integration/slurm_multidatapool/slurm_utils.py)
  - 输入裁剪：[data_minimizer.py](/tests/integration/slurm_multidatapool/data_minimizer.py)
- 最小配置模板（文件）：[tests/integration/slurm_multidatapool/configs](/tests/integration/slurm_multidatapool/configs)

## 2. 相关方

- 开发者：维护 `tests/integration` 用例与编排逻辑
- 平台维护：保障 Slurm 与 DeepMD 环境可用
- 使用者：参考本报告在集群上跑通链路并定位失败原因

## 3. 关键编排原则

### 3.1 Slurm 作业链式触发

DP-EVA 的 Train/Infer/Feature/Collect Worker 在成功结束时输出标准标记：

- `DPEVA_TAG: WORKFLOW_FINISHED`（定义于 [constants.py](/src/dpeva/constants.py)）

集成测试编排基于“监控日志出现该标记”推进下一步，从而：

- 避免依赖 `squeue` 的不稳定状态判断（排队/重排/延迟写日志等）
- 不需要 DP-EVA 内置实现 Slurm blocking wait（多数 Workflow 只负责提交）

### 3.2 CollectionWorkflow 完成标记补齐

Collect 阶段也必须具备统一完成锚点。当前已在 [collect.py](/src/dpeva/workflows/collect.py) 的本地执行路径末尾输出完成标记，使得 Slurm 自调用 worker 的 `collect_slurm.out` 可被稳定监控。

## 4. 输入裁剪策略（降本）

测试数据来自生产目录，但运行时会裁剪到最小规模（在 pytest 的 `tmp_path` 内生成），以显著降低 Feature/Infer/Collect 的计算与 IO。

裁剪规则（当前 smoke 用例）：

- 候选池（Multi Pool）保留 1 个 pool + 1 个 system：
  - `other_dpdata_all/mptrj-FeCOH/C0Fe4H0O8`
  - 截断为前 20 帧（原始 92 帧）
- 训练集（Single Pool）保留 1 个 system：
  - `sampled_dpdata/122`
  - 原始已是 1 帧，不再额外裁剪
- 训练 input.json：
  - 基于生产 input.json 生成最小训练 input（将 `numb_steps` 缩至 10，`disp_freq` 设为 1）

实现位置：[data_minimizer.py](/tests/integration/slurm_multidatapool/data_minimizer.py)

## 5. 集成测试用例设计

### 5.1 用例：Multi DataPool 全链路 Smoke（Slurm）

文件：[test_slurm_multidatapool_e2e.py](/tests/integration/test_slurm_multidatapool_e2e.py)

链路：

1. Feature（候选池）→ `desc_pool/<pool>/<system>.npy`
2. Feature（训练集）→ `desc_train/<system>.npy`
3. Train（3 模型）→ `0..2/model.ckpt.pt` + `train.out`（含完成标记）
4. Infer（3 模型）→ `0..2/test_val/results.e.out` + `test_job.log`（含完成标记）
5. Collect（CPU）→ `dpeva_uq_result/dataframe/final_df.csv` + `collect_slurm.out`（含完成标记）

核心断言（最小验收输出）：

- 描述符文件存在（Feature 输出）
- `model.ckpt.pt` 存在（Train 输出）
- `results.e.out` 存在（Infer 输出）
- `final_df.csv` 存在（Collect 输出）
- 每一步日志出现 `DPEVA_TAG: WORKFLOW_FINISHED`（串联锚点）

### 5.2 后续扩展用例（模板已覆盖）

- Collect Joint Sampling（训练集去重采样）：`collect_joint.json`
- 2-DIRECT（结构级 + 原子级两步采样）：对齐 `sampler_type="2-direct"` 的参数组

参考：

- ../../reference/config-schema.md

## 6. 运行说明（面向 Slurm 环境）

运行前置：

- `sbatch`、`squeue` 可用
- DeepMD-kit `dp` 命令可用

执行（默认跳过，需要显式开启）：

- `DPEVA_RUN_SLURM_ITEST=1`
- 可选设置：
  - `DPEVA_TEST_ENV_SETUP`
  - `DPEVA_TEST_GPU_PARTITION`、`DPEVA_TEST_GPU_QOS`
  - `DPEVA_TEST_CPU_PARTITION`、`DPEVA_TEST_CPU_QOS`
  - `DPEVA_SLURM_ITEST_TIMEOUT_S`

## 7. 风险与边界

- 该集成测试依赖 Slurm 与 DeepMD 运行环境，不建议默认纳入无 Slurm 的 CI。
- `collect_slurm.out`、`train.out`、`test_job.log` 等日志文件名来自当前 Workflow/Manager 的 JobConfig 固化设置；若未来变更，应同步更新编排器的监控路径。

## 8. 异常处理

- 作业提交失败：优先检查 `submission.slurm_config` 的 `partition/qos/account` 与资源字段。
- 日志无完成标记：直接查看日志末尾堆栈，确认是否为外部命令失败或输入数据问题。
- 目录结构不一致：以实际产物为准，在测试用例断言与本文档中同步更新路径约定。

## 9. 变更记录

- 2026-02-18：形成测试专题主页，统一交付物索引、链式编排原则与运行说明。
