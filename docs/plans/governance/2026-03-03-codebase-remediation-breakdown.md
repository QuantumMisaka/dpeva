---
title: Document
status: active
audience: Developers
last-updated: 2026-03-09
---

# DP-EVA 代码库修复任务分解与实施方案

- Status: active
- Audience: Maintainers / Developers
- Applies-To: >=0.4.8
- Owners: Workflow Owners / Infra Owner / Maintainer
- Last-Updated: 2026-03-03
- Related:
  - 开发规范：/docs/guides/developer-guide.md
  - 文档质量标准：/docs/policy/quality.md
  - 文档结构规范：/docs/policy/docs-structure.md
  - 历史规划：/docs/governance/plans/2026-02-18-doc-system-planning.md

## 1. 目标与范围

本计划用于落实最近一轮代码审查中识别出的高优先级问题，形成可执行的修复任务分解，并为每个任务给出具体修复方案、验收标准与文档更新要求。

本计划严格遵循开发流程规范：

1. Plan：先定义任务边界与验收口径。
2. Execute：按 DDD 边界落地修复，避免跨层侵入。
3. Verify：每个任务必须有对应测试与回归验证。
4. Document：同步更新开发与指南文档，避免文档漂移。

## 2. 修复任务分解（按优先级）

### P0（阻断级，优先本迭代完成）

| 任务ID | 主题 | 问题摘要 | 影响范围 | 负责人建议 |
|---|---|---|---|---|
| R1 | 退出码契约一致性 | workflow 失败路径可能返回 0，违背 CLI 契约 | `workflows/*`, `cli.py`, `tests/unit` | Workflow Owner |
| R2 | 裸异常与解析回退 | 存在裸 `except` 吞错；原子数解析回退过于宽松 | `io/dataproc.py`, `workflows/collect.py`, `utils/env_check.py` | IO/Workflow Owner |

### P1（高优先级，紧随 P0）

| 任务ID | 主题 | 问题摘要 | 影响范围 | 负责人建议 |
|---|---|---|---|---|
| R3 | 路径安全边界 | 输出目录拼接缺少规范化与根路径约束 | `io/collection.py` | IO Owner |
| R4 | 命令构建安全 | 命令参数字符串拼接缺少安全转义/参数化 | `utils/command.py`, `submission/manager.py` | Infra Owner |
| R5 | 完成标记语义 | 部分失败场景仍输出 `WORKFLOW_FINISHED` | `workflows/infer.py`, `tests/unit/workflows` | Workflow Owner |

### P2（治理优化，建议并行推进）

| 任务ID | 主题 | 问题摘要 | 影响范围 | 负责人建议 |
|---|---|---|---|---|
| R6 | 质量门禁自动化 | 质量标准存在但自动门禁不足 | CI 配置、`pytest.ini`、测试文档 | Maintainer |
| R7 | 文档-实现一致性 | 部分指南/矩阵与实现存在漂移 | `docs/guides/*`, `docs/governance/traceability/*` | Docs Owner |
| R8 | 常量治理 | MAGIC_NUMBER 数量多，影响可维护性 | `constants.py` 与高频模块 | Module Owners |

## 3. 每项任务的具体修复方案

### R1：退出码契约一致性（P0）

**修复目标**
- 任何业务失败路径都能稳定返回非 0。
- CLI 文档、行为、测试三者一致。

**具体方案**
- 将 `InferenceWorkflow.run`、`FeatureWorkflow.run` 的关键失败分支从“日志 + return”调整为“抛出带上下文信息的异常”。
- 统一在 CLI 顶层捕获异常并 `sys.exit(1)`，保持单一出口。
- 对可预期错误（如输入路径不存在）定义明确异常类型或统一错误码映射，避免隐式 `None` 语义。

**验证方案**
- 新增/更新单测：输入缺失、模型缺失、配置错误分别断言退出码与错误日志。
- 补充命令行回归用例，校验失败场景返回非 0。

**文档同步**
- 更新 `/docs/guides/cli.md` 的退出码约定与示例。
- 在 `/docs/guides/developer-guide.md` 的流程标准中补充“禁止 silent return 替代失败语义”。

### R2：裸异常与解析回退治理（P0）

**修复目标**
- 消除裸 `except`。
- 异常可观测且可定位，回退策略可解释。

**具体方案**
- 修复 `io/dataproc.py` 中解析逻辑依赖，避免未定义符号进入回退分支。
- 将裸 `except` 改为显式异常类型捕获，并记录必要上下文（输入值、阶段、建议动作）。
- 对“回退到默认值”的路径增加告警级日志，明确这是降级而非正常流程。

**验证方案**
- 增加参数化单测覆盖：正常命名、异常命名、边界输入。
- 验证异常分支日志中包含可检索关键词（模块、输入、原因）。

**文档同步**
- 在 `/docs/guides/developer-guide.md` 强化 “Errors Should Never Pass Silently” 的代码示例。

### R3：路径安全边界加固（P1）

**修复目标**
- 禁止越界写入。
- 所有导出路径均约束在预期根目录内。

**具体方案**
- 对 `sys_name` 做规范化处理（去除路径分隔符、拒绝 `..`、控制非法字符）。
- 使用 `pathlib.Path.resolve()` + 根路径前缀校验，写入前做边界断言。
- 对非法名称输入抛出显式异常并记录审计日志。

**验证方案**
- 单测覆盖：正常名称、含 `../`、含绝对路径片段、特殊字符。
- 回归验证导出目录结构与历史合法场景兼容。

**文档同步**
- 更新 `/docs/guides/configuration.md` 或相关导出说明，明确命名约束。

### R4：命令构建安全改造（P1）

**修复目标**
- 消除字符串拼接式命令执行风险。
- 统一命令构建接口，便于审计与测试。

**具体方案**
- 将命令构建从“单字符串”迁移为“参数列表”表示，并在执行层统一处理。
- 若保留字符串模式用于脚本输出，采用标准转义策略并限制可注入字段。
- 对模型路径、数据路径、prefix 等外部输入加白名单校验。

**验证方案**
- 增加单测覆盖带空格、特殊字符的路径参数。
- 回归验证 Local/Slurm 两种后端的命令生成与提交行为一致。

**文档同步**
- 更新 `/docs/guides/slurm.md` 与开发文档中命令构建约束。

### R5：完成标记语义修复（P1）

**修复目标**
- `WORKFLOW_FINISHED` 仅在全局成功条件满足时输出。
- 部分失败场景输出明确的失败标记或退出状态。

**具体方案**
- 在分析流程中累计失败计数并设置最终状态判定。
- 失败存在时：不输出完成标记，改为失败摘要并抛异常或返回失败状态。
- 对“可容忍部分失败”定义清晰阈值与策略，避免语义歧义。

**验证方案**
- 扩展现有完成标记单测，新增“部分失败/全失败/全成功”三类断言。

**文档同步**
- 更新 `/docs/governance/traceability/workflow-contract-test-matrix.md` 与 `/docs/guides/slurm.md`。

### R6：质量门禁自动化落地（P2）

**修复目标**
- 将质量规则从“文档声明”提升为“自动阻断”。

**具体方案**
- 增加最小门禁流水线：静态检查、单元测试、覆盖率阈值。
- 将 `tools/audit.py` 纳入可选或分级门禁（warning/error 分层）。
- 统一本地与 CI 的执行入口，避免双标。

**验证方案**
- 新增门禁演练清单：故意引入失败样例，验证阻断有效。

**文档同步**
- 更新 `/docs/policy/quality.md` 与 `/tests/unit/UNIT_TESTS.md` 的门禁执行说明。

### R7：文档-实现一致性治理（P2）

**修复目标**
- 消除关键用户路径上的文档漂移。

**具体方案**
- 建立“契约字段巡检清单”：退出码、完成标记、日志文件名、命令示例。
- 每次涉及 workflow/CLI 变更时，强制检查对应文档章节。
- 修复已识别的不一致项，并补充“最后验证日期”。

**验证方案**
- 文档自检：链接检查、示例命令抽检、关键条目对照源码。

**文档同步**
- 优先更新 `/docs/guides/cli.md`、`/docs/guides/slurm.md`、`/docs/guides/developer-guide.md`。

### R8：常量治理与可维护性提升（P2）

**修复目标**
- 减少高频魔法数字对理解与变更的阻力。

**具体方案**
- 对跨模块复用阈值抽取至 `constants.py` 或模块级命名常量。
- 对仅局部生效数值补充命名语义与单位，避免“裸值”。
- 结合审计结果分批治理，优先高风险算法阈值与 I/O 边界。

**验证方案**
- 确保重构前后数值行为一致（回归测试 + 样例对比）。

**文档同步**
- 对核心阈值在 `/docs/reference/validation.md` 给出语义说明与适用范围。

## 4. 执行顺序与里程碑

### 里程碑 M1（本迭代）
- 完成 R1、R2 并通过回归测试。
- 输出契约一致性变更说明（CLI/Workflow）。

### 里程碑 M2（下一迭代）
- 完成 R3、R4、R5，形成安全与状态语义闭环。
- 补齐相关文档与测试矩阵。

### 里程碑 M3（持续治理）
- 推进 R6、R7、R8，建立长期质量保障机制。

## 5. 验收标准（Definition of Done）

- 每个任务必须满足“代码修复 + 测试通过 + 文档同步”三条件。
- 不允许新增裸 `except`、隐式失败返回、未约束路径写入。
- 涉及 CLI/Workflow 行为变更时，必须同时更新 `docs/guides` 与 `docs/governance/traceability`。
- 文档元信息必须完整，命名与落位符合 `/docs/policy/docs-structure.md`。
- 项目单元测试和集成测试（local+slurm）均需完整且正确通过，确保核心功能不被破坏。

## 6. 风险与回滚策略

- 风险 1：行为语义收紧导致历史脚本依赖失效。  
  应对：在发布说明中给出兼容性提示，并保留短期兼容开关。
- 风险 2：命令执行方式改造影响 Slurm 模板兼容。  
  应对：先在测试环境做 Local/Slurm 双后端回归，再逐步推广。
- 风险 3：文档更新滞后。  
  应对：将文档检查加入合并前清单，未同步不得合并。

