---
title: 文档质量标准 (Documentation Quality Standard)
status: active
audience: Maintainers
last-updated: 2026-09-06
owner: Docs Owner
---

# 文档质量标准（Documentation Quality Standard）

## 1. 质量维度与评分（建议 0–2 分）

每篇文档按下列维度打分，推荐总分 ≥ 10/14 才能标记为 `active`：

1. 准确性（与当前代码一致）
2. 完整性（覆盖关键输入/输出/边界条件）
3. 可操作性（读者能按步骤复现/执行）
4. 单一权威来源（避免字段表/规则在多处重复）
5. 可导航性（有明确 TOC、相关链接、术语一致）
6. 可维护性（有 Owner、Last-Updated、适用版本/范围）
7. 示例质量（示例可运行、参数不过期、输出有预期描述）

## 2. 统一元信息（推荐写在文档开头）

- Status: draft | active | deprecated | archived
- Applies-To: 例如 `>=0.4.0` 或 “适用于 Slurm 后端”
- Owners: 角色或模块维护人
- Last-Updated: YYYY-MM-DD
- Related: 关键代码/配置/测试链接（尽量链接到 repo 内）

对于 `status: active` 文档，`owner` 或 `owners` 视为必填项。

## 3. 文档类型验收清单

### 3.1 Guide（操作指南）

- 明确“目标读者”和“前置条件”
- 有最小示例（含输入、命令、预期输出位置）
- 有排障入口（Troubleshooting 链接）
- 字段解释链接到 Reference，不复制粘贴字段表

### 3.2 Reference（查表/权威）

- 字段定义与代码模型一致（建议可自动生成）
- 描述“默认值/类型/约束/示例/是否弃用”
- 变更历史可追溯（至少标注“何时引入/弃用”）

### 3.3 Architecture（系统结构）

- 与当前 `src/` 模块边界一致
- 有数据流与目录结构图（可用 Mermaid 或静态图）
- 解释“为什么这样分层”，并链接到 ADR/Reports

### 3.4 ADR / Report（一次性结论）

- 结论必须明确（Decision/Result）
- 有适用范围与局限性
- 默认只追加不回写（除非修正事实错误）

## 4. 稳态化验收门槛（强制）

- **Docs profile**：`python scripts/run_gate.py docs_pr` 通过；该 profile 包含文档结构、
  新鲜度、warning-as-error 构建、产物和 PR linkcheck。
- **Release profile**：`python scripts/run_gate.py release` 通过；它还包含代码、单元、
  integration、traceability 和可选 extra 检查。
- **Ownership Gate**：`active` 文档 owner 覆盖率=100%，并与 `docs/governance/inventory/owners-matrix.md` 一致
- **PR Gate**：PR 模板须如实填写变更影响、必要同步、验证证据与风险；若为接口变更，
  必须包含 docs 更新说明。仅当新增或更新 active 文档时要求 owner/owners；仅当工作属于
  重大架构、迁移或发布时按文档生命周期维护计划/报告。

门禁目录与命令只维护在 `scripts/gates.toml`；本页解释
验收含义，不复制 argv。`release` profile 包含无写入的版本面一致性检查；Sphinx
release identity 从包版本导入。DeepMD 资格门禁不属于普通文档/软件发布的默认证明，只有
发布声明 DeepMD 能力时才单独调用，并且必须引用真实资格证据。

eval-card 只索引已存在的模型、谱系与指标证据，不是模型重验证、排名或新增科学门禁。
数据 bundle 的不覆盖发布契约限定为 Linux `renameat2(RENAME_NOREPLACE)` 的进程可见
原子性；不支持该原语的平台 fail closed，本版本不提供跨平台 fallback。
