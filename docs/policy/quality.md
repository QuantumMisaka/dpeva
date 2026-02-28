# 文档质量标准（Documentation Quality Standard）

- Status: active
- Audience: Maintainers
- Last-Updated: 2026-02-18

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

