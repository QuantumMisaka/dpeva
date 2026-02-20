# 文档完整性自检报告（Self-Check）

- Date: 2026-02-18
- Scope: `docs/` 全量 Markdown

本报告记录对“文档格式、占位符、交叉引用”的自动化自检结果，并给出后续治理建议。

## 1. 自检项与结果

### 1.1 占位符/待办检查

- 检查项：扫描 `TODO/待补充/TBD`
- 结果：未发现占位符残留

### 1.2 交叉引用与链接可达性

- 检查项：扫描 Markdown 链接（仅允许项目内相对路径），验证指向的文件/目录存在
- 结果：
  - Markdown 文件数：50
  - 链接检查数：128
  - 断链数：0

### 1.3 空文件检查

- 检查项：空文件/仅标题文件
- 结果：发现并修复 1 个空文件：
  - `docs/architecture/design-report.md`：已转为占位页并指向现行设计报告

## 2. 风险项与建议

### 2.1 元信息（Status/Owner/Last-Updated）覆盖率不一致

现状：部分历史文档与索引文档未包含统一元信息字段，容易导致“现行规范 vs 历史报告”边界不清。

建议：

- 对 `docs/guides/*`、`docs/reference/*`、`docs/policy/*` 的所有 `active` 文档强制补齐元信息。
- `docs/archive/*` 允许缺失元信息，但建议至少标注 `Status: archived/deprecated` 与适用版本。

### 2.2 设计类文档的“现状一致性”治理

现状：`docs/design/DP-EVA_Design_Report.md` 同时包含历史建议与已落地重构说明。

建议：

- 在后续迭代中将“与代码一致的架构概览”拆为 `docs/architecture/overview.md`（持续维护）。
- 将一次性决策下沉到 `docs/architecture/decisions/*`（ADR）。

## 3. 自检脚本说明

自检脚本逻辑：

- 解析所有 `docs/**/*.md` 的 Markdown 链接写法（示例：`[text] (path)`）
- 跳过 `http(s)`、`mailto`、`#anchor`
- 对相对路径按文档所在目录解析并验证存在性
- 禁止项目内使用文件系统绝对路径链接（例如 `file://...` 与 `/abs/path`）
