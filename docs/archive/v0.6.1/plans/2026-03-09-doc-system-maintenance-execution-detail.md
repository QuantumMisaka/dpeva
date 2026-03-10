---
title: 文档系统维护开发执行细则 (Document System Maintenance & Development Execution Plan)
status: completed
audience: Developers
last-updated: 2026-03-10
---

# 文档系统维护开发执行细则 (Document System Maintenance & Development Execution Plan)

**版本**: v1.1
**日期**: 2026-03-10
**状态**: Completed
**责任人**: Trae AI Agent (Acting Docs Owner)
**关联文档**:
- `docs/governance/reviews/DocReview_20260309_v1.0.md` (审查报告)
- `docs/archive/v0.6.1/plans/2026-02-18-doc-system-planning.md` (历史规划)

---

## 1. 目标与范围 (Goals & Scope)

### 1.1 核心目标
本计划旨在基于 v0.6.1 的治理基础，进一步解决文档系统中残留的结构性冲突、治理规范落地不足及自动化缺失问题。通过引入强制性 CI 门禁与明确的 RACI 流程，构建一个**自愈合、可度量、低熵增**的文档生态系统。

### 1.2 验收标准 (KPIs)
1.  **结构合规率 100%**: 所有目录均包含索引文件 (`README.md`)，无孤儿文件，无深度 > 4 的 Active 目录。
2.  **治理闭环率 100%**: 所有 `reviews` 产物归档至 `governance` 体系，无冗余路径。
3.  **自动化覆盖**: 建立 CI 流水线，包含链接检查 (LinkCheck)、结构校验 (StructureCheck) 与元数据检查 (MetaCheck)。
4.  **职责清晰度**: 核心模块 (Train/Infer/Collect/Label) 文档 Owner 100% 明确并公示。

---

## 2. 现状差距矩阵 (Gap Analysis Matrix)

| ID | 问题描述 (Gap) | 根因分析 | 建议方案 (Proposal) | 计划任务 (Task ID) | 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| G-01 | `docs/reviews` 与 `governance/audits` 职能重叠 | 新增目录未遵循既有治理架构 | **合并**: 将 `reviews` 迁移至 `governance/reviews`，统一入口 | T-STRUCT-01 | **Completed** |
| G-02 | 多个核心目录 (`reports`, `plans`) 缺失索引文件 | 缺乏脚手架与强制检查 | **补全**: 批量创建标准 `README.md` 模板 | T-CONTENT-01 | **Completed** |
| G-03 | 归档目录 (`archive`) 深度过深 (5层+) | 版本化归档策略过于机械 | **扁平化**: 优化归档命名规则，减少嵌套 | T-STRUCT-02 | **Completed** |
| G-04 | 缺乏可视化的 RACI 职责矩阵 | 仅在文字中描述，检索困难 | **可视化**: 在 `maintenance.md` 中绘制表格 | T-GOV-01 | **Completed** |
| G-05 | 缺乏自动化 CI 检查 | 依赖人工 Review，易遗漏 | **集成**: 引入 `doc-check` 脚本至 CI | T-AUTO-01 | **Completed** |

---

## 3. 详细工作分解 (WBS)

### 3.1 结构重构 (Structure Refactoring)
*   **T-STRUCT-01** (High, 2h): 迁移 `docs/reviews` 至 `docs/governance/reviews`。
    *   状态: **Completed**
    *   输出: 目录移动，更新引用链接。
*   **T-STRUCT-02** (Medium, 1h): 优化 `docs/archive` 结构。
    *   状态: **Completed**
    *   操作: 将 `docs/archive/plans/v0.6.0/` 扁平化为 `docs/archive/v0.6.0/plans/` (按版本归档而非按类型)。
*   **T-STRUCT-03** (High, 1h): 清理 `docs/reports` 根目录。
    *   状态: **Completed**
    *   操作: 将非 Active 报告移入 `docs/archive/reports`，仅保留模板或当前季度报告。
*   **T-STRUCT-04** (Critical, 1h): 消除 `docs/source` 与 `docs/` 的内容重复。
    *   状态: **Completed** (2026-03-09)
    *   操作: 将 `source/{guides,policy,reference,architecture,plans}` 替换为指向 `docs/` 对应目录的软链接，确保单一真实源 (SSOT)。
*   **T-CONFIG-01** (High, 0.5h): 修正 Sphinx 语言配置。
    *   状态: **Completed** (2026-03-09)
    *   操作: 将 `conf.py` 中的 `language` 设置为 `zh_CN`。

### 3.2 内容补全 (Content Completion)
*   **T-CONTENT-01** (High, 2h): 为缺失目录创建 `README.md`。
    *   范围: `docs/governance/reviews/`, `docs/reports/`, `docs/plans/`, `docs/archive/reports/`。
    *   内容: 目录用途、文件命名规范、贡献指南。
    *   状态: **Completed**
*   **T-CONTENT-02** (Medium, 1h): 更新 `docs/README.md`。
    *   操作: 同步新的目录结构，增加 `governance/reviews` 入口。
    *   状态: **Completed**
*   **T-CONTENT-03** (High, 2h): 增强 `Quickstart` 指南。
    *   操作: 补充“一键安装命令”和“Hello World”配置片段，减少用户在 Examples 目录间的跳转成本。
    *   状态: **Completed**

### 3.3 治理体系 (Governance System)
*   **T-GOV-01** (High, 1h): 落地 RACI 矩阵。
    *   状态: **Completed**
    *   操作: 将审查报告中的 RACI 表格写入 `docs/policy/maintenance.md`。
*   **T-GOV-02** (Medium, 1h): 定义审计日志格式。
    *   状态: **Completed**
    *   操作: 在 `docs/governance/reviews/README.md` 中定义 Review 报告的 Front Matter 标准。

### 3.4 自动化与工具 (Automation)
*   **T-AUTO-01** (High, 4h): 开发 `scripts/doc-check.py`。
    *   状态: **Completed**
    *   功能: 检查死链、检查 `README.md` 是否存在、检查 Front Matter。
*   **T-AUTO-02** (Medium, 2h): 集成 GitHub Actions / Pre-commit。
    *   状态: **Completed**
    *   操作: 配置 `.github/workflows/doc-lint.yml`。

---

## 4. 后续执行计划

### 4.1 剩余任务清单
1.  **归档结构扁平化 (T-STRUCT-02)**: 将 `docs/archive/plans/v0.6.0/` 重组为 `docs/archive/v0.6.0/plans/`，保持与新归档策略（v0.6.1）一致。
2.  **README 补全 (T-CONTENT-01)**: 扫描所有无 `README.md` 的目录，批量创建索引文件。

### 4.2 验证与收尾
1.  运行 `python3 scripts/doc_check.py` 确保所有检查项通过。
2.  运行 `make html` 验证 Sphinx 构建无警告。
3.  提交最终 PR 并合并。
