---
title: Release Notes v0.6.1
status: archived
audience: Users / Developers
last-updated: 2026-03-09
---

# Release Notes v0.6.1

**发布日期**: 2026-03-09
**版本类型**: Patch Release (文档治理与部署增强)

## 1. 核心变更 (Highlights)

本次发布重点聚焦于**文档系统的工程化治理**与**自动化部署能力的构建**。虽然不包含核心算法逻辑的变更，但对于项目的长期可维护性与用户体验具有里程碑意义。

*   **GitHub Actions 自动部署**: 正式支持将 Sphinx 文档自动构建并发布至 GitHub Pages。
*   **文档语言修正**: Sphinx 配置从 `en` 修正为 `zh_CN`，解决了搜索与索引的语言不匹配问题。
*   **链接完整性修复**: 修复了大量文档间的断链问题，消除了 Sphinx 构建过程中的 Cross-reference 警告。

## 2. 详细变更清单 (Changelog)

### 2.1 部署与工具 (Deployment & Tools)
*   **[New]** 新增 GitHub Actions 工作流 (`.github/workflows/docs-deploy.yml`)，支持多版本文档发布。
*   **[New]** 新增 `docs/guides/deployment.md`，详细说明了部署架构与配置策略。
*   **[Script]** 新增 `scripts/fix_links.py` 工具，用于批量修复 Markdown 中的绝对路径链接，确保 Sphinx 兼容性。

### 2.2 文档治理 (Docs Governance)
*   **[Fix]** 修正 `docs/source/conf.py` 中的 `language = 'zh_CN'`。
*   **[Refactor]** 移除了 `docs/source` 下的冗余实体目录，统一替换为指向 `docs/` 根目录的软链接，确保 "Single Source of Truth"。
*   **[Guide]** 更新 `developer-guide.md`，追加 v0.6.1 版本记录。

## 3. 升级指南 (Upgrade Guide)

本次更新为纯文档与工具链更新，不涉及 Python API 或 CLI 接口变更。
开发者只需拉取最新代码即可享受文档构建与部署的改进：

```bash
git pull origin main
```

若需本地构建文档：

```bash
cd docs
python scripts/fix_links.py
make html
```
