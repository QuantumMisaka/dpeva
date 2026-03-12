---
title: Document
status: active
audience: Developers
last-updated: 2026-03-12
owner: Docs Owner
---

# 开发计划 (Development Plans)

- Status: active
- Audience: Developers
- Last-Updated: 2026-03-12

本目录用于存放“项目级、可共享、可追溯”的开发计划文档，不承载 AI 执行期草案。

## 1. 计划类型

- **Iteration Plan**: 迭代周期计划（目标、范围、验收标准）。
- **Governance Plan**: 文档/代码治理计划（规则、迁移、门禁）。
- **Feature Plan**: 新功能实施计划（设计、测试、回滚策略）。

## 2. 维护策略

- **Active**: 当前有效且需要团队协作的计划。
- **Completed**: 完成后归档至 `docs/archive/vX.Y.Z/plans/`，并从活跃索引移除。
- **Naming**: `YYYY-MM-DD-<topic>-plan.md` 或 `vX.Y.Z-<topic>-plan.md`。
- **Ingestion Rule**: 草案先放 `.trae/documents/`；仅“收敛计划”才迁入本目录。

## 3. AI 执行文档边界

为避免过程文档污染项目导航，统一按以下边界执行：

- `.trae/documents/`：任务执行草案、临时计划、探索记录。
- `docs/plans/`：团队共享、可复盘、可长期追踪的计划版本。
- 迁入 `docs/plans/` 的前提：
  1. 目标与验收标准明确；
  2. 影响范围与风险说明完整；
  3. 可被非当前执行者理解并复用。

## 4. 模板 (Template)

```markdown
# 计划标题 (Plan Title)

- Status: active
- Audience: Maintainers
- Last-Updated: 2026-03-09

## 1. 目标 (Goals)
## 2. 任务 (Tasks)
## 3. 验收标准 (Acceptance Criteria)
```

## 5. 归档与索引要求

- 文档移动或重命名时，必须同步更新 `docs/source/**/*.rst` 的 `toctree`。
- 归档后必须更新对应版本目录 `docs/archive/vX.Y.Z/README.md`。
- 提交前必须通过：
  - `python3 scripts/doc_check.py`
  - `python3 scripts/check_docs_freshness.py --days 90`
  - `make -C docs html SPHINXOPTS="-W --keep-going"`
