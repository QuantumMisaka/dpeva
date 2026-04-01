---
title: Document
status: active
audience: Developers
last-updated: 2026-03-30
---

# Plans（开发计划）

本页用于给 Sphinx 文档系统提供开发计划导航入口。计划的规范落盘位置仍然是 `docs/plans/`。

## 当前关注主题

- Enhanced Parity Plot 设计规划
  - 核心内容：解释为何 `parity_cohesive_energy_enhanced` 与 `parity_force_enhanced` 在当前版本中出现明显视觉差异，并将这种差异沉淀为 quantity-aware 设计契约。
  - 规范文档：`docs/plans/2026-03-30-enhanced-parity-plot-design-plan.md`

## 维护说明

- `docs/plans/` 保存团队共享的正式计划文档；
- `docs/source/plans/README.md` 仅承担导航职责，避免 Sphinx 索引遗漏；
- 新增计划时，应同步更新本页与 `docs/source/index.rst` 的相关 `toctree`。
