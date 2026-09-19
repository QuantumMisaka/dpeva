---
title: Superpowers Execution Plans
status: historical-execution-records
audience: Developers / AI Agents
last-updated: 2026-09-20
owner: Docs Owner
---

# Superpowers Execution Plans

Execution plans in this directory preserve task-level checklists for agentic
implementation work. They are not a replacement for project-level plans in
`docs/plans/`. A plan's original checkbox list captures the approved execution
design at that time; it is not a live status tracker.

## Current status authority

- For the FT2DP-v2.2 campaign, the non-Git workspace tracker
  `docs/tasks/ft2dp-v2.2.md` is the sole
  current status and acceptance authority. Its checkpoint index is
  `docs/checkpoints/INDEX.md`.
- These historical plans have execution-status prose at their end and must
  link back to the workspace tracker for any continuing work:
  - `2026-07-10-sai-dpa4-env-rename-finetune.md`
  - `2026-07-11-dpa4-multitask-forgetting-study.md`
  - `2026-09-02-pr-template-and-dpa4c-review-governance.md`
- v0.8.1 archived execution plans are listed in [docs/archive/v0.8.1/plans/README.md](../../archive/v0.8.1/plans/README.md).

## Completed, pending integration

- [v0.8.2 compatibility closeout](2026-09-05-v082-compatibility-closeout.md): six tasks and independent final re-review completed; branch retained without merge, push, tag, or publication. [Verified closeout](../../reports/2026-09-05-v082-compatibility-closeout.md).

## Proposed plans

- [LMDB 数据格式兼容](2026-09-20-lmdb-format-compatibility.md)（status: proposed）: 读侧契约、失败语义、能力矩阵 `data_format` 维度、消费链对齐与写出侧的完整编排。**Phase 1 已实施**（`bf1b58d`、`ac58940`，见计划 §9.3）；Phase 2–5 与决策门 D2–D4 仍待确认。需求来源见 [LMDB 读取支持提案](../../reports/2026-09-19-lmdb-read-support.md)。

## Superseding status pointer

The Plans A–E files remain historical execution records and are not rewritten when
later governance or release rulings change. Their current completion state and the
approved v0.8.2 patch positioning are recorded in
[`../specs/2026-09-04-project-governance-and-deepmd-3-2-design.html`](../specs/2026-09-04-project-governance-and-deepmd-3-2-design.html)
and [`../../reports/2026-09-05-v082-compatibility-closeout.md`](../../reports/2026-09-05-v082-compatibility-closeout.md).
Future independent-family/cross-model review is optional under the later policy;
historical review results retain their original scope and evidence value.
