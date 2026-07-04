---
title: v0.8.1 Archive
status: archived
audience: Historians
last-updated: 2026-07-04
owner: Docs Owner
---

# v0.8.1 归档 (Archive)

- Status: archived
- Audience: Historians
- Last-Updated: 2026-07-04

本文档目录用于存放 v0.8.1 周期已从 active 区移出的计划、执行记录与规格快照。

## 目录结构

- `plans/`: 项目级计划、Superpowers 执行计划与操作交接记录。
- `reports/`: 本轮无 active 报告正文迁入，保留版本索引占位。
- `specs/`: Superpowers 设计规格快照。

## 本次归档摘要

- **[文档治理]** 归档 docs audit 与 DP-EVA operator skill 计划，作为本轮文档治理机制调整的历史入口。
- **[Labeling / SAI ABACUS]** 归档 FP11 first-principles labeling 执行记录；其中 full production rerun 仍为 blocked handoff，不作为生产完成声明。
- **[Dataset Audit]** 归档 dataset audit 设计规格与实现计划快照，供后续版本重新激活或拆分实施。
- **[Slurm Array Backend]** 归档 Slurm array backend refactor 计划快照，记录 SAI Slurm array 方向与边界。
- **[MAP_OPT 责任边界]** v0.8.1 代码侧确认普通 `abacus` launcher 不读取 `MAP_OPT`；只有显式 `launcher_mode="mpi_abacus"` 的 task class 才 source SAI rank-map 并使用 `--map-by $MAP_OPT`。
