---
title: Upstream Software
status: active
audience: Users / Developers
last-updated: 2026-06-10
owner: Docs Owner
---

# 上游软件与核心依赖（Upstream Software）

- Status: active
- Audience: Users / Developers
- Last-Updated: 2026-06-10

本文档汇总 DP-EVA 的核心上游软件，说明其仓库位置与在本项目中的职责边界。

## 1. DeePMD-kit

- 仓库地址：https://github.com/deepmodeling/deepmd-kit/
- 核心功能：机器学习势训练和推理平台。
- 在 DP-EVA 中的作用：
  - 作为训练、测试与描述符评估的核心计算后端。
  - 通过 `dp` 命令参与 `train / infer / feature` 等流程。

## 2. dpdata

- 仓库地址：https://github.com/deepmodeling/dpdata
- 核心功能：处理 `deepmd/npy`、`deepmd/npy/mixed` 等机器学习势结构数据格式。
- 在 DP-EVA 中的作用：
  - 负责数据集加载、结构读写与多系统数据组织。
  - 为采样、标注、分析等流程提供统一的数据结构接口。

## 3. ABACUS

- 仓库地址：https://github.com/deepmodeling/abacus-develop
- 核心功能：开源第一性原理计算软件。
- 在 DP-EVA 中的作用：
  - 作为 Labeling 阶段的 DFT 计算后端。
  - 承担从候选结构到高精度标注数据的关键计算步骤。

## 4. ASE

- 仓库地址：https://gitlab.com/ase/ase
- 核心功能：原子结构对象、结构读写与计算器生态。
- 在 DP-EVA 中的作用：
  - 作为 `ase.Atoms` 的核心结构表示。
  - v0.8.0 起核心依赖下限为 `ase>=3.28.0`，与 `atst-tools` 运行环境对齐。

## 5. atst-tools（可选）

- 仓库地址：本项目 `test/atst-tools` 参考仓库；发布包为 `atst-tools`。
- 核心功能：基于 ASE 的轨迹探索与过渡态工具。
- 在 DP-EVA 中的作用：
  - 作为可选 exploration backend，首版支持 `md` 与 `relax`。
  - 通过 `dpeva[explore]` 安装，不进入核心依赖。
  - DPEVA 内部 ABACUS writer 参考其 vendored `abacuslite/io/generalio.py` 的 INPUT/KPT/STRU 子集。

## 6. ase-abacus（Legacy）

- 仓库地址：https://gitlab.com/1041176461/ase-abacus
- 状态：历史依赖。v0.8.0 当前主链路不再推荐安装，也不再要求 `ase.io.abacus`。
- 在 DP-EVA 中的历史作用：
  - 曾用于 Labeling 工作流的 ABACUS 输入生成。
  - 已由 `src/dpeva/labeling/abacus_io.py` 的内部最小 writer 替代。

## 7. 依赖分工总览

| 依赖 | 主要阶段 | 角色定位 |
|---|---|---|
| DeepMD-kit | Train / Infer / Feature | 机器学习势训练与推理核心引擎 |
| dpdata | Data IO / Labeling / Analysis | 结构数据格式与系统组织层 |
| ABACUS | Labeling | 第一性原理计算后端 |
| ASE | Labeling / Exploration | 原子结构对象与结构读写基础 |
| atst-tools | Exploration（可选） | md/relax 轨迹探索后端 |
| ase-abacus | Legacy | 历史 ABACUS 输入生成依赖 |
