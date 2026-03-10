---
title: Upstream Software
status: active
audience: Users / Developers
last-updated: 2026-03-09
---

# 上游软件与核心依赖（Upstream Software）

- Status: active
- Audience: Users / Developers
- Last-Updated: 2026-03-09

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

## 4. ase-abacus

- 仓库地址：https://gitlab.com/1041176461/ase-abacus
- 核心功能：用于 ABACUS 计算的数据预处理。
- 在 DP-EVA 中的作用：
  - 负责 ABACUS 计算前的数据准备与输入转换。
  - 为 Labeling 工作流中的结构预处理提供支持。

## 5. 依赖分工总览

| 依赖 | 主要阶段 | 角色定位 |
|---|---|---|
| DeepMD-kit | Train / Infer / Feature | 机器学习势训练与推理核心引擎 |
| dpdata | Data IO / Labeling / Analysis | 结构数据格式与系统组织层 |
| ABACUS | Labeling | 第一性原理计算后端 |
| ase-abacus | Labeling 预处理 | ABACUS 输入准备与结构转换 |
