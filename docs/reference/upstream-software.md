---
title: Upstream Software
status: active
audience: Users / Developers
last-updated: 2026-09-05
owner: Docs Owner
---

# 上游软件与依赖边界（Upstream Software）

- Status: active
- Audience: Users / Developers
- Last-Updated: 2026-09-05

本文档汇总 DP-EVA 的上游软件，说明其仓库位置、安装层级与在本项目中的职责边界。

## 1. DeePMD-kit（默认有界运行时依赖）

- 仓库地址：https://github.com/deepmodeling/deepmd-kit/
- 核心功能：机器学习势训练和推理平台。
- 安装层级：默认核心安装提供 `deepmd-kit>=3.1.2,<3.3`；需要明确 3.2
  qualification lane 时安装 `dpeva[deepmd]`。
- 版本边界：默认包络保留 3.1.2 起的运行兼容性，用户 extra 为
  `deepmd-kit>=3.2,<3.3`。这些范围是依赖解析边界，不能据此宣称其中每个版本
  的行为完全等价；3.1 运行时也不因此获得新的科学验证。
- 研究生产锁定：正式结果使用独立环境锁定 `deepmd-kit==3.2.0`，并在运行记录中保存 `dp --version`；不能用宽范围 extra 替代研究环境锁。
- 在 DP-EVA 中的作用：
  - 作为训练、测试与描述符评估的计算后端。
  - 通过 `dp` 命令参与 `train / infer / feature` 等流程。

### 1.1 3.2 compatibility qualification

DeepMD-kit 3.2 的能力状态由
`src/dpeva/compatibility/deepmd-3.2.json` 的精确 operation/backend/model/
artifact/data/environment key 管理。当前有 3 条 `supported` 记录（DPA4
pt test/eval-desc/embed），其余能力仍保持明确的
experimental/unsupported/blocked 边界。SAI V100 qualification
JobID `1128260` 与 CPU contract JobID `1128442` 的 producer-issued JSON
证据已落库；完整边界见仓库报告
`docs/reports/2026-09-04-deepmd-3.2-compatibility.md`。

发布或研究运行不得仅因版本号落在 `>=3.2,<3.3` 就晋级能力。只有精确验证命令、
CPU evidence，以及需要时的 SAI evidence 均存在并通过，Compatibility Owner 才能
在同一变更中更新 manifest 和报告。后续 SAI qualification 仍需新的显式授权、
不可变证据目录和完整 collector gate；当前结果不外推为科学精度或普遍 GPU/runtime
正确性。

`dpeva doctor` 将两条诊断 lane 分开：稳定的 `>=3.1.2,<3.3` 运行时包络用于
保留旧环境可用性，稳定 3.2 版本另行报告 qualification；3.1 不会被重新标记为
3.2 已验证能力。legacy runtime 上缺少 3.2 才有的 CLI surface 属于信息性检查，
不会把仍可用的旧环境整体判为 incompatible。

当前 manifest 共 17 条（3 supported、8 experimental、4 unsupported、2
blocked-upstream）。每条记录还声明 `verification_status` 与
`required_evidence`；`implemented` 必须绑定可收集的 pytest node，尚未具备测试的
train/fine-tune/freeze、LMDB、pretrained alias 和 deploy 路线保持 planned，不能把
占位命令当作已执行证据。`candidate-evaluation` 是供上层 generic preflight 使用的
policy capability，不映射为 DeepMD CLI command。

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
| DeepMD-kit | Train / Infer / Feature（默认有界；显式 3.2 lane） | 机器学习势训练与推理计算引擎；默认 `>=3.1.2,<3.3`，用户 extra `dpeva[deepmd]`，研究生产锁定 `==3.2.0` |
| dpdata | Data IO / Labeling / Analysis | 结构数据格式与系统组织层 |
| ABACUS | Labeling | 第一性原理计算后端 |
| ASE | Labeling / Exploration | 原子结构对象与结构读写基础 |
| atst-tools | Exploration（可选） | md/relax 轨迹探索后端 |
| ase-abacus | Legacy | 历史 ABACUS 输入生成依赖 |
