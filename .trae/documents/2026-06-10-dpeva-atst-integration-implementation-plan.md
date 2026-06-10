---
title: DPEVA 去除 ase-abacus 并接入 atst-tools exploration backend 实施计划
status: draft
audience: Full-stack Engineer / Maintainers
last-updated: 2026-06-10
owner: Engineering
---

# DPEVA 去除 ase-abacus 并接入 atst-tools exploration backend 实施计划

## 1. 文档目的

本文档用于交付给工程师开展真实开发，目标是把前期已经达成一致的技术判断收敛为一份可执行的任务清单、实施顺序、文件级改造方案、验收标准与风险控制方案。

本次任务需要同时解决两个问题：

1. 将 DPEVA 当前对 GitLab `ase-abacus` fork 的运行时依赖移除。
2. 为 DPEVA 未来接入 `atst-tools` 作为结构探索/物理采样 backend 建立稳定边界，但不把 `atst-tools` 变成 DPEVA 的核心硬依赖。

## 2. 已确认的设计结论

### 2.1 结论 A：DPEVA 不应继续依赖 GitLab `ase-abacus`

当前 DPEVA 对 `ase-abacus` 的真实使用范围非常小，只集中在 Labeling 输入生成阶段，主要能力是：

- 写 `INPUT`
- 写 `STRU`
- `KPT` 实际由 DPEVA 自己生成

因此没有必要继续维护对旧 fork 的运行时绑定。

### 2.2 结论 B：DPEVA 应在本项目内部自持最小 ABACUS writer

对 DPEVA 而言，更合理的做法是在项目内部实现最小闭环的 ABACUS writer，仅覆盖当前 Labeling 真实需要的能力：

- `write_input_file`
- `write_stru_file`
- `write_kpt_file`

这比把 Labeling 入口直接切换到 `atst-tools` 更稳，因为：

- DPEVA 当前只需要极小能力子集
- `atst-tools` 是完整 workflow toolkit，不是轻量 I/O 库
- 直接依赖 `atst-tools` 会引入更大依赖树与配置模型

### 2.3 结论 C：`atst-tools` 可以作为 DPEVA 的可选 exploration backend

`atst-tools` 可作为 DPEVA 将来“结构探索/物理采样”能力的外部 backend，承担例如：

- MD
- Relax
- NEB / AutoNEB
- Dimer
- IRC

但其定位应为：

- 可选依赖
- 外部 backend
- 独立子系统

而不是 DPEVA 的核心基础依赖。

### 2.4 结论 D：DPEVA 现有 `sampling` 与 `atst-tools` 的关系是互补而非替代

DPEVA 当前已经具备候选筛选采样能力，其 `sampling` 包负责：

- DIRECT
- 2-DIRECT
- 候选特征筛选
- 与 UQ/Collection 联动

`atst-tools` 补充的是“结构探索/候选生成”能力，而不是对现有 `sampling` 的替换。

因此未来的统一闭环应是：

`exploration(atst-tools)` -> `feature` -> `UQ` -> `DIRECT` -> `label`

## 3. 本次任务目标

### 3.1 主目标

在不破坏 DPEVA 当前主链路的前提下，完成以下目标：

1. 从 DPEVA Labeling 主链路中移除 `ase-abacus` 运行时依赖。
2. 在 DPEVA 内部建立最小 ABACUS writer。
3. 在架构层为 `atst-tools` 接入预留稳定 exploration backend 边界。
4. 首版只完成 `atst-tools` backend 的最小闭环接入能力设计与骨架实现，优先支持 `md` / `relax`。
5. 将 `atst-tools` 控制为可选依赖，而不是默认硬依赖。

### 3.2 成功标准

满足以下条件视为本任务完成：

- `dpeva label` 不再依赖 GitLab `ase-abacus`
- DPEVA 内部 writer 能稳定生成 `INPUT/STRU/KPT`
- 关键 Labeling 单元测试通过
- exploration 抽象层落地
- `ATSTToolsBackend` 骨架可运行，至少覆盖 `md` / `relax`
- `atst-tools` 未被加入主依赖，仅在启用 exploration 时要求安装
- 文档与配置入口更新，工程师能根据文档继续扩展

## 4. 本次任务非目标

以下内容不属于本次交付范围：

- 不在本次任务中完整替换 `dpdata` 的 ABACUS 结果解析链路
- 不在本次任务中完成 `neb/autoneb/dimer/irc` 的全量接入
- 不在本次任务中重写 `atst-tools`
- 不在本次任务中实现 DPEVA 全新的物理采样算法
- 不在本次任务中立即彻底清理 `atst-tools` 自身的 vendored backend

## 5. 目标架构

### 5.1 DPEVA 内部边界

本次改造后，DPEVA 内部职责划分如下：

- `labeling/`
  - 负责 ABACUS 输入生成
  - 负责 ABACUS 结果收集与导出
  - 不依赖 `ase-abacus`

- `sampling/`
  - 继续负责 DIRECT / 2-DIRECT 候选筛选
  - 不承担物理采样运行职责

- `exploration/`（新增）
  - 负责结构探索 backend 抽象
  - 统一接入 `atst-tools`
  - 负责外部 workflow 的 prepare / run / collect

### 5.2 运行时依赖边界

- `dpeva` 基础安装：
  - 不依赖 `atst-tools`
  - 不依赖 `ase-abacus`

- `dpeva[explore]`：
  - 增加 `atst-tools`
  - exploration 启用时才需要

### 5.3 数据流边界

#### Labeling 主链路

`dataset -> generator -> ABACUS input -> ABACUS run -> dpdata parse -> export`

#### Exploration 扩展链路

`atst-tools exploration -> structures/trajectory -> DPEVA feature/UQ/sampling -> label`

## 6. 总体实施策略

实施分为六个阶段，必须按顺序推进，避免一次性引入过多不确定性。

### 阶段顺序

1. 先完成 DPEVA 内部 writer 替换，切掉 `ase-abacus`
2. 再补齐回归测试
3. 再引入 exploration 抽象层
4. 再接 `ATSTToolsBackend`
5. 再增加配置、可选依赖和文档
6. 最后做小规模集成验证

## 7. 详细任务清单

---

## Task Group 0: 基线固化与开发前准备

### TG0-1 记录现状基线

**目标**

在改动前固化当前 Labeling 行为，便于后续比较。

**执行内容**

- 记录当前 `generator.py` 的输出行为
- 准备至少一组可重复使用的 `Atoms + config` 测试输入
- 固化当前 `INPUT/STRU/KPT` 输出样例作为 golden baseline

**建议文件**

- `tests/unit/labeling/fixtures/`
- `tests/unit/labeling/golden/`

**交付物**

- 最小样例输入
- golden 输出文件

**完成标准**

- 测试中可以稳定复用这些基线样例

---

## Task Group 1: DPEVA 内部最小 ABACUS writer

### TG1-1 新增内部 writer 模块

**目标**

在 DPEVA 内新增最小 ABACUS writer，作为 `ase-abacus` 的替代。

**新文件**

- `src/dpeva/labeling/abacus_io.py`

**需要实现的函数**

- `write_input_file(path: Path, parameters: dict) -> Path`
- `write_stru_file(path: Path, atoms: Atoms, pp_map: dict[str, str], orb_map: dict[str, str] | None) -> Path`
- `write_kpt_file(path: Path, kpoints: list[int]) -> Path`

**实现约束**

- 仅支持当前 DPEVA Labeling 真正使用的 ABACUS 输入子集
- 不引入多余抽象
- 保持 ASCII 输出
- 对于缺失的 `pp_map/orb_map` 提供清晰报错

**关键点**

- `INPUT`
  - 写出 `INPUT_PARAMETERS`
  - 按 `key value` 输出
  - 保留 `orbital_dir` / `pseudo_dir`

- `STRU`
  - 支持 species 分组
  - 保留元素首次出现顺序
  - 支持 `magmom`
  - 支持 `pp_file` / `orb_file`
  - 坐标统一用 Cartesian

- `KPT`
  - 沿用当前 DPEVA 逻辑写 `Gamma`

**参考来源**

- DPEVA 现有 `generator.py`
- `atst-tools` 中 `generalio.write_input/write_stru/write_kpt`

### TG1-2 替换 Labeling 入口

**目标**

让 Labeling 生成阶段全面切换到内部 writer。

**修改文件**

- `src/dpeva/labeling/generator.py`

**执行内容**

- 移除 `from ase.io.abacus import write_input, write_abacus`
- 改为导入内部 `abacus_io`
- 删除旧的 ImportError 提示文案
- `generate()` 中改为调用：
  - `write_input_file(...)`
  - `write_stru_file(...)`
  - `write_kpt_file(...)`

**注意事项**

- 保持现有 `StructureAnalyzer`、`mag_map`、`kpt_criteria` 行为不变
- 不要修改 `manager.py` 和 `postprocess.py` 的逻辑

### TG1-3 清理旧依赖提示

**目标**

清理项目中对旧 GitLab 安装提示的直接依赖描述。

**修改范围**

- `src/dpeva/labeling/generator.py`
- `.github/workflows/docs-check.yml`
- `.github/workflows/docs-deploy.yml`
- 相关文档中 `ase-abacus` 的安装说明

**处理原则**

- 若文档中提及旧 fork 为“历史背景”，可以保留，但必须明确标注 legacy
- 不再保留“请安装 git+gitlab ... ase-abacus”作为当前推荐路径

---

## Task Group 2: 回归测试与行为校验

### TG2-1 新增 writer 单测

**目标**

为新 writer 提供稳定回归保护。

**建议新增文件**

- `tests/unit/labeling/test_abacus_io.py`

**测试覆盖**

- `INPUT` 基本写入
- `INPUT` 关键参数写入
- `STRU` species 顺序
- `STRU` `pp_map/orb_map` 写入
- `STRU` 初始磁矩写入
- `KPT` Gamma 格式写入
- cluster / layer 对输入参数的影响

### TG2-2 修订现有 generator 单测

**目标**

把旧测试从 mock `ase.io.abacus` 调整到 mock 内部 writer。

**修改文件**

- `tests/unit/labeling/test_labeling_generator.py`

**执行内容**

- 移除对 `write_abacus` / `write_input` 的旧 patch 路径依赖
- 改成 patch `dpeva.labeling.abacus_io.*` 或 `generator` 内新的导入路径

### TG2-3 Golden output 契约测试

**目标**

保证新 writer 输出与旧基线兼容。

**建议文件**

- `tests/unit/labeling/test_abacus_io_golden.py`

**执行内容**

- 对固定输入生成 `INPUT/STRU/KPT`
- 与 golden baseline 做文本比对
- 明确允许差异项（若存在）

---

## Task Group 3: 引入 exploration 抽象层

### TG3-1 定义 exploration 基础对象

**目标**

为未来所有 exploration backend 提供统一协议。

**新文件**

- `src/dpeva/exploration/base.py`

**需要定义的对象**

- `ExplorationRequest`
- `ExplorationResult`
- `ExplorationArtifact`
- `ExplorationBackend` 抽象类

**建议字段**

`ExplorationRequest`

- `backend`
- `workflow_type`
- `work_dir`
- `config_path`
- `input_structures`
- `metadata`

`ExplorationResult`

- `status`
- `structures`
- `artifacts`
- `metrics`
- `error_message`

### TG3-2 新增 exploration manager

**目标**

统一创建和调用 backend。

**新文件**

- `src/dpeva/exploration/manager.py`

**需要实现**

- backend registry
- `get_backend(...)`
- `run_exploration(...)`

**要求**

- DPEVA 主流程只能依赖 manager，不直接依赖具体 backend

---

## Task Group 4: 接入 `ATSTToolsBackend`

### TG4-1 新增 atst backend 实现

**目标**

建立 DPEVA 对 `atst-tools` 的首个外部 exploration backend。

**新文件**

- `src/dpeva/exploration/atst_backend.py`

**首版范围**

- 只支持 `md`
- 只支持 `relax`

**不在首版范围**

- `neb`
- `autoneb`
- `dimer`
- `irc`
- `d2s`

### TG4-2 运行方式

**原则**

优先使用 CLI，而不是 import 深层内部模块。

**建议调用方式**

- `atst run CONFIG.yaml`
- 需要时使用：
  - `atst abacus prepare`
  - `atst abacus collect`

**原因**

- CLI 是 `atst-tools` 的稳定公开接口
- 可以降低对其内部包结构变动的脆弱性

### TG4-3 结果回收

**目标**

将 `atst-tools` 结果统一转换为 DPEVA 可消费对象。

**新文件**

- `src/dpeva/exploration/io.py`

**建议实现**

- 从 `traj/extxyz/cif/STRU` 读取结果
- 统一转成 `ASE Atoms`
- 输出 `ExplorationResult`

**要求**

- 结构解析错误集中在这一层处理
- 不要把格式兼容逻辑散落到 workflow 层

### TG4-4 可选依赖检查

**目标**

在 exploration 启用时给出清晰的依赖提示。

**执行内容**

- 新增 dependency check helper
- 若启用 `backend=atst-tools` 但环境缺失：
  - 抛显式错误
  - 给出安装建议

**建议错误文案**

- `Exploration backend 'atst-tools' is enabled but package/CLI 'atst' is not available. Install dpeva[explore] or pip install atst-tools.`

---

## Task Group 5: 配置模型与主流程接线

### TG5-1 增加最小 exploration 配置模型

**目标**

只为 DPEVA 侧暴露必要配置，不复制 `atst-tools` 全量 schema。

**建议修改文件**

- `src/dpeva/config.py`

**建议新增字段**

- `exploration_enabled: bool = False`
- `exploration_backend: Literal["atst-tools"] | None`
- `exploration_workflow_type: Literal["md", "relax"] | None`
- `exploration_config_path: Path | None`
- `exploration_work_dir: Path | None`

### TG5-2 接入主工作流

**目标**

在 DPEVA 中为 exploration 预留入口，但不扰动当前默认主链路。

**建议方式**

- 首版不强行修改现有 `train/infer/collect/label` 默认流程
- 可新增：
  - 新 workflow
  - 或新 CLI 子命令
  - 或在 collection 前增加可选阶段

**推荐**

先新增独立入口，避免和当前主链路强耦合。

### TG5-3 明确链路衔接

**目标**

把 exploration 结果送入 DPEVA 的既有体系。

**建议衔接方式**

- exploration 输出结构集合
- 经 `feature` 生成描述符
- 经 `UQ + sampling` 进行筛选
- 进入 `label`

**原则**

- exploration 负责“生成候选”
- sampling 负责“筛选候选”
- 两层职责不混用

---

## Task Group 6: 依赖治理与文档更新

### TG6-1 调整依赖声明

**目标**

把 `atst-tools` 作为可选依赖，而不是主依赖。

**建议修改文件**

- `pyproject.toml`

**建议形式**

- `[project.optional-dependencies]`
- `explore = ["atst-tools>=2.1.0"]`

### TG6-2 更新文档

**需要更新的文档类型**

- 开发者文档
- Labeling/配置文档
- 安装说明
- 迁移说明

**建议至少更新**

- `docs/guides/developer-guide.md`
- `docs/guides/configuration.md`
- `docs/guides/cli.md`
- `README.md`

**需要表达清楚的事实**

- `ase-abacus` 已不再是当前推荐依赖
- DPEVA 内部已自持最小 writer
- `atst-tools` 用于 exploration backend，且为可选依赖

### TG6-3 清理 CI 与示例

**目标**

确保 CI 不再安装旧 GitLab fork。

**建议修改文件**

- `.github/workflows/docs-check.yml`
- `.github/workflows/docs-deploy.yml`
- `examples/scripts/labeling/run_labeling.sh`

**原则**

- Labeling 示例不再要求用户手工安装旧 `ase-abacus`

---

## Task Group 7: 最小集成验证

### TG7-1 Labeling 回归验证

**检查项**

- `dpeva label --stage prepare` 正常运行
- `INPUT/STRU/KPT` 生成正确
- 现有 Labeling 测试通过

### TG7-2 Exploration 最小闭环验证

**首版验证范围**

- `md`
- `relax`

**检查项**

- `atst-tools` 缺失时给出清晰错误
- `atst-tools` 存在时 backend 能正常启动
- 结果可回收为 `ASE Atoms`
- 后续可送入 DPEVA feature/UQ 流程

## 8. 文件级改造清单

### 必改文件

- `src/dpeva/labeling/generator.py`
- `src/dpeva/config.py`
- `pyproject.toml`
- `.github/workflows/docs-check.yml`
- `.github/workflows/docs-deploy.yml`
- `examples/scripts/labeling/run_labeling.sh`

### 必增文件

- `src/dpeva/labeling/abacus_io.py`
- `src/dpeva/exploration/base.py`
- `src/dpeva/exploration/manager.py`
- `src/dpeva/exploration/atst_backend.py`
- `src/dpeva/exploration/io.py`
- `tests/unit/labeling/test_abacus_io.py`

### 建议新增文件

- `tests/unit/labeling/test_abacus_io_golden.py`
- `tests/unit/exploration/test_atst_backend.py`
- `tests/unit/exploration/test_exploration_manager.py`

## 9. 开发顺序建议

工程师应严格按以下顺序开发：

1. 写 `abacus_io.py`
2. 改 `generator.py`
3. 补 Labeling 单测与 golden tests
4. 清理旧依赖提示与 CI
5. 新增 `exploration/base.py`
6. 新增 `exploration/manager.py`
7. 新增 `ATSTToolsBackend`
8. 加最小配置模型
9. 接 exploration 最小入口
10. 做 `md/relax` 最小集成验证
11. 更新文档

不建议跳步并行推进，否则容易把“去旧依赖”和“接新 backend”两个问题混在一起，导致回归定位困难。

## 10. 验收清单

### 10.1 功能验收

- [ ] `dpeva label` 不再导入 `ase.io.abacus`
- [ ] 内部 writer 可生成 `INPUT/STRU/KPT`
- [ ] 关键测试通过
- [ ] `atst-tools` 仅在 exploration 启用时需要
- [ ] exploration 首版支持 `md/relax`
- [ ] exploration 结果能进入 DPEVA 后续流程

### 10.2 架构验收

- [ ] `sampling` 与 `exploration` 职责分离
- [ ] DPEVA 未直接 import `atst_tools.external.*`
- [ ] DPEVA 与 `atst-tools` 的耦合点集中在 backend 层
- [ ] `atst-tools` 未成为主依赖

### 10.3 文档验收

- [ ] 安装说明已更新
- [ ] 配置说明已更新
- [ ] exploration 的边界与用途已写清
- [ ] 旧 `ase-abacus` 已被标注为 legacy

## 11. 风险与缓解方案

### 风险 1：内部 `write_stru` 兼容性不足

**表现**

- 生成的 `STRU` 无法被 ABACUS 正确识别
- 磁矩、species 顺序、orb/pp 路径处理出错

**缓解**

- 参考 `atst-tools/generalio`
- 增加 golden tests
- 至少准备 2 到 3 组典型结构样例

### 风险 2：`atst-tools` CLI 接口变化

**表现**

- DPEVA backend 调用命令失败

**缓解**

- 只依赖公开 CLI
- 在 backend 中集中构造命令
- 增加最小契约测试

### 风险 3：把 exploration 与 sampling 职责混淆

**表现**

- 生成候选与筛选候选耦合在一起
- 未来扩展困难

**缓解**

- 严格保持两个子系统分离
- `ExplorationResult` 只输出候选结构，不输出筛选决策

### 风险 4：可选依赖误变成强依赖

**表现**

- 未启用 exploration 时仍要求安装 `atst-tools`

**缓解**

- 仅在 runtime 检查
- `pyproject.toml` 放到 optional dependencies

## 12. 工程实施建议

### 12.1 PR 拆分建议

建议至少拆成三个 PR：

#### PR-1：Labeling 去 `ase-abacus`

- 新增 `abacus_io.py`
- 修改 `generator.py`
- 补测试
- 清理旧依赖提示

#### PR-2：exploration 抽象层

- 新增 `base.py`
- 新增 `manager.py`
- 配置骨架

#### PR-3：`ATSTToolsBackend`

- 新增 `atst_backend.py`
- 新增 `io.py`
- 可选依赖接入
- 最小 `md/relax` 验证

### 12.2 Review 重点

代码审查时重点检查：

- `STRU` 输出兼容性
- exploration 边界是否足够薄
- 是否出现对 `atst-tools` 内部模块的深层耦合
- 是否误伤现有 Labeling 主链路

## 13. 交付物清单

本次任务最终应产出以下交付物：

1. 代码实现
2. 新增与更新的测试
3. 安装与配置文档更新
4. migration 说明
5. CI 清理结果
6. exploration backend 最小可运行示例

## 14. 给工程师的执行指令

1. 先完成 PR-1，不要同时开做 exploration。
2. 在 PR-1 合并前，不要改动主 workflow 编排。
3. PR-2 只做抽象层，不直接引入复杂 backend。
4. PR-3 首版只做 `md/relax`，严禁 scope creep。
5. 每个阶段结束后都执行对应测试并记录结果。

## 15. 最终判断摘要

本次任务的推荐实施路线是：

- **DPEVA 内部自持最小 ABACUS writer**
- **`atst-tools` 作为可选 exploration backend**
- **保留 DPEVA 现有 sampling 作为候选筛选层**
- **两者职责分离，逐步打通主动学习闭环**

该路线兼顾了：

- 去除旧 `ase-abacus` 依赖
- 可持续维护
- 控制改动风险
- 为后续结构探索能力扩展保留清晰架构边界
