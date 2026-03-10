---
title: Archived Document
status: archived
audience: Historians
last-updated: 2026-03-09
---

# DP-EVA v0.6 迭代开发方案 (Iteration Plan)

**编制日期**: 2026-03-08  
**当前版本**: v0.5.3  
**目标版本**: v0.6.0  
**状态**: 已验收 (Accepted)

---

## 1. 复核结论与问题重定义

基于对 `docs/`、`src/`、`tests/` 的复核，本轮 v0.6 的核心判断如下：

1. `GAP-01`（Analysis 功能单一）与 `GAP-02`（Data Integration 缺失）为发布阻断项，优先级保持 **P0**。  
2. `GAP-04`（Analysis Skill 缺失）为低成本高收益项，优先级保持 **P0**。  
3. `GAP-03`（文档系统混乱）重定义为：**职责边界与入口一致性问题**，不是目录数量问题。  
4. 文档侧不在 v0.6 执行大规模目录迁移（如 `docs/project`），改为“边界澄清 + 索引治理 + 规范收敛”。  
5. `GAP-05`（全链路集成测试）按“契约测试优先、E2E 增量扩展”的方式推进，避免被外部环境阻塞。

---

## 2. 迭代目标、非目标与范围

### 2.1 迭代目标 (Goals)

本轮迭代目标为：**补齐数据闭环、扩展分析能力、完成文档边界治理、形成可发布证据链**。

1. `dpeva analysis` 支持 `dataset` 模式并可输出统计与图表。  
2. `dpeva label` 在工作流尾部支持自动数据整合，产出下一轮可训练数据。  
3. `analysis` skill 文档补齐并与 CLI/指南一致。  
4. 文档集完成职责边界修复（尤其是 `governance`/`archive`、`plans`/`archive/specs`）。

### 2.2 非目标 (Non-Goals)

1. 不在 v0.6 执行 `docs/` 大规模目录迁移。  
2. 不在 v0.6 引入新的顶层文档域（如 `docs/project`）。  
3. 不在 v0.6 重写历史归档文档内容，只补边界标识与入口指引。  

### 2.3 范围 (Scope)

- **功能范围（必须交付）**
  - Dataset Analysis MVP
  - Data Integration MVP
  - Analysis Skill 文档
  - 测试补齐（unit/contract + 增量 E2E）
- **文档治理范围（必须交付）**
  - 明确 active 与 archive 的职责边界
  - 修复索引/结构树/引用漂移
  - 明确 spec 的 active 落点策略（采用规范声明，不做大迁移）

---

## 3. 差异项与优先级

| ID | 差异项 | 复核结论 | 优先级 | 处理策略 |
| :--- | :--- | :--- | :--- | :--- |
| GAP-01 | Analysis 功能单一 | 成立 | P0 | 增加 dataset 模式与统一配置分流 |
| GAP-02 | 数据闭环缺失 | 成立 | P0 | 在 Labeling 末端接入 Integration |
| GAP-03 | 文档系统混乱 | 成立（重定义） | P1 | 做边界治理，不做目录大迁移 |
| GAP-04 | Analysis Skill 缺失 | 成立 | P0 | 新增 skill，并与 CLI/guide 对齐 |
| GAP-05 | 全链路测试不足 | 部分成立 | P1 | 先补契约测试，再增量 E2E |

---

## 4. 技术实施方案

### 4.1 Dataset Analysis MVP

- **目标**: 在现有 `AnalysisWorkflow` 上增加 dataset 分析能力。  
- **参考脚本**：`utils/fp/converged_data_view.py`。需要确保对该脚本数据分析和可视化部分的全面覆盖和超越。
- **建议路径**:
  - `src/dpeva/analysis/dataset/`（新增）
  - `src/dpeva/workflows/analysis.py`（扩展模式分流）
  - `src/dpeva/config.py`（扩展 `AnalysisConfig`）
- **核心设计**:
  - `DatasetLoader`: 负责 deepmd/npy(/mixed) 的系统加载与标准化。
  - `DatasetStats`: 统计指标计算（count/mean/std/percentiles/min/max）。
  - `DatasetVisualizer`: 输出核心分布图（至少能量、力、压力）。
  - `AnalysisWorkflow`: 基于配置选择 `model_test` 或 `dataset` 路径。
- **配置契约**:
  - 保持 `result_dir` 模式兼容。
  - 新增 dataset 模式输入字段，不破坏现有调用。

### 4.2 Data Integration MVP

- **目标**: 自动把 Labeling 有效产物整合为“下一代训练集”。
- **参考脚本**：`utils/dataset_tools/dpdata_addtrain.py`。需要确保对该脚本数据分析和可视化部分的全面覆盖和超越。  
- **建议路径**:
  - `src/dpeva/labeling/integration.py`（新增）
  - `src/dpeva/workflows/labeling.py`（在 `collect_and_export` 后接入）
  - `src/dpeva/config.py`（补充整合所需字段）
- **输入/输出定义**:
  - 输入：`New Labeled Data` + `Existing Training Data`（可选）
  - 输出：`Merged Training Data`（可直接给 train workflow 使用）
- **最小规则**:
  - 系统一致性检查（dataset/system 维度）
  - 可选去重开关（默认关闭，先确保正确性）
  - 整合日志与统计（输入量、输出量、过滤量）

### 4.3 文档边界治理（替代目录迁移）

- **目标**: 解决职责交叉和误读风险，而非重排目录。  
- **执行点**:
  - 在 `docs/README.md` 增加“现行文档与归档文档判别规则”。
  - 在 `docs/archive/README.md` 强化“只读历史，不作现行规范来源”说明。
  - 修复 `docs/policy/docs-structure.md` 与实际结构漂移项。
  - 统一 `governance` 与 `archive/governance` 的引用边界。
  - 明确 `spec` 的 active 落点策略并固化到 policy。

### 4.4 Analysis Skill 补全

- **路径**: `dpeva/.trae/skills/analysis.md`  
- **内容要求**:
  - `model_test` 与 `dataset` 双模式示例配置
  - 常见错误与排障提示
  - 与 `docs/guides/cli.md` 的参数命名一致性检查项

### 4.5 测试策略

- **新增/更新测试**
  - `tests/unit/analysis/*`：dataset 模式统计与可视化单测
  - `tests/unit/labeling/*`：integration 规则与异常路径
  - `tests/integration/test_e2e_cycle.py`：增量闭环（含 integration）
- **执行顺序**
  1. Unit/Contract（快速反馈）
  2. Integration（本地 backend）
  3. Slurm 条件 E2E（环境可用时）

---

## 5. 分阶段执行计划（可直接执行）

> 原则：每阶段都有“输入前置、执行任务、退出标准”；未达退出标准不得进入下一阶段。

### Phase 0：契约冻结与风险前置

- **输入前置**
  - 确认 `AnalysisConfig` 新旧字段兼容策略
  - 确认 `LabelingConfig` integration 所需最小字段
  - 确认文档治理边界策略（不做目录大迁移）
- **执行任务**
  1. 输出 Dataset Analysis 与 Integration 的配置契约草案
  2. 列出向后兼容矩阵（旧配置是否可直接运行）
  3. 固化文档边界策略（active/archive/spec 落点）
- **退出标准**
  - 配置契约评审通过
  - 风险清单具备 owner 与缓解动作
  - 后续阶段无阻断前提缺失

### Phase 1：功能 MVP 开发

- **输入前置**
  - Phase 0 退出标准全部满足
- **执行任务**
  1. 实现 dataset 分析主干（加载、统计、输出）
  2. 实现 integration 主干（读取、合并、导出）
  3. 接入 workflow（analysis/labeling）
  4. 保持旧功能行为不变
- **退出标准**
  - `analysis` 新旧两模式均可运行
  - `label` 结束后可产出 merged training data
  - 本地 smoke 测试通过

### Phase 2：文档与技能收敛

- **输入前置**
  - Phase 1 功能接口冻结
- **执行任务**
  1. 新增 `.trae/skills/analysis.md`
  2. 更新 `docs/guides/cli.md`、`developer-guide.md` 对应章节
  3. 修复 docs 入口、结构树、引用漂移
  4. 明确 archive 边界与 spec 落点说明
- **退出标准**
  - 文档与实现一致性检查通过
  - `AGENTS.md`、skills、CLI 指南术语一致
  - 不新增目录迁移债务

### Phase 3：测试闭环与发布验收

- **输入前置**
  - Phase 2 文档与接口一致
- **执行任务**
  1. 完成 unit/contract 测试并通过
  2. 补齐 `test_e2e_cycle.py` 最小闭环
  3. 运行回归测试并整理发布证据
- **退出标准**
  - 关键测试全通过
  - 发布验收清单完成
  - 满足 v0.6 发版门槛

---

## 6. 验收标准（Definition of Done）

### 6.1 功能验收

1. ✅ `dpeva analysis` 支持 `dataset` 模式并输出统计结果文件。  
   - 证据：`tests/unit/analysis/test_dataset_manager.py`，输出 `dataset_stats.json` 与 `dataset_frame_summary.csv`。  
2. ✅ `dpeva label` 可自动生成 `Merged Training Data`。  
   - 证据：`tests/unit/labeling/test_integration.py`，输出 `integration_summary.json` 与 merged 目录。  
3. ✅ 旧版 `analysis(result_dir)` 用法保持可用。  
   - 证据：`tests/unit/workflows/test_analysis_workflow.py::test_run_success`。  

### 6.2 文档验收

1. ✅ 存在 `analysis` skill 文档并可直接复用。  
2. ✅ `docs/README.md`、`policy/docs-structure.md`、`archive/README.md` 三者职责表述一致。  
3. ✅ 消除已识别的关键链接/结构漂移项。  

### 6.3 测试验收

1. ✅ 新增/更新的 unit 测试通过。  
2. ✅ 最小全链路 E2E（含 integration）通过。  
3. ✅ 回归测试通过，且无新增阻断级失败。  

---

## 7. 风险前置清单（Pre-Mortem）

| 风险ID | 风险描述 | 触发信号 | 前置缓解动作 | 责任角色 |
| :--- | :--- | :--- | :--- | :--- |
| R-01 | Analysis 新旧模式冲突 | 旧配置运行失败 | Phase 0 冻结兼容矩阵与默认策略 | Core Dev |
| R-02 | Integration 合并规则不清 | 输出数据不可复用 | 先做 MVP 最小规则，去重默认关闭 | Core Dev |
| R-03 | 文档边界继续漂移 | 同题多入口再次出现 | policy 明确“单一现行入口”并在 README 固化 | Doc Owner |
| R-04 | E2E 受外部环境阻塞 | Slurm/DeepMD 不可用 | 先完成 unit/contract，再条件执行 Slurm E2E | QA/Testing |
| R-05 | 技能文档与 CLI 不一致 | Agent 调用失败 | 增加发布前一致性检查项 | Doc Owner |

---

## 8. 里程碑与发布门槛

| 里程碑 | 达成条件 | 交付物 |
| :--- | :--- | :--- |
| M1（Contract Ready） | Phase 0 退出标准达成 | 配置契约与风险清单 |
| M2（Feature MVP Ready） | Phase 1 退出标准达成 | Dataset Analysis + Data Integration MVP |
| M3（Doc Aligned） | Phase 2 退出标准达成 | skill + docs 边界治理结果 |
| M4（Release Ready） | Phase 3 退出标准达成 | 测试证据与 v0.6 发布包 |

**发布门槛（必须全部满足）**:

1. 功能验收、文档验收、测试验收全部通过。  
2. 无阻断级已知缺陷。  
3. 发布说明覆盖新增能力、兼容性与已知限制。  

---

## 9. 分工与执行责任

- **Core Dev**: 负责 Phase 0/1 的契约与功能实现。  
- **Doc Owner**: 负责 Phase 2 的技能补齐与文档边界治理。  
- **QA/Testing**: 负责 Phase 3 的测试实施、回归与证据归档。  
- **Release Manager**: 负责发版门槛核查与发布流程收口。  

---

## 10. 交付清单

1. **代码**:  
   - `src/dpeva/analysis/`（dataset 模式支持）  
   - `src/dpeva/labeling/`（integration 支持）  
   - `tests/unit/` 与 `tests/integration/` 对应新增/更新测试  
2. **文档**:  
   - `.trae/skills/analysis.md`  
   - `docs/guides/*`、`docs/policy/*`、`docs/README.md`、`docs/archive/README.md` 的一致性更新  
3. **验收资产**:  
   - 配置契约说明  
   - 风险清单与关闭记录  
   - 测试结果与发布验收记录  

---

## 11. 可执行任务清单（Developer Checklist）

> 执行方式：按任务 ID 顺序推进。每完成一项，必须更新“状态”与“验收证据”。  
> 状态取值：`TODO` / `DOING` / `DONE` / `BLOCKED`。

### 11.1 Phase 0：契约冻结与风险前置

| ID | 任务 | 输入 | 输出 | 状态 | 验收证据 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| P0-C01 | 冻结 Analysis 双模式配置契约 | 现有 `AnalysisConfig` | 新旧字段兼容清单 | DONE | `/docs/plans/v0.6-phase0-contract-and-risk.md` 第2章 |
| P0-C02 | 冻结 Integration 最小配置契约 | 现有 `LabelingConfig` | Integration 字段清单 | DONE | `/docs/plans/v0.6-phase0-contract-and-risk.md` 第3章 |
| P0-C03 | 定义向后兼容矩阵 | 现有 recipes/tests | 兼容矩阵（旧配置可运行性） | DONE | `/docs/plans/v0.6-phase0-contract-and-risk.md` 第4章 |
| P0-C04 | 固化文档边界策略 | docs 现状与 policy | active/archive/spec 边界规则 | DONE | `/docs/plans/v0.6-phase0-contract-and-risk.md` 第5章 |
| P0-C05 | 生成风险与应对清单 | 风险表草案 | 风险 owner+缓解动作 | DONE | `/docs/plans/v0.6-phase0-contract-and-risk.md` 第6章 |

### 11.2 Phase 1：功能 MVP 开发

| ID | 任务 | 输入 | 输出 | 状态 | 验收证据 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| P1-A01 | 新增 dataset 加载组件 | deepmd/npy 数据样例 | DatasetLoader | DONE | `src/dpeva/analysis/dataset.py` |
| P1-A02 | 新增 dataset 统计组件 | DatasetLoader 输出 | DatasetStats | DONE | `src/dpeva/analysis/dataset.py` |
| P1-A03 | 新增 dataset 可视化组件 | DatasetStats 输出 | DatasetVisualizer | DONE | `src/dpeva/analysis/dataset.py` |
| P1-A04 | 扩展 AnalysisConfig 模式字段 | 现有 config 结构 | 兼容的新配置模型 | DONE | `src/dpeva/config.py` |
| P1-A05 | 改造 AnalysisWorkflow 分流 | model_test 逻辑 | model_test/dataset 双分流 | DONE | `src/dpeva/workflows/analysis.py` |
| P1-I01 | 新增 Integration 模块骨架 | labeling 输出路径 | integration.py | DONE | `src/dpeva/labeling/integration.py` |
| P1-I02 | 实现合并规则与一致性校验 | 新旧训练集输入 | merged dataset 输出 | DONE | `tests/unit/labeling/test_integration.py` |
| P1-I03 | 在 LabelingWorkflow 接入 Integration | `collect_and_export` 后置点 | 自动整合流程 | DONE | `src/dpeva/workflows/labeling.py` |
| P1-I04 | 增加 Integration 运行统计 | 合并过程数据 | 输入/输出/过滤统计 | DONE | `integration_summary.json` + `DataIntegrationManager` 日志 |

### 11.3 Phase 2：文档与技能收敛

| ID | 任务 | 输入 | 输出 | 状态 | 验收证据 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| P2-D01 | 新增 analysis skill 文档 | Analysis 双模式配置 | `.trae/skills/analysis.md` | DONE | `/.trae/skills/analysis.md` |
| P2-D02 | 同步 CLI 指南参数说明 | 新增配置字段 | `docs/guides/cli.md` 更新 | DONE | `/docs/guides/cli.md` analysis/label 章节更新 |
| P2-D03 | 同步开发者指南流程 | workflow 变更 | `developer-guide.md` 更新 | DONE | `/docs/guides/developer-guide.md` CLI 章节更新 |
| P2-D04 | 修复 docs 索引与引用漂移 | `docs/README` 与 policy | 一致化索引 | DONE | `/docs/README.md`、`/docs/architecture/README.md` 更新 |
| P2-D05 | 固化 archive 与 governance 边界 | 边界策略 | README/policy 对齐 | DONE | `/docs/README.md`、`/docs/archive/README.md` 更新 |
| P2-D06 | 明确 spec active 落点策略 | `plans/specs/archive` 现状 | 规范声明更新 | DONE | `/docs/policy/docs-structure.md` 新增策略条目 |

### 11.4 Phase 3：测试闭环与发布验收

| ID | 任务 | 输入 | 输出 | 状态 | 验收证据 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| P3-T01 | 补齐 analysis dataset 单测 | Phase 1 代码 | `tests/unit/analysis/*` 更新 | DONE | `pytest tests/unit/workflows/test_analysis_workflow.py -q` |
| P3-T02 | 补齐 integration 单测 | Phase 1 代码 | `tests/unit/labeling/*` 更新 | DONE | `pytest tests/unit/labeling/test_integration.py -q` |
| P3-T03 | 新增最小闭环 E2E | workflows 可运行 | `test_e2e_cycle.py` | DONE | `pytest tests/integration/test_e2e_cycle.py -q` |
| P3-T04 | 执行回归测试集 | unit+integration 用例 | 回归通过结论 | DONE | `pytest tests/unit/workflows/test_analysis_workflow.py tests/unit/workflows/test_labeling_workflow.py tests/unit/labeling/test_integration.py -q` |
| P3-T05 | 汇总发布验收资产 | 测试与文档变更 | 发布验收包 | DONE | `/docs/plans/v0.6-acceptance-record.md` + `/docs/plans/release_note_v0.6.0.md` |

---

## 12. 开发者执行与验收机制

### 12.1 日常执行规则

1. 任一时刻仅允许一个任务处于 `DOING`。  
2. 任务状态从 `TODO -> DOING -> DONE` 顺序流转，若受阻改为 `BLOCKED`。  
3. 每个 `DONE` 任务必须附带“验收证据”与“影响范围”。  
4. 若任务实现引入配置变更，必须同步更新对应文档与样例配置。  
5. 若任务失败回滚，必须在同 ID 下记录失败原因与补救动作。  

### 12.2 任务完成定义（Task DoD）

- 代码已实现并通过对应测试。  
- 文档已同步，且术语与参数命名一致。  
- 对上游/下游影响已记录。  
- 无阻断级遗留问题。  

### 12.3 执行记录模板

| 字段 | 填写要求 |
| :--- | :--- |
| 任务ID | 对应 11 章任务 ID |
| 开始时间 | 实际开始执行时间 |
| 完成时间 | 实际完成时间 |
| 执行人 | 责任开发者 |
| 变更文件 | 关键代码/文档路径 |
| 验收证据 | 测试输出、日志、产物路径 |
| 风险与备注 | 遇到的问题与处理方式 |

### 12.4 发布前总验收清单

- [x] 所有 P0/P1 任务状态为 `DONE`。  
- [x] `dpeva analysis` 双模式可用并通过验证。  
- [x] `dpeva label` 自动整合流程可用并通过验证。  
- [x] `analysis` skill 可直接使用。  
- [x] 文档边界治理项全部落地。  
- [x] Unit/Integration/E2E 结果满足发布门槛。  
- [x] 发布验收资产归档完整。  
