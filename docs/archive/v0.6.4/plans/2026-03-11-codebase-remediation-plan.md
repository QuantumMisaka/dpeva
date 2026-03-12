---
title: Codebase Remediation Plan
status: archived
audience: Developers
last-updated: 2026-03-11
owner: Engineering
---

# 2026-03-11 全量代码修复方案（可回溯/可验收）

## 1. 目标与原则

- 目标：基于 [2026-03-10-combined-review](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/docs/archive/v0.6.4/reports/2026-03-10-combined-review.md#L55-L164) 的全部问题，建立并执行可追踪、可验收、可回溯的修复闭环。
- 原则：先 P0 再 P1/P2；每项修复后执行单元、集成、回归测试；全部修复必须有“变更记录 + 测试证据 + 风险记录”。
- 完成定义：问题清单全部进入已关闭状态；测试门禁通过；修复总结报告完成。

## 2. 问题基线（全量列表）

来源：综合审查报告问题 1~37。

### P0（必须先完成）

- R01: `labeling/manager.py` shell 执行存在注入面
- R02: `labeling/manager.py` 裸 `except` 吞错
- R12: `labeling/postprocess.py` 覆盖率 14%
- R13: `labeling/strategy.py` 覆盖率 12%
- R26: `examples/README.md` 命令指向不存在配置
- R27: `examples/recipes/README.md` 命名漂移
- R28: `analysis_recipe.py` 默认配置文件不存在
- R29: `collect_recipe.py` 默认配置文件不存在

### P1（高优先级）

- R03 R04 R05 R06: 超长函数与职责耦合
- R09 R10: 日志读取与输出流实现问题
- R11: `cli.py` 覆盖率 0%
- R14: `uncertain/visualization.py` 覆盖不足
- R16 R18 R19 R21: 测试稳定性/可复现性缺陷
- R22 R23 R24 R25 R30: 文档一致性与版本漂移
- R31 R33: `.gitignore` 与 docs 检查脚本问题
- R37: `tools/dpdata_addtrain.py` 顶层执行副作用

### P2（中优先级）

- R07 R08: 重复逻辑抽取
- R17 R20: 测试结构优化
- R32 R34 R35 R36: 指引文案与打包策略一致性优化

## 3. 可追溯映射（Issue → 修复任务 → 验收证据）

| 问题ID | 修复任务ID | 目标文件 | 验收证据 |
|---|---|---|---|
| R01-R02 | T04 | `src/dpeva/labeling/manager.py` | 代码 diff + 安全回归测试 |
| R03-R08 | T04/T05 | `analysis/managers.py` `inference/managers.py` `workflows/*` | 复杂度下降 + 现有测试通过 |
| R09-R10 | T04 | `labeling/postprocess.py` `utils/logs.py` | 大文件处理/日志刷新测试 |
| R11-R14 | T05 | `tests/unit/*` 新增覆盖 | 覆盖率报表 |
| R16-R21 | T05 | `pytest.ini` `tests/integration/*` | 稳定性回归记录 |
| R22-R30 | T02 | `docs/guides/*` `examples/*` | 文档构建与示例命令验证 |
| R31-R37 | T03 | 根目录配置与脚本 | lint/audit/打包与脚本运行记录 |

## 4. 任务分解与执行步骤

### T01 方案落地

- 编写本方案文档，冻结问题基线（R01-R37）。
- 建立 Issue ID 与代码变更、测试报告之间的一对一映射。

### T02 文档与示例修复

- 修复失效链接、命名漂移、版本信息漂移、示例默认配置错误。
- 验收：`make -C docs html` 成功；示例入口命令可定位到真实文件。

### T03 工程资产与脚本修复

- 修复 `.gitignore`、`AGENTS.md`、`scripts/check_docs.py`、`scripts/gate.sh`、`scripts/audit.py`、`tools/dpdata_addtrain.py` 等一致性问题。
- 验收：脚本帮助与路径一致；审计脚本与 docs 检查脚本通过。

### T04 高优先级代码修复

- 安全：去除 `shell=True` 拼接风险，严格参数化执行。
- 可靠性：消除裸 `except` 吞错，改为显式异常处理与可追踪日志。
- 稳定性：优化大日志读取、`flush` 行为。
- 验收：新增针对性单测 + 单元/集成测试通过。

### T05 测试与门禁升级

- 增补 `cli.py`、`labeling/postprocess.py`、`labeling/strategy.py`、`uncertain/visualization.py` 的关键路径测试。
- 补充 `pytest.ini` 基础策略，建立覆盖率回归目标。
- 验收：测试全绿，覆盖率较基线显著提升并给出差异报告。

### T06 逐项验证与回溯

- 每完成一项执行：单元测试 → 集成测试 → 回归测试（docs 构建 + audit）。
- 形成“任务-测试-风险”三联记录。

### T07 总结收口

- 输出修复总结报告，逐项对照 R01-R37 给出关闭状态、证据链接、残余风险与后续优化建议。

## 5. 时间节点（按日历里程碑）

- M1（2026-03-11）：完成 T01、T02、T03。
- M2（2026-03-12）：完成 T04（安全/异常/日志）。
- M3（2026-03-13）：完成 T05（覆盖率关键补测）。
- M4（2026-03-13）：完成 T06、T07，并发布总结报告。

## 6. 验收标准

- 完整性：R01-R37 均有明确状态（Closed/Deferred），并提供理由与证据。
- 质量性：每项修复后必须附带测试结果；禁止仅代码变更无验证。
- 可追溯：所有变更可回溯到 Issue ID 与任务 ID。
- 可复现：测试命令、输出摘要、风险项登记齐全。

## 7. 测试要求（每项修复后执行）

- 单元测试：`pytest tests/unit -q`
- 集成测试：`pytest tests/integration -q`
- 回归测试：
  - `pytest tests/unit tests/integration --cov=src/dpeva --cov-branch --cov-report=term`
  - `python scripts/audit.py`
  - `make -C docs html`

## 8. 风险与应对

- 风险1：高耦合函数改动引发行为回归。
  - 应对：先补测试再改逻辑；分步提交，控制变更半径。
- 风险2：外部依赖（dp/abacus/slurm）导致集成波动。
  - 应对：保留可跳过逻辑并补充本地可复现实验路径。
- 风险3：文档与代码更新节奏不一致。
  - 应对：强制“同一变更包含文档更新 + docs 构建验证”。

## 9. 开发进展更新（2026-03-11 第二轮）

- 已完成：`labeling/manager.py` 的统计导出流程拆分（数据收集、过滤导出、异常导出、统计聚合、报告输出解耦）。
- 已完成：`workflows/collect.py` 的主流程拆分（UQ阶段、采样阶段、导出阶段方法化）。
- 已完成：新增回归测试
  - `tests/unit/workflows/test_collect_refactor.py`
  - `tests/unit/labeling/test_manager.py` 新增聚合统计断言
- 已验证：
  - `pytest tests/unit -q` 通过（178 passed）
  - `pytest tests/integration -q` 通过（7 passed, 1 skipped）
  - `pytest tests/unit tests/integration --cov=src/dpeva --cov-branch --cov-report=term` 通过（185 passed, 1 skipped）
  - `python scripts/check_docs.py`、`python scripts/audit.py`、`make -C docs html` 均通过

## 10. 开发进展更新（2026-03-11 第三轮）

- 已完成：`labeling/strategy.py` 默认参数鲁棒性修复（`attempt_params=None` 时安全回退空列表）。
- 已完成：新增 `tests/unit/labeling/test_strategy.py`，覆盖参数获取、缺失输入、参数追加与无效重试分支。
- 已完成：增强 `tests/unit/uncertain/test_visualization.py`，补齐 identity 缺失分支、截断分支、候选误差图与 PCA 汇总图分支。
- 已完成：`collect.py` 提炼 `_extract_unique_system_names` 以降低 UQ 阶段复杂度。
- 已验证：
  - `pytest tests/unit -q` 通过（187 passed）
  - `pytest tests/integration -q` 通过（7 passed, 1 skipped）
  - `pytest tests/unit tests/integration --cov=src/dpeva --cov-branch --cov-report=term` 通过（194 passed, 1 skipped，TOTAL 76%）
  - `python scripts/check_docs.py`、`python scripts/audit.py`、`make -C docs html` 均通过

## 11. 开发进展更新（2026-03-11 第四轮）

- 已完成：`workflows/analysis.py` 再解耦，拆分 `run` 为 `_run_dataset_mode`、`_run_model_mode`、`_resolve_composition_info`。
- 已完成：异常处理语义修正，由 `raise e` 改为 `raise`，保留原始 traceback。
- 已完成：增强 `tests/unit/workflows/test_analysis_workflow.py`，补齐：
  - data_path 组成信息加载分支断言
  - metrics 为空时跳过保存分支断言
- 已验证：
  - `pytest tests/unit -q` 通过（189 passed）
  - `pytest tests/integration -q` 通过（7 passed, 1 skipped）
  - `pytest tests/unit tests/integration --cov=src/dpeva --cov-branch --cov-report=term` 通过（196 passed, 1 skipped，TOTAL 76%）

## 12. 完成确认（归档结论）

- R01-R37 已全部关闭，计划目标全部达成。
- 本计划已结束执行并转入 v0.6.4 归档，只读保存。
