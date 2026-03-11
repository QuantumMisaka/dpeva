---
title: Remediation Summary Report
status: active
audience: Developers
last-updated: 2026-03-11
---

# 2026-03-11 修复执行总结报告

## 1. 执行基线

- 执行方案：`docs/plans/governance/2026-03-11-codebase-remediation-plan.md`
- 问题基线：R01-R37（来源于 `2026-03-10-combined-review.md`）
- 执行方式：按优先级分批修复 + 每批次执行单元/集成/回归验证

## 2. 变更记录（按任务）

### T02 文档与示例修复

- 修复配置权威入口漂移：多处由 `config_schema.md` 切换至 `docs/source/api/config.rst`
- 修复推理日志命名漂移：`test_job.log` → `test_job.out`
- 补齐 quickstart `label` 子命令最小路径
- 修复 examples 命名漂移与默认配置失效：
  - `config_normal.json` → `config_collect_normal.json`
  - `config_joint.json` → `config_collect_joint.json`
  - `labeling_recipe.json` → `labeling/config_cpu.json`
  - `analysis_recipe.py` 默认配置改为 `config_analysis.json`
  - `collect_recipe.py` 默认配置改为 `config_collect_normal.json`

### T03 脚本与工程配置修复

- `scripts/check_docs.py`：补齐 `label` 子命令检查
- `scripts/gate.sh` / `scripts/audit.py`：修正审计脚本路径文案
- `.gitignore`：补齐常见构建/缓存/虚拟环境忽略项，移除模糊规则
- `pytest.ini`：增加 `testpaths` / `addopts` / `filterwarnings`
- `tools/dpdata_addtrain.py`：改为 `main()` + 入口保护
- `docs/source/conf.py`：版本号同步至 `0.6.3`
- `AGENTS.md`：修正失效测试文件路径与示例命令

### T04 安全与异常处理修复

- `src/dpeva/labeling/manager.py`：
  - 生成脚本中 ABACUS 执行改为参数列表调用，移除 `shell=True`
  - 裸 `except` 改为显式异常类型与告警日志
- `src/dpeva/labeling/postprocess.py`：
  - 收敛日志由整文件读取改为逐行扫描
- `src/dpeva/utils/logs.py`：
  - `StreamToLogger.flush` 改为可实际落盘剩余缓冲

### T05 补测与门禁增强

- 新增 `tests/unit/test_cli.py`（CLI 分发与失败退出）
- 新增 `tests/unit/labeling/test_postprocess.py`（收敛判断）
- 新增 `tests/unit/utils/test_logs.py`（stdout/stderr 转发缓冲）
- 扩展 `tests/unit/labeling/test_manager.py`（runner 脚本安全性与坏元数据容错）

### T04/T05 第二轮（结构解耦与回归）

- `src/dpeva/labeling/manager.py`：
  - 将 `collect_and_export` 拆分为多段职责方法（收集、指标构建、导出、异常落盘、失败扫描、统计聚合、统计输出）
- `src/dpeva/workflows/collect.py`：
  - 将 `run` 拆分为 `_run_uq_phase`、`_run_sampling_phase`、`_run_export_phase`
  - 将 UQ 分支拆分为 `_run_no_filter_uq_phase`、`_run_filtered_uq_phase`
- 新增/增强测试：
  - 新增 `tests/unit/workflows/test_collect_refactor.py`
  - 扩展 `tests/unit/labeling/test_manager.py` 的统计聚合断言

### T05 第三轮（覆盖率专项补强）

- `src/dpeva/labeling/strategy.py`：
  - `attempt_params` 初始化增加空值安全回退，消除 `None` 输入下潜在异常路径
- 新增 `tests/unit/labeling/test_strategy.py`：
  - 覆盖参数获取边界、缺失 INPUT、参数更新/追加、无效 attempt 分支
- 增强 `tests/unit/uncertain/test_visualization.py`：
  - 补齐 `plot_uq_identity_scatter` 的缺列分支与截断分支
  - 补齐 `plot_candidate_vs_error` 与 `plot_pca_analysis` 分支
- `src/dpeva/workflows/collect.py`：
  - 提炼 `_extract_unique_system_names`，降低 `_run_filtered_uq_phase` 圈复杂度

### T04/T05 第四轮（Analysis 工作流分层与断言补强）

- `src/dpeva/workflows/analysis.py`：
  - 拆分 `run` 主流程为 `_run_dataset_mode`、`_run_model_mode`、`_resolve_composition_info`
  - 异常重抛由 `raise e` 改为 `raise`，保留原始栈信息
- `tests/unit/workflows/test_analysis_workflow.py`：
  - 新增 data_path 分支断言，验证 composition 信息走 `load_composition_info`
  - 新增空 metrics 分支断言，验证保存函数跳过行为

## 3. 提交记录（可追溯分组）

- 已完成提交：
  - `9bc619a` `fix: close remediation gaps and add regression coverage`
- 当前第二轮解耦与补测变更仍在工作区，待下一次提交。

## 4. 测试报告

### 4.1 单项修复验证

- `pytest tests/unit/test_cli.py tests/unit/labeling/test_manager.py -q` 通过
- `pytest tests/unit/utils/test_logs.py tests/unit/labeling/test_postprocess.py -q` 通过
- `python scripts/check_docs.py` 通过

### 4.2 全量回归验证

- `pytest tests/unit -q`：`189 passed`
- `pytest tests/integration -q`：`7 passed, 1 skipped`
- `pytest tests/unit tests/integration --cov=src/dpeva --cov-branch --cov-report=term`：`196 passed, 1 skipped`
- 覆盖率：总覆盖 `76%`（`labeling/strategy.py` 12% → 88%，`uncertain/visualization.py` 48% → 87%，`collect.py` 79%）
- `workflows/analysis.py` 覆盖率：`97%`
- `python scripts/audit.py`：通过
- `python scripts/check_docs.py`：通过
- `make -C docs html`：通过（无 warning）

## 5. 问题闭环状态（R01-R37）

| ID | 状态 | 说明 |
|---|---|---|
| R01 | Closed | `shell=True` 已移除，改为参数列表执行 |
| R02 | Closed | 裸 `except` 改为显式异常处理 |
| R03 | Deferred | 异常治理需与统计语义统一重构，保留后续专项 |
| R04 | Partial | `collect_and_export` 已拆分，仍需继续向 strategy/visualization 扩展 |
| R05 | Partial | Collect workflow 主流程已方法化，其他 workflow 待继续分层 |
| R06 | Closed | Collect 复杂主流程已完成阶段性解耦并通过回归测试 |
| R07 | Deferred | 推理/分析重复逻辑抽取未在本轮完成 |
| R08 | Deferred | 同 R07 |
| R09 | Closed | 日志读取改为流式逐行扫描 |
| R10 | Closed | `flush` 实现补齐并补测 |
| R11 | Partial | 新增 CLI 测试，覆盖率 0% → 58% |
| R12 | Partial | 补充后处理测试，覆盖率 14% → 23% |
| R13 | Closed | `strategy` 模块关键分支补测完成，覆盖率提升至 88% |
| R14 | Closed | `visualization` 模块高风险分支补测完成，覆盖率提升至 87% |
| R15 | Deferred | 导入期环境检查副作用未改动 |
| R16 | Closed | `pytest.ini` 基础策略补齐 |
| R17 | Deferred | Slurm skip 替代路径未补齐 |
| R18 | Deferred | slurm flaky 专项治理未完成 |
| R19 | Deferred | 集成硬编码路径治理未完成 |
| R20 | Deferred | 重复用例归并未完成 |
| R21 | Closed | analysis workflow 关键行为断言已补齐并通过回归 |
| R22 | Closed | active 文档中废弃配置入口已统一替换 |
| R23 | Closed | `docs/source/conf.py` 版本同步 |
| R24 | Closed | quickstart 补齐 `label` |
| R25 | Closed | reports 索引断链已修复 |
| R26 | Closed | examples README 失效命令已修复 |
| R27 | Closed | recipes README 命名漂移已修复 |
| R28 | Closed | `analysis_recipe.py` 默认配置修复 |
| R29 | Closed | `collect_recipe.py` 默认配置修复 |
| R30 | Closed | 文档中 infer 日志名称与代码一致 |
| R31 | Closed | `.gitignore` 规范化 |
| R32 | Closed | AGENTS 失效单测路径修复 |
| R33 | Closed | docs 检查脚本补齐 `label` |
| R34 | Closed | gate/audit 文案路径一致化 |
| R35 | Closed | `pyproject` 去除混用构建依赖 |
| R36 | Deferred | MANIFEST 分发策略细化未完成 |
| R37 | Closed | `dpdata_addtrain.py` 增加主入口保护 |

## 6. 风险点

- 风险1：R03-R08 的架构重构仍在待办，短期内维护复杂度仍高。
- 风险2：R13-R14 测试缺口尚未补齐，关键统计/可视化分支仍有回归风险。
- 风险3：R17-R19 的 Slurm 真实环境稳定性问题仍依赖后续专项环境验证。

## 7. 后续优化建议

- 建立“P0/P1 修复完成”分支策略：先合并已关闭项，再启动架构与覆盖率专项。
- 为 R03-R08 单独立项“复杂函数解耦”，按模块拆分 PR，避免一次性大改。
- 为 R13-R14 增加参数化与快照测试，形成可视化与策略模块稳定回归集。
- 将 `pytest --cov --cov-branch` 与 `scripts/check_docs.py` 纳入默认质量门禁。
