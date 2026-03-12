---
title: Document
status: active
audience: Developers
last-updated: 2026-03-12
---

# 2026-03-12 Labeling 解耦迭代总结

- Status: active
- Audience: Developers
- Last-Updated: 2026-03-12
- Related:
  - 计划文档：`/.trae/documents/2026-03-11-labeling-decoupling-bugfix-plan.md`

## 1. 任务完成确认

计划文档中的目标均已完成并通过验证：

1. 已修复 `atom_names` 顺序敏感误报，支持“元素集合一致、顺序不同”场景的兼容合并。
2. 已将 Labeling workflow 解耦为 `prepare`、`execute`、`postprocess` 三阶段独立入口。
3. 已扩展 `dpeva label --stage` 阶段化调用能力，默认 `all` 保持兼容。
4. 已基于 `test/fp5-test/config_gpu.json` 验证前处理独立执行与后处理复用执行。
5. 已补齐单元测试并完成回归验证。

## 2. 本轮关键迭代内容

### 2.1 集成兼容性修复

- 在 `DataIntegrationManager` 中将兼容性逻辑升级为“集合校验 + 顺序归一化”。
- 当 `atom_names` 集合一致但顺序不一致时，自动重排：
  - `atom_names`
  - `type_map`
  - `atom_numbs`
  - `atom_types`
- 当元素集合不一致或类型索引异常时，保持显式失败。

### 2.2 Labeling 三阶段解耦

- 在 `LabelingWorkflow` 中新增并稳定化：
  - `run_prepare()`
  - `run_execute()`
  - `run_postprocess()`
- `run()` 作为兼容编排入口，保持旧调用行为不变。
- 阶段前置条件不足时提供清晰错误提示。

### 2.3 CLI 阶段化增强

- `dpeva label` 新增参数：
  - `--stage all`
  - `--stage prepare`
  - `--stage execute`
  - `--stage postprocess`
- 命令分发逻辑已在单元测试覆盖下通过验证。

### 2.4 Prepare 幂等增强

- `prepare_tasks` 在每次执行前重置 `work_dir/inputs`，消除历史残留任务目录引发的重复打包冲突。
- 幂等增强仅作用于 `inputs` 目录，不影响 `outputs` 与 `CONVERGED` 核心数据目录。

### 2.5 三阶段独立日志

- 已支持将三阶段执行日志分别落盘为：
  - `labeling_prepare.log`
  - `labeling_execute.log`
  - `labeling_postprocess.log`
- 阶段结束后恢复 stdout/stderr，避免跨阶段日志串扰。

## 3. 验证结果

- 目标模块单测通过：
  - `tests/unit/labeling/test_integration.py`
  - `tests/unit/workflows/test_labeling_workflow.py`
  - `tests/unit/test_cli.py`
  - `tests/unit/labeling/test_manager.py`
- 全量单测通过：`pytest tests/unit`（202 passed）。
- 场景验证通过：
  - `fp5-test` 下连续执行两次 `label --stage prepare` 无冲突报错。
  - `label --stage postprocess` 可复用已有计算结果并完成 integration，不再出现 atom 顺序误报。

## 4. 影响评估

- 兼容性：保留默认 `dpeva label <config>` 全流程行为，不破坏历史用法。
- 可维护性：阶段职责边界更清晰，测试与排障颗粒度更细。
- 可运维性：阶段独立日志使问题定位路径更加直接，减少混合日志排查成本。
