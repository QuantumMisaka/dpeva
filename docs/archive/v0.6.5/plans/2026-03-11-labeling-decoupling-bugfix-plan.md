# 2026-03-11 Labeling 解耦与 atom_names 兼容性修复计划

## 背景与问题确认

- 现场报错来自 `dpeva label` 的数据集成阶段：`Incompatible atom_names at new[0]: ['C', 'Fe', 'H', 'O'] != ['H', 'C', 'O', 'Fe']`。
- 调用链已定位为：`cli.handle_label -> LabelingWorkflow.run -> DataIntegrationManager.integrate -> _validate_system_compatibility`。
- 当前实现把 `atom_names` 当作有序列表做严格相等比较；当训练集与新标注集仅元素顺序不同（元素集合一致）时，会被误判为不兼容并中断流程。
- 本场景属于“顺序不一致”而非“元素种类不一致”，应支持兼容处理并继续流程。

## 总体目标

1. 修复 `atom_names` 顺序敏感误报，保证兼容性校验符合项目数据实际。
2. 将 labeling workflow 解耦为三段可单独调用能力：前处理、计算、后处理，并挂载在 `dpeva label` 命令下。
3. 支持在不重新执行 ABACUS 的前提下，单独运行后处理并完成导出与 integration。
4. 以 `test/fp5-test/config_gpu.json` 为样例验证前处理阶段可独立执行。
5. 为三个解耦阶段补齐单元测试并完成回归验证。

## 设计原则

- 显式优于隐式：阶段入口与行为边界清晰。
- 可读性优先：阶段名称、参数、错误提示可直接理解。
- 实用优先：在元素集合一致时允许顺序归一化，避免误报。
- 错误不可静默：真正不兼容（集合差异、关键字段缺失）必须显式失败。

## 实施步骤

### 步骤 1：修复 atom_names 兼容性与顺序归一化

**改动文件：**
- `src/dpeva/labeling/integration.py`

**改动内容：**
- 将兼容性逻辑拆分为“元素集合校验”与“顺序归一化处理”两层。
- 对“集合一致、顺序不同”的系统在合并前进行原子类型映射重排，使其与 reference 顺序一致。
- 对“集合不一致”继续抛出 `ValueError`，并提供明确差异信息。

**验证要点：**
- 顺序不同但集合一致：应成功合并。
- 元素集合不同：应失败并报错。
- 空 `atom_names`：应失败并报错。

### 步骤 2：将 LabelingWorkflow 解耦为三阶段接口

**改动文件：**
- `src/dpeva/workflows/labeling.py`

**改动内容：**
- 增加三段显式方法：
  - `run_prepare()`：加载数据 + `prepare_tasks`
  - `run_execute()`：提交计算 + 监控 + 重试 + 收敛检查
  - `run_postprocess()`：`collect_and_export` + 可选 integration
- 保留 `run()` 作为编排器，串联三段方法，确保默认全流程行为不变。
- 每个阶段增加前置条件检查，缺失时给出可操作错误信息。

### 步骤 3：扩展 `dpeva label` 命令支持阶段执行

**改动文件：**
- `src/dpeva/cli.py`

**改动内容：**
- 为 `label` 子命令添加阶段参数（`--stage`）：
  - `all`（默认）
  - `prepare`
  - `execute`
  - `postprocess`
- 在 `handle_label` 中根据阶段分发到对应 workflow 方法。
- 默认行为保持兼容：`dpeva label <config>` 仍执行全流程。

### 步骤 4：按用户场景验证解耦能力

**样例配置：**
- `test/fp5-test/config_gpu.json`

**验证动作：**
- 执行 `label prepare`，确认仅生成前处理产物（输入文件与任务打包），不触发 ABACUS 计算。
- 在已有结果目录上执行 `label postprocess`，确认完整后处理可运行，且不再出现 atom_names 顺序误报。

### 步骤 5：为三阶段补充单元测试

**计划测试文件：**
- `tests/unit/workflows/test_labeling_workflow.py`
  - 覆盖 `run_prepare`、`run_execute`、`run_postprocess` 三段调用链与关键分支。
- `tests/unit/test_cli.py`
  - 覆盖 `label --stage` 四种分发行为（prepare/execute/postprocess/all）。
- `tests/unit/labeling/test_integration.py`
  - 增补 atom_names 顺序兼容回归测试与集合不一致失败测试。

### 步骤 6：回归验证与质量门禁

**验证顺序：**
1. 目标新增测试文件与相关模块测试。
2. `pytest tests/unit`
3. 视影响范围补跑 `pytest tests/integration` 的 labeling 相关测试。
4. `ruff check .` 与必要时 `ruff format .`

**通过标准：**
- bug 场景在修复后不再报错。
- 三阶段可独立调用且默认全流程不回归。
- 单元测试通过并覆盖核心新逻辑。

## 风险与缓解

- 原子顺序重排若实现不完整，可能导致类型映射错位。  
  缓解：对 `atom_names/type_map/atom_types` 联动建立回归测试。
- 阶段化执行存在状态依赖断裂风险。  
  缓解：阶段入口统一做前置条件校验并提示下一步操作。
- CLI 变更影响历史脚本调用。  
  缓解：默认 `--stage all`，不改变原调用方式。

## 预期交付

1. `atom_names` 顺序兼容修复及对应回归测试。
2. labeling 三阶段 workflow 接口与 CLI 阶段调用能力。
3. 基于 `fp5-test/config_gpu.json` 的前处理阶段可运行验证结果。
4. 三阶段单元测试与整体回归验证报告。
