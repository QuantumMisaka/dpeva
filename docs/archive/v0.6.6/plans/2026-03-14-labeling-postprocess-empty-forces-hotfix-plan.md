# Labeling Postprocess 空 forces 紧急修复与长期治理计划

## Summary
- 目标：先完成 `fp7` 工作目录的短期止血（提取坏样本并重跑 postprocess 产出可用数据），再落地长期代码修复，避免同类 `IndexError: index 0 is out of bounds for axis 0 with size 0` 再次发生。
- 成功标准：
  - `test/fp7/nohup.out` 不再出现 `postprocess.py:forces = s["forces"][fi]` 越界错误。
  - `labeling_workdir/outputs/cleaned` 正常产出，且 `labeling_postprocess.log` 有统计结果。
  - 新增回归测试可覆盖“电子收敛但力数据缺失”场景。

## Current State Analysis
- 现有报错调用链已确认：
  - `LabelingWorkflow.run_postprocess -> LabelingManager.collect_and_export -> _build_metrics_data -> AbacusPostProcessor.compute_metrics`
  - 崩溃点在 `compute_metrics` 对 `s["forces"][fi]` 的索引访问。
- 数据根因已确认（来自 `test/fp7/labeling_workdir/CONVERGED`）：
  - 异常目录：`DeepCNT/cluster/C108Fe55_0`、`DeepCNT/cluster/C114Fe55_0`
  - 两者都出现“charge density convergence is achieved”，但 `running_scf.log` 缺失 `TOTAL-FORCE` 块。
  - `dpdata` 解析表现为 `nframes=1` 且 `forces` 长度为 0，最终触发越界。
- 机制缺口：
  - `check_convergence` 当前只检查“电子收敛字符串”，未校验力输出完整性。
  - `compute_metrics` 对 `energies/forces/virials/cells` 完整性缺少防御式检查。
  - 对应单元测试尚未覆盖空 `forces`/缺失输出块的回归场景。

## Assumptions & Decisions
- 假设 A：当前 `fp7` 任务的主要目标是尽快恢复 postprocess 数据产出，短期可接受“先跳过坏样本”。
- 假设 B：长期修复不改变配置结构与用户接口，不新增额外配置项。
- 决策 1：短期止血优先数据恢复，采用“坏样本隔离 + 仅重跑 postprocess”策略，避免重算 DFT。
- 决策 2：长期修复采用“双保险”：
  - 上游：收敛判定加入关键输出块校验；
  - 下游：metrics 计算加入数组完整性校验并跳过坏样本。
- 决策 3：遵循现有项目风格，保持显式日志与最小侵入修改，不引入额外复杂抽象。

## Proposed Changes

### 阶段一：短期止血（先执行）

#### 1) 提取坏样本并隔离
- 目标目录：`test/fp7/labeling_workdir/CONVERGED`
- 执行策略：
  - 扫描所有 converged 任务目录，识别条件：
    - `dpdata.LabeledSystem(..., fmt="abacus/scf")` 可加载但 `get_nframes()>0` 且 `forces` 长度为 0；
    - 或 `running_scf.log` 缺失 `TOTAL-FORCE`。
  - 将坏样本从 `CONVERGED` 移至 `labeling_workdir/BROKEN_FORCE/<dataset>/<type>/<task>`。
  - 生成清单文件：`labeling_workdir/BROKEN_FORCE/manifest.csv`（包含原路径、判定原因、时间戳）。
- 设计理由：最小成本恢复 postprocess，且保留异常样本供后续追踪。

#### 2) 仅重跑 postprocess
- 使用原配置对 `fp7` 目录执行 `label --stage postprocess`，不触发 prepare/execute。
- 验证输出：
  - `labeling_postprocess.log` 无崩溃堆栈；
  - `outputs/cleaned` 成功产出；
  - 统计日志中 converged/cleaned 计数合理，且坏样本不再参与计算。

### 阶段二：长期修复（代码）

#### 3) 强化收敛判定的“可用性门控”
- 文件：[postprocess.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/src/dpeva/labeling/postprocess.py)
- 修改内容：
  - 增加私有方法：检查 `running_scf.log` 是否包含 `TOTAL-FORCE`（必要条件）。
  - `check_convergence` 在“电子收敛”基础上叠加“力块存在”校验；不满足则返回 `False` 并记录 warning。
- 设计理由：从源头减少“伪收敛任务”进入 `CONVERGED`。

#### 4) 强化 load/metrics 防御式校验
- 文件：[postprocess.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/src/dpeva/labeling/postprocess.py)
- 修改内容：
  - 在 `load_data` 后校验必需数组（`energies/forces/virials/cells`）存在且长度与 `nframes` 一致。
  - 校验失败返回 `None`，并输出包含任务路径与原因的 warning。
  - `compute_metrics` 增加逐帧保护：单帧异常跳过并统计；若最终无有效帧，返回结构化空 DataFrame（避免后续 KeyError）。
- 设计理由：错误显式处理，不让单个坏样本拖垮整批流程。

#### 5) 管理层统计可观测性补强
- 文件：[manager.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/src/dpeva/labeling/manager.py)
- 修改内容：
  - 在 `_build_metrics_data` 增加“跳过无效 converged 任务数”日志。
  - 保持现有数据流不变，避免额外接口变更。
- 设计理由：增强可观测性，便于后续定位数据质量问题。

### 阶段三：回归测试补齐

#### 6) postprocess 单测
- 文件：[test_postprocess.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/tests/unit/labeling/test_postprocess.py)
- 新增覆盖：
  - `running_scf.log` 有收敛但缺 `TOTAL-FORCE` 时，`check_convergence` 返回 `False`。
  - `load_data` 面对空 `forces` 或字段缺失时返回 `None`，且不抛异常。
  - `compute_metrics` 在无有效帧场景下返回空 DataFrame（含预期列），流程可继续。

#### 7) manager 单测
- 文件：[test_manager.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/tests/unit/labeling/test_manager.py)
- 新增覆盖：
  - “混合好坏 converged 样本”场景：坏样本被跳过、好样本仍可产出统计与导出调用。
  - 校验新增“跳过数”日志或对应调用行为，确保回归可见。

## Verification

### A. 短期止血验收（fp7 实目录）
- 扫描并隔离坏样本后，确认以下目录为空：
  - `.../CONVERGED/DeepCNT/cluster/C108Fe55_0`
  - `.../CONVERGED/DeepCNT/cluster/C114Fe55_0`
- 确认以下目录存在对应样本：
  - `.../BROKEN_FORCE/DeepCNT/cluster/C108Fe55_0`
  - `.../BROKEN_FORCE/DeepCNT/cluster/C114Fe55_0`
- 重跑 postprocess 后确认：
  - `labeling_postprocess.log` 无 IndexError；
  - `outputs/cleaned` 生成结果；
  - `manifest.csv` 与日志计数一致。

### B. 代码回归验收
- 运行：
  - `pytest tests/unit/labeling/test_postprocess.py`
  - `pytest tests/unit/labeling/test_manager.py`
  - `pytest tests/unit`
- 验收点：
  - 空 `forces` 样本不再导致流程崩溃；
  - 新增门控不影响正常样本路径；
  - 全量 unit 通过。

## Risks & Rollback
- 风险：更严格的 converged 判定可能降低“converged 数量”。
- 缓解：通过 warning 与跳过计数提供可观测性，必要时可仅保留下游防御（不回退短期止血数据隔离）。
- 回滚：若出现非预期影响，可回退 `check_convergence` 新门控，仅保留 `load_data/compute_metrics` 防御式检查。

## Execution Status (Completed)

### 任务完成确认
- [x] 短期止血：从 `CONVERGED` 提取坏样本并生成清单，重跑 `postprocess` 获取可用 `outputs/cleaned`。
- [x] 长期修复：补齐收敛判定与数据完整性防御，避免 `forces` 空数组导致整批崩溃。
- [x] 回归测试：补充 postprocess/manager/workflow/CLI 相关用例并通过全量 unit。
- [x] 功能解耦：新增 `extract` 独立阶段，支持 `dpeva label --stage extract` 单独执行结果提取。

### 最终修复方案（已落地）
- `AbacusPostProcessor` 新增 `classify_task_status(task_dir)`，将任务状态明确为 `converged`、`bad_converged`、`failed`。
- “坏 converged”定义为：电子步收敛但缺失 `TOTAL-FORCE` 输出块；该类任务不再进入后处理。
- `load_data` 增加 labeled 数据完整性校验（`energies/forces/virials/cells` 长度一致性）；不完整样本直接跳过并记录原因。
- `compute_metrics` 增加系统级/帧级防御式跳过与空结果安全返回，确保错误不再中断整批流程。
- `LabelingManager` 新增 `BAD_CONVERGED` 分流目录与原因计数日志，支持坏样本复查与单独重跑。

### 当前 Labeling Workflow 解耦现状
- `--stage prepare`：仅负责从输入数据生成并打包任务到 `work_dir/inputs/N_*`。
- `--stage execute`：负责提交/重试计算，并在每轮后执行结果提取分流（converged/bad_converged/failed）。
- `--stage extract`：可独立基于 `inputs/N_*` 执行提取分流，不触发提交计算与后处理。
- `--stage postprocess`：仅消费 `CONVERGED` 数据做指标清洗与导出（可附带 integration）。
- `--stage all`：顺序为 `prepare -> execute -> extract -> postprocess`。

### 验证结论
- `dpeva label --stage extract` 在相同输入快照下重复执行结果一致，确认幂等。
- 全量单测通过：`pytest tests/unit` 结果 `231 passed`。
