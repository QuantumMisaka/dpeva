# DIRECT no_filter 多数据池兼容修复计划

## Summary
- 目标：修复 `dpeva collect` 在 `uq_trust_mode=no_filter` 且 `sampler_type=direct` 时对多数据池描述符目录（嵌套子目录）无法加载的问题。
- 结果对齐：让 DIRECT-only（no_filter）与 UQ+DIRECT 在 Collection Workflow 的候选池识别、命名与导出行为保持一致，不再因目录层级差异中断。
- 约束：保持现有配置语义不变，仅修复描述符发现与系统命名映射链路；必须确保单数据池场景（输入目录本身可被 `dpdata.MultiSystems` 识别）行为不变；补充单元测试覆盖并确保现有测试不回归。

## Current State Analysis
- 复现日志已确认：`collection.log` 报 `No candidate descriptors found in .../desc-255` 并停止。
- 根因定位到 `CollectionWorkflow._run_no_filter_uq_phase()` 调用 `CollectionIOManager.load_descriptors(desc_dir)` 时未传 `target_names`，导致走 `load_descriptors` 的“glob 全量加载”分支。
- 当前 glob 分支仅匹配 `desc_dir/*.npy`（非递归），无法发现多数据池结构中的 `desc_dir/<pool>/<system>.npy`。
- 且 glob 分支使用 `basename` 作为 `sys_name`，在多池场景会丢失层级信息（`pool/system`），后续导出与统计的系统名一致性存在风险。
- UQ+DIRECT 路径通常可通过 `target_names` 逐系统加载（含层级名），因此该问题主要暴露在 DIRECT-only no_filter 路径。

## Proposed Changes

### 1) 修复描述符递归发现与层级命名保持
- 文件：[collection.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/src/dpeva/io/collection.py)
- 变更点：
  - 在 `load_descriptors` 的非 `target_names` 分支中，将 `*.npy` 扫描扩展为递归发现（支持 `**/*.npy`）。
  - 生成 `sys_name` 时改为相对 `desc_dir` 的无扩展路径（例如 `008/94`），而非仅 `basename`。
  - 对“平铺目录”保持兼容：当 `.npy` 位于 `desc_dir` 根层时，相对路径仍是 `sys1` 形式，不改变现有单池命名。
  - 对显式 glob 入参（`desc_dir` 含 `*`）保持现有语义：仅按用户给定模式匹配，不强行改写扫描策略。
  - 保持现有归一化、帧数校验与异常处理逻辑不变。
- 目的：
  - 让 no_filter 模式能发现多数据池描述符。
  - 让 `dataname` 维持 `pool/system-index` 形式，与 UQ+DIRECT 及导出链路一致。

### 2) 对齐原子特征加载的路径解析韧性（2-direct兼容防回归）
- 文件：[collection.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/src/dpeva/io/collection.py)
- 变更点：
  - 在 `load_atomic_features` 中复用与描述符加载一致的路径解析策略（优先完整层级名，其次兼容 basename 回退）。
  - 不改变函数接口，确保现有调用点无侵入。
- 目的：
  - 保证多层级命名在 `direct` 与 `2-direct` 下行为一致，避免后续功能差异。

### 3) 增加单元测试覆盖 no_filter 多池核心路径
- 文件：[test_collection_io_full.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/tests/unit/io/test_collection_io_full.py)
- 新增测试：
  - `load_descriptors` 在嵌套目录下可递归加载 `.npy`。
  - 返回 dataname 保留层级（例如 `poolA/sys1-0`）。
  - 向后兼容：平铺目录加载行为保持不变（单池 `sys1-0` 形式不变）。
  - 向后兼容：`desc_dir` 为显式 glob 模式时行为不变。
- 文件：[test_collect_workflow_routing.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/tests/unit/workflows/test_collect_workflow_routing.py)
- 新增测试：
  - 构造 no_filter 路由，mock `load_descriptors` 返回多池 dataname，验证 `_run_no_filter_uq_phase` 能生成候选并提取层级系统名。
  - 构造 no_filter 单池路由，验证 `_run_no_filter_uq_phase` 在单池命名下输出与既有逻辑一致。
- 文件：[test_dataset.py](file:///home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/tests/unit/io/test_dataset.py)
- 复用与必要补强测试：
  - 保留并执行 `load_systems_mixed_format`，确保“输入目录本身可被 `dpdata.MultiSystems` 识别”的路径不受此次改动影响。
  - 若当前覆盖不足，则补充一个最小回归用例，显式断言该路径在本次改动后仍可工作（不修改 `dataset.py` 实现）。
- 目的：
  - 直接锁定本次缺陷入口与关键行为。
  - 确保 DIRECT-only 与 UQ+DIRECT 在“系统命名/候选识别”维度一致。

### 4) 回归验证与质量门禁
- 验证顺序：
  - 运行新增与相关单测：`tests/unit/io/test_collection_io_full.py`、`tests/unit/workflows/test_collect_workflow_routing.py`。
  - 运行 collection 相关单测集合（若存在相关模块测试一并覆盖）。
  - 运行 `pytest tests/unit` 进行全量单测回归（按仓库建议优先单测）。
- 通过标准：
  - no_filter 多池测试通过，且不再出现“候选描述符为空”导致的中断。
  - 既有单测通过，未引入接口破坏或行为回退。

## Assumptions & Decisions
- 假设多数据池描述符命名与测试池目录层级语义一致（`<pool>/<system>.npy` 对应 `<testdata>/<pool>/<system>/`）。
- 决策：不改配置字段、不引入新开关，优先通过 I/O 层递归发现与命名统一完成修复。
- 决策：保持“层级优先 + basename 回退”的兼容策略，降低历史数据组织差异带来的风险。
- 决策：本次不修改 `dpeva.io.dataset.load_systems` 与 `CollectionWorkflow` 的路由判定逻辑，仅在 `CollectionIOManager` 内收敛修复，确保单数据池 `MultiSystems` 支持面不被扩大或收缩。

## Verification Steps
- 在测试中构造嵌套描述符目录与最小输入，验证 no_filter 路径能生成非空候选集。
- 验证导出的系统名与层级保持一致（通过现有导出相关测试与新增断言侧面保障）。
- 验证单池兼容性：确认平铺 `.npy`、显式 glob、以及 `dpdata.MultiSystems` 自动识别相关单测全部通过。
- 执行 unit 回归，确认 Collection 其他功能（含已有前缀归整、路径解析）未受破坏。
