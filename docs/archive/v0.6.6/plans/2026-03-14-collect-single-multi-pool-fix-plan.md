# CollectionWorkflow 单/多数据池采样未完成问题修复计划

## Summary
- 目标：修复 `dpeva collect` 在单数据池场景下“已采样但导出为 0 帧”导致流程失败的问题，同时确保多数据池路径行为不回退。
- 根因：`load_systems(..., target_systems=...)` 在目标系统名带有数据根目录前缀时，直接拼接 `data_dir/sys_name`，产生重复路径段（如 `.../other_dpdata/other_dpdata/C...`），导致导出阶段无法加载原始系统。
- 策略：增强目标系统路径解析的鲁棒性（保留层级语义与目标名一致性），并补齐单元测试覆盖该回归场景与多池兼容场景。

## Current State Analysis
- 复现配置：`test/dpeva-iter1/config_collect_normal.json` 中 `testdata_dir="test/dpeva-iter1/other_dpdata"`、`desc_dir=".../desc_pool"`，UQ+DIRECT 正常执行到采样完成。
- 失败现象：`collection.log` 在采样后报错 `CRITICAL: No frames were exported despite candidates being selected!`，随后抛出运行时失败。
- 关键日志证据：大量 `Data directory not found for system: other_dpdata/... at .../other_dpdata/other_dpdata/...`，说明目标系统名已含根目录前缀，拼接后路径重复。
- 代码定位：
  - `CollectionWorkflow._run_export_phase` 调用 `CollectionIOManager.export_dpdata`，当导出总帧数为 0 时强制失败。
  - `CollectionIOManager.export_dpdata` 将 `unique_system_names` 传给 `load_systems(testdata_dir, target_systems=...)`。
  - `load_systems` 目标分支当前仅执行 `os.path.join(data_dir, sys_name)`，没有处理“目标名包含根目录前缀”的情况。
- 影响范围：
  - 单数据池：若系统名来自上游数据链路并带前缀，导出必然失败。
  - 多数据池：层级名（`pool/system`）必须保持可用，不可被扁平化或误裁剪。

## Proposed Changes

### 1) 修复目标系统路径解析（核心逻辑）
- 文件：`src/dpeva/io/dataset.py`
- 改动内容：
  - 新增内部解析逻辑（函数或内联步骤），针对每个 `sys_name` 生成候选相对路径并按顺序尝试：
    1. 原始 `sys_name`（保持多池层级语义）；
    2. 若以 `basename(data_dir)/` 开头，去掉该前缀后的相对路径；
    3. 可选：重复前缀场景下进行一次“去重根段”纠偏（仅在目录存在时接受）。
  - 仅当候选目录实际存在时采用，并继续把 `target_name` 保留为原始 `sys_name`（保证与 `df_final` 的 `dataname` 键一致）。
  - 当候选均不存在时，输出更可诊断的 warning（包含尝试过的候选路径）。
- 设计理由：
  - 遵循“显式优于隐式”：先用原路径，再做最小必要纠偏。
  - 不破坏多池层级：优先保留 `pool/system`，不做 basename 扁平化替换。
  - 与现有导出映射兼容：保持 `sampled_indices_map` 键空间不变。

### 2) 增加针对性单元测试（防回归）
- 文件：`tests/unit/io/test_dataset.py`
- 新增测试点：
  - `target_systems=["other_dpdata/C..."]` 且 `data_dir=.../other_dpdata` 时，能够正确解析到 `.../other_dpdata/C...` 并加载成功。
  - 验证加载后 `target_name` 仍为原始目标名（含前缀），确保导出映射一致。
  - 保留并验证多池目标名（如 `poolA/sys1`）路径行为不受影响。

- 文件：`tests/unit/io/test_collection_io_full.py`
- 新增测试点：
  - 构造 `df_final.dataname` 含前缀系统名（`other_dpdata/sys1-0`）场景；
  - mock `load_systems` 返回对应 `target_name`，断言 sampled/other 的分帧导出调用与计数正确。
  - 覆盖“有选中帧时导出总帧数>0”的成功路径，确保不会触发 workflow 末端 0 导出异常。

### 3) 可观测性与错误信息优化（轻量）
- 文件：`src/dpeva/io/dataset.py`（同核心逻辑文件内完成）
- 改动内容：
  - 对“目标系统未找到”日志补充候选路径列表摘要，便于一眼识别“重复根目录段”类配置/命名问题。
- 设计理由：
  - 错误不应静默，且应可直接定位到路径映射问题。

## Assumptions & Decisions
- 决策：以“兼容单池前缀误入 + 保持多池层级原语义”为第一原则，不改变现有配置字段定义与上游 dataname 结构。
- 决策：不在本轮引入新的配置开关，避免增加使用复杂度。
- 假设：`dataname` 与 `target_systems` 的系统名分隔规则仍为 `"{sys_name}-{frame_idx}"`。
- 假设：多池系统名中的 `/` 语义合法，后续导出路径安全由既有 `normalize_sys_name + safe_join` 继续保障。

## Verification
- 单元测试（最小回归集）：
  - `pytest tests/unit/io/test_dataset.py -k "target or duplicate or fallback"`
  - `pytest tests/unit/io/test_collection_io_full.py -k "export_dpdata"`
- 工作流级校验（若本地/环境允许）：
  - 使用给定配置再次执行 collect，确认 `collection.log` 出现 `WORKFLOW_FINISHED`，且 `dpdata/sampled_dpdata`、`dpdata/other_dpdata` 目录均有有效导出帧。
- 回归关注点：
  - 单池：不再出现 `.../other_dpdata/other_dpdata/...` 路径。
  - 多池：`pool/system` 层级导出结构保持不变。
  - 当系统确实不存在时，日志能明确显示尝试路径，便于排障。
