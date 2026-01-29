# DP-EVA 多数据池支持方案

## 问题
当前的架构假设单一数据池且系统名称唯一。引入“多数据池”（Dataset/System 结构）会导致以下问题：
1.  **命名冲突**: `dp test` 输出解析 (`dataproc.py`) 仅提取基本名称 (System)，丢失了 Dataset 上下文（例如 `PoolA/Sys1` 变为 `Sys1`），如果 `PoolB` 也有 `Sys1` 则会导致冲突。
2.  **顺序不匹配**: `CollectWorkflow` 通过 `glob`（文件系统顺序）加载描述符，但依赖 `dp test` 输出顺序进行对齐。这两种顺序在多数据池中无法保证一致。
3.  **特征生成**: `generator.py` 中的 Python 模式和 CLI 模式尚未完全支持多数据池描述符所需的递归 3 层结构 (`Root/Dataset/System`)。

## 解决方案

### 1. 增强 `dataproc.py` (解析)
- **目标**: 在系统名称中保留 Dataset 上下文。
- **变更**: 修改 `TestResultParser._get_dataname_info`，当在 `dp test` 输出注释中检测到嵌套结构时，提取 `Dataset/System` 路径（最后两级目录组件），而不仅仅是 `System`（基本名称）。
- **影响**: `dataname` 变得唯一（例如 `mptrj-FeCOH/C0Fe4H0O8`），从而实现精确匹配。

### 2. 升级 `collect.py` (对齐)
- **目标**: 严格将数据和描述符与推理结果对齐。
- **变更**:
    - **有序加载**: 修改 `run()`，使用从 `dp test` 结果中提取的唯一 `dataname` 列表作为排序的“基准事实 (Ground Truth)”。
    - **描述符加载**: 更新 `_load_descriptors` 以接受此有序列表，并加载特定文件 (`desc_dir/Dataset/System.npy`)，而不是使用 `glob` 通配符。
    - **测试数据加载**: 更新数据加载逻辑，根据有序名称查找并加载 `testdata_dir/Dataset/System`，而不是遍历目录。

### 3. 升级 `generator.py` (生成)
- **目标**: 支持 3 层结构 (`Root/Dataset/System`)。
- **变更**:
    - **CLI 模式**: 更新 `run_cli_generation` 以检测 `data_path` 是否包含 Dataset（子目录）。如果是，生成一个脚本迭代 Dataset 并为每个 Dataset 运行 `dp eval-desc`，确保 `desc_pool` 镜像 `Dataset/System` 结构。
    - **Python 模式**: 如果检测到 Dataset，更新逻辑以递归进入 Dataset，确保 `dpdata` 加载对于 3 层结构的鲁棒性。

## 验证
- 使用 `test-for-multiple-datapool` 运行 `CollectWorkflow`。
- 验证 `df_uq` 是否包含正确的 `Dataset/System` 名称。
- 验证描述符和测试数据是否按正确对应关系加载。

## 阶段 2: 联合采样与扩展性支持 (Slurm Backend)

在此阶段，我们进一步增强了 DP-EVA 对大规模数据和持续学习场景的支持，主要包括联合采样策略、Slurm 作业调度支持以及可视化优化。

### 1. Joint DIRECT Sampling (联合采样)
- **背景**: 在主动学习迭代中，仅考虑候选池的多样性是不够的。为了最大化数据利用率，新采样的点应当尽可能补充现有训练集未覆盖的化学空间。
- **实现**:
  - 更新 `CollectWorkflow` 支持 `joint_sampling` 模式。
  - **输入**: 接受 `other_dpdata_all` (候选池) 和 `sampled_dpdata` (现有训练集)。
  - **算法**: 修改 `DIRECT_sampler`，将训练集数据作为 `fixed_set` (固定基底) 传入。在进行 Maximin 距离优化（FPS-like）时，确保新选取的点不仅彼此远离，且远离现有的训练集点。
- **配置**: 在 `collect_config.json` 中自动识别输入路径，无需额外复杂配置。

### 2. Slurm Backend Support (作业调度)
- **背景**: 当数据池规模增大（数十万/百万帧）时，本地串行处理 `CollectWorkflow`（尤其是描述符加载和聚类分析）可能导致内存溢出或长时间占用登录节点资源。
- **实现**:
  - **Self-Submission (自提交模式)**: `CollectWorkflow` 新增 `backend="slurm"` 选项。
  - **流程**:
    1. 主程序冻结当前配置为 `collect_config_frozen.json`。
    2. 自动生成执行脚本 `run_collect_slurm.py`。
    3. 自动生成并提交 SBATCH 脚本 (`submit_collect.sbatch`)。
  - **Logging 优化**: 针对 Slurm 输出缓冲问题，在生成的命令中添加 `python -u` (Unbuffered) 并显式配置 `logging.basicConfig`，确保 `collect_slurm.out` 实时显示进度日志。

### 3. Visualization Optimization (可视化改进)
- **背景**: 联合采样模式下，数据来源增多（Pool, Candidate, Training, Selected），导致 PCA 投影图要素过多，难以辨识。
- **改进**:
  - **分层展示**:
    - `DIRECT_PCA_feature_coverage.png`: 专注于展示覆盖率，保留 Training Set 和 Selected Data 的关系。
    - `Final_sampled_PCAview.png`: 专注于最终选取结果，移除 Training Set 干扰，仅展示 "All Data in Pool" (背景), "Candidate" (候选), "Final Selected" (结果)。
  - **图例优化**: 统一将 Legend 位置设为 `loc='best'`，并重命名标签（"All Data" -> "All Data in Pool", "UQ Selected" -> "Candidate"）以消除歧义。
