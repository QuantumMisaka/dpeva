# DP-EVA Joint(Mixed) DIRECT Sampling 功能开发报告

## 1. 功能概述
本功能旨在优化主动学习的数据采集策略。通过将**已有训练集**的结构描述符与**待选数据池**（Candidates）的描述符合并，共同进行 DIRECT 采样（降维+聚类+分层采样），从而确保新采集的数据尽可能覆盖训练集尚未覆盖的化学空间，最大化数据多样性并减少冗余。

## 2. 开发内容

### 2.1 核心逻辑 (`dpeva.workflows.collect`)
在 `CollectionWorkflow.run` 中增加了混合采样逻辑：
1.  **加载训练集描述符**: 读取 `config.json` 中的 `training_desc_dir`。
2.  **一致性检查**: 读取 `training_data_dir`，统计帧数，确保与描述符数量一致。若不一致则抛出异常。
3.  **合并特征**: 将 `candidate_descriptors` 与 `training_descriptors` 垂直堆叠（vstack）。
4.  **联合采样**: 使用 `DIRECTSampler` 对合并后的特征进行聚类和采样。
5.  **结果过滤**: 从采样结果的索引中，**剔除**属于训练集的部分，仅保留属于 Candidates 的索引。
    *   **代码原理**:
        1.  **数据合并**: 训练集描述符被堆叠在 Candidates 描述符之后。设 Candidates 数量为 $N$，Training 数量为 $M$，合并后总数量为 $N+M$。
        2.  **DIRECT 采样**: 算法在联合空间进行降维和聚类，返回一组选中样本的索引 `selected_indices`。
        3.  **索引过滤**: 遍历 `selected_indices`，若索引值 $i < N$，则判定该样本属于 Candidates，予以**保留**；若 $i \ge N$，则判定属于 Training Set，予以**剔除**。
    *   **设计意图**: 若 DIRECT 算法选中的代表性样本（如聚类中心）属于训练集，说明该化学空间区域已被现有训练数据充分覆盖，无需重复采集；反之，若选中了 Candidate，说明该区域是训练集的盲区，具有极高的采集价值。
6.  **可视化适配**: 更新 PCA 可视化逻辑，使用联合训练的 PCA 投影 Candidate 数据，确保可视化空间的一致性。

### 2.2 配置参数
在 `config.json` 中新增以下可选参数：
*   `training_data_dir`: 训练集数据路径（用于帧数校验）。
*   `training_desc_dir`: 训练集描述符路径（必需，用于混合采样）。

若未提供上述参数，工作流将自动回退到仅针对 Candidates 进行采样的原有逻辑。

## 3. 测试验证

### 3.1 测试环境
*   **测试脚本**: `dpeva/test/verification_test_run/run_mixed_direct_test.py`
*   **配置文件**: `dpeva/test/verification_test_run/config_mixed_direct.json`
*   **数据来源**:
    *   Candidates: `test_val` (2283 frames)
    *   Training: `sampled_dpdata` (4197 frames)

### 3.2 测试结果
*   **日志验证**:
    ```
    INFO:dpeva.workflows.collect:Mixed sampling enabled: 2283 candidates + 4197 training samples.
    INFO:dpeva.workflows.collect:Combined feature shape: (6480, 128)
    INFO:dpeva.sampling.clustering:BirchClustering... gives 50 clusters.
    INFO:dpeva.sampling.stratified_sampling:Finally selected 50 configurations.
    INFO:dpeva.workflows.collect:DIRECT selected 50 total samples. After filtering training samples, 4 candidates remain.
    ```
*   **结果分析**:
    *   系统成功加载了 4197 帧训练数据和 2283 帧候选数据。
    *   DIRECT 算法在联合空间中划分了 50 个聚类。
    *   在选出的 50 个代表性样本中，有 46 个来自训练集（说明大部分聚类已被训练集覆盖）。
    *   最终仅输出了 **4 个** 新的 Candidate 样本。这符合“最大化差异”的预期——只有真正独特的结构才会被选中。

### 3.3 异常处理测试
*   代码中包含了对 `training_data_dir` 帧数的校验逻辑。
*   若 `training_desc_dir` 为空或加载失败，系统会输出 Warning 并安全回退到仅 Candidate 采样模式。

## 4. 交付物清单
1.  **代码**: `dpeva/src/dpeva/workflows/collect.py` (已修改)。
2.  **测试脚本**: `dpeva/test/verification_test_run/run_mixed_direct_test.py`。
3.  **示例配置**: `dpeva/runner/dpeva_collect/config_mixed_sampling_example.json`。
4.  **开发报告**: 本文档。
