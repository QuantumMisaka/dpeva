
## 调试与修复报告：SamplingManager 接口与 n_candidates 参数传递问题

### 1. 问题描述
在运行 `dpeva collect` 工作流时，Slurm 任务在采样阶段崩溃。

*   **日志位置**: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/test/dpeva-iter-try2-0222/collect_slurm.err`
*   **错误信息**: 
    ```text
    ValueError: Pipeline.fit does not accept the n_candidates parameter. You can pass parameters to specific steps of your pipeline using the stepname__parameter format...
    ```
*   **发生阶段**: Phase 2: Sampling (Joint Sampling)

### 2. 原因分析
该问题源于 Joint Sampling（联合采样）功能的引入，涉及参数 `n_candidates` 在调用链中的传递：

1.  **Workflows 层 (`collect.py`)**: 
    在 `run()` 方法中，工作流尝试支持 Joint Sampling。
    *(注：早期版本曾尝试直接将 `n_candidates` 传递给 `execute_sampling`，导致 `TypeError`，该问题已在相关 PR 中修正，但引出了深层问题)*

2.  **Manager 层 (`dpeva/sampling/manager.py`)**:
    `SamplingManager` 在 `prepare_features` 阶段计算并存储了 `self.n_candidates`。
    在 `_run_direct` 方法中，它尝试将此参数传递给底层的 Sampler：
    ```python
    res = sampler.fit_transform(features, n_candidates=self.n_candidates)
    ```

3.  **Algorithm 层 (`dpeva/sampling/direct.py`)**:
    `DIRECTSampler` 继承自 `sklearn.pipeline.Pipeline`。Sklearn 的 Pipeline 机制默认**不接受** `fit` 或 `fit_transform` 方法中的额外关键字参数（除非用于路由到特定步骤）。因此，抛出 `ValueError`。

### 3. 修复方案决策

#### 方案 A：修改底层 Sampler (已否决)
尝试修改 `DIRECTSampler` 和 `SelectKFromClusters` 的 `fit_transform` 接口以接受 `n_candidates`。
*   **缺点**: 侵入式修改核心算法代码，违反了 **Zen of Python** (Explicit is better than implicit) 和项目关于模块职责分离的设计原则。`sampling` 模块应保持通用性，不应感知业务层的 `n_candidates` 概念。

#### 方案 B：在 Manager 层进行后处理 (已采纳)
保持底层 Sampler 接口不变，在 `SamplingManager` 中处理过滤逻辑。
*   **实现**: 
    1. 调用 `sampler.fit_transform(features)`（不传额外参数）。
    2. 获取返回的 `selected_indices`。
    3. 在 Manager 中根据 `self.n_candidates` 显式过滤掉训练集数据的索引。

### 4. 验证
编写并运行了单元测试 `tests/unit/sampling/test_sampling_manager.py::test_execute_sampling_joint`。
*   **测试结果**: Pass (OK)
*   **验证点**: 
    *   确认 `fit_transform` 调用时不再包含 `n_candidates` 参数。
    *   确认返回的 `selected_indices` 已正确过滤掉大于等于 `n_candidates` 的索引。
*   **集成验证**: 在 Local Backend 下成功运行 `config_multi_joint.json` 案例，数据导出正常。

### 5. 代码变更
**文件**: `dpeva/src/dpeva/sampling/manager.py`

```python
<<<<
        # Use stored n_candidates if available (for Joint Sampling)
        res = sampler.fit_transform(features, n_candidates=self.n_candidates)
        
        # Calculate scores and random baseline for visualization
        selected_indices = res["selected_indices"]
====
        # Use stored n_candidates if available (for Joint Sampling)
        res = sampler.fit_transform(features)
        
        # Calculate scores and random baseline for visualization
        selected_indices = res["selected_indices"]
        pca_features = res["PCAfeatures"]
        
        # Filter out training data (Joint Sampling post-processing)
        if self.n_candidates is not None:
            original_len = len(selected_indices)
            # Training data is appended after candidates, so indices >= n_candidates are training data
            selected_indices = [idx for idx in selected_indices if idx < self.n_candidates]
            filtered_len = len(selected_indices)
            if filtered_len < original_len:
                self.logger.info(f"Joint Sampling: Filtered out {original_len - filtered_len} selections from training data.")
>>>>
```
