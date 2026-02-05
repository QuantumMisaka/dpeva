通过对核心代码库 `dpeva/src` 的详细审计，我已确认 **DIRECT** 采样工作流已完全集成并可经由 CLI 调用，但 **2-DIRECT** 采样策略尚未集成。

以下是详细的审计结果：

### 1. DIRECT 采样工作流
- **状态：** <font color="green">**已集成**</font>
- **核心实现：** 类 `DIRECTSampler` 已在 `src/dpeva/sampling/direct.py` 中实现。
- **CLI 支持：** `CollectionWorkflow` (`src/dpeva/workflows/collect.py`) 已直接导入并使用了 `DIRECTSampler`。
- **调用方式：** 可通过 `dpeva collect <config_path>` 命令直接调用，配置中通过 `num_selection` 和 `direct_k` 等参数控制。

### 2. 2-DIRECT (Two-Step) 采样工作流
- **状态：** <font color="red">**未集成**</font>
- **缺失分析：**
    - 在 `src/` 目录下未发现涉及 "TwoStep", "Partition", "Atomic Feature Clustering" 等 2-DIRECT 核心逻辑的代码。
    - 该策略目前仅存在于 `examples/sampling/2-direct.ipynb` 笔记本中，作为一段独立的实验性代码存在。
- **技术差距：**
    - 2-DIRECT 需要引入“两步聚类”（先基于结构聚类，再基于原子环境聚类）的逻辑，目前的 `CollectionWorkflow` 仅支持基于结构的单层聚类。
    - 缺乏相应的配置参数（如 `step1_k`, `step2_k` 等）来支持这种两阶段操作。

### 下一步建议 (Next Steps)
为了完成重构并将 2-DIRECT 能力赋予 CLI，建议执行以下操作：

1.  **核心迁移**：将 `2-direct.ipynb` 中的逻辑提取并封装为 `src/dpeva/sampling/two_step_direct.py`，实现一个兼容 scikit-learn 风格的 `TwoStepDIRECTSampler` 类。
2.  **工作流集成**：修改 `CollectionWorkflow`，使其支持通过配置文件选择 `sampler_type: "direct"` (默认) 或 `sampler_type: "2-direct"`。
3.  **CLI 暴露**：在 `CollectionConfig` 中增加 2-DIRECT 所需的特定参数。

请确认是否需要我**立即开始**将 2-DIRECT 策略从 notebook 迁移到核心代码库中？