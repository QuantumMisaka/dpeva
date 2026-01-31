# DP-EVA 功能迁移完整性确认表

| 原始功能点 | 原始文件路径 | 重构后对应位置 | 迁移状态 | 差异描述 | 验证方式 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **多模型并行训练** | `utils/dptrain/run_parallel.py`<br>`utils/dptrain/gpu_DPAtrain-multigpu.sbatch` | `runner/dpeva_train/run_train.py`<br>`src/dpeva/training/parallel_trainer.py` | ✅ **已完美纳入** | 重构为 `ParallelTrainer` 类，支持 `init`/`cont` 模式，使用 `JobManager` 统一管理 Slurm/Local 提交，不再依赖硬编码脚本。 | 单元测试 (`tests/unit/training/test_trainer_input.py`)<br>单元测试 (`tests/unit/submission/test_job_manager.py`) |
| **描述符生成** | `utils/dpdesc/gen_desc.py`<br>`utils/dpdesc/gen_desc_stru.py` | `runner/dpeva_evaldesc/run_evaldesc.py`<br>`src/dpeva/feature/generator.py` | ✅ **已完美纳入** | 重构为 `DescriptorGenerator` 类，支持 CLI (`dp eval-desc`) 和 Python API 双模式，解决了递归目录处理和 OMP 线程控制问题。 | 单元测试 (`tests/unit/feature/test_generator.py`) |
| **数据采集工作流** | `utils/run_collect_workflow.py` | `runner/dpeva_collect/run_uq_collect.py`<br>`src/dpeva/workflows/collect.py` | ✅ **已完美纳入** | 迁移为标准 Runner，配置文件由 Python 字典改为 JSON，增加了联合采样 (Joint Sampling) 和 Auto-UQ 功能。 | 单元测试 (`tests/unit/workflows/test_collect_workflow_routing.py`)<br>单元测试 (`tests/unit/test_sampling.py`) |
| **UQ 结果可视化** | `utils/uq/uq-post-view.py`<br>`utils/uq/uq-post-view-2.py` | `src/dpeva/workflows/collect.py`<br>`src/dpeva/uncertain/visualization.py` | ✅ **已完美纳入** | `CollectionWorkflow` 完整集成了 UQ 计算、筛选、DIRECT 采样及可视化逻辑。原脚本的所有功能（含绘图）均已被 `dpeva.workflows.collect` 和 `dpeva.uncertain` 模块覆盖，无需保留独立脚本。 | 单元测试 (`tests/unit/test_calculator_uq.py`)<br>单元测试 (`tests/unit/test_filter_uq.py`) |
| **PCA 分析** | `utils/dpdesc/pca_desc_stru.py` | `src/dpeva/sampling/pca.py` | ⚠️ **部分纳入** | 核心 PCA 算法已封装为 `PCASampler` 类。原始脚本包含特定的 "FeCHO" 生成能计算和特定绘图逻辑，作为定制化分析脚本未被通用化。 | 代码审查 |
| **数据集合并工具** | `utils/dataset_tools/dpdata_addtrain.py`<br>`utils/others/dpdata_update.py` | `src/dpeva/workflows/collect.py` (部分逻辑) | ⚠️ **部分纳入** | `CollectWorkflow` 已内置了从池中移除已选样本并导出 `other_dpdata` 的逻辑。显式的数据合并功能（Add Train）目前建议通过配置多数据路径实现，该脚本作为独立工具保留。 | 代码审查 |
| **DFT 计算准备** | `utils/fp/reprepare.py`<br>`utils/fp/*.sh` | N/A | ❌ **未纳入** | 属于 "Labeling" (标注) 环节的 DFT 计算准备逻辑，涉及 ABACUS 特定输入/结构处理，超出当前 `dpeva` (主动学习循环控制) 的核心范围。 | N/A |
| **Legacy 测试脚本** | `utils/dptest/*.sh` | `runner/dpeva_test/run_inference.py` | ✅ **已完美纳入** | Shell 脚本逻辑已被 Python 化的 `InferenceWorkflow` 完全替代，支持自动模型发现和批量提交。 | 单元测试 (`tests/unit/workflows/test_infer_workflow_exec.py`)<br>单元测试 (`tests/unit/test_parser.py`) |

## 缺失/差异清单

1.  **DFT Labeling 模块 (`utils/fp`)**:
    *   **现状**: 当前代码库专注于主动学习的 *筛选* 与 *训练* 闭环，未包含 *标注* (Labeling/DFT Calculation) 的具体实现。
    *   **建议**: 维持现状。标注通常高度依赖具体的计算软件 (VASP, CP2K, ABACUS) 和集群环境，建议作为独立项目或通过接口集成，而非硬编码在 DP-EVA 中。

2.  **定制化分析脚本**:
    *   **现状**: `pca_desc_stru.py` 等脚本包含硬编码的原子能量参数（如 Fe, C, H, O 的参考能量），无法直接通用化。
    *   **建议**: 保留为示例脚本 (`examples/`) 或用户自定义脚本，不建议并入核心库。

3.  **数据合并工具**:
    *   **现状**: `dpdata_addtrain.py` 是一个方便的 CLI 工具。
    *   **建议**: 可以在 `dpeva.io` 中添加类似的合并函数，但目前通过配置多路径已能满足训练需求，优先级较低。

## 结论

核心业务逻辑（训练、推理、采样、筛选）已 **100% 迁移** 至新的模块化架构中。`utils` 目录下剩余的主要是特定场景的分析脚本、Legacy 脚本或属于上下游（如 DFT 标注）的辅助工具。重构后的代码库在功能完备性、扩展性和鲁棒性上均优于原始脚本集合。
