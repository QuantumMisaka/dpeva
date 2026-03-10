# 变量描述审查报告 (Docstring Review Report)

**Review Date**: 2026-03-10
**Target**: `src/dpeva/config.py`
**Reviewer**: Code Audit Agent

## 1. 总体评价
代码库使用了 Pydantic V2 进行配置管理，整体结构清晰，所有字段均包含 `description` 参数，符合自动化文档生成的基本要求。但在详细程度、类型说明和业务含义方面仍有优化空间。

## 2. 发现的问题与优化建议

### 2.1 模糊或缺失的描述
部分字段描述过于简略，缺乏具体的取值范围或业务上下文。

| 类名 | 字段名 | 当前描述 | 建议优化 |
| :--- | :--- | :--- | :--- |
| `SubmissionConfig` | `env_setup` | "Environment setup commands." | "List of shell commands to execute before running the task (e.g., `module load cuda`)." |
| `FeatureConfig` | `output_mode` | "Output mode." | "Descriptor output format. Options: `atomic` (per-atom features) or `structural` (global features)." |
| `LabelingConfig` | `pp_map` | "Pseudopotential mapping." | "Dictionary mapping element symbols to pseudopotential filenames (e.g., `{'Fe': 'Fe.upf'}`)." |
| `LabelingConfig` | `kpt_criteria` | "K-point density criteria." | "K-point density parameter (K*L). Determines grid density: `k_i = ceil(criteria / lattice_i)`." |
| `CollectionConfig` | `uq_select_scheme` | "Selection scheme (e.g., 'strict', 'loose')." | "Strategy for selecting high-uncertainty data. Options: `tangent_lo`, `strict`, `circle_lo`, etc." |

### 2.2 类型信息不明确
虽然 Pydantic 提供了类型校验，但在文档字符串中明确类型有助于用户在不看代码的情况下理解。

- **建议**: 在 `description` 中提及复杂类型的结构，例如 `List[int]` 或 `Dict[str, float]`。

### 2.3 业务含义缺失
部分参数缺乏对“为什么需要这个参数”的解释。

- `tasks_per_job`: 仅描述为 "Number of tasks per packed job"，未说明这对 Slurm 调度效率的影响（减少小作业数量）。
- `cleaning_thresholds`: 未说明如果不满足阈值会发生什么（被丢弃或标记）。

## 3. 变更计划 (Action Items)

建议对 `src/dpeva/config.py` 进行如下批量更新（以 Pull Request 形式提交）：

1.  **LabelingConfig**: 详细说明 `pp_map`, `orb_map`, `dft_params` 的键值对结构。
2.  **CollectionConfig**: 补充 `uq_trust_ratio` 和 `uq_trust_width` 在不同 `uq_trust_mode` 下的行为差异。
3.  **通用字段**: 统一 `Path` 类型字段的描述，明确是“文件路径”还是“目录路径”。

## 4. 结论
当前文档质量处于 **B+** 水平。通过实施上述优化，可提升至 **A** 级，显著降低用户的认知负荷。
