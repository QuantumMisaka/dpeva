## 8. 标注工作流 (Labeling Workflow)

**配置类**: `LabelingConfig`
**功能**: 第一性原理计算 (ABACUS) 数据标注。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `input_data_path` | `Path` | 是 | - | 输入候选结构路径 (dpdata 兼容格式) |
| `output_dir` | `Path` | 是 | - | 结果输出目录 |
| `dft_params` | `Dict` | 否 | `{}` | ABACUS INPUT 参数字典 |
| `pp_map` | `Dict` | 否 | `{}` | 元素到赝势文件名映射 (如 `{'Fe': 'Fe_ONCV_PBE-1.0.upf'}`) |
| `orb_map` | `Dict` | 否 | `{}` | 元素到轨道文件名映射 |
| `pp_dir` | `str` | 是 | - | 赝势文件所在目录 |
| `orb_dir` | `str` | 是 | - | 轨道文件所在目录 |
| `kpt_criteria` | `int` | 否 | `20` | K点生成标准 (K_spacing = criteria/lattice_const) |
| `vacuum_thickness` | `float` | 否 | `10.0` | 真空层厚度判断阈值 |
| `tasks_per_job` | `int` | 否 | `50` | 每个 Slurm 作业打包的任务数 |
| `mag_map` | `Dict` | 否 | `{}` | 初始磁矩映射 |
| `cleaning_thresholds` | `Dict` | 否 | (见常量) | 结果清洗阈值 (force, energy 等)。支持 `null` 跳过检查。 |
| `attempt_params` | `List[Dict]` | 否 | (见常量) | 自动重试策略参数列表 |
| `ref_energies` | `Dict` | 否 | `{}` | 元素参考能量 (用于 Cohesive Energy 计算) |
| `output_format` | `str` | 否 | `deepmd/npy` | 输出数据集格式 |