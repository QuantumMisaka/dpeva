---
title: Document
status: active
audience: Developers
last-updated: 2026-03-11
---

# 配置编写指南（Configuration）

- Status: active
- Audience: Users / Developers
- Last-Updated: 2026-03-11

## 1. 目的与范围

本页说明“如何写配置、如何组织路径、如何最小化配置”。全量字段字典与校验规则请以 Reference 为准：

- ../source/api/config.rst
- ../reference/validation.md

## 2. 相关方

- 使用者：编写配置并运行工作流
- 开发者：维护配置模型、路径解析与默认值
- 平台维护：提供 Slurm 队列/环境初始化建议

## 3. 路径解析规则（强烈建议使用相对路径）

### 3.1 规则

- 配置文件中推荐写相对路径，便于迁移与复现。
- CLI 会把相对路径按“配置文件所在目录”解析为绝对路径。

参考实现：`src/dpeva/utils/config.py` 的 `resolve_config_paths`。

### 3.2 常见路径字段

- `work_dir`：Train/Infer/Feature/Analysis 常用的工作目录
- `project`：Collect 常用的项目目录
- `data_path`：Feature/Infer 常用输入数据目录
- `savedir`：Feature 描述符输出目录
- `root_savedir`：Collect 输出目录

## 4. Submission 配置（Local / Slurm）

### 4.1 Local 最小配置

```json
{
  "submission": {
    "backend": "local"
  }
}
```

### 4.2 Slurm 最小配置

```json
{
  "submission": {
    "backend": "slurm",
    "env_setup": [
      "source /path/to/env.sh"
    ],
    "slurm_config": {
      "nodes": 1,
      "ntasks": 1,
      "walltime": "00:30:00"
    }
  }
}
```

常用扩展字段：`partition/qos/gpus_per_node/cpus_per_task/account`。

## 5. 各 Workflow 最小配置示例

以下示例用于“快速跑通”，完整字段见 Reference。

### 5.1 Feature

```json
{
  "work_dir": "./",
  "data_path": "./other_dpdata_all",
  "model_path": "./DPA-3.1-3M.pt",
  "savedir": "./desc_pool",
  "mode": "cli",
  "submission": { "backend": "slurm" }
}
```

### 5.2 Train

```json
{
  "work_dir": "./",
  "input_json_path": "input.json",
  "training_data_path": "./sampled_dpdata",
  "base_model_path": "DPA-3.1-3M.pt",
  "num_models": 3,
  "training_mode": "init",
  "submission": { "backend": "slurm" }
}
```

### 5.3 Infer

```json
{
  "work_dir": "./",
  "data_path": "./other_dpdata_all",
  "task_name": "test_val",
  "results_prefix": "results",
  "auto_analysis": false,
  "submission": { "backend": "slurm" }
}
```

### 5.4 Collect

```json
{
  "project": "./",
  "desc_dir": "./desc_pool",
  "testdata_dir": "./other_dpdata_all",
  "testing_dir": "test_val",
  "results_prefix": "results",
  "root_savedir": "dpeva_uq_result",
  "sampler_type": "direct",
  "direct_n_clusters": 20,
  "direct_k": 1,
  "submission": { "backend": "slurm" }
}
```

### 5.5 Analysis

```json
{
  "mode": "model_test",
  "result_dir": "0/test_val",
  "results_prefix": "results",
  "data_path": "sampled_dpdata",
  "output_dir": "analysis",
  "type_map": ["Fe", "C", "O", "H"],
  "enable_cohesive_energy": true,
  "allow_ref_energy_lstsq_completion": false,
  "enhanced_parity_renderer": "auto"
}
```

Analysis 相关建议：

- Inference 若自定义了 `results_prefix`，Analysis 的 `results_prefix` 必须保持一致。
- `model_test` 模式下若需要 Cohesive Energy，请优先提供 `data_path` 指向与 `result_dir` 对应的原始数据集，以避免仅靠文件名推断组分失败。
- `enable_cohesive_energy` 控制是否启用 Cohesive Energy 统计与作图。
- `allow_ref_energy_lstsq_completion` 控制当 `ref_energies` 不完整时是否允许最小二乘补全缺失元素参考能。
- `enhanced_parity_renderer` 控制 enhanced parity 主图区渲染入口：
  - `auto`：沿用 quantity-aware 默认策略，energy/cohesive 使用 scatter，force/virial 使用 hexbin。
  - `scatter`：强制 enhanced parity 主图区全部使用 scatter。
  - `hexbin`：强制 enhanced parity 主图区全部使用 hexbin。
- `dataset` 模式若希望输出 Cohesive Energy 图，可将 `"enable_cohesive_energy": true` 并提供合理 `ref_energies`（或开启 `allow_ref_energy_lstsq_completion`）。
- 新增图谱会额外输出：
  - `dist_<quantity>_overlay.png`（Pred/True 叠加分布）
  - `dist_<quantity>_with_error.png`（主分布 + 小幅 error 分布）
  - `parity_<quantity>_enhanced.png`（含边缘分布的增强 parity）
- 图中统计信息默认仅显示 `count/mean/std/min/max`，不再显示分位数。
- `Error Distribution` 与 `parity_*_enhanced.png` 默认不显示统计信息框。
- 单变量分布图默认不显示 `All Data` 图例；dataset 元素占比/存在性使用多色饼图。
- quantity-aware 默认下，Force / Virial 的 hexbin enhanced parity 会在右侧信息栏同时展示 Error Density 与 colorbar，colorbar 表示每个 hexbin 中样本数量。

### 5.6 Labeling

```json
{
  "input_data_path": "./candidate_dpdata",
  "work_dir": "./labeling_work",
  "pp_dir": "./abacus_pp",
  "orb_dir": "./abacus_orb",
  "pp_map": { "Fe": "Fe.upf", "C": "C.upf", "O": "O.upf", "H": "H.upf" },
  "orb_map": { "Fe": "Fe.orb", "C": "C.orb", "O": "O.orb", "H": "H.orb" },
  "submission": { "backend": "slurm" }
}
```

## 6. 异常处理

- 配置校验失败：对照 `/docs/reference/validation.md` 的跨字段依赖与范围约束
- 输入路径不存在：检查相对路径是否以 config 所在目录为基准
- 历史字段不兼容：优先以 `sampler_type + direct_* / step*_*` 的新参数体系为准

## 7. 变更记录

- 2026-03-30：Analysis 配置新增 `enhanced_parity_renderer`，并补充 quantity-aware parity 渲染策略说明。
- 2026-03-16：Analysis 最小配置增加 `data_path`、`enable_cohesive_energy`、`allow_ref_energy_lstsq_completion`，并补充 Cohesive Energy 配置建议。
- 2026-03-11：配置字段权威入口改为 API Reference，并补充 Labeling 最小配置示例。
- 2026-02-18：补齐路径解析、Submission 结构与最小配置示例。
