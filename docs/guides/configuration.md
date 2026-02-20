# 配置编写指南（Configuration）

- Status: active
- Audience: Users / Developers
- Last-Updated: 2026-02-18

## 1. 目的与范围

本页说明“如何写配置、如何组织路径、如何最小化配置”。全量字段字典与校验规则请以 Reference 为准：

- ../reference/config-schema.md
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
  "result_dir": "0/test_val",
  "output_dir": "analysis",
  "type_map": ["Fe", "C", "O", "H"]
}
```

## 6. 异常处理

- 配置校验失败：对照 `/docs/reference/validation.md` 的跨字段依赖与范围约束
- 输入路径不存在：检查相对路径是否以 config 所在目录为基准
- 历史字段不兼容：优先以 `sampler_type + direct_* / step*_*` 的新参数体系为准

## 7. 变更记录

- 2026-02-18：补齐路径解析、Submission 结构与最小配置示例。
