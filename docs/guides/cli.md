# CLI 使用指南

- Status: active
- Audience: Users / Developers
- Applies-To: CLI 模式（推荐）
- Last-Updated: 2026-02-18

## 1. 目的与范围

本页说明 DP-EVA 的统一命令行入口、子命令职责、配置文件约定、输出与失败定位方式。

范围：

- `dpeva train / infer / feature / collect / analysis`
- `--no-banner`

## 2. 相关方

- 使用者：通过 CLI 运行工作流
- 开发者：维护 CLI 接口、配置解析与错误信息
- 平台维护：为 Slurm/DeepMD 环境提供支持

## 3. 总体用法

### 3.1 帮助信息

```bash
dpeva --help
dpeva train --help
```

### 3.2 通用命令格式

```bash
dpeva <subcommand> <config_path>
```

可选参数：

```bash
dpeva --no-banner <subcommand> <config_path>
```

实现入口：`src/dpeva/cli.py`（基于 `argparse`）。

## 4. 子命令职责、输入输出与配置

所有子命令的第一个参数均为配置 JSON 路径。配置字段的权威查表入口：

- ../reference/config-schema.md
- ../reference/validation.md

### 4.1 train（并行微调训练）

- 输入
  - `TrainingConfig`（训练配置 JSON）
  - `input_json_path`（DeepMD 训练 input.json）
  - `training_data_path`（训练 dpdata，需能量/力等标注）
  - `base_model_path`（基础模型）
- 输出
  - `work_dir/0..N-1/`（每个子目录一个模型）
  - `work_dir/<i>/train.out`（Slurm 时常用监控锚点）

示例配置：`examples/recipes/training/config_train.json`

### 4.2 infer（并行推理与误差分析）

- 输入
  - `InferenceConfig`
  - `data_path`（候选 dpdata，可无标注）
  - `work_dir`（包含 `0..N-1/` 模型目录）
- 输出
  - `work_dir/<i>/<task_name>/results.*.out`
  - `work_dir/<i>/<task_name>/test_job.log`

示例配置：`examples/recipes/inference/config_infer.json`

### 4.3 feature（描述符生成）

- 输入
  - `FeatureConfig`
  - `data_path`（dpdata：训练集或候选池）
  - `model_path`（用于 `dp eval-desc` 的模型）
- 输出
  - `savedir/`（描述符目录）
  - `savedir/eval_desc.log`（常用监控锚点）

示例配置：`examples/recipes/feature_generation/config_feature.json`

### 4.4 collect（UQ + Filtering + Sampling + Export）

`collect` 在 Slurm 后端采用自调用方式提交 worker（配置路径会被传入，避免“冻结配置”写盘）。

- 输入（核心）
  - `CollectionConfig.desc_dir`（候选描述符）
  - `CollectionConfig.testdata_dir`（候选 dpdata）
  - `CollectionConfig.testing_dir`（推理输出目录名，如 `test_val`）
  - 采样参数（`sampler_type` 与 direct/2-direct 参数组）
- 输出
  - `root_savedir/dataframe/*.csv`
  - `root_savedir/view/*.png`
  - `root_savedir/dpdata/*`（导出 dpdata）

示例配置：`examples/recipes/collection/config_multi_normal.json`、`examples/recipes/collection/config_multi_joint.json`

### 4.5 analysis（推理结果统计）

- 输入：`AnalysisConfig.result_dir`（例如 `0/test_val`）
- 输出：`AnalysisConfig.output_dir`

示例配置：`examples/recipes/analysis/config.json`

## 5. 完成标记与链式编排

DP-EVA 在核心工作流成功结束时会输出统一标记：

```text
DPEVA_TAG: WORKFLOW_FINISHED
```

建议外部编排器通过监控日志出现该标记推进下一步（尤其是 Slurm 场景）。

## 6. 异常处理与退出码

- 退出码
  - 正常执行：0
  - 捕获异常退出：1（CLI 会打印堆栈）
- 常见异常类型
  - 配置校验失败（字段缺失/类型不匹配/跨字段依赖不满足）
  - 输入路径不存在（数据目录、模型文件等）
  - 外部命令失败（`dp` 命令不可用、作业脚本执行失败）

排障入口：

- ./troubleshooting.md

## 7. 变更记录

- 2026-02-18：补齐子命令 I/O、完成标记与退出码说明，并建立与 recipes/api 的权威链接。
