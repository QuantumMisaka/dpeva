---
title: Document
status: active
audience: Developers
last-updated: 2026-09-04
owner: Docs Owner
---

# CLI 使用指南

- Status: active
- Audience: Users / Developers
- Applies-To: CLI 模式（推荐）
- Last-Updated: 2026-09-04

## 1. 目的与范围

本页说明 DP-EVA 的统一命令行入口、子命令职责、配置文件约定、输出与失败定位方式。

范围：

- `dpeva train / infer / feature / explore / collect / label / analysis / clean / doctor / eval-card`
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

### 3.2 工作流命令格式

```bash
dpeva <workflow> <config_path>
```

可选参数：

```bash
dpeva --no-banner <workflow> <config_path>
```

`feature` 与 `infer` 还支持运行证据选项：`--run-id ID` 固定本次运行身份，
`--resume` 恢复未完成的本地运行；已进入 `submitted` 的 Slurm 运行会在提交前拒绝
resume（调度器轮询/恢复不在本试点范围），不会创建新作业。使用带必填 `--reason`
的 `--force` 创建显式新尝试。`--resume` 与 `--force` 互斥；运行清单写入配置工作目录下的
`.dpeva/runs/<run-id>/run.json`。

除 `doctor` 外的工作流都要求提供 `<config_path>`；CLI 会在参数解析阶段对它执行统一前置校验（存在性、可读性、JSON 文件后缀）。`doctor` 是不需要配置文件的独立环境检查命令，格式见下节。

实现入口：`src/dpeva/cli.py`（基于 `argparse`）。

### 3.3 doctor（环境能力检查）

```bash
dpeva doctor
dpeva doctor --json
```

`doctor` 显式检查当前运行环境的 DeepMD 能力。默认输出每项检查的人类可读结果；`--json` 输出 schema 版本为 `1.0` 的机器可读报告。JSON 模式不会输出欢迎 banner，因此标准输出始终只有 JSON。

报告顶层 `status` 为 `ok` 时命令退出码为 `0`；任一必需检查不是 `ok` 时退出码为 `1`。
检查状态可能包括 `ok`、`missing`、`error`、`unknown`、`incompatible` 和
`unavailable`。每项可带 `required=false` 表示信息性能力（例如 CUDA/GPU 和可选
后端），其不可用不会阻止 CPU-safe 工作流。默认检查 DeepMD 版本以及 `test`、
`eval-desc`、`embed` CLI surface，并显式报告 dpdata、Torch/CUDA、GPU 可见性和
可选后端。

## 4. 子命令职责、输入输出与配置

除 `doctor` 外，所有工作流子命令的第一个参数均为配置 JSON 路径；`doctor` 不接收配置路径。配置字段的权威查表入口：

- API Reference（Sphinx 生成的配置字段文档）
- ../reference/validation.md

### 4.9 eval-card（候选评估卡片）

`eval-card` 将已存在的模型引用、数据谱系和评测证据组装为一个机器可读的候选交接卡片；
它不会启动评测、复制活动中的 FT2DP 任务清单或推断科学排名。六个固定维度始终存在，
未提供证据标记为 `not-run`，配置了但无法读取或校验的证据标记为 `failed`，并保留证据路径。

`examples/recipes/evaluation/config_eval_card.json` 是配置模板，不是可直接执行的
candidate artifact。填入真实 model-ref、dataset manifest 和 metric 文件后，再运行：

```bash
dpeva eval-card path/to/filled-eval-card.json
```

配置文件中的相对路径均相对该配置文件所在目录解析，包括 `output_path`、
`model_ref_path`、六个可选 metric 路径和 `dataset_manifest_paths`。输出卡片以原子方式发布，
不会覆盖已存在的目标文件；如需生成新的候选卡片，请使用新的输出路径。配置模板见
`examples/recipes/evaluation/config_eval_card.json`。
卡片中的本地 `model_ref`、`dataset_refs`、metric `evidence_ref` 和本地下游反馈引用均是
相对于卡片目录的 POSIX 逻辑引用；将包含这些目标的 candidate package 整体搬迁后仍可解析。
相对路径只解决可移植定位，不单独提供不可变性；不可变性由引用目标自身的 no-overwrite、校验和
及其验证契约承担。HTTP/其他下游 URI 保持原样。

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

### 4.2 infer（并行推理）

- 输入
  - `InferenceConfig`
  - `data_path`（候选 dpdata，可无标注）
  - `work_dir`（包含 `0..N-1/` 模型目录）
- 输出
  - `work_dir/<i>/<task_name>/results.*.out`
  - `work_dir/<i>/<task_name>/test_job.out`
- 自动分析规则
  - 仅当 `submission.backend=local` 且 `auto_analysis=true` 时触发链式分析
  - `slurm` 场景建议单独执行 `dpeva analysis`

示例配置：`examples/recipes/inference/config_infer.json`

例如指定可复查的运行身份：

```bash
dpeva infer config.json --run-id infer-20260904-a1b2c3
```

### 4.3 feature（描述符生成）

- 输入
  - `FeatureConfig`
  - `data_path`（dpdata：训练集或候选池）
  - `model_path`（用于 `dp eval-desc` 的模型）
- 输出
  - `savedir/`（描述符目录）
  - `savedir/eval_desc.log`（常用监控锚点）

示例配置：`examples/recipes/feature_generation/config_feature.json`

`feature` 同样支持上述 `--run-id`、`--resume`、`--force` 和 `--reason` 选项。

### 4.4 explore（轨迹探索，可选）

- 输入
  - `ExplorationConfig`
  - `backend_config_path`（后端原生配置，例如 ATST YAML）
  - 可选 `result_structure_paths`（后端完成后需要回收的结构文件）
- 功能
  - 通过可选 backend 启动轨迹探索。
  - v0.8.0 首个 backend 为 `atst-tools`，支持 `workflow_type=md|relax`。
  - `atst-tools` 不属于核心依赖；需要时安装 `dpeva[explore]`。
  - 成功或失败都会在 `work_dir/dpeva_exploration_result.json` 写入 manifest。
  - `input_structure_paths` 会被快照到 `work_dir/dpeva_inputs/`，`result_structure_paths` 用于声明 backend 运行后需要回收的结构文件。
- 输出
  - 由后端原生配置决定；DPEVA 只收集显式配置的结果结构路径。

### 4.5 collect（UQ + Filtering + Sampling + Export）

`collect` 在 Slurm 后端采用自调用方式提交 worker（配置路径会被传入，避免“冻结配置”写盘）。

- 输入（核心）
  - `CollectionConfig.desc_dir`（候选描述符）
  - `CollectionConfig.testdata_dir`（候选 dpdata）
  - `CollectionConfig.testing_dir`（推理输出目录名，如 `test_val`）
  - 采样参数（`sampler_type` 与 direct/2-direct 参数组）
- 输出
  - `root_savedir/dataframe/*.csv`，关键结果通常包括 `df_uq_desc.csv`、`df_uq.csv`、`final_df.csv`
  - `root_savedir/view/*.png`
  - `root_savedir/dpdata/sampled_dpdata/*` 与 `root_savedir/dpdata/other_dpdata/*`

示例配置：`examples/recipes/collection/config_collect_normal.json`、`examples/recipes/collection/config_collect_joint.json`

### 4.6 label（标注）

- 输入
  - `LabelingConfig`
- 命令参数
  - `--stage {all,prepare,execute,extract,postprocess}`（默认 `all`）
- 功能
  - 执行主动学习中的标注工作流 (LabelingWorkflow)
  - 将 `dpdata` 格式的候选结构转化为 DFT (ABACUS) 计算任务
  - 支持自动 K 点生成、任务打包 (Packing) 和 Slurm 并行提交
  - 自动处理任务失败重试与结果回收
- 输出
  - `work_dir/outputs/cleaned`（主清洗结果）
  - `work_dir/outputs/anomalies`（异常或被筛出结果）
  - 当 `integration_enabled=true` 时，额外输出 `merged_training_data_path`（未指定时默认 `work_dir/outputs/merged_training_data`）
  - 整合统计文件：`<merged_training_data_path>/integration_summary.json`
  - 数据谱系清单：`<merged_training_data_path>/dataset-manifest.json` 是当前代的兼容指针，
    并保留不可变的 `dataset-manifest-<generation>.json`；统计文件中的
    `dataset_manifest_path` 是相对于输出目录的不可变清单引用，同时记录 generation 和
    SHA-256。清单记录父数据集、合并后的帧/体系数、去重移除数、type map 和逻辑来源引用；
    当前整合不会伪造不可解析的 parent manifest ref，仅保留逻辑来源标签；
    未声明来源、未解释的重复/交集或 type map 冲突会在下游交接前失败；显式去重会保存
    overlap/removal evidence 与机器可读 `validation_result`（含 rule version）。
  - 发布采用 sibling staging 目录与 Linux `renameat2(RENAME_NOREPLACE)`，在进程可见范围内
    原子且不覆盖并发产生的目标目录；不支持该原语的平台直接失败，不回退到普通 rename。
    该发布语义不提供 crash 后目录持久化或递归 fsync dpdata 树的保证；JSON 文件可能进行
    文件级 flush，不能外推为整个 bundle 的 durability 保证。

### 4.7 analysis（双模式分析）

- `model_test` 模式（默认）
  - 输入：`AnalysisConfig.result_dir`（例如 `0/test_val`）
  - 建议：若启用 Cohesive Energy，配置 `AnalysisConfig.data_path` 指向原始测试数据集，避免仅依赖文件名推断组分
  - 输出：`AnalysisConfig.output_dir`（指标、误差统计、基础图、增强图）
- `dataset` 模式
  - 输入：`AnalysisConfig.dataset_dir`
  - 输出：`dataset_stats.json`、`dataset_frame_summary.csv`、元素占比/覆盖图、分布图（可选 cohesive）

示例配置：`examples/recipes/analysis/config_analysis.json`

### 4.8 clean（基于推理误差阈值的数据清洗）

- 输入
  - `DataCleaningConfig.dataset_dir`（带标注数据集）
  - `DataCleaningConfig.result_dir`（与数据集对应的推理结果目录）
  - 可选阈值：`energy_diff_threshold`、`force_max_diff_threshold`、`stress_max_diff_threshold`
- 输出
  - `output_dir/cleaned_dpdata`
  - `output_dir/filtered_out_dpdata`
  - `output_dir/frame_metrics.csv`
  - `output_dir/cleaning_summary.json`

示例配置：`examples/recipes/data_cleaning/config_clean_all_thresholds.json`

关键参数：

- `enable_cohesive_energy`：开启/关闭 Cohesive Energy 统计与作图
- `allow_ref_energy_lstsq_completion`：当 `ref_energies` 不完整时，是否允许最小二乘补全缺失元素参考能
- `results_prefix`：必须与推理阶段 `InferenceConfig.results_prefix` 一致
- `plot_level`：`basic` 生成日常分析核心图（parity、单变量分布、error distribution、dataset 元素统计）；`full` 在此基础上增加论文/补充图（enhanced parity、overlay、with_error，以及条件满足时的 dataset cohesive energy 分布）
- 图谱文件命名：`dist_<quantity>_overlay.png`、`dist_<quantity>_with_error.png`、`parity_<quantity>_enhanced.png`
- 统计可视化约定：Analysis 分布图统一采用轻量 histogram + KDE 模板；统计项仅 `count/mean/std/min/max`；Error Distribution 与增强 Parity 不展示统计框。
- 标签约定：非 Pred/True 对比图默认不展示 `All Data` 图例；dataset 元素图使用多色饼图。
- 推荐取舍：`parity_*_enhanced` 与关键 `dist_*_with_error` 优先作为论文主文候选；`dist_*_overlay` 与 `dist_dataset_cohesive_energy` 更适合作为补充图；基础 `parity_*`、单变量 `dist_*`、`error_dist_*` 与 dataset 元素统计图保留为日常诊断图。

## 5. 完成标记与链式编排

DP-EVA 在多数核心工作流及其实际执行日志中会输出统一标记：

```text
DPEVA_TAG: WORKFLOW_FINISHED
```

`WORKFLOW_FINISHED` is written only after the guarded command returns zero and all declared artifacts pass validation. Consumers MUST require both a successful process/job state and the marker; the marker alone is not proof of success. `sbatch` returning a JobID establishes only `submitted`, not `finished`.

说明：

- `train`、`collect`、`label` 等主流程会在成功结束时输出该标记。
- `infer` 在 `auto_analysis=true` 且本地链式分析完成时会由分析阶段输出该标记；Slurm 场景更稳妥的推进锚点仍是各模型 `test_job.out` 完成后再显式执行 `dpeva analysis`。

## 6. 异常处理与退出码

- **退出码契约**
  - **正常执行**：0。
  - **参数解析失败**：2（例如 config 文件不存在、不可读、路径不是文件，或参数形态错误）。
  - **运行期失败**：1（配置内容不合法、业务逻辑失败、外部命令失败等）。
- **doctor 环境检查**：报告 `status=ok` 时为 0，否则为 1；`doctor --json` 的标准输出仅包含 JSON 报告。
- **局部完成**：本地 infer 只要有一个模型成功、另一个失败，就写入 `partial`
  清单并以退出码 `1` 返回；所有模型失败则为 `failed`。输出缺失/为空时清单
  顶层类别为 `ARTIFACT`，命令异常为 `EXECUTION`；混合失败类别确定性归为
  `EXECUTION`，但每个 `JobRecord` 保留具体类别。Slurm 的 `sbatch` 回执只表示
  `submitted`，不会伪造 `finished`。Slurm 多模型提交若部分 JobID 成功、部分提交失败，
  父清单仍保持 `submitted`（保留成功 JobID 与失败子记录），但工作流立即以退出码 `1`
  报告提交不完整；只有全部提交失败才将父清单记为 `failed`。
  - 注意：CLI 对用户输入类错误优先给出可操作提示，避免无意义堆栈噪音；内部异常仍会保留堆栈用于排障。

- 常见异常类型
  - 参数级配置文件错误（argparse）：`config_path` 不存在/不可读/非 JSON 文件。
  - 配置内容校验失败（`ValidationError`）：字段缺失/类型不匹配。
  - 路径/文件错误（`FileNotFoundError` / `WorkflowError`）：数据目录、模型文件未找到。
  - 运行时错误（`RuntimeError` / `WorkflowError`）：DeepMD 版本不兼容、外部命令执行失败。

- 常见误用示例

```bash
# 错误：把 stage 词放在 config 位置
dpeva label prepare

# 正确：显式提供 config，并通过 --stage 指定阶段
dpeva label config.json --stage prepare
```

排障入口：

- ./troubleshooting.md

## 7. 变更记录

- 2026-09-04：新增 `doctor` 环境能力检查及 `--json` 稳定 JSON 输出契约。
- 2026-06-11：补充 `dpeva explore` manifest、输入结构快照和结果结构回收契约。
- 2026-06-10：新增 `dpeva explore` 可选轨迹探索入口，记录 `atst-tools` backend 的 md/relax 边界。
- 2026-03-03：更新退出码契约说明，明确 `WorkflowError` 会导致退出码 1。
- 2026-03-12：新增 config 路径前置校验说明，补充 label `--stage` 参数和参数解析失败退出码 2。
- 2026-03-11：更新配置权威入口为 API Reference，并同步 infer 日志文件名为 `test_job.out`。
- 2026-03-08：补充 analysis 双模式与 labeling integration 输出说明。
- 2026-02-18：补齐子命令 I/O、完成标记与退出码说明，并建立与 recipes/api 的权威链接。
- 2026-04-05：补充 `clean` 子命令，修正 collection recipe 文件名，并对齐 labeling 输出与 infer 完成标记语义。
