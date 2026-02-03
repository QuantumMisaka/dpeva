# DP-EVA 输入参数文档

## 1. 概述
本文档详细描述了 DP-EVA 系统中各个 Workflow 的输入参数。所有配置均基于 Pydantic V2 模型定义，支持严格的类型校验和自动默认值填充。

## 2. 通用配置 (Common Configuration)
适用于所有 Workflow 的基础配置参数。

| 参数名 (Identifier) | 类型 | 默认值 | 说明 | 约束/验证 |
| :--- | :--- | :--- | :--- | :--- |
| `work_dir` | Path | `CWD` (当前目录) | 工作目录，用于存放运行时生成的临时文件和日志。 | 必须是有效路径。 |
| `omp_threads` | int \| "auto" | `4` | OpenMP 并行线程数。`auto` 自动使用所有核心。 | `>= 1` (若为 int) |
| `submission` | Object | (见下文) | 任务提交配置，包含后端选择和资源调度参数。 | - |

### 2.1 任务提交配置 (SubmissionConfig)
嵌套在 `submission` 字段下的配置。

| 参数名 | 类型 | 默认值 | 说明 | 约束/验证 |
| :--- | :--- | :--- | :--- | :--- |
| `backend` | string | `"local"` | 执行后端类型。`local` 为本地直接执行，`slurm` 为提交到集群。 | 枚举: `["local", "slurm"]` |
| `slurm_config` | dict | `{}` | Slurm 调度器的具体参数 (见下表)。 | 键值对字典 |
| `env_setup` | string \| list[str] | `""` | 运行前需要执行的环境初始化命令 (如 `module load`, `source env`)。 | 列表会自动拼接为多行字符串。 |

#### Slurm 参数详情 (slurm_config)
以下参数仅当 `backend="slurm"` 时有效，且需配置在 `slurm_config` 字典中：

| 参数名 | 类型 | 默认值 | Slurm Flag | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `partition` | string | `None` | `-p` | 分区名称。若为 `None`，则省略 `-p` 标志，使用集群默认分区。 |
| `nodes` | int | `1` | `-N` | 申请节点数。 |
| `ntasks` | int | `1` | `-n` | 总任务数 (Processes)。 |
| `gpus_per_node` | int | `None` | `--gpus-per-node` | 每节点 GPU 数。若设置，DP-EVA 会尝试适配多卡命令。 |
| `cpus_per_task` | int | `None` | `--cpus-per-task` | 每任务 CPU 核心数 (OpenMP 线程数)。 |
| `walltime` | string | `"24:00:00"` | `-t` | 作业时间限制。格式: `HH:MM:SS` 等。 |
| `qos` | string | `None` | `--qos` | 服务质量 (QoS)。 |
| `nodelist` | string | `None` | `-w` | 指定节点列表。 |
| `output_log` | string | `None` | `-o` | 标准输出日志文件名。`None` 使用默认格式。 |
| `error_log` | string | `None` | `-e` | 错误输出日志文件名。`None` 使用默认格式。 |
| `custom_headers` | list[str] | `""` | - | 自定义头部参数列表 (如 `["#SBATCH --exclusive"]`)。 |

---

## 3. 训练工作流 (Training Workflow)
对应配置类: `TrainingConfig`

| 参数名 | 类型 | 默认值 | 说明 | 约束/验证 |
| :--- | :--- | :--- | :--- | :--- |
| `base_model_path` | Path | **必填** | 基础模型/预训练模型路径。 | - |
| `num_models` | int | `4` | 训练的模型数量 (Ensemble Size)。 | `>= 3` (UQ要求) |
| `mode` | string | `"cont"` | 训练模式。`init` (从头初始化) 或 `cont` (继续训练)。 | 枚举: `["init", "cont"]` (Alias: `training_mode`) |
| `model_head` | string | **必填** | 微调的 Model Head 名称。 | (Alias: `finetune_head_name`) |
| `input_json_path` | Path | `"input.json"` | DeepMD-kit 训练输入参数文件路径。 | - |
| `training_data_path` | Path | `None` | 训练数据根目录。 | - |
| `seeds` | list[int] | `None` | 全局随机种子列表。 | - |
| `training_seeds` | list[int] | `None` | 训练专用随机种子列表。 | - |

---

## 4. 推理工作流 (Inference Workflow)
对应配置类: `InferenceConfig`

| 参数名 | 类型 | 默认值 | 说明 | 约束/验证 |
| :--- | :--- | :--- | :--- | :--- |
| `data_path` | Path | **必填** | 待推理的测试数据集路径。 | 必须存在。 |
| `model_head` | string | `None` | 使用的模型 Head 名称。若为 Frozen Model 可选。 | (Alias: `head`) |
| `results_prefix` | string | `"results"` | 输出结果文件的前缀 (如 `results.e.out`)。 | - |
| `task_name` | string | `"test"` | 任务子目录名称。 | - |

---

## 5. 特征生成工作流 (Feature Workflow)
对应配置类: `FeatureConfig`

| 参数名 | 类型 | 默认值 | 说明 | 约束/验证 |
| :--- | :--- | :--- | :--- | :--- |
| `data_path` | Path | **必填** | 输入数据集路径。 | 必须存在。 |
| `model_path` | Path | **必填** | 模型文件路径。 | (Alias: `modelpath`) |
| `model_head` | string | **必填** | 模型 Head 名称。 | (Alias: `head`) |
| `format` | string | `"deepmd/npy"` | 数据格式。 | - |
| `output_mode` | string | `"atomic"` | 输出模式。`atomic` (原子级) 或 `structural` (结构级)。 | 枚举: `["atomic", "structural"]` |
| `batch_size` | int | `32` | 推理批次大小。 | `> 0` |
| `savedir` | Path | `None` | 结果保存目录。 | 若未指定，自动根据模型和数据名生成。 |

---

## 6. 采集工作流 (Collection Workflow)
对应配置类: `CollectionConfig`

### 6.1 基础路径与项目信息
| 参数名 | 类型 | 默认值 | 说明 | 约束/验证 |
| :--- | :--- | :--- | :--- | :--- |
| `project` | string | `"./"` (当前目录) | 项目名称/标识符。 | - |
| `desc_dir` | Path | **必填** | 描述符 (Descriptor) 文件所在的目录。 | 必须存在。 |
| `testdata_dir` | Path | **必填** | 测试数据/候选池数据目录。 | 必须存在。 |
| `training_data_dir` | Path | `None` | (可选) 训练数据目录，用于联合采样 (Joint Sampling)。 | 若启用 Joint 模式则必填。 |
| `training_desc_dir` | Path | `None` | (可选) 训练数据描述符目录。 | - |
| `root_savedir` | Path | `"dpeva_uq_post"` | 结果保存的根目录。 | - |

### 6.2 UQ (不确定性量化) 参数
| 参数名 | 类型 | 默认值 | 说明 | 约束/验证 |
| :--- | :--- | :--- | :--- | :--- |
| `uq_select_scheme` | string | `"tangent_lo"` | UQ 选择策略/方案。 | 枚举: `["tangent_lo", "strict", "circle_lo", "crossline_lo", "loose"]` |
| `uq_trust_mode` | string | `"auto"` | 信任区域边界确定模式。`auto` 为自动计算，`manual` 为手动指定。 | 枚举: `["auto", "manual"]` |
| `uq_trust_ratio` | float | `0.33` | 全局信任比例 (用于自动计算边界)。 | `0.0 <= x <= 1.0` |
| `uq_trust_width` | float | `0.25` | 信任区域宽度 (用于自动计算或手动推导)。 | `> 0.0` |

#### 手动模式专用参数 (Manual Mode Overrides)
仅当 `uq_trust_mode="manual"` 时生效。

| 参数名 | 类型 | 默认值 | 说明 | 约束/验证 |
| :--- | :--- | :--- | :--- | :--- |
| `uq_qbc_trust_lo` | float | `None` | QbC (Query by Committee) 信任区域下界。 | Manual 模式下必填。 |
| `uq_qbc_trust_hi` | float | `None` | QbC 信任区域上界。 | 若未填，由 `lo + width` 自动计算。 |
| `uq_rnd_rescaled_trust_lo` | float | `None` | RND (Random Network Distillation) 信任区域下界。 | Manual 模式下必填。 |
| `uq_rnd_rescaled_trust_hi` | float | `None` | RND 信任区域上界。 | 若未填，由 `lo + width` 自动计算。 |

### 6.3 采样与测试参数
| 参数名 | 类型 | 默认值 | 说明 | 约束/验证 |
| :--- | :--- | :--- | :--- | :--- |
| `num_selection` | int | `100` | 每一轮采集的样本数量。 | `> 0` |
| `direct_k` | int | `1` | DIRECT 算法的 K 值参数。 | `>= 1` |
| `direct_thr_init` | float | `0.5` | DIRECT 算法的初始阈值。 | `>= 0.0` |
| `testing_dir` | string | `"test_results"` | 测试集子目录名称。 | - |
| `results_prefix` | string | `"results"` | 测试集结果文件名前缀 (Alias: `testing_head`)。 | - |
| `fig_dpi` | int | `300` | 可视化图片的 DPI 分辨率。 | - |
