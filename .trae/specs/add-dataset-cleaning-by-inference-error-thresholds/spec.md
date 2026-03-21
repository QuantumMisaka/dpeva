# Dataset Cleaning by Inference Error Thresholds Spec

## Why
当前 DP-EVA 在 Collection 阶段主要基于 UQ 指标进行筛选，但缺少“基于已标注数据与推理误差结果进行规则化清洗”的能力。对于含 label 的数据集，用户希望按可解释阈值直接剔除预测误差过大的结构，以便得到更稳定、更高质量的训练数据子集。

## What Changes
- 新增一个“基于推理误差阈值的数据清洗”能力：输入含 label 数据集与对应推理结果，输出清洗后的数据集。
- 支持三类可选阈值：能量预测差值、原子受力最大预测差值、晶格应力最大预测差值。
- 采用“阈值可选生效”策略：某指标阈值未配置时，不参与清洗判定。
- 清洗判定为“任一启用指标超阈值即剔除该结构”；全部启用指标均不超阈值时保留。
- 产出清洗统计：总结构数、保留数、剔除数、各指标触发剔除计数与阈值配置回显。
- 增加单元测试，覆盖阈值组合、缺失阈值、缺失结果文件、帧对齐异常等关键路径。
- 提供独立 CLI 子命令、独立 Config 模型、独立 Workflow 与 recipes 入口，避免影响现有 `collect` 主链路。

## Impact
- Affected specs: 数据清洗能力、配置模型、CLI 调用链、workflow 组织、recipes 示例、测试覆盖
- Affected code:
  - `src/dpeva/cli.py`
  - `src/dpeva/config.py`
  - `src/dpeva/constants.py`
  - `src/dpeva/workflows/data_cleaning.py`（新增）
  - `src/dpeva/workflows/__init__.py`
  - `src/dpeva/io/dataproc.py`
  - `src/dpeva/io/collection.py` 或 `src/dpeva/io/data_cleaning.py`（新增）
  - `examples/recipes/data_cleaning/*`（新增）
  - `tests/unit/**`

## Design Details
### CLI 设计
- 新增子命令：`dpeva clean <config.json>`，与 `train/infer/collect/analysis` 保持一致的“单 config 位置参数”风格。
- 在 `src/dpeva/cli.py` 增加：
  - `handle_clean(args)`：加载并解析配置后，实例化并运行清洗 workflow。
  - `p_clean = subparsers.add_parser("clean", ...)`：绑定 `validate_config_path` 与 `set_defaults(func=handle_clean)`。
- 失败策略对齐现有 CLI：参数错误抛 `CLIUserInputError`，运行期异常走统一 `logging.error(..., exc_info=True)` 并 `exit(1)`。

### Config 与常量设计
- 在 `src/dpeva/config.py` 新增 `DataCleaningConfig(BaseWorkflowConfig)`，字段分组：
  - 输入：`dataset_dir`, `result_dir`, `results_prefix`
  - 输出：`output_dir`
  - 阈值：`energy_diff_threshold`, `force_max_diff_threshold`, `stress_max_diff_threshold`
  - 运行控制：`strict_alignment`（默认开启）
- 阈值字段使用 `Optional[float]`：
  - `None` 表示该指标未启用；
  - 若提供值，要求 `>= 0`。
- 在 `src/dpeva/constants.py` 新增默认值常量：
  - `DEFAULT_CLEAN_OUTPUT_DIR = "cleaned_dpdata"`
  - `DEFAULT_CLEAN_RESULTS_PREFIX = DEFAULT_RESULTS_PREFIX`
  - `DEFAULT_CLEAN_STRICT_ALIGNMENT = True`
  - 三个阈值默认值设为 `None`（通过 config 默认体现，不用魔法数）。
- 文档化要求：所有新字段必须写明单位与语义
  - 能量阈值：eV/atom（与 `results.e_peratom.out` 对齐）
  - 受力阈值：eV/A（逐帧最大原子力误差）
  - 应力阈值：与 virial 输出一致的约定单位，文档显式注明来源文件（`v_peratom.out` 或 `v.out`）

### 模块与 Workflow 设计
- 新增独立 workflow 文件：`src/dpeva/workflows/data_cleaning.py`，主类 `DataCleaningWorkflow`。
- workflow 分层遵循现有模式（Config + IO + 业务判定）：
  - `DataCleaningWorkflow.run()`：编排流程
  - `_load_predictions()`：读取推理结果
  - `_compute_frame_metrics()`：构建每帧三类误差指标
  - `_build_keep_mask()`：执行阈值判定
  - `_export_clean_dataset()`：导出保留结构
  - `_write_cleaning_summary()`：输出统计
- IO 设计二选一，优先可复用：
  - 复用 `CollectionIOManager.export_dpdata` 的帧筛选导出逻辑并抽象通用接口；
  - 或新增 `src/dpeva/io/data_cleaning.py`，封装帧映射、掩码切片、导出与统计写出。
- 对齐校验设计：
  - 默认严格对齐：系统名 + 帧号双键一致才允许清洗；
  - 任一 mismatch 直接 fail-fast，报错包含 system/frame 上下文；
  - 不允许静默跳过或隐式重排。

### 用户案例与 recipes 入口
- 新增目录：`examples/recipes/data_cleaning/`
- 新增示例：
  - `config_clean_all_thresholds.json`：三阈值全启用
  - `config_clean_force_only.json`：仅力阈值启用
  - `config_clean_passthrough.json`：三阈值均不启用（旁路）
- 示例路径使用当前用户数据作为参考模板：
  - `dataset_dir`: `test/s-20w-cosine/sampled_dpdata`
  - `result_dir`: `test/s-20w-cosine/test-50k-trn`
- 在 `examples/recipes/README.md` 增加 “Data Cleaning (`clean`)” 节，提供最小命令示例。

## ADDED Requirements
### Requirement: 提供独立 CLI 清洗入口
系统 SHALL 提供 `dpeva clean` 子命令，支持通过 JSON 配置执行清洗流程。

#### Scenario: 命令行触发清洗
- **WHEN** 用户执行 `dpeva clean <config.json>`
- **THEN** 系统按配置加载数据集与推理结果并运行清洗
- **AND** 输出清洗统计与导出目录

### Requirement: 支持基于推理误差阈值清洗含 label 数据集
系统 SHALL 支持读取含 label 的数据集与推理测试结果，并按阈值规则对结构进行逐帧过滤后导出清洗结果。

#### Scenario: 三类阈值同时启用
- **GIVEN** 用户提供数据集路径与推理结果路径，并配置能量、受力最大差值、应力最大差值阈值
- **WHEN** 系统执行清洗
- **THEN** 对每一帧计算并比对三类误差指标
- **AND** 任一指标超阈值则剔除该帧
- **AND** 仅保留全部启用指标均满足阈值的帧

#### Scenario: 部分阈值未配置
- **GIVEN** 用户只配置了部分阈值（例如仅配置受力阈值）
- **WHEN** 系统执行清洗
- **THEN** 未配置阈值的指标不参与判定
- **AND** 判定逻辑仅基于已启用指标执行

#### Scenario: 三类阈值均未配置
- **GIVEN** 用户未配置任何清洗阈值
- **WHEN** 系统执行清洗
- **THEN** 系统不执行误差驱动清洗
- **AND** 返回显式提示，说明无指标启用并保持数据不变

### Requirement: 推理结果与数据集帧必须可对齐
系统 SHALL 在清洗前验证推理结果与数据集的帧级对齐关系，避免错配过滤。

#### Scenario: 对齐成功
- **WHEN** 推理结果帧数、帧顺序与数据集可一一映射
- **THEN** 系统按对应关系执行过滤并导出结果

#### Scenario: 对齐失败
- **WHEN** 推理结果缺帧、重复帧或无法映射到数据集结构
- **THEN** 系统显式报错并终止清洗
- **AND** 错误信息包含无法对齐的系统/帧上下文

### Requirement: 输出清洗结果与可审计统计
系统 SHALL 输出清洗后数据集与可审计统计信息，确保清洗可复现、可追踪。

#### Scenario: 清洗完成
- **WHEN** 清洗流程结束
- **THEN** 输出目录包含保留结构数据
- **AND** 日志/统计文件包含总数、保留数、剔除数、按指标触发次数与阈值回显

### Requirement: 提供可复用的配置默认值与单位语义
系统 SHALL 在配置与常量中显式定义清洗默认值，并在字段描述中标注单位、阈值语义与未配置行为。

#### Scenario: 用户仅提供最小配置
- **GIVEN** 用户仅提供必填输入路径与输出路径
- **WHEN** 系统加载配置
- **THEN** 自动应用默认 `results_prefix` 与对齐策略
- **AND** 三类阈值保持未启用状态

### Requirement: 提供用户可直接运行的 recipes 示例
系统 SHALL 在 `examples/recipes` 中提供 `clean` 功能的配置样例，覆盖常见阈值组合。

#### Scenario: 用户复制样例运行
- **WHEN** 用户复制并修改 `examples/recipes/data_cleaning/*.json`
- **THEN** 可直接通过 `dpeva clean <config>` 运行
- **AND** 能快速对照三种阈值启用模式

## MODIFIED Requirements
### Requirement: Workflow 扩展方式
系统 SHALL 优先以独立 workflow + 独立 CLI 子命令方式扩展数据清洗能力，不将其强耦合进 `collect` 的 UQ+DIRECT 主路径。未使用 `clean` 命令时，现有行为保持完全兼容。

## REMOVED Requirements
- None
