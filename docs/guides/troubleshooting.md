---
title: Document
status: active
audience: Developers
last-updated: 2026-10-22
owner: Docs Owner
---

# Troubleshooting（常见问题与排查）

- Status: active
- Audience: Users / Developers / Infra
- Last-Updated: 2026-06-10

## 1. 目的与范围

本页用于快速定位 DP-EVA 在 Train/Infer/Feature/Collect/Analysis 过程中出现的问题，给出“先看哪里、再排什么”的最短路径。

## 2. 相关方

- 使用者：运行工作流、收集日志与复现问题
- 开发者：维护错误提示、输入校验与日志锚点
- 平台维护：处理 Slurm 队列与 DeepMD 环境问题

## 3. 总体排查顺序（强制建议）

1. 确认当前工作目录与配置文件路径（相对路径解析是否正确）
2. 定位对应工作流的日志文件（Train/Infer/Feature/Collect）
3. 判断是否出现完成标记（见 3.1）
4. 若失败来自外部命令（`dp`/Slurm），先定位外部命令报错，再回到配置与输入数据

### 3.1 任务完成锚点

DP-EVA 以统一完成标记作为应用层完成锚点：

```text
DPEVA_TAG: WORKFLOW_FINISHED
```

没有该标记时，优先认为任务“未成功结束”，需要继续追踪日志中的异常堆栈或外部命令错误输出。

### 3.2 选择重跑方式

先保存错误日志，确认旧进程或 Slurm 作业已结束，再决定是否重用输出目录。
`--force` 或移动目录都不会取消仍在运行的作业。Quickstart 的默认重新开始方式见
{doc}`入门指南 </guides/quickstart>`。

| 场景 | 当前支持的做法 |
| --- | --- |
| 想从头重做小型示例 | 备份整个示例工作目录，从第一阶段重新运行 |
| Feature/Infer 中断，清单仍为未终结且非 `submitted` 状态 | 配置、输入、运行代码证据一致时，可指定原 `--run-id ID --resume`；这会创建新尝试，不保证跳过已成功的子任务 |
| Feature/Infer 已记录 `failed` 或 `finished` | 不能 resume；排除原因后，用原 `--run-id ID --force --reason "原因"` 显式创建新尝试，或使用新工作目录 |
| Slurm 清单为 `submitted` | 先查调度器状态与作业日志，不能把 CLI 返回当作计算完成；当前不支持 scheduler resume，重新提交前确认旧作业结束 |
| Train 或 Collect 失败 | 没有统一的 `--resume/--force` 接口；先修复输入或环境，隔离旧输出后重新执行相应工作流。Train 不能据此声称从 checkpoint 续训 |

保留成功阶段时，只复用已确认成功且与当前配置、输入匹配的产物。沿
`feature → train → infer → collect` 的示例操作时，实际依赖还需分别核对：
更换候选池要重建对应描述符和推理结果；更换训练数据或训练模型后应重新训练、
推理，再采集。更换 Feature 基座模型则需要重建描述符。不要仅凭目录存在判断
上游已成功，也不要从未结束的 Slurm 作业输出中启动下游阶段。

## 4. 环境类问题

### 4.1 `dp: command not found`

症状：

- 作业日志出现 `dp: command not found`

原因：

- DeepMD-kit 环境未加载或未写入 `PATH`

处理：

- Local：在当前 shell 环境中先 `dp --version` 验证
- Slurm：将环境加载写入 `submission.env_setup`（不要依赖交互式 shell）

### 4.2 DeepMD 版本不匹配/行为差异

症状：

- `dp` 命令存在但输出/参数行为与预期不一致

处理：

- 记录 `dp --version` 输出与运行环境信息（CUDA/驱动）
- 若为集群，建议固定模块版本或 Conda 环境版本

## 5. 数据类问题（dpdata / Multi DataPool）

### 5.1 dpdata 目录结构不符合预期

症状：

- Feature/Infer/Collect 报“找不到系统/空数据/系统数为 0”

处理：

- 检查数据目录是否包含 `type.raw/type_map.raw` 与 `set.000/*.npy`
- Multi DataPool 场景确认目录层级为：`Pool/System/set.000`

### 5.2 `type_map` 不一致导致元素映射错误

症状：

- 推理结果异常（能量/力量级明显不对）
- 部分 system 加载失败

处理：

- 确认训练集与候选池在 `type_map.raw` 的元素顺序一致
- Analysis 的 `type_map` 必须与训练/数据一致

### 5.3 `deepmd/lmdb` 输入无法加载或被拒绝

症状：

- `load_systems` 报 `DatasetLoadError`，提示 "Reading 'deepmd/lmdb' requires dpdata>=1.1"
- Feature 的 `eval-desc`/`embed` 在提交前报 "cannot consume a DeepMD LMDB dataset"
- `dpeva infer` 在解析旧 LMDB 明细时可能报缺少 `frame_nlocs`/`frame_system_ids`
- Clean 在 LMDB 缺少 `frame_system_ids`/`system_info` 时会拒绝猜测原始 system 划分

处理：

- 先跑 `dpeva doctor --json`：`dpdata.lmdb` 检查为 `ok` 表示当前环境可读 LMDB；
  为 `unavailable` 时升级到 `dpdata>=1.1`（预置环境需重建或补装）
- 数据路径按 deepmd-kit 的判定规则识别：目录名以 `.lmdb` 结尾，或目录内含 `data.mdb`
- LMDB 不保存目录级体系身份，读回结果按**组成分组**；需要逐 system 统计或
  `target_systems=` 筛选时，改用同一数据集的 `deepmd/npy`/`npy/mixed` 副本
- `dp eval-desc`/`dp embed` 上游不支持 LMDB；Feature 工作流请指向 npy/mixed 副本，
  或使用进程内 `DescriptorGenerator` 路径。上游跟踪 issue：
  [deepmodeling/deepmd-kit#6034](https://github.com/deepmodeling/deepmd-kit/issues/6034)
- 训练（`dp train`）支持 LMDB，但 `training_data.systems` 必须是**单个 LMDB 路径**；
  不要把多个 LMDB 放在一个容器目录里让 DP-EVA 展开。SAI 200-step execution
  A/B（JobID `1893783`）已证明两条训练路径均可完成并写出 checkpoint；短程
  lcurve 受随机采样影响，不能据此声称训练曲线逐位等价。
- `dp test` 自报的聚合指标在 LMDB 上有效；带 `frame_nlocs`/`frame_system_ids`
  的 LMDB 可由 `dpeva infer`/`analysis` 解析逐帧明细。缺失元数据时 parser 会
  fail-loud，不会猜测归属
- `test + deepmd/lmdb` 已由 DeepMD 3.2.0 CPU contract 与 SAI npy↔lmdb A/B
  证据晋级 supported。DeepMD-kit 3.2.0 的 mixed-nloc `-d` 输出可能只有首个
  `[nloc=...]` header；parser 仅在总行数与帧索引完全一致时按组序归属，否则
  报错。`clean` 在 LMDB 带 `frame_system_ids` 元数据时可用；缺失时请补齐元数据或使用
  npy/mixed 副本

## 6. 作业与调度类问题（Slurm）

### 6.1 `sbatch` 提交失败

症状：

- `sbatch: error: ...`

处理：

- 检查 `partition/qos/account` 是否符合策略
- GPU/CPU 资源字段是否与分区匹配
- 检查提交目录权限与配额

### 6.2 长时间 pending

处理：

- 使用 `squeue -j <jobid>` 查看原因（资源不足/策略限制）
- 适当降低资源申请（缩短 walltime、减少 GPU/CPU）

### 6.3 日志不落盘/找不到日志文件

处理：

- 以提交目录为准确认 `.out` 位置（建议固定 `work_dir` 并在其中提交）
- 对照 `docs/source/guides/slurm.md` 的常见日志命名约定

## 7. 数值类问题（NaN/Inf/聚类不收敛）

### 7.1 UQ 分布异常或全 NaN/Inf

处理：

- 优先检查推理输出是否完整（`results.*.out` 是否存在）
- 检查是否存在失败模型（某个模型推理输出缺失会影响方差/UQ）
- 若输入数据包含异常结构（box/coord 形状不一致），先修复数据

### 7.2 聚类不收敛或采样结果为空

处理：

- 检查 `sampler_type` 与参数组是否匹配（direct vs 2-direct）
- 降低 `direct_n_clusters` 或调整阈值相关参数
- 若候选池数量过少或被联合采样去重，采样数少于目标是预期行为

## 8. 变更记录

- 2026-02-18：补齐 Troubleshooting 的结构化排查路径与四类常见问题处理建议。
