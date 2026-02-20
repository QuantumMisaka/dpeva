# Troubleshooting（常见问题与排查）

- Status: active
- Audience: Users / Developers / Infra
- Last-Updated: 2026-02-18

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
- 对照 [slurm.md](/docs/guides/slurm.md) 的常见日志命名约定

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
