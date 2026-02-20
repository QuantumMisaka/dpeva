# Slurm 使用与排障指南

- Status: active
- Audience: Users / Developers / Infra
- Applies-To: `submission.backend="slurm"`
- Last-Updated: 2026-02-18

## 1. 目的与范围

本页说明 DP-EVA 如何在 Slurm 队列上运行各工作流、如何配置 `submission`、如何通过日志与完成标记进行监控与链式编排，以及常见故障的排查路径。

## 2. 相关方

- 使用者：提交作业、指定资源、查看产物
- 开发者：维护脚本模板、日志命名、完成标记一致性
- 平台维护：维护 partition/qos/account 与环境模块

## 3. Submission 配置结构

权威字段定义：

- ../reference/config-schema.md
- ../reference/validation.md

### 3.1 最小 Slurm 配置

```json
{
  "submission": {
    "backend": "slurm",
    "env_setup": [
      "source /path/to/env.sh",
      "export DP_INTERFACE_PREC=high"
    ],
    "slurm_config": {
      "nodes": 1,
      "ntasks": 1,
      "walltime": "00:30:00"
    }
  }
}
```

### 3.2 常用扩展字段（按集群策略补齐）

- `partition`：队列/分区
- `qos`：QoS
- `account`：计费账号
- `gpus_per_node`：GPU 资源（Feature/Train/Infer 常用）
- `cpus_per_task`：CPU 资源（Collect 常用）

## 4. 工作流脚本与日志约定

### 4.1 完成标记（链式编排锚点）

DP-EVA 在成功结束时输出：

```text
DPEVA_TAG: WORKFLOW_FINISHED
```

建议用该标记监控应用层完成状态，而不是仅依赖 `squeue`。

### 4.2 常见日志文件名（用于监控）

日志位置与命名由工作流与 JobConfig 固化（随版本可能变更，以实际产物为准）：

- Train：`<work_dir>/<i>/train.out`
- Infer：`<work_dir>/<i>/<task_name>/test_job.log`
- Feature：`<savedir>/eval_desc.log`
- Collect：`collect_slurm.out`（Collect 的 Slurm 自调用 worker 输出）

集成测试实现参考：

- ./testing/integration-slurm.md
- `tests/integration/test_slurm_multidatapool_e2e.py`

## 5. 运行示例

### 5.1 提交单步作业

```bash
dpeva feature configs/feature.json
dpeva train configs/train.json
dpeva infer configs/infer.json
dpeva collect configs/collect.json
```

### 5.2 监控完成标记

```bash
tail -f 0/train.out | grep -F "DPEVA_TAG: WORKFLOW_FINISHED"
```

对于推理：

```bash
tail -f 0/test_val/test_job.log | grep -F "DPEVA_TAG: WORKFLOW_FINISHED"
```

## 6. 异常处理与排障

### 6.1 环境未初始化

症状：

- `dp: command not found`
- `ModuleNotFoundError: deepmd`

处理：

- 将环境加载命令写入 `submission.env_setup`（不要依赖交互式 shell）
- 在作业日志中确认 `dp --version`

### 6.2 资源/队列配置不正确

症状：

- `sbatch: error: Batch job submission failed`
- 长时间 pending

处理：

- 检查 `partition/qos/account` 是否符合集群策略
- GPU/CPU 字段是否与分区匹配

### 6.3 输出不落盘或日志找不到

处理：

- 确认 `work_dir/project/savedir/root_savedir` 的权限与磁盘配额
- 确认提交目录与期望输出目录一致

### 6.4 任务运行失败但未产出完成标记

处理：

- 直接查看对应 `.out`/`test_job.log` 的错误堆栈
- 若失败发生在 DeepMD 命令内部，优先检查输入数据、模型文件、CUDA/驱动与 DeepMD 版本

排障入口：

- ./troubleshooting.md

## 7. 变更记录

- 2026-02-18：补齐 Slurm 配置结构、日志命名与完成标记监控建议，并纳入常见故障排查路径。
