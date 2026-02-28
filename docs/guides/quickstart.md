# Quickstart（最短路径跑通）

- Status: active
- Audience: Users / Developers
- Applies-To: CLI 模式（推荐）
- Last-Updated: 2026-02-18

## 1. 目的与范围

目标：用最短路径跑通一次 DP-EVA 的关键链路，并明确：

- 配置文件写在哪里、关键字段是什么
- 输出产物在哪里、怎样判断任务完成
- 失败时去哪里看日志、如何快速定位问题

范围：

- 覆盖 CLI 子命令：`train / infer / feature / collect / analysis`
- 给出 Local 与 Slurm 两种运行方式的最小示例

## 2. 相关方

- 使用者：准备数据、运行工作流、查看结果
- 开发者：维护配置模型、CLI 行为、输出目录约定
- 集群/平台维护：提供 DeepMD 环境与 Slurm 队列配置

## 3. 前置条件

### 3.1 基础环境

- Python 环境可导入 `dpeva`（开发模式安装推荐）
- DeepMD-kit `dp` 命令可用（至少覆盖 `dp train/test/eval-desc`）

### 3.2 可选：Slurm 环境

- `sbatch/squeue` 可用
- `submission.backend="slurm"` 且可正常提交短作业

## 4. 最小目录与配置约定

推荐工作目录结构（示意）：

```text
work/
├── configs/
│   ├── feature.json
│   ├── train.json
│   ├── infer.json
│   ├── collect.json
│   └── analysis.json
├── sampled_dpdata/                 # 训练集（含能量/力等标注）
├── other_dpdata_all/               # 候选池（可无标注，多数据池结构）
└── DPA-3.1-3M.pt                   # 基础模型（示例）
```

关键输出约定（常见）：

- `work_dir/0..N-1/`：训练输出（模型与日志）
- `work_dir/<i>/<task_name>/`：推理输出（`results.*.out` 与 `test_job.log`）
- `savedir`：描述符输出目录（Feature）
- `root_savedir`：采集/筛选输出（Collect）

## 5. Quickstart：Local 一次跑通（建议先跑）

本节假设你在 `work/` 目录下执行，且配置文件写在 `work/configs/`。

参考最小配置模板：

- `examples/recipes/`：../examples/recipes/
- 配置字段字典：../reference/config-schema.md

### 5.1 Feature（生成描述符）

```bash
dpeva feature configs/feature.json
```

验证输出：

- `savedir/` 目录存在
- `eval_desc.log` 或对应日志文件包含完成标记（见 7.1）

### 5.2 Train（训练 N 个模型）

```bash
dpeva train configs/train.json
```

验证输出：

- `0/..N-1/` 目录生成
- `0/model.ckpt.pt`（或同等训练产物）存在

### 5.3 Infer（候选池推理）

```bash
dpeva infer configs/infer.json
```

验证输出：

- `0/<task_name>/results.e.out` 存在
- `0/<task_name>/test_job.log` 存在

### 5.4 Collect（UQ + Sampling + Export）

```bash
dpeva collect configs/collect.json
```

验证输出：

- `<root_savedir>/dataframe/df_uq_desc_sampled-final.csv` 存在
- `<root_savedir>/dpdata/` 下导出目录存在

### 5.5 Analysis（结果统计）

```bash
dpeva analysis configs/analysis.json
```

验证输出：

- `output_dir/` 下生成统计/图表文件

## 6. Quickstart：Slurm 一次跑通（链式编排）

当 `submission.backend="slurm"` 时，多数工作流会提交作业并返回。

建议用“完成标记 + 日志监控”的方式串联：

- 训练日志：`<work_dir>/<i>/train.out`
- 推理日志：`<work_dir>/<i>/<task_name>/test_job.log`
- Feature 日志：`<savedir>/eval_desc.log`
- Collect 日志：`collect_slurm.out`（以具体配置/脚本输出为准）

集成测试实现参考：

- `tests/integration/test_slurm_multidatapool_e2e.py`
- `docs/guides/testing/integration-slurm.md`

## 7. 异常处理与排查入口

### 7.1 如何判断“任务完成”

DP-EVA 以统一完成标记作为应用层完成锚点：

```text
DPEVA_TAG: WORKFLOW_FINISHED
```

建议外部编排系统通过监控日志出现该标记来推进后续任务。

### 7.2 常见错误类型

- `dp` 不可用 / 版本不匹配：检查 DeepMD 环境、`dp --version`
- 数据目录结构不符合预期：检查 Multi DataPool 目录结构与 `type_map.raw/type.raw`
- Slurm 提交失败：检查 partition/qos/gpu/cpu 资源字段与队列限制

排障入口：

- ./troubleshooting.md

## 8. 变更记录

- 2026-02-18：补齐 Local/Slurm Quickstart 的最小链路与完成标记约定，统一链接入口。
