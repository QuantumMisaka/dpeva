---
title: Smoke Demo（仓库自带夹具最小闭环）
status: active
audience: Users / Developers
last-updated: 2026-10-11
owner: Workflow Owner
---

# Smoke Demo（仓库自带夹具最小闭环）

- Status: active
- Audience: Users / Developers
- Applies-To: CLI + Local backend
- Last-Updated: 2026-10-11

## 1. 目的与范围

本页给出一条**不依赖集群、不依赖外部下载**的最短链路，用于在首次接触 DP-EVA 时
确认「环境能装、命令能跑、产物能落盘」：

```text
feature → train → infer → collect
```

所有输入都来自仓库自带夹具，因此在离线机器或登录节点上即可完成。

**这不是科学结果。** 夹具只有 1 帧训练数据与 5 帧候选数据，训练步数被压到几十步，
采样结果没有统计意义；本页只用于验证工具链本身。需要正式流程时请回到
[Quickstart](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/quickstart.md)
与 [Configuration](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/configuration.md)。

## 2. 自带夹具

以下文件随仓库跟踪，无需额外下载：

| 路径 | 内容 | 说明 |
|---|---|---|
| `tests/integration/data/DPA-3.1-3M.pt` | 预训练 DPA-3.1-3M 模型（47,176,032 字节） | 多任务头模型；本页使用 `OC22` 头 |
| `tests/integration/data/sampled_dpdata/122` | 训练集，1 帧，含 `energy/force/virial/spin` | 对应 `training_data_path` |
| `tests/integration/data/other_dpdata_all/amourC/C2` | 候选池，5 帧，仅 `coord/box` | 对应候选池与 `testdata_dir` |
| `tests/integration/data/input.json` | DeepMD-kit 训练输入模板 | 冒烟时需下调 `numb_steps` |

夹具清单与外部模型覆盖方式见
[tests/integration/data/README.md](https://github.com/QuantumMisaka/dpeva/blob/main/tests/integration/data/README.md)；
目录语义与各 workflow 的字段映射见
[multi-datapool-artifacts.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/testing/multi-datapool-artifacts.md)。

## 3. 前置条件

- 已按 [安装指南](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/installation.md) 完成 `pip install -e .`；
- `dp`（DeepMD-kit）在 `PATH` 上，且与 `dpeva doctor` 报告的运行时包络一致；
- 一个可用 GPU 或可接受的 CPU 训练时间（`dp train` 建议 GPU）；
- 建议在**独立工作目录**中执行，例如 `smoke/`，不要写入仓库目录。

自检：

```bash
dpeva doctor
dp --version
```

## 4. 准备最小工作目录

```text
smoke/
├── configs/
│   ├── feature.json
│   ├── train.json
│   ├── infer.json
│   └── collect.json
├── input.json          # 由 tests/integration/data/input.json 复制并下调 numb_steps
├── desc_pool/          # Feature 输出
├── 0/ 1/ 2/            # Train 输出的 3 个模型
└── dpeva_uq_result/    # Collect 输出
```

先把训练输入复制到工作目录，并把 `training.numb_steps` 降到 30 左右（示例值，仅为缩短冒烟时间）：

```bash
cp <repo>/tests/integration/data/input.json smoke/input.json
# 编辑 smoke/input.json：把 training.numb_steps 改为 30
```

四个配置都使用 Local backend（`"submission": {"backend": "local"}`）。字段含义以
`docs/reference/*` 为单一权威来源，完整模板见
[integration-config-templates.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/testing/integration-config-templates.md)。

`configs/feature.json`（候选池描述符）：

```json
{
  "data_path": "<repo>/tests/integration/data/other_dpdata_all",
  "model_path": "<repo>/tests/integration/data/DPA-3.1-3M.pt",
  "savedir": "desc_pool",
  "model_head": "OC22",
  "mode": "cli",
  "omp_threads": 2,
  "submission": {"backend": "local"}
}
```

`configs/train.json`（3 个模型；schema 要求 `num_models >= 3`）：

```json
{
  "work_dir": ".",
  "input_json_path": "input.json",
  "num_models": 3,
  "training_mode": "init",
  "seeds": [19090, 42, 10032],
  "training_seeds": [19090, 42, 10032],
  "model_head": "OC22",
  "base_model_path": "<repo>/tests/integration/data/DPA-3.1-3M.pt",
  "training_data_path": "<repo>/tests/integration/data/sampled_dpdata",
  "omp_threads": 2,
  "submission": {"backend": "local"}
}
```

`configs/infer.json`（候选池推理）：

```json
{
  "data_path": "<repo>/tests/integration/data/other_dpdata_all",
  "work_dir": ".",
  "task_name": "test_val",
  "model_head": "OC22",
  "results_prefix": "results",
  "auto_analysis": false,
  "omp_threads": 2,
  "submission": {"backend": "local"}
}
```

`configs/collect.json`（普通采样；`num_models` 必须与训练出的模型数一致，
`testing_dir` 必须与 `infer.task_name` 一致）：

```json
{
  "project": ".",
  "uq_select_scheme": "tangent_lo",
  "testing_dir": "test_val",
  "results_prefix": "results",
  "desc_dir": "desc_pool",
  "testdata_dir": "<repo>/tests/integration/data/other_dpdata_all",
  "root_savedir": "dpeva_uq_result",
  "num_models": 3,
  "sampler_type": "direct",
  "direct_n_clusters": 3,
  "direct_k": 1,
  "direct_thr_init": 0.1,
  "fig_dpi": 150,
  "uq_trust_mode": "auto",
  "uq_trust_ratio": 0.5,
  "uq_trust_width": 0.15,
  "submission": {"backend": "local"}
}
```

## 5. 运行

在 `smoke/` 目录下依次执行：

```bash
dpeva feature configs/feature.json
dpeva train   configs/train.json
dpeva infer   configs/infer.json
dpeva collect configs/collect.json
```

Local backend 会同步执行并在当前终端返回；每步都应以 `DPEVA_TAG: WORKFLOW_FINISHED`
作为应用层完成锚点（收集阶段见下表说明）。

## 6. 预期产物与完成标记

| 步骤 | 主要产物 | 完成标记所在日志 |
|---|---|---|
| `feature` | `desc_pool/amourC/C2.npy` | `desc_pool/feature.log` |
| `train` | `0/`、`1/`、`2/`（含 `model.ckpt.pt`、`lcurve.out`、`train.log`） | `0..2/train.log` 与终端 |
| `infer` | `0..2/test_val/results.{e,e_peratom,f,v,v_peratom}.out`、`0..2/test_val/test.log` | `0..2/test_val/test.log` 与终端 |
| `collect` | `<root_savedir>/dataframe/*.csv`、`<root_savedir>/view/*.png`、`<root_savedir>/dpdata/{sampled_dpdata,other_dpdata}/` | `<root_savedir>/collection.log` |

其它 Workflow 产物路径约定见
[multi-datapool-artifacts.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/testing/multi-datapool-artifacts.md)。

本地实测参考（单卡，`numb_steps=30`，3 个模型）：feature ≈ 45 s；train ≈ 1 min 50 s；
infer ≈ 1 min 30 s；collect ≈ 10 s。耗时随硬件与 `numb_steps` 变化，仅供判断量级。

## 7. 限制与常见问题

- **仅冒烟**：`direct_n_clusters=3` 配 5 帧候选池时可能选出 0 帧，`collect` 仍会正常
  以 exit 0 结束并把全部候选帧导出到 `dpdata/other_dpdata/`。这是夹具太小导致的预期
  行为，不代表采样算法失效。
- **模型数量必须对齐**：`collect.num_models` 与 `train.num_models` 不一致时，
  collect 会去找不存在的 `N/test_val/results.e_peratom.out` 并报 `Energy file not found`。
- **目录名必须对齐**：`infer.task_name`、`collect.testing_dir` 与 `infer.results_prefix`、
  `collect.results_prefix` 需成对一致。
- **重跑前清理旧产物**：v0.8.2 对「本次尝试」的产物新鲜度是 fail-closed 的，陈旧文件
  不会被当作本次成功。重跑前请清空上一步的输出目录。
- **`dp` 不在 `PATH`**：生成脚本直接调用 `dp`，缺少时会以
  `dp: command not found`（exit 127）失败。把 DeepMD 环境的 `bin` 目录加入 `PATH`，
  或在 `submission.env_setup` 中显式加载环境。

## 8. 相关文档

- 最短路径与字段说明：[Quickstart](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/quickstart.md)、[Configuration](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/configuration.md)
- 安装：[Installation](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/installation.md)
- Slurm 与集群运行：[Slurm](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/slurm.md)、[SAI 快速上手](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/sai-quickstart.md)
- 测试专题：[Testing](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/testing/README.md)

## 9. 变更记录

- 2026-10-11：新增仓库自带夹具的最小冒烟闭环（Local backend），纳入入职文档补丁。
