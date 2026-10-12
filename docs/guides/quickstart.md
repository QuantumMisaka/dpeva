---
title: Quickstart（自带示例跑通 Local 工作流）
status: active
audience: Users / Developers
last-updated: 2026-10-12
owner: Workflow Owner
---

# Quickstart（最短路径跑通）

本页使用仓库自带的模型、数据与 `examples/recipes/quickstart/` 中的四份配置，
依次运行 `feature → train → infer → collect`。示例只用于熟悉命令和产物；训练数据只有
1 帧、候选数据只有 5 帧，不能据此判断模型精度或采样效果。正式运行时请换用自己的
模型与数据，并阅读 {doc}`配置与路径说明 </guides/configuration>`。

## 准备

克隆仓库后，按 {doc}`安装指南 </guides/installation>` 准备含 PyTorch 后端的 DeepMD-kit 环境。
确认 `dpeva`、`dp` 都来自当前环境：

```bash
git clone https://github.com/QuantumMisaka/dpeva.git
cd dpeva
dpeva doctor
dp --version
```

四份配置与训练输入位于 `examples/recipes/quickstart/`；它们引用仓库内
`tests/integration/data/` 的小型夹具。配置中的文件路径相对**配置文件所在目录**解析，
无需复制夹具、手改 `input.json` 或建立第二套工作目录。从仓库外执行时，把下方的
`examples/recipes/quickstart/*.json` 换成对应配置文件的绝对路径即可。

## 依次运行

在仓库根目录执行：

```bash
dpeva feature examples/recipes/quickstart/feature.json
dpeva train   examples/recipes/quickstart/train.json
dpeva infer   examples/recipes/quickstart/infer.json
dpeva collect examples/recipes/quickstart/collect.json
```

Local backend 同步执行。先等当前命令成功返回，再运行下一条。训练包括 3 个短步数模型，
仍可能耗时；可用 GPU 加速。需要在 Slurm 上执行时，参阅
{doc}`SAI 快速上手 </guides/sai-quickstart>` 和 {doc}`Slurm 指南 </guides/slurm>`。

## 查看产物

所有输出位于 `examples/recipes/quickstart/work/`：

| 阶段 | 主要产物 |
| --- | --- |
| Feature | `desc_pool/amourC/C2.npy` |
| Train | `0/`、`1/`、`2/` 下的模型与训练日志 |
| Infer | `0..2/test_val/` 下的 `results.*.out` 和日志 |
| Collect | `dpeva_uq_result/dataframe/` 下的 CSV、`dpdata/` 导出目录 |

成功的工作流会输出 `DPEVA_TAG: WORKFLOW_FINISHED`；具体日志位置见
{doc}`产物约定 </guides/testing/multi-datapool-artifacts>`。此小夹具的 collect 阶段可能选出
0 帧，仍会导出候选数据；这是输入规模所致，不是科学效果验证。若失败，先查看
{doc}`排障指南 </guides/troubleshooting>`，尤其检查 `dp` 是否在 `PATH`、数据路径以及相邻阶段
的模型数量和任务目录是否一致。重跑前应清理这次示例生成的旧产物，以免旧文件混入
当前结果。

接下来可按 {doc}`配置参考 </reference/index>` 准备自己的工作目录；标注与分析等其它
工作流的参数和示例见 {doc}`CLI 指南 </guides/cli>` 与仓库中的
`examples/recipes/README.md`。
