---
title: Quickstart（自带示例跑通 Local 工作流）
status: active
audience: Users / Developers
last-updated: 2026-10-15
owner: Workflow Owner
---

# Quickstart（最短路径跑通）

本页使用仓库自带的模型、数据与 `examples/recipes/quickstart/` 中的四份配置，
依次运行 `feature → train → infer → collect`。示例只用于熟悉命令和产物；训练数据只有
1 帧、候选数据只有 5 帧，不能据此判断模型精度或采样效果。正式运行时请换用自己的
模型与数据，并阅读 {doc}`配置与路径说明 </guides/configuration>`。

## 准备

先按 {doc}`安装指南 </guides/installation>` 获取源码并准备含 PyTorch 后端的 DeepMD-kit 环境。
完成后进入仓库根目录，确认 `dpeva`、`dp` 都来自当前环境：

```bash
dpeva doctor
dp --version
```

四份配置与训练输入位于 `examples/recipes/quickstart/`；它们引用仓库内
`tests/integration/data/` 的小型夹具。配置中的文件路径相对**配置文件所在目录**解析，
无需复制夹具、手改 `input.json` 或建立第二套工作目录。从仓库外执行时，把下方的
`examples/recipes/quickstart/*.json` 换成对应配置文件的绝对路径即可。

## 选择运行位置

Local backend 会在启动命令的当前机器上直接计算，不会自动申请 Slurm 资源。
独占 Linux 服务器可在确认资源可用后执行；共享集群用户应先按站点规则取得
交互式计算资源，再在计算节点运行下面的 Local 示例，或改用 Slurm 提交配置。
不要在登录节点直接执行这组计算命令。通用提交方法见
{doc}`Slurm 指南 </guides/slurm>`，SAI 的环境与队列配置见
{doc}`SAI 快速上手 </guides/sai-quickstart>`。

示例虽然只有几帧数据，仍会加载预训练模型并训练 3 个模型；数据帧数不能作为
内存需求的估计。当前没有可供承诺的统一耗时或峰值内存数字，应按实际模型、
硬件和 DeepMD 环境安排资源。先运行 `dpeva doctor` 检查实际计算环境。

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
的模型数量和任务目录是否一致。重新开始的方法见下节。

接下来可按 {doc}`配置参考 </reference/index>` 准备自己的工作目录；标注与分析等其它
工作流的参数和示例见 {doc}`CLI 指南 </guides/cli>` 与仓库中的
`examples/recipes/README.md`。


## 从头重新运行示例

先确认上一次 Local 进程已退出；如果改成了 Slurm 提交，则先确认关联作业均已结束
或已取消。移动输出目录不会停止计算任务。排查问题时保留日志与运行清单，
不要先删除出错现场。

对于这份小型示例，默认做法是备份整个 `work/` 后，从 Feature 开始重跑。
在仓库根目录执行以下命令，只移动本示例的输出，不移动模型、数据或配置：

```bash
if [ -d examples/recipes/quickstart/work ]; then
    mkdir -p "$HOME/scratch" &&
    quickstart_backup_dir=$(mktemp -d "$HOME/scratch/dpeva-quickstart.XXXXXX") &&
    mv examples/recipes/quickstart/work "$quickstart_backup_dir/work" &&
    printf 'Previous outputs saved to %s\n' "$quickstart_backup_dir/work"
fi
```

原来存在 `work/` 时，只有看到 `Previous outputs saved to ...` 且命令无报错，
才继续重跑；备份失败时先处理错误，不要覆盖原输出。原来没有 `work/` 则可直接开始。
然后重新执行上文四条工作流命令。该方法保留旧日志、模型与 `.dpeva` 清单，
新运行使用空工作目录。需要保留已成功阶段、恢复中断运行或重新提交作业时，
先阅读 {doc}`排障与重跑说明 </guides/troubleshooting>`；四个阶段并不共享统一的
断点续跑能力。

## 换用自己的模型与数据

熟悉示例后，按以下两种情况准备自己的配置：

- **沿用本示例模型，只替换数据**：修改训练数据与候选数据路径；Feature、Infer、
  Collect 应指向同一候选池。核对训练标签、元素映射和训练设置，不要把示例的
  30 步训练当作科研生产参数。
- **更换模型或模型家族**：同时选择匹配的 DeepMD 训练模板、模型头与 `dp_backend`。
  本例的 `input.json` 绑定 DPA3 架构，不能仅替换模型文件路径就视为适用于 DPA4。
  DPA4 的现有模板位于 `examples/recipes/training/dpa4/`，使用前需要准备外部模型、
  数据和相应训练统计输入，并调整其提交配置。

具体字段之间的匹配关系见 {doc}`配置指南 </guides/configuration>`。
公开资源包括：

| 资源 | 入口 | 用途 |
| --- | --- | --- |
| DPA-3.1-3M 预训练模型 | [AIS Square 模型库](https://www.aissquare.com/models/343) | 用作微调基座 `base_model_path`，按模型选择 `model_head` |
| FT2DP-dataset-FeCHO-v1 | [AIS Square 数据集](https://www.aissquare.com/datasets/306) | 铁基 FTS 场景的训练集与候选池示例 |

普通 CLI 工作流通过 JSON 配置指定模型和数据路径。测试夹具文档中的
`DPEVA_DPA3_MODEL_PATH`、`DPEVA_DPA4_MODEL_PATH` 属于测试入口约定，
不是所有 CLI 工作流通用的模型路径覆盖开关。字段与路径规则见
{doc}`配置指南 </guides/configuration>`；安装、后端选择与科研环境版本锁定见
{doc}`安装指南 </guides/installation>`。
