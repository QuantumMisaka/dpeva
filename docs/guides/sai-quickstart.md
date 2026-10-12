---
title: SAI 快速上手（Slurm 用户）
status: active
audience: Users / Developers
last-updated: 2026-10-12
owner: Workflow Owner
---

# SAI 快速上手（Slurm 用户）

- Status: active
- Audience: Users / Developers / Platform Maintainers
- Applies-To: `submission.backend="slurm"` 的 SAI 运行环境
- Last-Updated: 2026-10-12

## 1. 目的与范围

本页把「在 SAI 上第一次把 DP-EVA 跑起来」需要的最少信息集中到一处，并逐条给出
权威出处，避免重复维护字段定义。字段语义请查阅
{doc}`validation.md </reference/validation>`
与 Sphinx API Reference；Slurm 行为与排障细节的单一权威来源是
{doc}`slurm.md </guides/slurm>`。

## 2. 环境

| 项目 | 取值 | 出处 |
|---|---|---|
| Conda 环境 | 预置 `dpeva-dpa4` | {doc}`SAI 构建记录 </guides/developer/deepmd-kit-sai-build>` |
| 环境加载脚本 | `source "${DPEVA_REPO:?}/scripts/env/dpeva-dpa4.env"` | 仓库内 `scripts/env/dpeva-dpa4.env` |
| DeepMD-kit | 与 `dpeva doctor` 报告的运行时包络一致 | `dpeva doctor` |

{doc}`SAI 构建记录 </guides/developer/deepmd-kit-sai-build>` 以已有 `dpeva-dpa4` 环境和未入库的
`test/deepmd-kit` 源码检出为前提，说明的是该环境的升级与验证。新账号从零建环境
请先阅读 {doc}`安装指南 </guides/installation>`，再向集群维护者确认 GPU、模块和队列配置。

## 3. 安装与自检

```bash
git clone https://github.com/QuantumMisaka/dpeva.git
cd dpeva
export DPEVA_REPO="$(pwd -P)"
source "${DPEVA_REPO:?}/scripts/env/dpeva-dpa4.env"
python -m pip install --no-deps -e .
dpeva doctor
dp --version
```

这里的 `--no-deps` 只适用于已由平台完整预置依赖的环境。安装方式与依赖边界见
{doc}`安装指南 </guides/installation>`。
`dpeva doctor` 会报告 DeepMD 运行时包络、
`dp` 子命令可用性、`dpdata`/`torch`/CUDA 可见性等；提交前检查本次工作流所需项。

## 4. 提交配置要点

Slurm 相关字段都挂在 `submission` 下（`submission.backend`、`submission.slurm_config`、
`submission.env_setup`）。最小结构：

```json
{
  "submission": {
    "backend": "slurm",
    "env_setup": [
      "source \"${DPEVA_REPO:?}/scripts/env/dpeva-dpa4.env\"",
      "export DP_INTERFACE_PREC=high"
    ],
    "slurm_config": {
      "partition": "4V100",
      "qos": "flood-1o2gpu",
      "nodes": 1,
      "ntasks": 1,
      "gpus_per_node": 1,
      "walltime": "02:00:00"
    }
  }
}
```

- `submission.env_setup`：作业内的环境加载命令。Slurm 作业是非交互 shell，
  不能依赖提交 shell 的当前目录或已 source 的环境。提交 `dpeva` 前在 shell 中
  `export DPEVA_REPO="$(pwd -P)"`，确保这是源码仓库的真实绝对路径；默认 `sbatch`
  会将已导出的变量传给作业。若集群使用 `--export=NONE` 或过滤提交环境，需在作业
  环境中显式提供相同的绝对路径。`${DPEVA_REPO:?}` 在变量缺失时立即失败，避免
  作业误从 `work/0` 等任务目录寻找脚本。
- `partition` / `qos` / `account`：按集群策略补齐，字段清单见
  {doc}`validation.md </reference/validation>`。
- 端到端可复制的示例：`examples/recipes/labeling/config_sai_abacus_slurm.json`
  （ABACUS 标注，含 class-aware 的 `labeling_task_classes`）与 `examples/recipes/`
  下的其它 workflow 模板。
- 首次使用可按 {doc}`Quickstart </guides/quickstart>` 用仓库自带夹具跑 Local 链路。

## 5. 已知坑（逐条给出处）

以下每一条都来自
{doc}`slurm.md </guides/slurm>`
的实测记录，这里只做索引：

1. **QOSMinGRES 拒绝 1GPU 请求**：`flood-gpu` / `rush-gpu` 上实测 1 GPU 申请会被
   `QOSMinGRES` 拒绝；批量 fallback 使用 4GPU MPI 类。见 slurm.md §3.3。
2. **`source /etc/profile` 的顺序**：必须出现在 `#SBATCH` 指令之后、`module load`
   之前；`launcher_mode="mpi_abacus"` 会在其后自动插入 SAI rank-map。见 slurm.md §3.3。
3. **`launcher_mode` 语义**：`abacus` 不应加载 SAI rank-map，也不应引用 `MAP_OPT`；
   只有 `mpi_abacus` 才插入 rank-map。见 slurm.md §3.3。
4. **array 提交上限**：`slurm_array_task_limit` 只控制并发，不改变 array 元素总数；
   集群对 array 总数有限制时应调整 QOS、拆分数据或降低批量规模。见 slurm.md §3.3。
5. **长作业与大分区**：单卡排队过久时可用 `8V100V0`，该分区通常要求每节点 GPU 数量
   按 4 的倍数申请（可用 `--gpus-per-node=4` 跑单进程验证）。见
   {doc}`deepmd-kit-sai-build.md </guides/developer/deepmd-kit-sai-build>` §9.3 与 GPU 验证小节。
6. **登录节点没有 GPU**：登录节点 `nvidia-smi` 失败属正常现象，GPU 可用性结论必须
   来自 Slurm 分配的计算节点。见
   {doc}`deepmd-kit-sai-build.md </guides/developer/deepmd-kit-sai-build>` §9.4。

## 6. 完成标记与监控

DP-EVA 以 `DPEVA_TAG: WORKFLOW_FINISHED` 作为应用层完成锚点；部分失败时不会输出该
标记，且 CLI 返回非零退出码。监控命令与日志命名见
{doc}`slurm.md </guides/slurm>` §4–§5。

## 7. 相关文档

- Slurm 使用与排障：{doc}`slurm.md </guides/slurm>`
- 最短路径与字段说明：{doc}`quickstart.md </guides/quickstart>`、{doc}`configuration.md </guides/configuration>`
- 排障总入口：{doc}`troubleshooting.md </guides/troubleshooting>`

## 8. 变更记录

- 2026-10-12：作业环境脚本改用仓库绝对路径，区分预置环境与从零安装。
