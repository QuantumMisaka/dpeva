---
title: SAI 快速上手（Slurm 用户）
status: active
audience: Users / Developers
last-updated: 2026-10-11
owner: Workflow Owner
---

# SAI 快速上手（Slurm 用户）

- Status: active
- Audience: Users / Developers / Platform Maintainers
- Applies-To: `submission.backend="slurm"` 的 SAI 运行环境
- Last-Updated: 2026-10-11

## 1. 目的与范围

本页把「在 SAI 上第一次把 DP-EVA 跑起来」需要的最少信息集中到一处，并逐条给出
权威出处，避免重复维护字段定义。字段语义请查阅
[validation.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/reference/validation.md)
与 Sphinx API Reference；Slurm 行为与排障细节的单一权威来源是
[slurm.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/slurm.md)。

## 2. 环境

| 项目 | 取值 | 出处 |
|---|---|---|
| Conda 环境 | `dpeva-dpa4` | [deepmd-kit-sai-build.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/developer/deepmd-kit-sai-build.md) |
| 环境加载脚本 | `source scripts/env/dpeva-dpa4.env` | 仓库内 `scripts/env/dpeva-dpa4.env` |
| DeepMD-kit | 与 `dpeva doctor` 报告的运行时包络一致 | `dpeva doctor` |

从源码构建 DeepMD-kit（而不是使用 PyPI/Conda 预构建包）的完整流程、CUDA/PyTorch
后端选择与验证方法，见
[deepmd-kit-sai-build.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/developer/deepmd-kit-sai-build.md)；
本页不复制其步骤。

## 3. 安装与自检

```bash
git clone https://github.com/QuantumMisaka/dpeva.git
cd dpeva
pip install -e .
dpeva doctor
```

安装方式与依赖边界（默认 DeepMD 运行时包络、`.[deepmd]` extra、`--no-deps` 的含义）
见 [installation.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/installation.md)。
`dpeva doctor` 会报告 DeepMD 运行时包络、
`dp` 子命令可用性、`dpdata`/`torch`/CUDA 可见性等；**先让它全绿再提交作业**。

## 4. 提交配置要点

Slurm 相关字段都挂在 `submission` 下（`submission.backend`、`submission.slurm_config`、
`submission.env_setup`）。最小结构：

```json
{
  "submission": {
    "backend": "slurm",
    "env_setup": [
      "source scripts/env/dpeva-dpa4.env",
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
  **不要依赖登录时已 source 的环境**。
- `partition` / `qos` / `account`：按集群策略补齐，字段清单见
  [validation.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/reference/validation.md)。
- 端到端可复制的示例：`examples/recipes/labeling/config_sai_abacus_slurm.json`
  （ABACUS 标注，含 class-aware 的 `labeling_task_classes`）与 `examples/recipes/`
  下的其它 workflow 模板。
- 先用仓库自带夹具跑通最小闭环（Local backend 也可）再上集群：
  [smoke-demo.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/smoke-demo.md)。

## 5. 已知坑（逐条给出处）

以下每一条都来自
[slurm.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/slurm.md)
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
   [deepmd-kit-sai-build.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/developer/deepmd-kit-sai-build.md) §9.3 与 GPU 验证小节。
6. **登录节点没有 GPU**：登录节点 `nvidia-smi` 失败属正常现象，GPU 可用性结论必须
   来自 Slurm 分配的计算节点。见
   [deepmd-kit-sai-build.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/developer/deepmd-kit-sai-build.md) §9.4。

## 6. 完成标记与监控

DP-EVA 以 `DPEVA_TAG: WORKFLOW_FINISHED` 作为应用层完成锚点；部分失败时不会输出该
标记，且 CLI 返回非零退出码。监控命令与日志命名见
[slurm.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/slurm.md) §4–§5。

## 7. 相关文档

- Slurm 使用与排障：[slurm.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/slurm.md)
- 最短路径与字段说明：[quickstart.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/quickstart.md)、[configuration.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/configuration.md)
- 自带夹具冒烟：[smoke-demo.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/smoke-demo.md)
- 排障总入口：[troubleshooting.md](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/troubleshooting.md)

## 8. 变更记录

- 2026-10-11：新增 SAI 用户快速上手页，汇总既有出处并交叉链接 slurm.md / smoke-demo.md。
