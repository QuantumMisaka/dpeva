# 教程维护：验证与重建数据

学员直接使用[教程入口](../README.md)中的 `run.py`。本目录供维护者检查数据、
更新 notebook 静态输出，以及从作者实验归档重建教学输入。

## 脚本分工

| 文件 | 职责 | 是否需要作者归档 |
|---|---|---|
| `verify.py` | 核验数据身份、UQ 复算、选数和导出，并独立执行四章 notebook | 否 |
| `diagnose_reproduction.py` | 保存采样各阶段数组、环境和名单差异，支持固定 PCA 的跨环境对照 | 否 |
| `assess_conclusions.py` | 核对论文采样趋势：20 次随机覆盖对照、UQ 盲区及同数量 joint/normal 新增信息诊断 | 否 |
| `prepare_data.py` | 数据包重建总入口：整理特征、UQ、索引和来源信息，并调用结构打包 | 是 |
| `pack_full_structures.py` | 打包全部结构与标签，核验原始编号和训练/池划分 | 是 |
| `pack_sampling_features.py` | 从 init / iter1 原子描述符生成两阶段特征，保存原始加载顺序和来源；由总入口调用 | 是 |
| `archive_io.py` | 共用的无损 NPZ 写入函数；由重建脚本导入，无命令行入口 | 否 |

## 验证与更新静态图像

先按主 README 完成环境安装。以下命令从**仓库根目录**开始：

```bash
cd examples/tutorials/paper_case
source .venv-tutorial/bin/activate
python scripts/verify.py
# 将执行结果写回四章 notebook，供 GitHub 静态阅读：
python scripts/verify.py --write-notebooks
```

普通验证把 notebook 副本写到 `work/executed/`，检查报告写到 `work/verification.json`；
`--write-notebooks` 会更新教程源 notebook 的输出，保留 Markdown 中的论文图附件。
论文图的原文件及同步维护说明见 [figures/README.md](../figures/README.md)。两种方式均会生成选数导出，
不提交训练或 DFT 作业。默认 `run.py` 已执行同一套检查。

核对采样结论时，运行 `python scripts/assess_conclusions.py`，检查覆盖优势、UQ 互补性与新增信息趋势。
结果写入 `work/conclusions/assessment.json`；脚本仅进行这些诊断，不执行模型训练或精度评估。
解释指标时以其定义、实际差异和科学意义为依据，正文可使用适当有效数字，保存结果保留计算精度。

需要进一步追溯选点身份时，按 [复现核查](../REPRODUCIBILITY.md)运行 `diagnose_reproduction.py`。
上述补充核查均为可选步骤，不在一键教学主线中自动执行；名单完全一致不是教学验收要求。

## 从原始归档重建

在上述教学目录与环境中执行：

```bash
python scripts/prepare_data.py --source /path/to/CE-2025-DPEVA-paper/test-for-paper
python scripts/verify.py --write-notebooks
```

`--source` 必须指向作者的完整实验归档，包含原始 `ft2dp-v1-30656`、初始和迭代
采样表、原子描述符、委员会预测及统计；仅下载公开结构数据不足以重建全部教学输入。
脚本默认写入并覆盖本教程的 `data/`，也可用 `--output /path/to/output` 指定重建目录。
`paper_metrics.csv` 是独立维护的论文表格转录，不由重建脚本生成；新输出目录需要另行放入它。

通常只需运行 `prepare_data.py`。`pack_full_structures.py` 是其中一个步骤，
单独运行时要求输出目录已有 `manifest.csv.gz` 和 `provenance.json`。
数据字段、索引及科学使用范围见 [data/README.md](../data/README.md)。
