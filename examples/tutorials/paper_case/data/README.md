# FeCHO 论文教学数据

本目录包含论文全部 30,656 帧的结构、DFT 标签和结构特征，所有数值来自真实数据或论文表格。
读取无需原始工作区、网络或 checkpoint。原始位置以归档相对路径写入 `provenance.json`。
结构覆盖 737 个 system、1,251,299 个逐帧累计原子；float64 数值采用无损压缩，
不重复携带原目录中等价的 `.raw` 文本。模型权重及全量四模型原子预测不包含在本目录。
`features.npz`、`iteration_features.npz` 和 `structures.npz` 使用标准 ZIP LZMA，NumPy 可直接读取；
`manifest.csv.gz` 和 `uq.csv.gz` 使用 gzip，pandas 可自动读取。压缩转换已逐字节核对数组
及解压后的 CSV，未使用 float32、舍入或抽帧。整个教学目录不进入 PyPI wheel/sdist。

| 文件 | 内容与索引 |
|---|---|
| `features.npz` | 初始阶段完整 `features`：30,656 × 128，float64，按 manifest 对齐；`paper_order` 恢复论文冷启动加载顺序 |
| `iteration_features.npz` | 迭代阶段完整 `features`：同样按 manifest 对齐；`training_order` 恢复 iter1 训练描述符的加载顺序 |
| `manifest.csv.gz` | 每行对应一个特征行；`feature_row` 为行号；`frame_id` 区分 train/pool；`source_dataname` 唯一对应完整原始数据中的组成与帧号；元素计数与原子数 |
| `uq.csv.gz` | 27,656 个池中点，与特征第 3,000 行起逐行对齐；包含原始/缩放 RND、QbC、真实最大力误差及历史分类 |
| `small_subset_indices.npz` | 可选 `small` 模式：`train` 为前 3,000 行中的 256 个索引，`pool` 为 UQ 表中的 512 个索引；二者都从 0 开始 |
| `structures.npz` | 全部 30,656 帧的原子序数、坐标、晶胞和隐藏 DFT 能量/力/virial；`feature_rows` 与 manifest 逐行对应，`offsets` 指定每帧的原子切片 |
| `committee_probe.npz` | 前八个池中帧的四模型原子力数据；六列依次为 data_fx/fy/fz、pred_fx/fy/fz；单位 eV/Å |
| `paper_selected.csv` | 论文 DPA3 + UQ + joint-DIRECT 的 843 个新增点身份 |
| `paper_metrics.csv` | 手工转录发表版 Table 1/2；能量单位 meV/atom，力单位 eV/Å；new/train 帧数分别列出 |
| `dataset_stats.json` | 全量原始数据统计，移除机器路径 |
| `provenance.json` | 来源文件校验值、切片种子、精度、检查范围与实验边界 |
| `reproduction_audit.json` | 2026-10-21 跨环境选数诊断的实测摘要，不参与教学选点，也不是论文原始数据 |
| `reproduction_trace.json` | 后续追溯：原作业身份、输入顺序、旧 NumPy 池化、用户包隔离与历史选点差异 |
| `conclusion_audit.json` | 采样结论的实测快照：覆盖、UQ 与新增信息诊断；由 `scripts/assess_conclusions.py` 重算，不是论文训练结果 |

结构池和训练集各自使用其原始 `dataname` 编号。**不能把 pool 的帧号拿到完整原数据
同名目录中直接读取**：划分后局部帧号会变化。`scripts/pack_full_structures.py` 从
`ft2dp-v1-30656` 读取完整结构，并用初始训练名单的补集映射候选池。全部 27,656 个池中点的
坐标、晶胞与 E/F/virial 都与 `dpeva-iter1/other_dpdata` 逐项精确核对。
`pack_sampling_features.py` 分别从 init 的全量原子描述符和 iter1 的 train/pool 原子描述符
计算均值与 L2 归一化。两个阶段独立存储，不混用；源文件 SHA-256 存在 `provenance.json`。
每份特征仍以 manifest 行号对应结构，额外顺序数组只决定送入 BIRCH 的行顺序。
为保存两个阶段的完整原始数值，较原先单份特征增加约 29 MiB，无需新增结构或模型文件。


全部候选池已有 ABACUS 标签。选择逻辑只读 UQ 与特征；真实误差用于事后诊断，
`structures.npz` 中的标签保留供课后核查，导出函数不将标签写入标注输入。
历史精度指标是原实验报告值；全量重算以覆盖、UQ 和新增信息等指标核对采样趋势，名单用于追溯身份。
正文中的数值舍入只用于展示，数据文件保持原有精度；小切片用于操作练习。

重建数据的命令、原始归档要求和脚本职责见 [维护说明](../scripts/README.md)。
`paper_metrics.csv` 是独立维护的论文转录表，准备脚本不覆盖它。
`reproduction_audit.json` 和 `reproduction_trace.json` 同样独立维护，记录本次诊断；
解释和复查命令见 [复现核查](../REPRODUCIBILITY.md)。
`conclusion_audit.json` 是本轮结论核查的快照，重建原始数据的脚本不生成或覆盖它。
原始数据下载入口与引用信息见 [教程来源](../README.md#来源)。
