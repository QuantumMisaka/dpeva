# DP-EVA 论文案例：FeCHO 主动学习交互教程

用论文真实数据完成一次选数决策：**看结构 → 看覆盖 → 调 UQ → 选新增结构 → 查看科学结果**。
适合 20–30 分钟讲者演示，也可供听众课后逐格运行。FeCHO 指 Fe–C–H–O（铁、碳、氢、氧）体系。
`paper_case` 表示 DP-EVA 自身论文的配套案例，数据范围为论文所用的 FT2DP FeCHO v1。

第一次使用：按下节启动环境，再依次打开 01–04。数据字段见 [data/README.md](data/README.md)，
接入计算工作流见 [configs/README.md](configs/README.md)，重建和检查见 [scripts/README.md](scripts/README.md)。
采样趋势的实测依据见 [复现核查](REPRODUCIBILITY.md)，其中附有可选的逐点追溯记录。

主线使用 DPA-3.1-3M / OC22 的预计算结果，CPU 即可，无需 GPU、DeepMD、模型权重、
SAI 账号或原作者工作区。UQ 与采样直接调用当前仓库的实现；安装依赖后可离线演示。
论文全部 30,656 帧的结构与 DFT 标签已包含在教学目录中，冷启动和迭代选数默认使用全量。

## 一键运行与环境

先克隆完整仓库，准备 Python 3.10–3.13（含 `venv` 和 `pip`）。
从仓库根目录执行一条命令，自动创建专用环境、安装依赖、检查数据并运行全部四本 notebook：

```bash
python3 examples/tutorials/paper_case/run.py
```

首次安装需要访问 Python 软件源，不需要 Conda、CUDA 或集群环境。
脚本按自身位置定位文件，也可从其他目录通过绝对路径运行。已有环境会复用；
全部执行结果写入 `work/executed/`，检查报告为 `work/verification.json`，
实际安装版本保存到 `work/environment-installed.txt`。

现场打开交互界面：

```bash
python3 examples/tutorials/paper_case/run.py lab
```

打开终端显示的本地链接。依赖已安装时，加 `--offline` 可以离线验证或启动：

```bash
python3 examples/tutorials/paper_case/run.py --offline
python3 examples/tutorials/paper_case/run.py lab --offline
```

`run.py` 使用 `.venv-tutorial/`，不修改已有 Conda 环境。`requirements.txt` 给出直接依赖及兼容范围，
其中数值分析、结构读写、JupyterLab、控件和 notebook kernel 均已包含。
若希望手动管理环境，也可按以下等价步骤操作：

```bash
cd examples/tutorials/paper_case
python3 -m venv .venv-tutorial
source .venv-tutorial/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

打开 `01_data_and_representation.ipynb`，选择这个环境的 Python kernel。
依次执行全部单元格，再操作滑块和下拉菜单。图中文字用英文标注，讲解用中文，避免依赖
本机中文绘图字体。每个 notebook 可独立从头运行，03 与 04 各自按显式参数生成导出。

仓库中的 notebook 保存了执行结果，可在 GitHub 静态阅读；控件需要本地 Jupyter。
即使不使用控件，也可以修改普通代码单元格中的参数后运行。

讲解、操作提示、讨论题和图注以中文为主。八张图由真实数据运行生成，包含组成与大小、
特征分布、原子结构、采样对比、UQ 筛选和论文指标；论文表格重绘图注明 Table 1/2 来源。
另有两张论文原图辅助方法与应用讲解，分别放在 01 与 04；来源及许可见
[图片说明](figures/README.md)，阅读路线见 [参考资料](REFERENCES.md)。
图像嵌入 `.ipynb`，静态阅读无需图片外链或原始论文附件。

## 实际执行输出在哪里

四章 notebook 均保存了真实执行后的单元格输出，包括文本日志、表格和图像；
在 GitHub 上打开即可查看。重新运行时，这些输出会按本次数据和参数更新。

| 章节 | 保存的运行信息 |
|---|---|
| 01 | 数据规模、结构分布与实际查看的结构编号 |
| 02 | 冷启动输入规模、聚类预算、实际选中数、保留主成分数和覆盖率 |
| 03 | 八帧 UQ 复算核对、全池筛选统计、已有/新增点数量和导出位置 |
| 04 | 导出目录、结构数量、后续分析命令及论文结果对照 |

这些是 CPU 数据分析、UQ 和采样的实际输出；论文误差指标来自表格转录，
本教程未运行模型训练或 DFT，因此没有本次训练曲线、推理或 DFT 作业日志。
静态输出中的 `work/selection-*/` 是保存时的运行目录，不随 Git 分发；本地重新运行会生成新目录。

`run.py` 的终端会报告各章是否通过及耗时；执行后的 notebook 副本在 `work/executed/`，
结构化检查报告在 `work/verification.json`。维护者更新本目录 notebook 的静态输出，
使用 [维护说明](scripts/README.md)中的 `--write-notebooks`。

## 25 分钟讲解路线

| 顺序 | Notebook | 时间 | 现场动作 |
|---|---|---:|---|
| 1 | [数据与表示](01_data_and_representation.ipynb) | 5 分钟 | 看组成、原子数、全量 PCA 分布，查看真实结构 |
| 2 | [冷启动选数](02_cold_start.ipynb) | 5 分钟 | 在全量数据上把聚类预算从 300 调到 3,000，比较同数量随机选择 |
| 3 | [不确定度与联合采样](03_uq_and_joint_sampling.ipynb) | 10 分钟 | 从原始力预测复算 UQ；切换边界、揭示误差；切换 joint；导出 |
| 4 | [结构导出与论文结果](04_export_and_paper_results.ipynb) | 5 分钟 | 查看交接目录，解释 3,000 + 843 帧与论文评估 |

方法解释融入上述时段：01 用约 1 分钟读框架图；03 用训练设置表区分浅层委员会与最终模型；
04 用约 1 分钟读反应误差表和表面稳定性图。第 03 章末尾公式推导留作课后阅读。
若只有 20 分钟，01 跳过结构控件，02 只比较默认预算，04 选读一项应用结果，把主要时间留给 03。

## 演示前准备

首次使用先完成上面的联网安装。演示前从仓库根目录运行：

```bash
python3 examples/tutorials/paper_case/run.py --offline
python3 examples/tutorials/paper_case/run.py lab --offline
```

第一条命令应报告四章均 `passed`，最终输出 `All tutorial checks passed`；
确认后再用第二条命令打开 JupyterLab。按 01–04 顺序打开 notebook，先运行全部单元格，
再调整控件；02 的预算和 03 的采样参数改动后需点击“运行采样”。

现场优先演示全量默认配置；机器较慢时用小切片展示操作，科学结论结合保存的全量结果讲解。
若现场无法启动交互，可直接讲解 notebook 已保存的日志、表格和图片；只有重新运行后控件才可操作。
分享给听众时提供完整仓库及本 README，环境和 `work/` 导出目录由听众本地生成。

## 目录与文件

```text
paper_case/
├── README.md                            # 本教程入口：启动与讲解路线
├── REPRODUCIBILITY.md                   # 论文结论与逐点复现核查
├── REFERENCES.md                        # 论文、推介与方法实现的阅读路线
├── figures/                             # 论文图及来源许可（亦嵌入 notebook）
├── 01_data_and_representation.ipynb      # 数据、结构与特征表示
├── 02_cold_start.ipynb                   # 冷启动：DIRECT 与随机选择
├── 03_uq_and_joint_sampling.ipynb        # 不确定度筛选与联合采样
├── 04_export_and_paper_results.ipynb     # 导出结构、衔接工作流、解读论文结果
├── run.py                               # 一键检查或启动交互界面
├── requirements.txt                     # CPU 教学环境依赖
├── tutorial_support.py                  # 数据加载、绘图与导出辅助
├── data/                                # 随 Git 分发的输入及数据说明
├── configs/                             # 接入完整工作流的模板与说明
├── scripts/                             # 维护者重建、压缩和验证工具
├── .gitignore                           # 排除环境、缓存和运行输出
├── .venv-tutorial/                      # 首次运行创建的环境（Git 忽略）
└── work/                                # 执行结果与导出（Git 忽略）
```

01–04 保留在本目录，方便按编号打开；文件名采用英文，正文以中文讲解。
`data/` 是固定教学输入，`work/` 是可重新生成的输出。
辅助模块直接调用 DP-EVA 的 UQ 与采样实现；学员无需先阅读或修改它们。
各数据文件的含义和索引关系见 [数据说明](data/README.md)，各脚本职责见 [维护说明](scripts/README.md)。

教学选择的结构会导出为 `selected.extxyz` 和无标签 DeepMD `label_input/`。
`selected.csv` 记录池索引、局部帧号、完整原数据编号 `source_dataname` 和导出目录；
`parameters.json` 记录实际选择参数。
结构文件本身不附带已有 DFT 标签，模拟交给标注环节的输入。

## 怎样判断教学结果

以论文方法、数据预算和科学指标为主线，在有意义的报告精度下判断结论是否保持。
当前与历史选点的冷启动覆盖率约为 98.3% 与 98.5%，均高于补充核查中 20 次随机选择
（平均 89.7%）；UQ 分类与历史一致，联合采样后的总训练量均约为全量数据的 12.5%。
这些结果支持教学中的覆盖优势与选数机制。近等距代表点可发生替换，名单交集用于追溯，
不作为课堂复现成功的必要条件。

正文按讲解需要舍入；输入数组、计算结果和论文转录表保留原有精度。
现场复算涵盖数据分析、UQ 与采样，模型精度及应用效果由论文原实验结果说明。
有关有效数字、指标定义与结论依据的详细说明见 [复现核查](REPRODUCIBILITY.md)。

## 实验口径

1. **全量历史数据**：论文冷启动 3,000 帧，剩余池 27,656 帧；归档 UQ 分类为
   accurate 22,217 / candidate 5,339 / failed 100，最终新增 843 帧。
2. **现场全量与小切片**：默认冷启动从 30,656 帧中选择 3,000 帧；迭代使用全部 3,000 个历史
   训练点和 27,656 个池中点，默认 joint 聚类预算 3,000（normal 为 1,000）。控件可切换 `small`：固定种子 20261021，
   从池和训练集分别均匀抽取 512 与 256 点，不按真实误差挑样。小切片改变了训练/候选比例，
   仅用于快速操作练习，其新增数量不能外推到全量。改完参数后点击“运行采样”。
3. **论文自动阈值**：默认从全池独立计算 QbC/RND 的 KDE 阈值，峰高比例 0.50、宽度 0.25 eV/Å，
   使用 `tangent_lo`；逐帧核对历史分类。切换“手动练习”后才使用滑块区间。参数见
   [paper_sampling.json](configs/paper_sampling.json)。迭代采用原始 iter1 特征和加载顺序；
   按原方法重新选数，名单交集用于事后追溯。UQ 分类逐帧一致；本地验证新增 833 帧，
   历史新增 843 帧，两者加上初始训练集后均约占全量的 12.5%。跨环境细节见复现核查附录。
4. **评估结果**：`paper_metrics.csv` 转录论文 Table 1/2，各策略在各自采样后剩余数据上评估。
   第 04 章另摘录 Table 3 的应用误差，并展示 Figure 5；这些内容解释原实验的训练与应用效果。
   现场重新选数后的精度需通过后续训练评估。
5. **UQ 原始证据**：附带八帧、四个模型的原子力预测，可用当前 `UQCalculator` 复算。
   模型 0 为基准，模型 1–3 为委员会；RND 的尺度对齐使用全池参考分布。
6. **模型文件现状**：原归档四个浅层委员会的 checkpoint 为零字节占位，预测文件可用。
   本教学不执行它们的推理。论文 SI Figure S2 是早期 DPA-2.3.1 案例；本教学的 UQ 图来自 DPA3 数据。

## 随 Git 仓库分发

教学输入全部位于本目录的 `data/`，包含真实数值文件、来源说明和索引，
不使用指向个人工作区的软链接。notebook、嵌入图像、数据、配置、辅助代码和依赖清单
均可随本目录一起提交并同步远端；教学包约 111 MiB，不要求额外下载模型或论文实验归档。
`.gitignore` 只排除环境、缓存和可重新生成的 `work/` 运行输出。

数值文件保留 float64：大型 NPZ 使用 ZIP LZMA 无损压缩，仍由 `numpy.load` 直接读取；
大型索引和 UQ 表使用 `.csv.gz`，由 pandas 自动解压。保留全部帧和嵌入图像，未降低数值精度。
本目录与 PyPI 发布包分开：`MANIFEST.in` 明确排除整个 `examples/tutorials/`，
wheel 仅从 `src/` 查找包。安装 `dpeva` 不会附带这套教学数据；教学仍从完整 Git 仓库运行。

听众应克隆**完整 DP-EVA 仓库**，然后按上述命令安装依赖、启动 notebook。
教学算法直接引用同仓库的 `src/dpeva`，因此单独下载本文件夹不能替代完整克隆。
安装依赖后，四本教学 notebook 可离线运行。`scripts/prepare_data.py` 仅供维护者重建数据包，
听众不需要运行它，也不需要访问作者的原始工作区。

## 验证与课后扩展

默认的 `run.py` 已包含数据、论文 UQ 分类、选数名单对照、导出和四章独立执行检查。
第 02 章重新做冷启动选择；第 03 章使用论文历史训练集及其委员会结果，未接续第 02 章重新训练。
维护者需要刷新 GitHub 静态图像或重建数据时，见 [scripts/README.md](scripts/README.md)。

需要真正运行 feature/train/infer/collect 时，按仓库[安装指南](../../../docs/guides/installation.md)
准备完整环境，先运行[四阶段 quickstart](../../recipes/quickstart/README.md)。它使用 1 帧训练、
5 帧候选及 30 步训练，仅用于接口教学。DPA4 可作为后续迁移练习，需要重新生成匹配的
描述符、委员会预测与评估，并处理模型头、ZBL 和能量参考。

## 来源

- DP-EVA 论文：[Clean Energy, zkag029](https://doi.org/10.1093/ce/zkag029)。
- 原始数据：AI Square 的 `FT2DP-dataset-FeCHO-v1`（dataset id 306），
  [数据入口](https://www.aissquare.com/datasets/detail?pageType=datasets&name=FT2DP-dataset-FeCHO-v1&id=306)。
- 源文件及转换规则见 `data/provenance.json`；代码沿用仓库许可证，数据使用请引用原论文与数据发布来源。
