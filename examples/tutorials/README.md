# DP-EVA 交互教程

- [DP-EVA 论文案例：FeCHO 主动学习交互教程](paper_case/README.md)：20–30 分钟，使用论文
  全部 30,656 帧数据与 DPA3 预计算结果，在 CPU 上交互运行不确定度筛选（UQ）和
  DIRECT / 联合采样（joint-DIRECT）。

Notebook 与数据随仓库提供，运行输出留在各教学目录的 `work/`。
这些教程不包含在 PyPI 安装包中，需要克隆完整 DP-EVA 仓库。

克隆完整仓库后，在仓库根目录用 Python 3.10–3.13 一键准备 CPU 环境并执行全部教学：

```bash
python3 examples/tutorials/paper_case/run.py
```

现场交互使用 `python3 examples/tutorials/paper_case/run.py lab`。
首次安装依赖需要联网；已安装后加 `--offline` 可直接复用本地环境。
