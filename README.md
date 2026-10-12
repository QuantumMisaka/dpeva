<div align="center">
  <img src="docs/img/DP-EVA-Logo-0312.png" alt="DP-EVA Logo" width="100%">
  <p>
    <em>The DP-EVA identity bridges deep potential scientific modeling ('DP') and radical AI evolution ('EVA'), connected by an acceleration beam that filters order from chaos. It symbolizes the awakening of critical data from the vast chemical space to accelerate scientific discovery.</em>
  </p>
</div>

# DP-EVA (Deep Potential EVolution Accelerator)

![Version](https://img.shields.io/badge/version-0.8.2-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![License](https://img.shields.io/badge/License-LGPL_v3-blue.svg)
![Unit Test Coverage](https://img.shields.io/badge/unit%20coverage-84%25-yellowgreen)

DP-EVA is an active learning framework designed for efficient fine-tuning of DPA universal machine learning interatomic potential. It integrates uncertainty quantification (UQ), representative sampling (DIRECT), and automated DFT labeling workflows to minimize data annotation costs while maximizing model performance via fully unraveling the knowledge of DPA pre-trained PES model.

## Key Features

*   **Active Learning Loop**: Fully pipeline for Training -> Inference -> Collection -> Labeling.
*   **Data Collection**: Collects data from target data pool based on DPA fine-tuning and descriptor.
    *   **2D Shallow UQ**: Supports 2-Dimensional Uncertainty Quantification (UQ) with Query by Committee (QbC) and Random Network Distillation (RND) in the shallow layer (fitting-net) of DPA model.
    *   **Smart Sampling**: Use DIRECT sampling based on DPA descriptor to select the most representative configurations in target data pool.
*   **HPC Ready**: Built-in `JobManager` supports seamless switching between Local and Slurm backends, with optimized task packing for massive labeling jobs.
*   **User and Agent Friendly**: Provides a unified CLI `dpeva` for all elementary workflows to both user and AI agent.
*   **Labeling Automation**: Integrated DFT workflow (ABACUS) for automatic input generation, error correction, and data cleaning.

## Quick Start

### Installation

For detailed installation instructions, please refer to the [Installation Guide](docs/guides/installation.md).

```bash
git clone https://github.com/QuantumMisaka/dpeva.git
cd dpeva
pip install -e .
```

默认安装包含有界的 DeepMD-kit 运行时依赖 `deepmd-kit>=3.1.2,<3.3`，以保留
现有 3.1.2 运行环境的兼容边界。它不等同于新的科学能力验证；需要明确使用
3.2 能力 lane 时，安装用户运行时 extra：

```bash
pip install -e '.[deepmd]'
```

该 extra 的依赖范围为 `deepmd-kit>=3.2,<3.3`；能力矩阵仍严格只记录已验证的
3.2 证据，不会把 3.1 运行时重新标记为已验证能力。研究生产环境请在独立环境
文件或锁文件中固定 `deepmd-kit==3.2.0`，并保留 `dp --version` 证据；范围不代表
其中所有版本行为等价。

如果平台已经预置并验证了完整运行环境，可跳过依赖解析，但调用方必须自行提供
`pyproject.toml` 中的全部依赖（包括 DeepMD）：

```bash
python -m pip install --no-deps -e .
```

`--no-deps` 只表示安装器不补齐依赖，不是 DP-EVA 的无依赖默认安装；安装后请用
`dpeva doctor` 检查当前环境。

Optional trajectory exploration through `atst-tools` is installed separately:

```bash
pip install -e '.[explore]'
```

### 模型与数据获取

DP-EVA 需要一个 DPA 预训练模型作为微调基座，以及用于主动学习循环的候选数据。公开资源：

| 资源 | 位置 | 说明 |
|---|---|---|
| DPA-3.1-3M 预训练模型 | [AIS Square 模型库](https://www.aissquare.com/models/343)（`aissquare.com/models`，条目 id=343） | 通用 DPA-3.x 多任务预训练模型，用作 `base_model_path`；使用时需指定 `model_head` |
| FT2DP-dataset-FeCHO-v1 数据集 | [AIS Square 数据集](https://www.aissquare.com/datasets/306)（`aissquare.com/datasets`，条目 id=306） | 铁基 FTS 场景数据集，可作为训练集/候选池的使用示例 |

只想先确认环境可用时不必下载任何外部资源：仓库自带的
`tests/integration/data/DPA-3.1-3M.pt` 与几帧 dpdata 夹具即可跑通最小闭环，见
[Smoke Demo](docs/guides/smoke-demo.md)。

外部模型可用环境变量覆盖默认位置：`DPEVA_DPA3_MODEL_PATH`、`DPEVA_DPA4_MODEL_PATH`；
夹具说明与目录约定见 [tests/integration/data/README.md](tests/integration/data/README.md)。

### Usage

DP-EVA provides a unified CLI `dpeva` for all workflows. See [CLI Guide](docs/guides/cli.md) for more details.

```bash
# 1. Training
dpeva train config_train.json

# 2. Inference & UQ
dpeva infer config_infer.json

# 3. Feature Generation
dpeva feature config_feature.json

# 4. Collection (Sampling)
dpeva collect config_collect.json

# 5. Labeling (DFT)
dpeva label config_label.json

# 6. Analysis
dpeva analysis config_analyze.json

# Optional: trajectory exploration backend
cd examples/recipes/exploration
dpeva explore config_explore_md.json
```

For detailed configuration examples, see `examples/recipes/`.

## Documentation

> 📘 **Latest Documentation**
>
> Docs site: [**quantummisaka.github.io/dpeva/latest**](https://quantummisaka.github.io/dpeva/latest/)
>
> Repository index:
> [**Docs Index**](docs/README.md) | [**Guides**](docs/guides/README.md) | [**Reports**](docs/reports/README.md)

*   **Getting Started**: [Quickstart](docs/guides/quickstart.md) | [Installation](docs/guides/installation.md)
*   **Onboarding**: [Smoke Demo（自带夹具）](docs/guides/smoke-demo.md) | [SAI Quickstart](docs/guides/sai-quickstart.md)
*   **Reference**: [Reference Index](docs/reference/README.md) | [CLI](docs/guides/cli.md)
*   **Upstream**: [Upstream Software](docs/reference/upstream-software.md)
*   **Developer**: [Developer Guide](docs/guides/developer-guide.md) | [Architecture](docs/architecture/README.md)
*   **Governance**: [Policy](docs/policy/README.md) | [Plans](docs/plans/README.md)
*   **Review**: [Repository Audit 2026-04-01](docs/archive/v0.8.0/reports/2026-04-01-Code-Review-Repository-Audit.md) | [Reports Index](docs/reports/README.md) | [Archive Index](docs/archive/README.md)

## Citation

If you use DP-EVA, please cite the project article:

Zhaoqing Liu, Zhe Deng, Huabo Zhao, Han Wang, Mohan Chen, Hong Jiang, DP-EVA: Data-Efficient Fine-Tuning Framework via Maximizing Pre-Trained Knowledge of Large Atomistic Models for Developing Domain-Specific Machine Learning Interatomic Potentials, *Clean Energy*, 2026, zkag029, https://doi.org/10.1093/ce/zkag029.

```bibtex
@article{liu2026dpeva,
  author = {Liu, Zhaoqing and Deng, Zhe and Zhao, Huabo and Wang, Han and Chen, Mohan and Jiang, Hong},
  title = {DP-EVA: Data-Efficient Fine-Tuning Framework via Maximizing Pre-Trained Knowledge of Large Atomistic Models for Developing Domain-Specific Machine Learning Interatomic Potentials},
  journal = {Clean Energy},
  year = {2026},
  pages = {zkag029},
  doi = {10.1093/ce/zkag029},
  url = {https://doi.org/10.1093/ce/zkag029},
}
```

## License

LGPL-v3 License
