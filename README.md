<div align="center">
  <img src="https://raw.githubusercontent.com/QuantumMisaka/dpeva/main/docs/img/DP-EVA-Logo-0312.png" alt="DP-EVA Logo" width="100%">
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

*   **Active Learning Loop**: Independent workflows for training, inference, feature generation, collection, and labeling.
*   **Data Collection**: Collects data from target data pool based on DPA fine-tuning and descriptor.
    *   **2D Shallow UQ**: Supports 2-Dimensional Uncertainty Quantification (UQ) with Query by Committee (QbC) and an RND-inspired deviation from a pretrained-initialized baseline in the shallow ensemble (fitting-net) of a DPA model.
    *   **Smart Sampling**: Use DIRECT sampling based on DPA descriptor to select the most representative configurations in target data pool.
*   **HPC Ready**: Built-in `JobManager` supports seamless switching between Local and Slurm backends, with optimized task packing for massive labeling jobs.
*   **User and Agent Friendly**: Provides a unified CLI `dpeva` for all elementary workflows to both user and AI agent.
*   **Labeling Automation**: Integrated DFT workflow (ABACUS) for automatic input generation, error correction, and data cleaning.

## Getting Started

Choose an entry point for your task:

| Goal | Start here | Requirements |
|---|---|---|
| Learn the paper's methods | [Paper case tutorial](https://github.com/QuantumMisaka/dpeva/blob/main/examples/tutorials/paper_case/README.md): four Chinese notebooks for a 20–30 minute session | CPU, Python 3.10–3.13; all 30,656 structures and precomputed features are included |
| Run actual model workflows | [Installation](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/installation.md), then [Quickstart](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/quickstart.md) | DeepMD-kit with a matching PyTorch backend; repository model and small datasets |
| Develop or build documentation | [Contributing](https://github.com/QuantumMisaka/dpeva/blob/main/CONTRIBUTING.md) | Source checkout with the `dev` / `docs` extras |

For the CPU tutorial, clone the complete repository and run:

```bash
git clone https://github.com/QuantumMisaka/dpeva.git
cd dpeva
python3 examples/tutorials/paper_case/run.py
```

The launcher creates a dedicated environment and executes all four notebooks.
Append `lab` to the same command to open JupyterLab; add `--offline` after the first installation.
The tutorial needs no GPU, DeepMD runtime or model download. It recomputes data
analysis and selection; model accuracy and application results come from the paper.

For training, inference and feature generation, follow the full installation
guide. The default package includes `deepmd-kit>=3.1.2,<3.3`; the `deepmd` extra
selects `>=3.2,<3.3`. Choose CPU/GPU PyTorch separately as described in the guide.
Tutorial data and notebooks are distributed through Git and excluded from PyPI
packages. The Quickstart model also requires the complete repository checkout.

For cluster execution, see the [SAI guide](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/sai-quickstart.md) and
[Slurm guide](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/slurm.md). Reusable configurations live in
[examples/recipes](https://github.com/QuantumMisaka/dpeva/blob/main/examples/recipes/README.md).

## Documentation

The documentation is currently maintained primarily in Chinese. English
translations will be added incrementally toward bilingual documentation;
the repository README is maintained in English.

- **Documentation site**: [Latest documentation](https://quantummisaka.github.io/dpeva/latest/)
- **Start here**: [Documentation index](https://github.com/QuantumMisaka/dpeva/blob/main/docs/README.md) | [Installation](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/installation.md) | [Quickstart](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/quickstart.md)
- **Configure workflows**: [CLI guide](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/cli.md) | [Configuration guide](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/configuration.md) | [Reference](https://github.com/QuantumMisaka/dpeva/blob/main/docs/reference/README.md)
- **Develop and contribute**: [Developer guide](https://github.com/QuantumMisaka/dpeva/blob/main/docs/guides/developer-guide.md) | [Architecture](https://github.com/QuantumMisaka/dpeva/blob/main/docs/architecture/README.md) | [Testing](https://github.com/QuantumMisaka/dpeva/blob/main/tests/README.md)
- **Project records**: [Reports](https://github.com/QuantumMisaka/dpeva/blob/main/docs/reports/README.md) | [Archive](https://github.com/QuantumMisaka/dpeva/blob/main/docs/archive/README.md)

## Contributing

Start with [CONTRIBUTING.md](https://github.com/QuantumMisaka/dpeva/blob/main/CONTRIBUTING.md) for reporting bugs, proposing changes,
setting up a development environment, and opening your first pull request.
Questions and scientific workflow discussions belong in
[Discussions](https://github.com/QuantumMisaka/dpeva/discussions).

## Citation

If you use DP-EVA, please cite the project article:

Zhaoqing Liu, Zhe Deng, Huabo Zhao, Han Wang, Mohan Chen, Hong Jiang, DP-EVA: data-efficient fine-tuning framework via maximizing pretrained knowledge of large atomistic models for developing domain-specific machine-learning interatomic potentials, *Clean Energy*, 2026, **10**(4), 103–116, https://doi.org/10.1093/ce/zkag029.

```bibtex
@article{liu2026dpeva,
  author = {Liu, Zhaoqing and Deng, Zhe and Zhao, Huabo and Wang, Han and Chen, Mohan and Jiang, Hong},
  title = {DP-EVA: data-efficient fine-tuning framework via maximizing pretrained knowledge of large atomistic models for developing domain-specific machine-learning interatomic potentials},
  journal = {Clean Energy},
  year = {2026},
  volume = {10},
  number = {4},
  pages = {103--116},
  doi = {10.1093/ce/zkag029},
  url = {https://doi.org/10.1093/ce/zkag029},
}
```

## License

[LGPL-v3 License](https://github.com/QuantumMisaka/dpeva/blob/main/LICENCE)
