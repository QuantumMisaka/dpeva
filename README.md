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

*   **Active Learning Loop**: Independent workflows for training, inference, feature generation, collection, and labeling.
*   **Data Collection**: Collects data from target data pool based on DPA fine-tuning and descriptor.
    *   **2D Shallow UQ**: Supports 2-Dimensional Uncertainty Quantification (UQ) with Query by Committee (QbC) and Random Network Distillation (RND) in the shallow layer (fitting-net) of DPA model.
    *   **Smart Sampling**: Use DIRECT sampling based on DPA descriptor to select the most representative configurations in target data pool.
*   **HPC Ready**: Built-in `JobManager` supports seamless switching between Local and Slurm backends, with optimized task packing for massive labeling jobs.
*   **User and Agent Friendly**: Provides a unified CLI `dpeva` for all elementary workflows to both user and AI agent.
*   **Labeling Automation**: Integrated DFT workflow (ABACUS) for automatic input generation, error correction, and data cleaning.

## Getting Started

Start with the [installation guide](docs/guides/installation.md), then follow
[Quickstart](docs/guides/quickstart.md) to run a local four-stage workflow using
the model and small datasets included in the repository. No external data
download is needed for this example.

For cluster execution, see the [SAI guide](docs/guides/sai-quickstart.md) and
[Slurm guide](docs/guides/slurm.md). Reusable configurations live in
[examples/recipes](examples/recipes/README.md).

## Documentation

The documentation is currently maintained primarily in Chinese. English
translations will be added incrementally toward bilingual documentation;
the repository README is maintained in English.

- **Documentation site**: [Latest documentation](https://quantummisaka.github.io/dpeva/latest/)
- **Start here**: [Documentation index](docs/README.md) | [Installation](docs/guides/installation.md) | [Quickstart](docs/guides/quickstart.md)
- **Configure workflows**: [CLI guide](docs/guides/cli.md) | [Configuration guide](docs/guides/configuration.md) | [Reference](docs/reference/README.md)
- **Develop and contribute**: [Developer guide](docs/guides/developer-guide.md) | [Architecture](docs/architecture/README.md) | [Testing](tests/README.md)
- **Project records**: [Reports](docs/reports/README.md) | [Archive](docs/archive/README.md)

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md) for reporting bugs, proposing changes,
setting up a development environment, and opening your first pull request.
Questions and scientific workflow discussions belong in
[Discussions](https://github.com/QuantumMisaka/dpeva/discussions).

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

[LGPL-v3 License](LICENCE)
