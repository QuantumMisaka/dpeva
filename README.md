# DP-EVA (Deep Potential EVolution Accelerator)

![Version](https://img.shields.io/badge/version-0.6.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/License-LGPL_v3-blue.svg)

DP-EVA is an active learning framework designed for efficient fine-tuning of DPA universal machine learning interatomic potential. It integrates uncertainty quantification (UQ), diverse sampling (DIRECT), and labeling workflows to minimize data annotation costs while maximizing model performance.

## Key Features

*   **Active Learning Loop**: Fully pipeline for Training -> Inference -> UQ -> Sampling -> Labeling.
*   **Advanced UQ**: Supports Query by Committee (QbC) and Random Network Distillation (RND) with robust "Clamp-and-Clean" numerics.
*   **Smart Sampling**: Implements 2-Step DIRECT sampling (Structure -> Atomic Environment) to select the most representative and uncertain configurations.
*   **HPC Ready**: Built-in `JobManager` supports seamless switching between Local and Slurm backends, with optimized task packing for massive labeling jobs.
*   **Labeling Automation**: (v0.5.1+) Integrated DFT workflow (ABACUS) for automatic input generation, error correction, and data cleaning.

## Quick Start

### Installation

For detailed installation instructions, please refer to the [Installation Guide](docs/guides/installation.md).

```bash
git clone https://github.com/QuantumMisaka/dpeva.git
cd dpeva
pip install -e .
```

### Usage

DP-EVA provides a unified CLI `dpeva` for all workflows. See [CLI Guide](docs/guides/cli.md) for more details.

```bash
# 1. Training
dpeva train config_train.json

# 2. Inference & UQ
dpeva infer config_infer.json

# 3. Collection (Sampling)
dpeva collect config_collect.json

# 4. Labeling (DFT)
dpeva label config_label.json
```

For detailed configuration examples, see `examples/recipes/`.

## Documentation

*   **Getting Started**: [Quickstart](docs/guides/quickstart.md) | [Installation](docs/guides/installation.md)
*   **Reference**: [Config Schema](docs/reference/config-schema.md) | [CLI](docs/guides/cli.md)
*   **Upstream**: [Upstream Software](docs/reference/upstream-software.md)
*   **Developer**: [Developer Guide](docs/guides/developer-guide.md) | [Architecture](docs/architecture/README.md)
*   **Governance**: [Policy](docs/policy/README.md) | [Roadmap](docs/plans/docs-governance-roadmap.md)

## License

LGPL-v3 License
