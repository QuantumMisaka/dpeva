<div align="center">
  <img src="docs/img/DP-EVA-Logo-0312.png" alt="DP-EVA Logo" width="100%">
  <p>
    <em>The DP-EVA identity bridges deep potential scientific modeling ('DP') and radical AI evolution ('EVA'), connected by an acceleration beam that filters order from chaos. It symbolizes the awakening of critical data from the vast chemical space to accelerate scientific discovery.</em>
  </p>
</div>

# DP-EVA (Deep Potential EVolution Accelerator)

![Version](https://img.shields.io/badge/version-0.6.8-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![License](https://img.shields.io/badge/License-LGPL_v3-blue.svg)
![Unit Test Coverage](https://img.shields.io/badge/unit%20coverage-82%25-yellowgreen)

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
```

For detailed configuration examples, see `examples/recipes/`.

## Documentation

> 📘 **Latest Documentation**
>
> The full, automatically generated documentation is available online:
> [**Latest Docs**](https://dpeva.readthedocs.io) (Placeholder Link) | [**Config Reference**](docs/source/api/config.rst)

*   **Getting Started**: [Quickstart](docs/guides/quickstart.md) | [Installation](docs/guides/installation.md)
*   **Reference**: [Configuration (SSOT)](docs/source/api/config.rst) | [CLI](docs/guides/cli.md)
*   **Upstream**: [Upstream Software](docs/reference/upstream-software.md)
*   **Developer**: [Developer Guide](docs/guides/developer-guide.md) | [Architecture](docs/architecture/README.md)
*   **Governance**: [Policy](docs/policy/README.md) | [Roadmap](docs/plans/docs-governance-roadmap.md)
*   **Review**: [Combined Code Review 2026-03-10](docs/archive/v0.6.4/reports/2026-03-10-combined-review.md) | [Remediation Summary 2026-03-11](docs/reports/2026-03-11-remediation-summary.md) | [Labeling Iteration Summary 2026-03-12](docs/reports/2026-03-12-labeling-decoupling-iteration-summary.md)

## License

LGPL-v3 License
