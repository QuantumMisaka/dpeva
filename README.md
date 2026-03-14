<div align="center">
  <img src="docs/img/DP-EVA-Logo-0312.png" alt="DP-EVA Logo" width="100%">
  <p>
    <em>The DP-EVA identity bridges rigorous physical modeling ('DP') and radical AI evolution ('EVA'), connected by an acceleration beam that filters order from chaos. It symbolizes the awakening of critical data from the vast chemical space to accelerate scientific discovery.</em>
  </p>
</div>

# DP-EVA (Deep Potential EVolution Accelerator)

![Version](https://img.shields.io/badge/version-0.6.7-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![License](https://img.shields.io/badge/License-LGPL_v3-blue.svg)
![Unit Test Coverage](https://img.shields.io/badge/unit%20coverage-82%25-yellowgreen)
[![Code Review](https://img.shields.io/badge/review-2026--03--13-important)](docs/archive/v0.6.6/reports/2026-03-13-Code-Review-Collection-PCA-Background.md)

DP-EVA is an active learning framework designed for efficient fine-tuning of DPA universal machine learning interatomic potential. It integrates uncertainty quantification (UQ), diverse sampling (DIRECT), and labeling workflows to minimize data annotation costs while maximizing model performance.

## Key Features

*   **Active Learning Loop**: Fully pipeline for Training -> Inference -> UQ -> Sampling -> Labeling.
*   **Advanced UQ**: Supports Query by Committee (QbC) and Random Network Distillation (RND) with robust "Clamp-and-Clean" numerics.
*   **Smart Sampling**: Implements DIRECT sampling to select the most representative configurations in target data pool.
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
