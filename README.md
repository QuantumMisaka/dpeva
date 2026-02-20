# DP-EVA (Deep Potential EVolution Accelerator)

[![License](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**DP-EVA** is a high-efficiency active learning framework designed for the **Deep Potential (DPA3)** ecosystem. It automates the "Finetune-Explore-Label" loop by identifying the most valuable atomic configurations from massive datasets using advanced uncertainty quantification and representativeness sampling.

## 🚀 Vision
To minimize the data annotation cost for developing general-purpose machine learning potentials while maximizing model robustness across diverse chemical spaces.

## ✨ Core Features
- **Auto-UQ**: Adaptive uncertainty thresholding based on KDE (Kernel Density Estimation) to handle shifting model distributions.
- **2-dimension of UQ**: Using UQ-QbC and UQ-RND to construct a 2-dimension uncertainty space.
- **Joint Sampling**: **DIRECT** (Dimensionality Reduction and Clustering) sampling that considers existing training data to avoid redundancy.
- **Dual-Mode Scheduling**: Seamless switching between Local (Multiprocessing) and Slurm Cluster environments.
- **Modular Design**: Decoupled modules for Training, Inference, Uncertainty, and Sampling.

## 📦 Installation

### Prerequisites
- Python >= 3.8
- DeePMD-kit (installed via their official guides), PyTorch backend is most needed.

### Install from Source
```bash
git clone https://github.com/QuantumMisaka/dpeva.git
cd dpeva
pip install -e .
```


## ⚡ Quick Start (30 Seconds)

### 1. Prepare Configuration
Create a `config.json` for the collection task:

```json
{
    "project": "./my_project",
    "desc_dir": "./descriptors",
    "testdata_dir": "./unlabeled_data",
    "uq_select_scheme": "tangent_lo",
    "uq_trust_mode": "auto",
    "sampler_type": "direct",
    "direct_n_clusters": 100,
    "direct_k": 1
}
```

### 2. Run Collection
Execute the CLI command:

```bash
dpeva collect config.json
```

### 3. Check Results
Selected structures will be exported to `my_project/dpeva_uq_post/dpdata`.

## 🛠️ Usage Guide

### Training
```bash
dpeva train config_train.json
```

### Inference & Analysis
```bash
dpeva infer config_test.json
```

## 🧪 Advanced Usage (Python API)

For complex workflows requiring dynamic configuration or custom logic, you can use the Python API directly.
See `examples/recipes/` for template scripts.

Example:
```python
from dpeva.workflows.train import TrainingWorkflow
# Load or generate config dict
config = {...} 
workflow = TrainingWorkflow(config)
workflow.run()
```

## 🤝 Contribution
We welcome contributions! Please follow these steps:
1.  **Fork** the repository.
2.  **Create a branch** for your feature/fix.
3.  **Install dev dependencies**: `pip install -e ".[dev]"`
4.  **Run unit tests**: `pytest tests/unit`
5.  **Submit a PR**.

## 📄 License
This project is licensed under the **LGPL-v3 License**. See [LICENSE](LICENSE) for details.
