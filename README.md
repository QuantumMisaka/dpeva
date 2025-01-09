# DP-EVA
Deep Potential EVolution Accelerator

## Target:
Using **single model** cuncurrent learning method to accelerate the evolution of deep potential (and other machine learning interatomic potentials).

Methods used in this project:
- Data sampling:
- - DIRECT (from [maml](https://github.com/materialsvirtuallab/maml) package)
- - modified DIRECT (to be implemented)
- Uncertainty estimation based on descriptor space:
- - Random Network Distillation (RND)
- - Gaussian Mixture Model (GMM) (to be implemented)

## Installation:

Install the package via pip:
```bash
pip install git+https://github.com/quantummisaka/dpeva.git
```

Or clone the repository and install the package:
```bash
pip install .
```

## Usage
See example directory for usage examples.

More notebook and scripts will be added soon.