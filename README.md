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

## Notice

This project is still under development. Please feel free to open an issue or pull request if you have any suggestions or questions.

Especially, the detailed algorithm and way to use Random Network Distillation (RND) method is still in exploration.

## License
This project is licensed under the LGPL-v3 License - see the [LICENSE](LICENSE) file for details.
```