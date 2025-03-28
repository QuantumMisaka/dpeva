# DP-EVA
Deep Potential EVolution Accelerator

## Target:
Using **single model** cuncurrent learning method to accelerate the evolution of deep potential (and other machine learning interatomic potentials).

Methods used in this project:
- Data sampling baesd on encoder space:
- - DIRECT (from [maml](https://github.com/materialsvirtuallab/maml) package)
- - 2-DIRECT and atomic-DIECT (usage in notebook)
- Uncertainty estimation on atomic force evaluation in double variables:
- - Query-by-committee uncertainty
- - Random-Network-Distillation-like uncertainty

## Installation:

Note: the src libraries only implemented DIRECT method now.

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

## License
This project is licensed under the LGPL-v3 License - see the [LICENSE](LICENSE) file for details.
```